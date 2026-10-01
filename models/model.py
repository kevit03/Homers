"""GPT-style transformer over basketball events, with a win-probability head.

Input at each position = event token embedding + position embedding
                         + projection of game-state features (time, score diff, period, pre-game Elo).
Outputs: next-event logits and a home-win logit at every position.

With wp_residual the win-probability head is a correction on top of a logistic baseline's logit
(score diff + time left, plus the pre-game Elo when the model has it), zero-initialized, so training starts from the baseline and every learned
effect reads as a shift away from it. wp_temp is a temperature fitted on the validation season.
"""
import math
import pickle
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

N_FEAT = 6  # a checkpoint may use fewer: Config.n_feat takes the first n


def make_features(sec, diff, period, elo=0.0):
    """elo: the home team's pre-game edge as a logit (team_strength.py); it matters less as the clock runs."""
    minutes = np.maximum(sec, 0) / 60.0
    frac_left = np.clip(sec / 2880.0, 0, 1) * (period <= 4)
    return np.stack([sec / 2880.0, diff / 20.0, np.minimum(period, 6) / 4.0,
                     diff / np.sqrt(minutes + 1.0) / 5.0,  # lead relative to time left
                     np.full_like(sec, elo, dtype=np.float32), elo * frac_left],
                    axis=-1).astype(np.float32)


def load_data(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


class GameDataset(Dataset):
    def __init__(self, games, block_size):
        self.games, self.block_size = games, block_size

    def __len__(self):
        return len(self.games)

    def __getitem__(self, i):
        g, T = self.games[i], self.block_size
        return (torch.from_numpy(g["tokens"][:T].astype(np.int64)),
                torch.from_numpy(make_features(g["sec"][:T], g["diff"][:T], g["period"][:T], g.get("elo", 0.0))),
                float(g["home_win"]))


def collate(batch):
    T = max(len(t) for t, _, _ in batch)
    B = len(batch)
    tokens = torch.zeros(B, T, dtype=torch.long)  # 0 = <pad>
    feats = torch.zeros(B, T, N_FEAT)
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, (t, f, _) in enumerate(batch):
        tokens[i, :len(t)], feats[i, :len(t)], mask[i, :len(t)] = t, f, True
    y = torch.tensor([w for _, _, w in batch])
    return tokens, feats, mask, y


@dataclass
class Config:
    vocab_size: int
    block_size: int = 1024
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    n_feat: int = 4             # 4 = no Elo (older checkpoints), 6 = with pre-game Elo
    wp_residual: bool = False   # win-prob head adds to the baseline logit below
    base_coef: tuple = None     # logistic baseline (b, w_diff, w_diff/sqrt(min+1), w_min[, w_elo, w_elo_fading])
    wp_temp: float = 1.0        # win-prob logit is divided by this (temperature scaling)


class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)
        self.proj = nn.Linear(c.n_embd, c.n_embd)
        self.n_head, self.dropout = c.n_head, c.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        # Causal mask is enough: padding sits at the end, so real tokens never attend to it.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.n_embd), nn.LayerNorm(c.n_embd)
        self.attn = Attention(c)
        self.mlp = nn.Sequential(nn.Linear(c.n_embd, 4 * c.n_embd), nn.GELU(),
                                 nn.Linear(4 * c.n_embd, c.n_embd), nn.Dropout(c.dropout))
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        return x + self.mlp(self.ln2(x))


class Tempo(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.config = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos_emb = nn.Embedding(c.block_size, c.n_embd)
        self.feat_proj = nn.Linear(c.n_feat, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying
        self.wp_head = nn.Linear(c.n_embd, 1)
        self.apply(self._init)
        for n, p in self.named_parameters():  # GPT-2 style scaled init on residual projections
            if n.endswith("proj.weight") or n.endswith("mlp.2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * c.n_layer))
        if c.wp_residual:
            nn.init.zeros_(self.wp_head.weight)  # start exactly at the baseline

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def n_params(self):
        return sum(p.numel() for p in self.parameters()) - self.pos_emb.weight.numel()

    def base_logit(self, feats):
        """The logistic baseline's logit, recovered from the features: diff = 20 f1, diff/sqrt(min+1) = 5 f3, min = 48 f0."""
        b, w_diff, w_rel, w_min, *w_elo = self.config.base_coef
        z = b + w_diff * 20 * feats[..., 1] + w_rel * 5 * feats[..., 3] + w_min * 48 * feats[..., 0]
        if w_elo:  # the baseline with Elo (baseline.fit_elo_baseline): + w_elo * elo + w_fade * elo * share of game left
            z = z + w_elo[0] * feats[..., 4] + w_elo[1] * feats[..., 5]
        return z

    def forward(self, tokens, feats):
        T = tokens.size(1)
        pos = torch.arange(T, device=tokens.device)
        f = feats[..., :self.config.n_feat]
        x = self.drop(self.tok_emb(tokens) + self.pos_emb(pos) + self.feat_proj(f))
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        wp = self.wp_head(x).squeeze(-1)
        if self.config.wp_residual:
            wp = wp + self.base_logit(feats)
        return self.lm_head(x), wp / self.config.wp_temp


def wp_weights(feats, mask, late):
    """Win-prob loss weights: every game counts the same whatever its length, and within a game a play
    counts 1 + late * (share of regulation elapsed), so the late game, where the score says the most, pulls harder."""
    w = (1 + late * (1 - feats[..., 0].clamp(0, 1))) * mask
    return w / w.sum(1, keepdim=True) * mask.sum() / mask.size(0)


def compute_losses(model, tokens, feats, mask, y, wp_late=None):
    """wp_late=None: plain mean BCE over positions (what evaluation reports); a number: wp_weights()."""
    logits, wp = model(tokens, feats)
    tgt_mask = mask[:, 1:]
    ce = F.cross_entropy(logits[:, :-1][tgt_mask], tokens[:, 1:][tgt_mask])
    wp_target = y[:, None].expand_as(wp)
    if wp_late is None:
        bce = F.binary_cross_entropy_with_logits(wp[mask], wp_target[mask])
    else:
        w = wp_weights(feats, mask, wp_late)
        bce = (F.binary_cross_entropy_with_logits(wp, wp_target, reduction="none") * w)[mask].sum() / mask.sum()
    acc = (logits[:, :-1].argmax(-1)[tgt_mask] == tokens[:, 1:][tgt_mask]).float().mean()
    return ce, bce, acc, logits, wp
