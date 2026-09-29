"""Player / defender / coach shot model.

For every field-goal attempt it predicts
  - the shot zone   (layup, dunk, floater, hook, short mid, long mid, corner 3, above-the-break 3)
  - the shot style  (standard, pull-up, step-back, fadeaway, driving, cutting, putback, alley-oop, running)
  - make probability given the zone
from learned embeddings of the shooter, his four teammates, the likely primary defender, the five
defenders on the floor, both head coaches, both teams' Synergy play-type profiles, and the game state.

Each shooter starts from his own (smoothed) shot history: the zone logits and make logits are
that prior plus a learned adjustment from the context. So the model can never do worse than
"he shoots what he usually shoots", and every learned effect reads as a shift away from it.

The network is a small MLP on purpose: the dashboard runs it in the browser.

    python shot_model.py --shots data/processed/shots.parquet --out runs/shots
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from build_shots import STYLES, ZONES
from fetch_context import PLAY_TYPES
from tokenize_pbp import chrono_games

N_STATE = 5  # sec, margin, period, is_home, defender confidence


# ---------------------------------------------------------------- vocabularies & features

def build_vocab(ids, min_count):
    vc = pd.Series(ids).value_counts()
    keep = sorted(int(i) for i in vc[vc >= min_count].index if i > 0)
    return {pid: k + 2 for k, pid in enumerate(keep)}  # 0 = none, 1 = rare / unknown


def lookup(vocab, pid):
    return vocab.get(int(pid), 1) if pid and pid > 0 else 0


def team_profiles(ctx):
    """(season, teamId) -> standardized offensive / defensive play-type frequency vectors."""
    path = Path(ctx) / "playtypes.parquet"
    if not path.exists():
        return {}, {}, None
    pt = pd.read_parquet(path)
    pt = pt[pt["level"] == "T"]
    out = {}
    for grouping in ("offensive", "defensive"):
        piv = (pt[pt["grouping"] == grouping].pivot_table(index=["season", "TEAM_ID"], columns="PLAY_TYPE",
                                                          values="POSS_PCT").reindex(columns=PLAY_TYPES).fillna(0))
        mu, sd = piv.mean(), piv.std().replace(0, 1)
        z = (piv - mu) / sd
        out[grouping] = {(s, int(t)): row.to_numpy(np.float32) for (s, t), row in z.iterrows()}
        out[grouping + "_stats"] = (mu.to_list(), sd.to_list())
    return out["offensive"], out["defensive"], {"off": out["offensive_stats"], "def": out["defensive_stats"]}


def state_features(sec, margin, period, is_home, conf):
    return np.stack([np.asarray(sec) / 2880.0, np.clip(margin, -30, 30) / 20.0,
                     np.minimum(period, 5) / 4.0, is_home, conf], -1).astype(np.float32)


def encode(shots, vocabs, prof_off, prof_def):
    off_v, def_v, coach_v = vocabs["off"], vocabs["def"], vocabs["coach"]
    zi = {z: i for i, z in enumerate(ZONES)}
    si = {s: i for i, s in enumerate(STYLES)}
    n = len(shots)
    mates = np.zeros((n, 4), np.int64)
    defs = np.zeros((n, 5), np.int64)
    for k, (sid, o5, d5) in enumerate(zip(shots["shooterId"], shots["off5"], shots["def5"])):
        m = [lookup(off_v, p) for p in o5 if p != sid][:4]
        mates[k, :len(m)] = m
        d = [lookup(def_v, p) for p in d5][:5]
        defs[k, :len(d)] = d
    zero = np.zeros(len(PLAY_TYPES), np.float32)
    return {
        "shooter": torch.tensor([lookup(off_v, p) for p in shots["shooterId"]]),
        "mates": torch.from_numpy(mates),
        "defender": torch.tensor([lookup(def_v, p) for p in shots["defenderId"]]),
        "defs": torch.from_numpy(defs),
        "coach_off": torch.tensor([coach_v.get(int(c), 1) if c else 0 for c in shots["offCoachId"]]),
        "coach_def": torch.tensor([coach_v.get(int(c), 1) if c else 0 for c in shots["defCoachId"]]),
        "prof_off": torch.from_numpy(np.stack([prof_off.get((s, int(t)), zero) for s, t in zip(shots["season"], shots["offTeamId"])])),
        "prof_def": torch.from_numpy(np.stack([prof_def.get((s, int(t)), zero) for s, t in zip(shots["season"], shots["defTeamId"])])),
        "state": torch.from_numpy(state_features(shots["sec"].to_numpy(), shots["margin"].to_numpy(),
                                                 shots["period"].to_numpy(), shots["is_home"].to_numpy(),
                                                 shots["defenderConf"].to_numpy())),
        "zone": torch.tensor([zi[z] for z in shots["zone"]]),
        "style": torch.tensor([si[s] for s in shots["style"]]),
        "made": torch.tensor(shots["made"].to_numpy(), dtype=torch.float32),
    }


# ---------------------------------------------------------------- model

class ShotNet(nn.Module):
    def __init__(self, n_off, n_def, n_coach, prior_zone, prior_make, d_player=24, d_coach=12, hidden=128,
                 dropout=0.1):
        super().__init__()
        self.register_buffer("prior_zone", torch.as_tensor(prior_zone, dtype=torch.float32))  # [n_off, Z] log-probs
        self.register_buffer("prior_make", torch.as_tensor(prior_make, dtype=torch.float32))  # [n_off, Z] logits
        self.off_emb = nn.Embedding(n_off, d_player, padding_idx=0)
        self.def_emb = nn.Embedding(n_def, d_player, padding_idx=0)
        self.coach_off_emb = nn.Embedding(n_coach, d_coach, padding_idx=0)
        self.coach_def_emb = nn.Embedding(n_coach, d_coach, padding_idx=0)
        n_in = 4 * d_player + 2 * d_coach + 2 * len(PLAY_TYPES) + N_STATE
        self.mlp = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.zone_head = nn.Linear(hidden, len(ZONES))
        self.style_head = nn.Linear(hidden, len(STYLES))
        self.make_head = nn.Linear(hidden + len(ZONES), 1)
        for e in (self.off_emb, self.def_emb, self.coach_off_emb, self.coach_def_emb):
            nn.init.normal_(e.weight, std=0.05)
        for head in (self.zone_head, self.make_head):  # start exactly at the player's history
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)

    @staticmethod
    def masked_mean(emb, idx):
        m = (idx > 0).float().unsqueeze(-1)
        return (emb(idx) * m).sum(1) / m.sum(1).clamp(min=1)

    def hidden(self, b):
        x = torch.cat([self.off_emb(b["shooter"]), self.masked_mean(self.off_emb, b["mates"]),
                       self.def_emb(b["defender"]), self.masked_mean(self.def_emb, b["defs"]),
                       self.coach_off_emb(b["coach_off"]), self.coach_def_emb(b["coach_def"]),
                       b["prof_off"], b["prof_def"], b["state"]], -1)
        return self.mlp(x)

    def forward(self, b):
        h = self.hidden(b)
        zone_oh = F.one_hot(b["zone"], len(ZONES)).float()
        zone = self.prior_zone[b["shooter"]] + self.zone_head(h)
        make = self.prior_make[b["shooter"], b["zone"]] + self.make_head(torch.cat([h, zone_oh], -1)).squeeze(-1)
        return zone, self.style_head(h), make


def losses(model, b):
    zl, sl, ml = model(b)
    return (F.cross_entropy(zl, b["zone"]), F.cross_entropy(sl, b["style"]),
            F.binary_cross_entropy_with_logits(ml, b["made"]), ml)


def batch(enc, idx):
    return {k: v[idx] for k, v in enc.items()}


def player_priors(train, off_vocab, alpha=20.0, beta=30.0):
    """Smoothed per-shooter zone log-shares and per-zone make logits; row 0/1 = league average."""
    Z = len(ZONES)
    league = train["zone"].value_counts(normalize=True).reindex(ZONES, fill_value=1e-4).to_numpy()
    fg = train.groupby("zone")["made"].mean().reindex(ZONES).fillna(0.4).to_numpy()
    n = len(off_vocab) + 2
    pz, pm = np.tile(league, (n, 1)), np.tile(fg, (n, 1))
    cnt = train.groupby(["shooterId", "zone"])["made"].agg(["sum", "count"])
    for pid, idx in off_vocab.items():
        if pid not in cnt.index.get_level_values(0):
            continue
        c = cnt.loc[pid].reindex(ZONES).fillna(0)
        pz[idx] = (c["count"].to_numpy() + alpha * league) / (c["count"].sum() + alpha)
        pm[idx] = (c["sum"].to_numpy() + beta * fg) / (c["count"].to_numpy() + beta)
    pm = np.clip(pm, 0.01, 0.99)
    return np.log(pz), np.log(pm / (1 - pm))


# ---------------------------------------------------------------- baselines

def baseline_scores(train, test):
    """League-average and player-history baselines (smoothed toward league rates)."""
    Z = len(ZONES)
    zi = {z: i for i, z in enumerate(ZONES)}
    league = train["zone"].map(zi).value_counts(normalize=True).reindex(range(Z), fill_value=1e-6).to_numpy()
    ty = test["zone"].map(zi).to_numpy()
    nll_league = -np.mean(np.log(league[ty]))

    counts = train.groupby("shooterId")["zone"].value_counts().unstack(fill_value=0).reindex(columns=ZONES, fill_value=0)
    alpha = 20.0
    player = (counts.to_numpy() + alpha * league) / (counts.to_numpy().sum(1, keepdims=True) + alpha)
    pmap = dict(zip(counts.index, player))
    pz = np.stack([pmap.get(s, league) for s in test["shooterId"]])
    nll_player = -np.mean(np.log(pz[np.arange(len(ty)), ty]))

    fg_league = train.groupby("zone")["made"].mean()
    p_league = test["zone"].map(fg_league).to_numpy()
    g = train.groupby(["shooterId", "zone"])["made"].agg(["sum", "count"])
    beta = 30.0
    fg_player = ((g["sum"] + beta * fg_league.reindex(g.index.get_level_values(1)).to_numpy()) / (g["count"] + beta)).to_dict()
    p_player = np.array([fg_player.get((s, z), fg_league[z]) for s, z in zip(test["shooterId"], test["zone"])])
    y = test["made"].to_numpy()
    return {"zone_nll_league": float(nll_league), "zone_nll_player": float(nll_player),
            "make_brier_league": float(np.mean((p_league - y) ** 2)),
            "make_brier_player": float(np.mean((p_player - y) ** 2)), "league_zone_share": league.tolist(),
            "league_fg": fg_league.reindex(ZONES).fillna(0).tolist()}


# ---------------------------------------------------------------- main

def split_shots(shots):
    seasons = sorted(shots["season"].unique())
    if len(seasons) > 1:
        test = shots["season"] == seasons[-1]
    else:
        games = chrono_games(shots)
        test = shots["gameId"].isin(games[-max(1, len(games) // 10):])
    rest = shots[~test]
    games = chrono_games(rest)
    val = rest["gameId"].isin(games[-max(1, len(games) // 10):])
    return rest[~val].reset_index(drop=True), rest[val].reset_index(drop=True), shots[test].reset_index(drop=True)


@torch.no_grad()
def evaluate(model, enc, bs=8192):
    model.eval()
    n = len(enc["zone"])
    tot = {"zone": 0.0, "style": 0.0, "make_bce": 0.0, "make_brier": 0.0, "zone_acc": 0.0}
    for s in range(0, n, bs):
        b = batch(enc, torch.arange(s, min(s + bs, n)))
        zl, sl, ml = model(b)
        k = len(b["zone"])
        tot["zone"] += F.cross_entropy(zl, b["zone"], reduction="sum").item()
        tot["style"] += F.cross_entropy(sl, b["style"], reduction="sum").item()
        tot["make_bce"] += F.binary_cross_entropy_with_logits(ml, b["made"], reduction="sum").item()
        tot["make_brier"] += ((torch.sigmoid(ml) - b["made"]) ** 2).sum().item()
        tot["zone_acc"] += (zl.argmax(-1) == b["zone"]).sum().item()
    model.train()
    return {k: v / n for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="data/processed/shots.parquet")
    ap.add_argument("--context", default="data/context")
    ap.add_argument("--out", default="runs/shots")
    ap.add_argument("--min_off", type=int, default=100, help="min shots for a shooter to get his own embedding")
    ap.add_argument("--min_def", type=int, default=400, help="min on-floor shots faced for a defender embedding")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    shots = pd.read_parquet(args.shots)
    train, val, test = split_shots(shots)
    vocabs = {
        "off": build_vocab(train["shooterId"], args.min_off),
        "def": build_vocab(np.concatenate(train["def5"].to_numpy()), args.min_def),
        "coach": build_vocab(pd.concat([train["offCoachId"], train["defCoachId"]]), 1),
    }
    prof_off, prof_def, prof_stats = team_profiles(args.context)
    enc = {k: encode(df, vocabs, prof_off, prof_def) for k, df in (("train", train), ("val", val), ("test", test))}
    prior_zone, prior_make = player_priors(train, vocabs["off"])
    model = ShotNet(len(vocabs["off"]) + 2, len(vocabs["def"]) + 2, len(vocabs["coach"]) + 2, prior_zone, prior_make)
    print(f"shots train {len(train):,} / val {len(val):,} / test {len(test):,} | shooters {len(vocabs['off'])}, "
          f"defenders {len(vocabs['def'])}, coaches {len(vocabs['coach'])} | play-type profiles: {bool(prof_off)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = len(train)
    steps = args.epochs * math.ceil(n / args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    history, best, bad = [], float("inf"), 0
    for ep in range(args.epochs):
        perm, run = torch.randperm(n), 0.0
        for s in range(0, n, args.batch_size):
            b = batch(enc["train"], perm[s:s + args.batch_size])
            zl, sl, ml = losses(model, b)[:3]
            loss = zl + 0.5 * sl + ml
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            run += loss.item() * len(b["zone"])
        v = evaluate(model, enc["val"])
        total = v["zone"] + 0.5 * v["style"] + v["make_bce"]
        history.append({"epoch": ep + 1, "train_loss": run / n, "val_loss": total, **{f"val_{k}": x for k, x in v.items()}})
        print(f"ep {ep + 1:2d} | train {run / n:.4f} | val zone {v['zone']:.4f} style {v['style']:.4f} "
              f"make {v['make_bce']:.4f} (brier {v['make_brier']:.4f}) zone acc {v['zone_acc']:.3f}")
        if total < best:
            best, bad = total, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break

    model.load_state_dict(torch.load(out / "best.pt"))
    torch.save(model.state_dict(), out / "best.pt")
    t = evaluate(model, enc["test"])
    base = baseline_scores(train, test)
    metrics = {"test_shots": len(test), "zone_nll": t["zone"], "style_nll": t["style"], "zone_acc": t["zone_acc"],
               "make_brier": t["make_brier"], **base}
    print(json.dumps({k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}, indent=1))

    meta = {"vocabs": {k: {str(p): i for p, i in v.items()} for k, v in vocabs.items()},
            "dims": {"n_off": len(vocabs["off"]) + 2, "n_def": len(vocabs["def"]) + 2, "n_coach": len(vocabs["coach"]) + 2},
            "profile_stats": prof_stats, "zones": ZONES, "styles": STYLES, "play_types": PLAY_TYPES,
            "metrics": metrics, "history": history, "args": vars(args),
            "seasons": {"train": sorted(train["season"].unique()), "test": sorted(test["season"].unique())}}
    json.dump(meta, open(out / "meta.json", "w"))
    print(f"saved {out}/best.pt and meta.json")


if __name__ == "__main__":
    main()
