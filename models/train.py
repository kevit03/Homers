"""Train Tempo on next-event prediction + win probability.

The win-prob head is a residual on a logistic baseline (python -m models.baseline; with Elo in the data,
the baseline refit with the pre-game Elo edge from python -m models.team_strength), and after training a temperature is fitted
on the validation season and saved in the checkpoint's config.

    python -m models.train --out runs/base
    python -m models.train --out runs/small --n_layer 2 --n_embd 64 --n_head 2
"""
import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.baseline import fit_elo_baseline
from models.model import Config, GameDataset, Tempo, collate, compute_losses, load_data


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/games.pkl")
    ap.add_argument("--out", default="runs/base")
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--block_size", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--wp_weight", type=float, default=1.0, help="weight of win-prob loss")
    ap.add_argument("--wp_late", type=float, default=1.0,
                    help="win-prob loss weight grows from 1 at tip-off to 1 + this at the end of regulation")
    ap.add_argument("--baseline", default="runs/baseline.joblib", help="logistic baseline the win-prob head corrects")
    ap.add_argument("--no_residual", action="store_true", help="win-prob head from scratch, not on top of the baseline")
    ap.add_argument("--no_elo", action="store_true", help="leave out the pre-game Elo features")
    ap.add_argument("--patience", type=int, default=4, help="early stopping epochs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    return ap.parse_args(argv)


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def fit_temperature(model, loader, device):
    """One scalar T minimizing validation BCE of sigmoid(logit / T); fitted on log T so it stays positive."""
    model.eval()
    zs, ys = [], []
    with torch.no_grad():
        for batch in loader:
            tokens, feats, mask, y = (t.to(device) for t in batch)
            _, wp = model(tokens, feats)
            zs.append(wp[mask].cpu()); ys.append(y[:, None].expand_as(wp)[mask].cpu())
    z, y = torch.cat(zs).double(), torch.cat(ys).double()
    log_t = torch.zeros(1, dtype=torch.double, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z / log_t.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    before = F.binary_cross_entropy_with_logits(z, y).item()
    after = F.binary_cross_entropy_with_logits(z / log_t.exp(), y).item()
    return float(log_t.exp()), before, after


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot = {"ce": 0.0, "wp": 0.0, "acc": 0.0}
    n = 0
    for batch in loader:
        tokens, feats, mask, y = (t.to(device) for t in batch)
        ce, bce, acc, _, _ = compute_losses(model, tokens, feats, mask, y)
        tot["ce"] += ce.item(); tot["wp"] += bce.item(); tot["acc"] += acc.item(); n += 1
    model.train()
    return {k: v / n for k, v in tot.items()}


def main(argv=None):
    args = get_args(argv)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d = load_data(args.data)
    games, splits = d["games"], d["splits"]
    mk = lambda split, shuffle: DataLoader(
        GameDataset([games[i] for i in splits[split]], args.block_size),
        batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate)
    train_dl, val_dl = mk("train", True), mk("val", False)

    has_elo = any(g.get("elo") for g in games)
    if not has_elo and not args.no_elo:
        print("no pre-game Elo in the data (run models/team_strength.py); training without it")
    base_coef = None
    if not args.no_residual:
        # with Elo, start from the logistic baseline that also sees Elo, so the plays have to add something on top of it
        clf = (fit_elo_baseline([games[i] for i in splits["train"]]) if has_elo and not args.no_elo
               else joblib.load(args.baseline))
        base_coef = (float(clf.intercept_[0]), *map(float, clf.coef_[0]))
    cfg = Config(vocab_size=len(d["vocab"]), block_size=args.block_size, n_layer=args.n_layer,
                 n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout,
                 n_feat=6 if has_elo and not args.no_elo else 4, wp_residual=not args.no_residual, base_coef=base_coef)
    model = Tempo(cfg).to(device)
    print(f"device={device} params={model.n_params():,} train_games={len(splits['train'])}")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))
    total_steps = args.epochs * len(train_dl)
    warmup = max(1, int(0.05 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))

    log = {"config": asdict(cfg), "args": vars(args), "n_params": model.n_params(), "history": []}
    best, bad_epochs, step = float("inf"), 0, 0
    for epoch in range(args.epochs):
        t0, run = time.time(), 0.0
        for batch in train_dl:
            tokens, feats, mask, y = (t.to(device) for t in batch)
            ce, bce, _, _, _ = compute_losses(model, tokens, feats, mask, y, wp_late=args.wp_late)
            loss = ce + args.wp_weight * bce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            run += loss.item()

        val = evaluate(model, val_dl, device)
        val_total = val["ce"] + args.wp_weight * val["wp"]
        rec = {"epoch": epoch + 1, "step": step, "train_loss": run / len(train_dl),
               "val_ce": val["ce"], "val_wp_bce": val["wp"], "val_acc": val["acc"]}
        log["history"].append(rec)
        print(f"ep {epoch + 1:2d} | train {rec['train_loss']:.4f} | val ce {val['ce']:.4f} "
              f"(ppl {math.exp(val['ce']):.2f}) wp {val['wp']:.4f} acc {val['acc']:.3f} | {time.time() - t0:.0f}s")

        if val_total < best:
            best, bad_epochs = val_total, 0
            log["best"] = rec
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "vocab": d["vocab"]}, out / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("early stopping")
                break
        json.dump(log, open(out / "log.json", "w"), indent=1)

    ck = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    t, before, after = fit_temperature(model, val_dl, device)
    ck["config"]["wp_temp"] = log["config"]["wp_temp"] = t
    torch.save(ck, out / "best.pt")
    log["temperature"] = {"T": t, "val_bce_before": before, "val_bce_after": after}
    print(f"temperature {t:.3f}: val win-prob BCE {before:.4f} -> {after:.4f}")

    json.dump(log, open(out / "log.json", "w"), indent=1)
    return log


if __name__ == "__main__":
    main()
