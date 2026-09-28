"""Evaluate a trained model on the test split against the baseline.

Outputs (in --out): metrics.json, calibration.png, game_<id>.png

    python evaluate.py --ckpt runs/base/best.pt
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from baseline import baseline_features
from model import Config, GameDataset, NBAGPT, collate, load_data


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def calibration(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return [(float(p[idx == b].mean()), float(y[idx == b].mean()), int((idx == b).sum()))
            for b in range(bins) if (idx == b).any()]


@torch.no_grad()
def predict_games(model, games, block_size, device, bs=16):
    model.eval()
    ds = GameDataset(games, block_size)
    wps, correct, nll, n_tok = [], 0, 0.0, 0
    for s in range(0, len(ds), bs):
        tokens, feats, mask, _ = collate([ds[i] for i in range(s, min(s + bs, len(ds)))])
        tokens, feats, mask = tokens.to(device), feats.to(device), mask.to(device)
        logits, wp = model(tokens, feats)
        tm = mask[:, 1:]
        lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, tokens[:, 1:, None]).squeeze(-1)
        nll += -lp[tm].sum().item(); n_tok += tm.sum().item()
        correct += (logits[:, :-1].argmax(-1) == tokens[:, 1:])[tm].sum().item()
        prob = torch.sigmoid(wp).cpu().numpy()
        for i, L in enumerate(mask.sum(1).tolist()):
            wps.append(prob[i, :L])
    return wps, correct / n_tok, nll / n_tok


def plot_game(g, p_model, p_base, path):
    L = len(p_model)
    elapsed = np.array([2880 - s if pr <= 4 else 2880 + 300 * (pr - 4) - s
                        for s, pr in zip(g["sec"][:L], g["period"][:L])]) / 60
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(elapsed, p_model, label="NBAGPT")
    ax[0].plot(elapsed, p_base, label="baseline", alpha=0.7)
    ax[0].axhline(0.5, color="gray", lw=0.5)
    ax[0].set_ylabel("P(home win)"); ax[0].set_ylim(0, 1); ax[0].legend()
    ax[0].set_title(f"Game {g['gameId']} ({g['season']}) — home {'won' if g['home_win'] else 'lost'}")
    ax[1].plot(elapsed, g["diff"][:L], color="black")
    ax[1].axhline(0, color="gray", lw=0.5)
    ax[1].set_ylabel("home − away"); ax[1].set_xlabel("minutes elapsed")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base/best.pt")
    ap.add_argument("--data", default="data/processed/games.pkl")
    ap.add_argument("--baseline", default="runs/baseline.joblib")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n_game_plots", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location=device)
    cfg = Config(**ck["config"])
    model = NBAGPT(cfg).to(device)
    model.load_state_dict(ck["model"])

    d = load_data(args.data)
    games = [d["games"][i] for i in d["splits"]["test"]]
    wps, acc, nll = predict_games(model, games, cfg.block_size, device)
    clf = joblib.load(args.baseline)

    p_m, p_b, y, sec_all = [], [], [], []
    for g, w in zip(games, wps):
        L = len(w)
        p_m.append(w)
        p_b.append(clf.predict_proba(baseline_features(g["sec"][:L], g["diff"][:L]))[:, 1])
        y.append(np.full(L, g["home_win"]))
        sec_all.append(np.where(g["period"][:L] <= 4, g["sec"][:L], 0))
    p_m, p_b, y, sec_all = map(np.concatenate, (p_m, p_b, y, sec_all))

    # Brier by game phase (regulation quarter; OT counted with Q4)
    phase = np.clip(4 - (sec_all // 720).astype(int), 1, 4)
    by_q = {f"Q{q}": {"model": brier(p_m[phase == q], y[phase == q]),
                      "baseline": brier(p_b[phase == q], y[phase == q])} for q in range(1, 5)}

    metrics = {
        "test_games": len(games),
        "next_event_acc": acc,
        "next_event_perplexity": float(np.exp(nll)),
        "brier_model": brier(p_m, y),
        "brier_baseline": brier(p_b, y),
        "brier_constant": brier(np.full_like(y, y.mean(), dtype=float), y),
        "brier_by_quarter": by_q,
        "calibration_model": calibration(p_m, y),
    }
    json.dump(metrics, open(out / "metrics.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in metrics.items() if k != "calibration_model"}, indent=1))

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, p in (("NBAGPT", p_m), ("baseline", p_b)):
        c = calibration(p, y)
        ax.plot([a for a, _, _ in c], [b for _, b, _ in c], "o-", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("predicted P(home win)"); ax.set_ylabel("actual home win rate")
    ax.set_title("Calibration (test set)"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "calibration.png", dpi=120); plt.close(fig)

    rng = np.random.default_rng(0)
    for i in rng.choice(len(games), min(args.n_game_plots, len(games)), replace=False):
        g, w = games[i], wps[i]
        plot_game(g, w, clf.predict_proba(baseline_features(g["sec"][:len(w)], g["diff"][:len(w)]))[:, 1],
                  out / f"game_{g['gameId']}.png")
    print(f"saved plots to {out}/")


if __name__ == "__main__":
    main()
