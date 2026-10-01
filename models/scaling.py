"""Train several model sizes and plot validation loss vs. parameter count.

    python -m models.scaling --epochs 15
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import train
SIZES = [  # (n_layer, n_head, n_embd)
    (1, 2, 32),
    (2, 2, 64),
    (4, 4, 128),
    (6, 6, 192),
    (8, 8, 256),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/games.pkl")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default="runs/scaling")
    args = ap.parse_args()

    results = []
    for L, H, E in SIZES:
        run = f"{args.out}/L{L}_E{E}"
        print(f"\n=== {run} ===")
        log = train.main(["--data", args.data, "--out", run, "--n_layer", str(L), "--n_head", str(H),
                          "--n_embd", str(E), "--epochs", str(args.epochs)])
        results.append({"n_params": log["n_params"], **log["best"], "run": run})

    Path(args.out).mkdir(parents=True, exist_ok=True)
    json.dump(results, open(f"{args.out}/scaling.json", "w"), indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    n = [r["n_params"] for r in results]
    ax[0].plot(n, [r["val_ce"] for r in results], "o-"); ax[0].set_ylabel("val next-event loss")
    ax[1].plot(n, [r["val_wp_bce"] for r in results], "o-"); ax[1].set_ylabel("val win-prob BCE")
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("parameters")
    fig.suptitle("Scaling: model size vs. validation loss")
    fig.tight_layout(); fig.savefig(f"{args.out}/scaling.png", dpi=120)
    print(f"saved {args.out}/scaling.png")


if __name__ == "__main__":
    main()
