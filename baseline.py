"""Logistic-regression win-probability baseline (score diff + time remaining).

The transformer has to beat this to be interesting.

    python baseline.py --data data/processed/games.pkl
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from model import load_data


def baseline_features(sec, diff):
    minutes = np.maximum(sec, 0) / 60.0
    return np.stack([diff, diff / np.sqrt(minutes + 1.0), minutes], axis=-1)


def stack_games(games, max_rows=None, seed=0):
    X = np.concatenate([baseline_features(g["sec"], g["diff"]) for g in games])
    y = np.concatenate([np.full(len(g["sec"]), g["home_win"]) for g in games])
    if max_rows and len(y) > max_rows:
        idx = np.random.default_rng(seed).choice(len(y), max_rows, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/games.pkl")
    ap.add_argument("--out", default="runs/baseline.joblib")
    args = ap.parse_args()

    d = load_data(args.data)
    games = d["games"]
    Xtr, ytr = stack_games([games[i] for i in d["splits"]["train"]], max_rows=2_000_000)
    Xte, yte = stack_games([games[i] for i in d["splits"]["test"]])

    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    print(f"baseline test Brier: {np.mean((p - yte) ** 2):.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.out)


if __name__ == "__main__":
    main()
