"""Does game context (models/game_context.py) improve win probability over score + clock + Elo?

Gradient-boosted trees on every play's state, one feature group added at a time. To keep from fooling ourselves:
  - groups are chosen on the validation season only: a group is kept if it lowers validation Brier and the
    game-level bootstrap puts it ahead at least KEEP_PROB of the time
  - the test season (2025-26) is scored once, for the chosen set, next to the logistic + Elo bar
  - trees are small (min MIN_LEAF plays per leaf) and monotone where the direction is known
    (a bigger lead, a better Elo, a better five on the floor can only help)

    python -m models.wp_context                      # report + runs/wp_context/{report.json, model.joblib}
"""
import argparse
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from models.baseline import elo_features, fit_elo_baseline
from models.model import load_data

BASE = ["diff", "sec", "period", "elo", "elo_fading", "diff_rel"]
BASE_MONO = [1, 0, 0, 1, 1, 1]
MIN_LEAF = 2000
KEEP_PROB = 0.9
MAX_ROWS = 2_000_000


def base_matrix(g):
    sec, diff = g["sec"], g["diff"]
    frac = np.clip(sec / 2880, 0, 1) * (g["period"] <= 4)
    return np.column_stack([diff, sec, np.minimum(g["period"], 5), np.full(len(sec), g["elo"]), g["elo"] * frac,
                            diff / np.sqrt(np.maximum(sec, 0) / 60 + 1)]).astype(np.float32)


def stack(games, ctx):
    X = np.concatenate([np.column_stack([base_matrix(g), ctx["values"][g["gameId"]]]) for g in games])
    y = np.concatenate([np.full(len(g["sec"]), g["home_win"]) for g in games]).astype(float)
    gid = np.concatenate([np.full(len(g["sec"]), i) for i, g in enumerate(games)])
    return X, y, gid


def fit_gbm(X, y, cols, mono, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), min(len(y), MAX_ROWS), replace=False)
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=MIN_LEAF,
                                          l2_regularization=1.0, monotonic_cst=mono, early_stopping=False,
                                          random_state=seed).fit(X[idx][:, cols], y[idx])


def brier_by_game(p, y, gid):
    return np.bincount(gid, weights=(p - y) ** 2), np.bincount(gid)


def compare(p, ref, y, gid, B=1000, seed=0):
    """Brier difference p - ref (negative = p better), its 95% interval and P(p better), resampling games."""
    a, n = brier_by_game(p, y, gid)
    b, _ = brier_by_game(ref, y, gid)
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, len(n), len(n)) for _ in range(B)]
    d = np.array([(a[s].sum() - b[s].sum()) / n[s].sum() for s in draws])
    return {"brier": float(np.mean((p - y) ** 2)), "delta": float(a.sum() - b.sum()) / n.sum(),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_better": float(np.mean(d < 0))}


def by_quarter(p, y, games):
    s = np.concatenate([np.where(g["period"] <= 4, g["sec"], 0) for g in games])
    q = np.clip(4 - (s // 720).astype(int), 1, 4)
    return {f"Q{k}": float(np.mean((p[q == k] - y[q == k]) ** 2)) for k in range(1, 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/games.pkl")
    ap.add_argument("--context", default="data/processed/game_context.pkl")
    ap.add_argument("--out", default="runs/wp_context")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    d = load_data(args.data)
    ctx = pickle.load(open(args.context, "rb"))
    feats, groups, mono = ctx["features"], ctx["groups"], ctx["monotone"]
    names = BASE + feats
    col = {f: i for i, f in enumerate(names)}
    G = d["games"]
    tr, va, te = ([G[i] for i in d["splits"][k] if G[i]["gameId"] in ctx["values"]] for k in ("train", "val", "test"))
    Xtr, ytr, _ = stack(tr, ctx)
    Xva, yva, gva = stack(va, ctx)
    print(f"train {len(tr)} games / {len(ytr):,} plays, val {len(va)}, test {len(te)}", flush=True)
    for f in feats:
        print(f"  {f:16s} missing {np.isnan(Xtr[:, col[f]]).mean():6.1%}  sd {np.nanstd(Xtr[:, col[f]]):.3f}")

    def model_for(fs):
        cols = [col[f] for f in BASE + fs]
        return cols, fit_gbm(Xtr, ytr, cols, BASE_MONO + [mono[f] for f in fs])

    clf = fit_elo_baseline(tr)
    p_lr = lambda games: np.concatenate([clf.predict_proba(elo_features(g, len(g["sec"])))[:, 1] for g in games])
    cols0, m0 = model_for([])
    p0 = m0.predict_proba(Xva[:, cols0])[:, 1]
    report = {"val": {"logistic+Elo": compare(p_lr(va), p0, yva, gva), "GBM base": compare(p0, p0, yva, gva)}}
    print(f"\nvalidation ({len(va)} games), Brier vs GBM on score+clock+Elo:")
    print(f"  {'logistic+Elo':24s} {report['val']['logistic+Elo']['brier']:.5f}")
    print(f"  {'GBM base':24s} {report['val']['GBM base']['brier']:.5f}")
    kept = []
    for name, fs in groups.items():
        cols, m = model_for(fs)
        r = compare(m.predict_proba(Xva[:, cols])[:, 1], p0, yva, gva)
        report["val"][f"+{name}"] = r
        keep = r["delta"] < 0 and r["p_better"] >= KEEP_PROB
        print(f"  +{name:23s} {r['brier']:.5f}  {r['delta']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}]"
              f"  P(better) {r['p_better']:.2f}{'  keep' if keep else ''}", flush=True)
        if keep:
            kept += [f for f in fs if f not in kept]
    report["kept_features"] = kept
    if kept:
        cols, m = model_for(kept)
        r = compare(m.predict_proba(Xva[:, cols])[:, 1], p0, yva, gva)
        report["val"]["+kept"] = r
        print(f"  {'+kept together':24s} {r['brier']:.5f}  {r['delta']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}]")
    else:
        cols, m = cols0, m0
    # the one look at the test season
    Xte, yte, gte = stack(te, ctx)
    pt_lr, pt0, pt = p_lr(te), m0.predict_proba(Xte[:, cols0])[:, 1], m.predict_proba(Xte[:, cols])[:, 1]
    report["test"] = {"logistic+Elo": {**compare(pt_lr, pt_lr, yte, gte), **by_quarter(pt_lr, yte, te)},
                      "GBM base": {**compare(pt0, pt_lr, yte, gte), **by_quarter(pt0, yte, te)},
                      "GBM +kept": {**compare(pt, pt_lr, yte, gte), **by_quarter(pt, yte, te)}}
    print(f"\ntest ({len(te)} games), Brier vs logistic+Elo:")
    for k, r in report["test"].items():
        print(f"  {k:16s} {r['brier']:.5f}  {r['delta']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}]  "
              + " ".join(f"{q} {r[q]:.4f}" for q in ("Q1", "Q2", "Q3", "Q4")))
    joblib.dump({"model": m, "features": [names[c] for c in cols]}, out / "model.joblib")
    json.dump(report, open(out / "report.json", "w"), indent=1)
    print(f"saved {out}/model.joblib and report.json")


if __name__ == "__main__":
    main()
