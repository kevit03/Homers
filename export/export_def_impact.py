"""Defensive impact: a ridge-regressed adjusted plus-minus on every opponent shot, for the Defense tab's Impact axis.

The matchup model only sees the man a defender is assigned to, so it misses help defense, rim protection and shots that
are never taken, which is most of what a big does. This sees all of it. Each regular-season shot from the charted
shots (data/processed/shots.parquet, 2016-17 on) is one row: the points it scored, minus that shooter's average points
per shot that season (so who shoots is the offense's doing), against the ten players on the floor. Ridge (RIDGE, picked
on held-out games) shrinks small samples toward average. A defender's coefficient is points per 100 opponent shots he
adds, so below zero is good. It doesn't care about position: a guard who denies his man and a big who scares drives
off are both just fewer points.

Turnovers and free throws aren't shots, so steals and fouls aren't in it; the Wall keeps steals as their own axis.

One fit per period the page offers: the whole span, the last three seasons and each season alone. Payload:
{"periods": [...], "players": {id: [period index, opponent shots, points per 100, ...]}}, players with MIN_SHOTS or more.

Used by export/export_dashboard.py. Run it alone for a sanity check:
    python3 -m export.export_def_impact
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

RIDGE = 10000
MIN_SHOTS = 1000


def load_shots(shots="data/processed/shots.parquet"):
    s = pd.read_parquet(shots, columns=["gameId", "season", "shooterId", "made", "is3", "off5", "def5"])
    s = s[s["gameId"].astype(str).str.startswith("002") & (s["def5"].map(len) == 5) & (s["off5"].map(len) == 5)]
    pts = s["made"] * np.where(s["is3"] == 1, 3, 2)
    return s.assign(y=pts - pts.groupby([s["season"], s["shooterId"]]).transform("mean")).reset_index(drop=True)


def fit(s):
    """{player id: (opponent shots defended, points per 100 opponent shots)} for one span of shots."""
    d5, o5 = (np.stack(s[c].to_numpy()).astype(np.int64) for c in ("def5", "off5"))
    ids, col = np.unique(np.concatenate([d5.ravel(), o5.ravel()]), return_inverse=True)
    n, P = len(s), len(ids)
    rows = np.repeat(np.arange(n), 5)
    X = sparse.csr_matrix((np.ones(10 * n), (np.concatenate([rows, rows]), np.concatenate([col[:5 * n], P + col[5 * n:]]))), shape=(n, 2 * P))
    m = Ridge(alpha=RIDGE, solver="sparse_cg").fit(X, s["y"].to_numpy())
    nd = np.bincount(col[:5 * n], minlength=P)
    return {int(p): (int(nd[i]), float(m.coef_[i] * 100)) for i, p in enumerate(ids) if nd[i] >= MIN_SHOTS}


def build_def_impact_payload(shots="data/processed/shots.parquet"):
    s = load_shots(shots)
    if s.empty:
        return None
    seasons = sorted(s["season"].unique())
    spans = {"career": seasons, "last3": seasons[-3:], **{k: [k] for k in seasons}}
    periods, players = list(spans), {}
    for i, (k, ss) in enumerate(spans.items()):
        for pid, (n, d) in fit(s[s["season"].isin(ss)]).items():
            players.setdefault(str(pid), []).extend([i, n, round(d, 2)])
    return {"periods": periods, "ridge": RIDGE, "players": players}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="data/processed/shots.parquet")
    ap.add_argument("--profiles", default="", help="a built dashboard's DATA as JSON, for names")
    args = ap.parse_args()
    t = time.time()
    p = build_def_impact_payload(args.shots)
    print(f"{len(p['players'])} players, {len(p['periods'])} periods, {len(json.dumps(p, separators=(',', ':'))) / 1e3:.0f} KB, {time.time() - t:.0f}s")
    names = json.load(open(args.profiles))["profiles"]["players"] if args.profiles else {}
    for k in ("career", p["periods"][-1]):
        i = p["periods"].index(k)
        rows = [(pid, r[3 * j + 1], r[3 * j + 2]) for pid, r in p["players"].items() for j in range(len(r) // 3) if r[3 * j] == i]
        print(f"best in {k}:", ", ".join(f"{names.get(pid, {}).get('name', pid)} {d:+.1f}" for pid, n, d in sorted(rows, key=lambda r: r[2])[:12]))
