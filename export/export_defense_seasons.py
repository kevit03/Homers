"""Each defender's regular-season numbers one season at a time, so the Defense tab can rate any span: one season,
the last three, or the whole career.

The career wall uses the matchup model (models/matchup_model.py), which is fit once over every season. To rate a
single season the same way, this goes back to the raw matchup feed (BoxScoreMatchupsV3, 2017-18 on) and compares
what a defender allowed with what the same scorers did against everyone else that season (leave-one-out, so his
own possessions don't set the bar he's measured against):

  xpts  = his possessions on each scorer x that scorer's points per possession against every other defender
  xfgm  = the scorer's shots against him x that scorer's FG% against every other defender

The page sums any seasons it needs and shrinks small samples toward average. Rim protection and deflections come
from NBA.com tracking per season (fetch/fetch_context.py): rim makes, attempts and the shooters' usual makes on
those attempts (FGA x their normal FG% within 6 feet).

Per player, a flat list of 12-number rows: a season index into `seasons`, then COLS (poss, pts, xpts, fga, fgm,
xfgm, rim_m, rim_a, rim_xm, defl, hmin). Seasons where he had fewer than MIN_POSS matchup possessions and no
tracking rows are left out.

Used by export/export_dashboard.py. Run it alone for a size and sanity check:

    python3 -m export.export_defense_seasons
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["poss", "pts", "xpts", "fga", "fgm", "xfgm", "rim_m", "rim_a", "rim_xm", "defl", "hmin"]
MIN_POSS = 100


def matchup_rows(matchups="data/context/matchups"):
    """{(season, defender): [poss, pts, xpts, fga, fgm, xfgm]} from every regular-season game."""
    parts = []
    for d in sorted(p for p in Path(matchups).iterdir() if p.is_dir()):
        for f in sorted(d.glob("002*.parquet")):
            parts.append(pd.read_parquet(f, columns=["personIdOff", "personIdDef", "partialPossessions", "playerPoints",
                                                     "matchupFieldGoalsMade", "matchupFieldGoalsAttempted"]).assign(season=d.name))
    if not parts:
        return {}
    m = pd.concat(parts, ignore_index=True).rename(columns={"personIdOff": "off", "personIdDef": "dfn", "partialPossessions": "poss",
                                                            "playerPoints": "pts", "matchupFieldGoalsMade": "fgm", "matchupFieldGoalsAttempted": "fga"})
    m = m.groupby(["season", "off", "dfn"], as_index=False)[["poss", "pts", "fgm", "fga"]].sum()
    tot = m.groupby(["season", "off"])[["poss", "pts", "fgm", "fga"]].transform("sum")
    rest = tot - m[["poss", "pts", "fgm", "fga"]]  # the scorer against every other defender that season
    m["xpts"] = m["poss"] * (rest["pts"] / rest["poss"].where(rest["poss"] >= 20))
    m["xfgm"] = m["fga"] * (rest["fgm"] / rest["fga"].where(rest["fga"] >= 10))
    m = m.dropna(subset=["xpts", "xfgm"])  # scorers seen by no one else can't set a baseline
    g = m.groupby(["season", "dfn"])[["poss", "pts", "xpts", "fga", "fgm", "xfgm"]].sum()
    return {k: list(v) for k, v in zip(g.index, g.to_numpy())}


def tracking_rows(tracking="data/context/tracking.parquet"):
    """{(season, player): [rim_m, rim_a, rim_xm, defl, hmin]} from NBA.com's per-season player tracking."""
    if not Path(tracking).exists():
        return {}
    t = pd.read_parquet(tracking)
    t = t[(t["level"] == "P") & t["PLAYER_ID"].notna()]
    out = {}
    r = t[t["measure"] == "Less Than 6Ft"]
    for s, pid, m, a, ns in zip(r["season"], r["PLAYER_ID"], r["FGM_LT_06"], r["FGA_LT_06"], r["NS_LT_06_PCT"]):
        v = out.setdefault((s, int(pid)), [0.0] * 5)
        a = float(a or 0)
        v[0] += float(m or 0); v[1] += a; v[2] += a * float(ns or 0)
    h = t[t["measure"] == "Hustle"]
    for s, pid, dfl, mn in zip(h["season"], h["PLAYER_ID"], h["DEFLECTIONS"], h["MIN"]):
        v = out.setdefault((s, int(pid)), [0.0] * 5)
        v[3] += float(dfl or 0); v[4] += float(mn or 0)
    return out


def build_defense_seasons_payload(matchups="data/context/matchups", tracking="data/context/tracking.parquet"):
    mu, tr = matchup_rows(matchups), tracking_rows(tracking)
    if not mu and not tr:
        return None
    keep = {k for k in set(mu) | set(tr) if mu.get(k, [0])[0] >= MIN_POSS or tr.get(k, [0, 0])[1]}
    seasons = sorted({s for s, _ in keep})
    si = {s: i for i, s in enumerate(seasons)}
    tenths = {COLS.index(c) for c in ("xpts", "xfgm", "rim_xm")}  # expected counts keep a decimal; the rest are whole
    players = {}
    for season, pid in sorted(keep, key=lambda k: (k[1], k[0])):
        v = mu.get((season, pid), [0.0] * 6) + tr.get((season, pid), [0.0] * 5)
        players.setdefault(str(int(pid)), []).extend([si[season]] + [round(float(x), 1) if i in tenths else int(round(float(x))) for i, x in enumerate(v)])
    return {"cols": COLS, "seasons": seasons, "mu_seasons": sorted({s for s, _ in mu}), "players": players}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", default="data/context")
    args = ap.parse_args()
    p = build_defense_seasons_payload(f"{args.context}/matchups", f"{args.context}/tracking.parquet")
    print(f"{len(p['players'])} players, seasons {p['seasons'][0]} to {p['seasons'][-1]}, matchups from {p['mu_seasons'][0]}")
    print(f"{len(json.dumps(p, separators=(',', ':'))) / 1e3:.0f} KB")
    # sanity: league-wide, allowed should equal expected on average
    a = np.array([r for flat in p["players"].values() for r in np.reshape(flat, (-1, 1 + len(COLS)))])[:, 1:]
    print(f"points allowed / expected, all defenders: {a[:, 1].sum() / a[:, 2].sum():.3f}; FG made / expected: {a[:, 4].sum() / a[:, 5].sum():.3f}")
