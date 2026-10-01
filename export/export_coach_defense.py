"""Opponents' shooting against each team season, for the Coaches tab's defensive game plan.

Built straight from data/processed/shots.parquet (regular-season games only, like the Synergy and tracking numbers
next to it). For every team season it counts opponents' attempts and makes by

  - shot type (layup, dunk, floater, ..., above-the-break 3; build_shots.ZONES)
  - shot style (pull-up, step-back, driving, ...; build_shots.STYLES), from road games only: styles come from the
    play-by-play text, and each arena's scorer labels them differently (at Golden State in 2016-17, opponents' "pull-ups"
    went in 76% of the time against 44% on the road), so a team's home games would measure its scorer, not its defense
  - court zone (restricted area, paint, mid-range L/C/R, corner 3s, above-the-break 3s, heaves; export_shotcharts.LOC_ZONES)

Units are keyed "<season>|<tricode>" like export_tracking.team_units, so the page can join them to a coach's
Basketball-Reference stints. Shots' defCoachId is not used: it comes from NBA.com's roster feed, which names the
wrong head coach for some team seasons. Counts are stored as flat [attempts, makes, attempts, makes, ...] lists;
the page derives every percentage, and the per-season league totals it compares against.

Used by export/export_dashboard.py. Run it alone for a size check:

    python -m export.export_coach_defense --shots data/processed/shots.parquet
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.build_shots import STYLES, ZONES
from export.export_shotcharts import LOC_ZONES, loc_zone


def _counts(df, col, cats):
    """{(season, team): [a0, m0, a1, m1, ...]} over the categories in `cats`."""
    g = df.groupby(["season", "defTeam", col], observed=True)["made"].agg(["size", "sum"])
    out = {}
    for (season, team, cat), (a, m) in g.iterrows():
        v = out.setdefault((season, team), [0] * (2 * len(cats)))
        i = cats.index(cat)
        v[2 * i] += int(a); v[2 * i + 1] += int(m)
    return out


def build_coach_defense_payload(shots_path="data/processed/shots.parquet"):
    if not Path(shots_path).exists():
        return None
    s = pd.read_parquet(shots_path, columns=["gameId", "season", "defTeam", "is_home", "zone", "style", "made", "is3", "x", "y"])
    s = s[s["gameId"].str.startswith("002")]  # regular season
    s = s[s["zone"].isin(ZONES) & s["style"].isin(STYLES)]
    s["lz"] = loc_zone(s["x"].to_numpy(), s["y"].to_numpy(), s["is3"].to_numpy())

    road = s[s["is_home"] == 1]  # the shooter at home: the defense on the road, its opponents' styles labelled by other arenas
    parts = {"t": _counts(s, "zone", ZONES), "s": _counts(road, "style", STYLES), "z": _counts(s, "lz", list(range(len(LOC_ZONES))))}
    keys = sorted(set().union(*(p.keys() for p in parts.values())))
    units = {f"{season}|{team}": {k: parts[k].get((season, team), [0] * (2 * n)) for k, n in (("t", len(ZONES)), ("s", len(STYLES)), ("z", len(LOC_ZONES)))}
             for season, team in keys}
    league = {}
    for key, u in units.items():
        tot = league.setdefault(key.split("|")[0], {k: [0] * len(v) for k, v in u.items()})
        for k, v in u.items():
            tot[k] = [a + b for a, b in zip(tot[k], v)]
    return {"types": ZONES, "styles": STYLES, "zones": LOC_ZONES, "units": units, "league": league}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="data/processed/shots.parquet")
    args = ap.parse_args()
    p = build_coach_defense_payload(args.shots)
    print(f"{len(p['units'])} team seasons, league seasons {list(p['league'])}")
    k = next(k for k in p["units"] if k.startswith("2024-25|OKC"))
    u, L = p["units"][k], p["league"]["2024-25"]
    for i, z in enumerate(p["types"]):
        a, m, la, lm = u["t"][2 * i], u["t"][2 * i + 1], L["t"][2 * i], L["t"][2 * i + 1]
        print(f"  {k} {z:<14} {m:>4}-{a:<5} {m / a:.3f}  league {lm / la:.3f}")
    print(f"{len(json.dumps(p, separators=(',', ':'))) / 1e3:.0f} KB")
