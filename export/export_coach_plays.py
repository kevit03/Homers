"""How each team season's plays turn out, for the Coaches tab's playbook cards.

export_tracking.team_units already carries each Synergy play type's possessions and points. This adds what the
cards show next to them, for the team's offense and for what opponents ran against its defense (regular season,
data/context/playtypes.parquet from fetch/fetch_context.py):

  - field goals made and attempted on the play
  - possessions that scored, that ended in a turnover, and that drew free throws

Units are keyed "<season>|<tricode>" like team_units and hold, per play type in fetch_context.PLAY_TYPES order,
[possessions, fgm, fga, scored, turnovers, free throws] for "off" and "def". The page turns them into percentages
and compares them with the season's league totals.

Used by export/export_dashboard.py. Run it alone for a size check:

    python -m export.export_coach_plays
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fetch.fetch_context import PLAY_TYPES

FIELDS = ["poss", "fgm", "fga", "scored", "tov", "ft"]


def build_coach_plays_payload(context="data/context"):
    path = Path(context) / "playtypes.parquet"
    if not path.exists():
        return None
    p = pd.read_parquet(path)
    p = p[(p["level"] == "T") & p["PLAY_TYPE"].isin(PLAY_TYPES)].copy()
    poss = p["POSS"].fillna(0)
    # Synergy gives these as shares of the play's possessions; back to counts so seasons and the league add up
    p["scored"], p["tov"], p["ft"] = [(poss * p[c].fillna(0)).round() for c in ("SCORE_POSS_PCT", "TOV_POSS_PCT", "FT_POSS_PCT")]
    p["poss"], p["fgm"], p["fga"] = poss, p["FGM"].fillna(0), p["FGA"].fillna(0)
    p["side"] = np.where(p["grouping"] == "offensive", "off", "def")
    p["k"] = p["PLAY_TYPE"].map(PLAY_TYPES.index)

    units = {}
    for (season, tri, side), g in p.groupby(["season", "TEAM_ABBREVIATION", "side"]):
        rows = [[0] * len(FIELDS) for _ in PLAY_TYPES]
        for r in g.itertuples(index=False):
            rows[r.k] = [int(getattr(r, f)) for f in FIELDS]
        units.setdefault(f"{season}|{tri}", {})[side] = rows
    league = {}
    for key, u in units.items():
        tot = league.setdefault(key.split("|")[0], {s: [[0] * len(FIELDS) for _ in PLAY_TYPES] for s in ("off", "def")})
        for side, rows in u.items():
            tot[side] = [[a + b for a, b in zip(x, y)] for x, y in zip(tot[side], rows)]
    return {"play_types": PLAY_TYPES, "fields": FIELDS, "units": units, "league": league}


if __name__ == "__main__":
    P = build_coach_plays_payload()
    print(f"{len(P['units'])} team seasons, league seasons {list(P['league'])}")
    u, L = P["units"]["2024-25|OKC"]["off"], P["league"]["2024-25"]["off"]
    for name, r, l in zip(P["play_types"], u, L):
        if r[0]:
            print(f"  OKC 24-25 {name:<14} {r[0]:>5} poss  scored {r[3] / r[0]:.3f} (lg {l[3] / l[0]:.3f})  "
                  f"FG {r[1] / max(r[2], 1):.3f}  TOV {r[4] / r[0]:.3f}")
    print(f"{len(json.dumps(P, separators=(',', ':'))) / 1e3:.0f} KB")
