"""Offensive and defensive play data for players and teams, for the dashboard's Tendencies and Coaches tabs.

Everything is regular-season totals from data/context (fetch/fetch_context.py):

  - Synergy play types, offense (11) and defense (11 for teams, 7 for players): possessions and points
  - tracking actions: drives, catch-and-shoot, pull-ups, paint / post / elbow touches, screen assists
  - hustle: deflections, contested shots, charges drawn, loose balls recovered, box outs
  - rim and 3-point defense: opponents' FG% within 6 feet and from three, against what those shooters usually make

A "unit" is one player's career in the data or one team's season. Units hold totals; the page turns them
into shares, per-game or per-36 rates and points per play, and compares them with the league.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from fetch.fetch_context import PLAY_TYPES, PLAYER_DEF_PLAY_TYPES

# measure, count column, points column, makes, attempts
ACTIONS = [("Drives", "DRIVES", "DRIVE_PTS", "DRIVE_FGM", "DRIVE_FGA"),
           ("CatchShoot", "CATCH_SHOOT_FGA", "CATCH_SHOOT_PTS", "CATCH_SHOOT_FGM", "CATCH_SHOOT_FGA"),
           ("PullUpShot", "PULL_UP_FGA", "PULL_UP_PTS", "PULL_UP_FGM", "PULL_UP_FGA"),
           ("PaintTouch", "PAINT_TOUCHES", "PAINT_TOUCH_PTS", "PAINT_TOUCH_FGM", "PAINT_TOUCH_FGA"),
           ("PostTouch", "POST_TOUCHES", "POST_TOUCH_PTS", "POST_TOUCH_FGM", "POST_TOUCH_FGA"),
           ("ElbowTouch", "ELBOW_TOUCHES", "ELBOW_TOUCH_PTS", "ELBOW_TOUCH_FGM", "ELBOW_TOUCH_FGA"),
           ("Hustle", "SCREEN_ASSISTS", "SCREEN_AST_PTS", None, None)]
ACTION_KEYS = ["Drives", "CatchShoot", "PullUp", "PaintTouch", "PostTouch", "ElbowTouch", "ScreenAssist"]
HUSTLE = ["DEFLECTIONS", "CONTESTED_SHOTS", "CHARGES_DRAWN", "LOOSE_BALLS_RECOVERED", "BOX_OUTS"]
HUSTLE_KEYS = ["Deflections", "Contests", "Charges", "LooseBalls", "BoxOuts"]
RIM = [("Less Than 6Ft", "FGM_LT_06", "FGA_LT_06", "NS_LT_06_PCT"), ("3 Pointers", "FG3M", "FG3A", "NS_FG3_PCT")]
RIM_KEYS = ["Rim", "Three"]


def _i(v):
    return int(round(float(v))) if pd.notna(v) else 0


def _tricodes():
    from nba_api.stats.static import teams
    return {t["id"]: t["abbreviation"] for t in teams.get_teams()}


def _load(context):
    ctx = Path(context)
    tr = pd.read_parquet(ctx / "tracking.parquet") if (ctx / "tracking.parquet").exists() else None
    pt = pd.read_parquet(ctx / "playtypes.parquet") if (ctx / "playtypes.parquet").exists() else None
    return tr, pt


def _empty(n_def):
    return {"gp": 0, "min": 0.0, "hgp": 0, "hmin": 0.0,
            "off": [[0, 0] for _ in PLAY_TYPES], "def": [[0, 0] for _ in range(n_def)],
            "act": [[0, 0, 0, 0] for _ in ACTIONS], "hus": [0] * len(HUSTLE), "rim": [[0, 0, 0.0] for _ in RIM]}


def _fill(units, rows, key_fn, level, n_def, def_types, tracking=None, playtypes=None):
    """Add totals from tracking and play-type rows into units[key]."""
    if tracking is not None:
        for (measure, cnt, pts, fgm, fga) in ACTIONS:
            k = [a[0] for a in ACTIONS].index(measure) if measure != "Hustle" else len(ACTIONS) - 1
            for _, r in tracking[tracking["measure"] == measure].iterrows():
                u = units.setdefault(key_fn(r), _empty(n_def))
                u["act"][k][0] += _i(r[cnt]); u["act"][k][1] += _i(r[pts])
                if fgm:
                    u["act"][k][2] += _i(r[fgm]); u["act"][k][3] += _i(r[fga])
                if measure == "Drives":  # games and minutes once per season, from one measure
                    u["gp"] += _i(r["GP"]); u["min"] += float(r["MIN"] or 0)
        for _, r in tracking[tracking["measure"] == "Hustle"].iterrows():
            u = units.setdefault(key_fn(r), _empty(n_def))
            u["hus"] = [a + _i(r[c]) for a, c in zip(u["hus"], HUSTLE)]
            u["hmin"] += float(r["MIN"] or 0)
            u["hgp"] += _i(r["G"]) if "G" in r and pd.notna(r["G"]) else 0
        for k, (measure, fgm, fga, ns) in enumerate(RIM):
            for _, r in tracking[tracking["measure"] == measure].iterrows():
                u = units.setdefault(key_fn(r), _empty(n_def))
                a = _i(r[fga])
                u["rim"][k][0] += _i(r[fgm]); u["rim"][k][1] += a; u["rim"][k][2] += a * float(r[ns] or 0)
    if playtypes is not None:
        for grouping, dst, types in (("offensive", "off", PLAY_TYPES), ("defensive", "def", def_types)):
            g = playtypes[(playtypes["level"] == level) & (playtypes["grouping"] == grouping)]
            for _, r in g.iterrows():
                if r["PLAY_TYPE"] not in types:
                    continue
                u = units.setdefault(key_fn(r), _empty(n_def))
                k = types.index(r["PLAY_TYPE"])
                u[dst][k][0] += _i(r["POSS"]); u[dst][k][1] += _i(r["PTS"])


def _finish(u):
    u["min"] = round(u["min"], 1); u["hmin"] = round(u["hmin"], 1)
    u["rim"] = [[a, b, round(c, 1)] for a, b, c in u["rim"]]
    return u


def _league(units):
    """Sum of every unit: the page divides it the same way as a single unit to get league rates."""
    n_def = len(next(iter(units.values()))["def"])
    tot = _empty(n_def)
    for u in units.values():
        for k in ("gp", "min", "hgp", "hmin"):
            tot[k] += u[k]
        for k in ("off", "def", "act", "rim"):
            tot[k] = [[a + b for a, b in zip(x, y)] for x, y in zip(tot[k], u[k])]
        tot["hus"] = [a + b for a, b in zip(tot["hus"], u["hus"])]
    tot["n"] = len(units)
    return _finish(tot)


def team_units(context="data/context"):
    """{"<season>|<tricode>": unit} for every team season, plus per-season league totals and Synergy's
    percentile ranks (higher = better offense, or better defense) for each team season and play type."""
    tr, pt = _load(context)
    if tr is None and pt is None:
        return None
    tri = _tricodes()
    units = {}
    team_gp = {}
    if tr is not None:
        t = tr[tr["level"] == "T"].copy()
        t["TEAM_ID"] = t["TEAM_ID"].astype("Int64")
        d = t[t["measure"] == "Drives"]
        team_gp = {(s, int(i)): _i(g) for s, i, g in zip(d["season"], d["TEAM_ID"], d["GP"])}
        t.loc[t["measure"] == "Hustle", "G"] = [team_gp.get((s, int(i)), 0) for s, i in
                                                zip(t.loc[t["measure"] == "Hustle", "season"], t.loc[t["measure"] == "Hustle", "TEAM_ID"])]
    else:
        t = None
    p = pt[pt["level"] == "T"].copy() if pt is not None else None
    key = lambda r: f"{r['season']}|{tri.get(int(r['TEAM_ID']), '')}"
    _fill(units, None, key, "T", len(PLAY_TYPES), PLAY_TYPES, t, p)
    units = {k: _finish(u) for k, u in units.items() if not k.endswith("|")}
    pct = {}
    if p is not None:
        for grouping, dst in (("offensive", "off"), ("defensive", "def")):
            for _, r in p[p["grouping"] == grouping].iterrows():
                k = key(r)
                if k in units:
                    pct.setdefault(k, {"off": [None] * len(PLAY_TYPES), "def": [None] * len(PLAY_TYPES)})
                    pct[k][dst][PLAY_TYPES.index(r["PLAY_TYPE"])] = round(float(r["PERCENTILE"]), 3) if pd.notna(r["PERCENTILE"]) else None
    for k, v in pct.items():
        units[k]["pct"] = v
    seasons = sorted({k.split("|")[0] for k in units})
    league = {s: _league({k: u for k, u in units.items() if k.startswith(s + "|")}) for s in seasons}
    return {"units": units, "league": league}


def player_units(context="data/context", ids=None):
    """{personId: unit} over every regular season in the data (only `ids` if given), plus league totals over
    all players, and the regular seasons the totals cover."""
    tr, pt = _load(context)
    if tr is None and pt is None:
        return None
    units = {}
    t = tr[(tr["level"] == "P") & tr["PLAYER_ID"].notna()] if tr is not None else None
    p = pt[(pt["level"] == "P") & pt["PLAYER_ID"].notna()] if pt is not None else None
    key = lambda r: int(r["PLAYER_ID"])
    _fill(units, None, key, "P", len(PLAYER_DEF_PLAY_TYPES), PLAYER_DEF_PLAY_TYPES, t, p)
    units = {k: _finish(u) for k, u in units.items()}
    league = _league(units)
    seasons = sorted(set(t["season"]) if t is not None else set(p["season"]))
    if ids is not None:
        ids = {int(i) for i in ids}
        units = {k: u for k, u in units.items() if k in ids}
    return {"units": units, "league": league, "seasons": seasons}


def labels():
    return {"play_types": PLAY_TYPES, "def_play_types_player": PLAYER_DEF_PLAY_TYPES,
            "actions": ACTION_KEYS, "hustle": HUSTLE_KEYS, "rim": RIM_KEYS}


if __name__ == "__main__":
    T = team_units()
    print(f"{len(T['units'])} team seasons; league seasons {list(T['league'])}")
    k = next(k for k in T["units"] if k.endswith("GSW"))
    print(k, {kk: v for kk, v in T["units"][k].items() if kk != "pct"})
    P = player_units(ids=[2544, 203999])
    print(len(P["units"]), P["seasons"], P["units"].get(203999))
