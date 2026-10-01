"""An estimate of each team season's defensive coverage, for the Coaches tab's defensive game plan.

No public feed labels coverages (drop, switch, hedge, blitz, zone): Synergy and Second Spectrum tag them, but only in
paid data. NBA.com's matchup feed has a switchesOn column, but it is 0 on every row. So the scheme is inferred from
five signals the free data does have, and placed on two axes, each a z-score among the 30 teams that season:

  scramble    how often a scorer's defender changes. For every offensive player in a game, the Herfindahl index of
              his defenders' shares of his possessions (BoxScoreMatchupsV3, 2017-18 on): 1 = one man guarded him all
              night, low = the assignment kept changing, through switches or zone. scramble = -z(HHI).
  aggression  how far the defense comes out to meet the pick-and-roll, from four signs:
                + turnovers forced on pick-and-roll ball-handler plays (Synergy): traps and hedges force them
                - opponents' pull-up and step-back mid-range share (road games, like export_coach_defense.py):
                  what a drop big concedes
                + opponents' rim share: a big pulled up to the ball leaves the rim
                - roll-man possessions (Synergy): a drop big lets the screener roll free; traps and switches take him
                  away. Empirically the drop teams below allow the most (MIL 2019-20: 1st of 30).
              aggression = mean of the four signed z-scores.

Quadrants: locked + conservative = Drop, scrambled + conservative = Switch, locked + aggressive = Hedge / blitz,
scrambled + aggressive = Pressure / zone. It cannot tell zone from switching, since both scramble the assignments.

Checked against schemes reported at the time (rank among 30 that season; 1 = most):
  drop        MIL 2018-19 to 2020-21 (Lopez), UTA 2017-18, 2020-21 and 2021-22 (Gobert), MIN 2023-24 (Gobert):
              scramble 18th-25th, aggression 24th-30th. All land in Drop.
  switching   BOS 2021-22 to 2023-24: scramble 1st-3rd, aggression 26th-29th. All land in Switch.
              HOU 2017-18 and 2018-19: scramble 1st both years but aggression 3rd, so Pressure / zone: their switching
              came with ball pressure (many pick-and-roll turnovers forced, many rim shots allowed). Leaving the
              roll-man signal out still puts them 8th-9th.
  pressure    TOR 2020-21 and 2021-22 (Nurse: traps, zone, junk defenses): aggression 3rd and 1st, scramble 6th and 8th.

Used by export_dashboard.py. Run it alone to print the checks above:

    python export_coach_scheme.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["hhi", "pu_mid", "rim", "bh_tov", "rm_freq"]
SIGN = {"pu_mid": -1, "rim": 1, "bh_tov": 1, "rm_freq": -1}  # each feature's direction on the aggression axis
CHECKS = [("2017-18", "HOU"), ("2018-19", "HOU"), ("2021-22", "BOS"), ("2022-23", "BOS"), ("2023-24", "BOS"),
          ("2018-19", "MIL"), ("2019-20", "MIL"), ("2020-21", "MIL"), ("2017-18", "UTA"), ("2020-21", "UTA"),
          ("2021-22", "UTA"), ("2023-24", "MIN"), ("2020-21", "TOR"), ("2021-22", "TOR")]


def matchup_hhi(matchups="data/context/matchups"):
    """{(season, defending tricode): possession-weighted HHI of each scorer's defenders}, regular season only."""
    acc = {}
    for f in sorted(Path(matchups).glob("*/002*.parquet")):
        d = pd.read_parquet(f, columns=["teamTricode", "personIdOff", "partialPossessions"])
        teams = d["teamTricode"].unique()
        if len(teams) != 2:
            continue
        d["def"] = d["teamTricode"].map({teams[0]: teams[1], teams[1]: teams[0]})  # teamTricode is the scorer's team
        tot = d.groupby("personIdOff")["partialPossessions"].transform("sum")
        d["w"] = d["partialPossessions"] ** 2 / tot.where(tot > 0)  # share^2 * player's possessions
        for team, g in d.groupby("def"):
            a = acc.setdefault((f.parent.name, team), [0.0, 0.0])
            a[0] += g["w"].sum(); a[1] += g["partialPossessions"].sum()
    return {k: w / p for k, (w, p) in acc.items() if p > 0}


def scheme_table(shots="data/processed/shots.parquet", playtypes="data/context/playtypes.parquet",
                 matchups="data/context/matchups"):
    s = pd.read_parquet(shots, columns=["gameId", "season", "is_home", "defTeam", "zone", "style"])
    road = s[s["gameId"].str.startswith("002") & (s["is_home"] == 1)]  # the shooter at home: the defense on the road
    t = road.assign(pu_mid=road["zone"].isin(["SHORT_MID", "LONG_MID", "FLOATER"]) & road["style"].isin(["PULLUP", "STEPBACK", "FADEAWAY"]),
                    rim=road["zone"].isin(["LAYUP", "DUNK"])).groupby(["season", "defTeam"])[["pu_mid", "rim"]].mean()
    t.index.names = ["season", "team"]
    p = pd.read_parquet(playtypes)
    p = p[(p["level"] == "T") & (p["grouping"] == "defensive")].set_index(["season", "TEAM_ABBREVIATION"])
    p.index.names = ["season", "team"]
    t["bh_tov"] = p.loc[p["PLAY_TYPE"] == "PRBallHandler", "TOV_POSS_PCT"]
    t["rm_freq"] = p.loc[p["PLAY_TYPE"] == "PRRollman", "POSS_PCT"]
    t["hhi"] = pd.Series(matchup_hhi(matchups))
    t = t.dropna()  # matchup tracking starts in 2017-18
    z = t.groupby(level="season")[FEATURES].transform(lambda c: (c - c.mean()) / c.std())
    t["scramble"] = -z["hhi"]
    t["aggression"] = sum(SIGN[k] * z[k] for k in SIGN) / len(SIGN)
    return t


def build_coach_scheme_payload(shots="data/processed/shots.parquet", playtypes="data/context/playtypes.parquet",
                               matchups="data/context/matchups"):
    if not (Path(shots).exists() and Path(playtypes).exists() and Path(matchups).exists()):
        return None
    t = scheme_table(shots, playtypes, matchups)
    if t.empty:
        return None
    pct = t.groupby(level="season")[FEATURES].rank(pct=True)  # 1 = most in the league that season
    units = {f"{season}|{team}": {"s": round(float(r["scramble"]), 3), "a": round(float(r["aggression"]), 3),
                                  "f": [round(float(r[k]), 4) for k in FEATURES], "p": [round(float(pct.loc[(season, team), k]), 3) for k in FEATURES]}
             for (season, team), r in t.iterrows()}
    return {"features": FEATURES, "units": units, "seasons": sorted(t.index.get_level_values("season").unique())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="data/processed/shots.parquet")
    args = ap.parse_args()
    t = scheme_table(args.shots)
    r = t.groupby(level="season")[["scramble", "aggression"]].rank(ascending=False).astype(int)
    print(f"{len(t)} team seasons, {t.index.get_level_values('season').min()} to {t.index.get_level_values('season').max()}")
    print("rank among 30 (1 = most):")
    for k in CHECKS:
        if k in t.index:
            print(f"  {k[0]} {k[1]}  scramble {r.loc[k, 'scramble']:>2}  aggression {r.loc[k, 'aggression']:>2}")
