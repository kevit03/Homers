"""Download NBA.com shot charts (ShotChartDetail) for every field-goal attempt, 1996-97 on.

PlayByPlayV3 is missing court coordinates for about a quarter of the shots before 2010-11
(xLegacy = yLegacy = 0, shotDistance = 0). The shot-chart feed has LOC_X / LOC_Y, SHOT_DISTANCE and
NBA.com's own zones (SHOT_ZONE_BASIC / AREA / RANGE) for those seasons; models/build_shots.py joins
it to the play-by-play on (GAME_ID, GAME_EVENT_ID = actionNumber).

Playoffs: one league-wide request per season (team_id = 0, player_id = 0). The regular season is too big
for one request (NBA.com answers with an empty body), so it goes team by team (~5 s each; franchise IDs are
stable across relocations, and a team that didn't exist yet just returns no rows). Each team answer is
cached under data/context/shotchart/parts/teams/, each (season, season type) under parts/, and the season
file is written once both are in, so the script can be stopped and re-run safely.

    python -m fetch.fetch_shotchart                       # 1996-97 to 2025-26
    python -m fetch.fetch_shotchart --seasons 1996-97 1997-98
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import teams

SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(1996, 2026)]
SEASON_TYPES = ["Regular Season", "Playoffs"]
LEAGUE_WIDE = {"Playoffs"}  # season types small enough for one league-wide request
KEEP = ["GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "TEAM_ID", "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING",
        "EVENT_TYPE", "ACTION_TYPE", "SHOT_TYPE", "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE",
        "SHOT_DISTANCE", "LOC_X", "LOC_Y", "SHOT_MADE_FLAG", "GAME_DATE", "HTM", "VTM"]


def request(team_id, season, season_type, timeout):
    return shotchartdetail.ShotChartDetail(team_id=team_id, player_id=0, season_nullable=season,
                                           season_type_all_star=season_type, context_measure_simple="FGA",
                                           timeout=timeout).get_data_frames()[0]


def call(fn, retries=4, sleep=1.5):
    for attempt in range(retries):
        try:
            out = fn()
            time.sleep(sleep)
            return out
        except Exception as e:  # network hiccups / rate limiting
            print(f"    attempt {attempt + 1} failed: {e}")
            time.sleep(10 * (attempt + 1))
    return None


def fetch_part(season, season_type, sleep, cache):
    if season_type in LEAGUE_WIDE:
        df = call(lambda: request(0, season, season_type, timeout=180), retries=2, sleep=sleep)
        if df is not None and len(df):
            return df
        print(f"  {season} {season_type}: league-wide request failed, going team by team", flush=True)
    frames, failed = [], 0
    cache.mkdir(parents=True, exist_ok=True)
    for t in teams.get_teams():
        p = cache / f"{season}_{season_type.replace(' ', '')}_{t['id']}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
            continue
        d = call(lambda: request(t["id"], season, season_type, timeout=60), sleep=sleep)
        if d is None:
            print(f"  {season} {season_type} {t['abbreviation']}: giving up", flush=True)
            failed += 1
            continue
        d = d.reindex(columns=KEEP)
        d.to_parquet(p)
        frames.append(d)
    if failed:
        return None  # the teams that did come back stay cached for the re-run
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=SEASONS)
    ap.add_argument("--out", default="data/context/shotchart")
    ap.add_argument("--sleep", type=float, default=1.5)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    for season in args.seasons:
        final = out / f"{season}.parquet"
        if final.exists():
            continue
        parts = []
        for st in SEASON_TYPES:
            p = out / "parts" / f"{season}_{st.replace(' ', '')}.parquet"
            if not p.exists():
                df = fetch_part(season, st, args.sleep, out / "parts" / "teams")
                if df is None:
                    print(f"{season} {st}: failed, re-run later", flush=True)
                    continue
                df = df.reindex(columns=KEEP)
                df.to_parquet(p)
            parts.append(pd.read_parquet(p).assign(seasonType=st))
        if len(parts) == len(SEASON_TYPES):
            df = pd.concat(parts, ignore_index=True).drop_duplicates(["GAME_ID", "GAME_EVENT_ID"])
            df["season"] = season
            df.to_parquet(final)
            print(f"{season}: {len(df):,} shots in {df['GAME_ID'].nunique()} games", flush=True)


if __name__ == "__main__":
    main()
