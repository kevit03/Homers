"""Download context for the player/coach shot model:

  - head coach of every team (commonteamroster)          -> data/context/coaches.parquet
  - Synergy play-type frequencies for teams and players  -> data/context/playtypes.parquet
  - per-game defensive matchups (who guarded whom)       -> data/context/matchups/<season>/<gameId>.parquet

Everything is cached, so the script can be stopped and re-run safely.

    python fetch_context.py --seasons 2021-22 2022-23 2023-24 2024-25
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import boxscorematchupsv3, commonteamroster, synergyplaytypes
from nba_api.stats.static import teams

from fetch_data import get_game_ids

PLAY_TYPES = ["Isolation", "Transition", "PRBallHandler", "PRRollman", "Postup", "Spotup",
              "Handoff", "Cut", "OffScreen", "OffRebound", "Misc"]


def call(fn, retries=4, sleep=1.0):
    for attempt in range(retries):
        try:
            out = fn()
            time.sleep(sleep)
            return out
        except Exception as e:  # network hiccups / rate limiting
            print(f"  attempt {attempt + 1} failed: {e}")
            time.sleep(10 * (attempt + 1))
    return None


def fetch_coaches(seasons, out, sleep):
    path = out / "coaches.parquet"
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["season", "teamId"])
    rows = [have]
    for season in seasons:
        for t in teams.get_teams():
            if ((have["season"] == season) & (have["teamId"] == t["id"])).any():
                continue
            dfs = call(lambda: commonteamroster.CommonTeamRoster(team_id=t["id"], season=season, timeout=30)
                       .get_data_frames(), sleep=sleep)
            if dfs is None or len(dfs) < 2:
                continue
            head = dfs[1][dfs[1]["COACH_TYPE"] == "Head Coach"]
            if len(head):
                rows.append(pd.DataFrame([{"season": season, "teamId": t["id"], "team": t["abbreviation"],
                                           "coachId": int(head["COACH_ID"].iloc[0]),
                                           "coach": head["COACH_NAME"].iloc[0]}]))
        print(f"{season}: coaches done")
    pd.concat(rows, ignore_index=True).to_parquet(path)


def fetch_playtypes(seasons, out, sleep):
    path = out / "playtypes.parquet"
    have = pd.read_parquet(path) if path.exists() else None
    done = set() if have is None else set(zip(have["season"], have["level"], have["grouping"], have["PLAY_TYPE"]))
    rows = [] if have is None else [have]
    for season in seasons:
        for level, grouping in (("T", "offensive"), ("T", "defensive"), ("P", "offensive")):
            for pt in PLAY_TYPES:
                if (season, level, grouping, pt) in done:
                    continue
                dfs = call(lambda: synergyplaytypes.SynergyPlayTypes(
                    season=season, play_type_nullable=pt, player_or_team_abbreviation=level,
                    type_grouping_nullable=grouping, season_type_all_star="Regular Season",
                    per_mode_simple="Totals", timeout=30).get_data_frames(), sleep=sleep)
                if dfs is None:
                    continue
                df = dfs[0].assign(season=season, level=level, grouping=grouping, PLAY_TYPE=pt)
                rows.append(df)
        print(f"{season}: play types done")
        pd.concat(rows, ignore_index=True).to_parquet(path)


def fetch_matchups(seasons, season_type, out, sleep):
    for season in seasons:
        sdir = out / "matchups" / season
        sdir.mkdir(parents=True, exist_ok=True)
        ids = get_game_ids(season, season_type)
        print(f"{season}: {len(ids)} games")
        failed = []
        for i, gid in enumerate(ids):
            path = sdir / f"{gid}.parquet"
            if path.exists():
                continue
            dfs = call(lambda: boxscorematchupsv3.BoxScoreMatchupsV3(game_id=gid, timeout=30).get_data_frames(),
                       sleep=sleep)
            if dfs is None or not len(dfs[0]):
                failed.append(gid)
                continue
            df = dfs[0]
            obj = df.select_dtypes("object").columns
            df[obj] = df[obj].astype(str)
            df.to_parquet(path)
            if i % 100 == 0:
                print(f"  {i}/{len(ids)}")
        print(f"{season}: matchups done, {len(failed)} failed {failed[:10]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=["2021-22", "2022-23", "2023-24", "2024-25"])
    ap.add_argument("--season_type", default="Regular Season")
    ap.add_argument("--out", default="data/context")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--skip_matchups", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fetch_coaches(args.seasons, out, args.sleep)
    fetch_playtypes(args.seasons, out, args.sleep)
    if not args.skip_matchups:
        fetch_matchups(args.seasons, args.season_type, out, args.sleep)


if __name__ == "__main__":
    main()
