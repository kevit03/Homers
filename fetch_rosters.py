"""Download team rosters (name, number, position, height, weight, age, experience, school) for the
player profiles in the dashboard -> data/context/rosters.parquet. Cached, so re-running is cheap.

    python fetch_rosters.py --seasons 2021-22 2022-23 2023-24 2024-25
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams

from fetch_data import SEASONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=SEASONS)
    ap.add_argument("--out", default="data/context/rosters.parquet")
    ap.add_argument("--sleep", type=float, default=1.2, help="seconds between requests")
    args = ap.parse_args()

    path = Path(args.out)
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["season", "TeamID"])
    frames = [have]
    for season in args.seasons:
        for t in teams.get_teams():
            if ((have["season"] == season) & (have["TeamID"] == t["id"])).any():
                continue
            for attempt in range(4):
                try:
                    df = commonteamroster.CommonTeamRoster(team_id=t["id"], season=season, timeout=30).get_data_frames()[0]
                    df["season"] = season
                    frames.append(df.astype({c: str for c in df.select_dtypes("object").columns}))
                    break
                except Exception as e:  # network hiccups / rate limiting
                    print(f"  {season} {t['abbreviation']} attempt {attempt + 1} failed: {e}")
                    time.sleep(10 * (attempt + 1))
            time.sleep(args.sleep)
        print(f"{season}: rosters done")
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_parquet(path)


if __name__ == "__main__":
    main()
