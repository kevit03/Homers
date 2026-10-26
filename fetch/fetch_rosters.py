"""Download team rosters (name, number, position, height, weight, age, experience, school) for the
player profiles in the dashboard -> data/context/rosters.parquet. Cached, so re-running is cheap.

    python -m fetch.fetch_rosters --seasons 2021-22 2022-23 2023-24 2024-25
    python -m fetch.fetch_rosters --seasons 2026-27 --refresh   # re-download a season in progress
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams

from fetch.fetch_data import SEASONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=SEASONS)
    ap.add_argument("--out", default="data/context/rosters.parquet")
    ap.add_argument("--sleep", type=float, default=1.2, help="seconds between requests")
    ap.add_argument("--refresh", action="store_true", help="re-download these seasons even if cached")
    args = ap.parse_args()
    fetch_rosters(args.seasons, args.out, args.sleep, refresh=args.seasons if args.refresh else ())


def fetch_rosters(seasons, out, sleep=1.2, refresh=()):
    """NBA.com rosters for every team of `seasons` into `out`; `refresh` seasons are re-downloaded.
    Returns the number of team-seasons that failed."""
    path = Path(out)
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["season", "TeamID"])
    old = have[have["season"].isin(list(refresh))]
    have = have[~have["season"].isin(list(refresh))]
    frames, failed = [have], 0
    for season in seasons:
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
            else:
                failed += 1
                # keep yesterday's roster for this team rather than none
                frames.append(old[(old["season"] == season) & (old["TeamID"] == t["id"])])
            time.sleep(sleep)
        print(f"{season}: rosters done")
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_parquet(path)
    return failed


if __name__ == "__main__":
    main()
