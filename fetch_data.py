"""Download NBA play-by-play (PlayByPlayV3) for one or more seasons, regular season and playoffs.

Each game is cached as its own parquet file, so the script can be stopped and
re-run safely; finished games are skipped. Playoff games sit in the same season folder
as the regular season; the game ID tells them apart (002... regular season, 004... playoffs).

    python fetch_data.py --seasons 2021-22 2022-23 2023-24 2024-25
    python fetch_data.py --season_types Playoffs
"""
import argparse
import time
from pathlib import Path

from nba_api.stats.endpoints import leaguegamefinder, playbyplayv3

SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(2016, 2026)]  # 2016-17 to 2025-26
SEASON_TYPES = ["Regular Season", "Playoffs"]


def get_game_ids(season: str, season_type: str) -> list[str]:
    df = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",
        season_type_nullable=season_type,
    ).get_data_frames()[0]
    return sorted(df["GAME_ID"].unique())


def fetch_game(gid: str, season: str, path: Path, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            df = playbyplayv3.PlayByPlayV3(game_id=gid, timeout=30).get_data_frames()[0]
            df["gameId"] = gid
            df["season"] = season
            obj_cols = df.select_dtypes("object").columns
            df[obj_cols] = df[obj_cols].astype(str)  # avoid mixed-type parquet errors
            df.to_parquet(path)
            return True
        except Exception as e:  # network hiccups / rate limiting
            print(f"  {gid} attempt {attempt + 1} failed: {e}")
            time.sleep(5 * (attempt + 1))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=SEASONS)
    ap.add_argument("--season_types", nargs="+", default=SEASON_TYPES, help='"Regular Season", "Playoffs" or "PlayIn"')
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--sleep", type=float, default=0.6, help="seconds between requests")
    args = ap.parse_args()

    for season in args.seasons:
        sdir = Path(args.out) / season
        sdir.mkdir(parents=True, exist_ok=True)
        for season_type in args.season_types:
            ids = get_game_ids(season, season_type)
            print(f"{season} {season_type}: {len(ids)} games")
            failed = []
            for i, gid in enumerate(ids):
                path = sdir / f"{gid}.parquet"
                if path.exists():
                    continue
                if not fetch_game(gid, season, path):
                    failed.append(gid)
                time.sleep(args.sleep)
                if i % 100 == 0:
                    print(f"  {i}/{len(ids)}")
            print(f"{season} {season_type}: done, {len(failed)} failed {failed[:10]}")


if __name__ == "__main__":
    main()
