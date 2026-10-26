"""Download NBA play-by-play (PlayByPlayV3) for one or more seasons, regular season and playoffs.

Each game is cached as its own parquet file, so the script can be stopped and
re-run safely; finished games are skipped. Playoff games sit in the same season folder
as the regular season; the game ID tells them apart (002... regular season, 004... playoffs).

    python -m fetch.fetch_data                     # every season NBA.com has play-by-play for: 1996-97 on
    python -m fetch.fetch_data --seasons 2021-22 2022-23 2023-24 2024-25
    python -m fetch.fetch_data --season_types Playoffs
    python -m fetch.fetch_data --seasons 2026-27   # a season in progress: only finished games are saved

A game is cached only once it is over: for the current season the game list keeps games with a result dated before
today (US Eastern), and every feed is checked for a final period end before it is written (PlayByPlayV3 has no
game-end event; the last row is the end of the 4th period or of an overtime, with the scores apart).
The daily update for the current season is fetch/update_season.py.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv3

from fetch.seasons import current_season, today_et

SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(2016, 2026)]  # 2016-17 to 2025-26: the context scripts' default
# Play-by-play goes back further, to where NBA.com's starts. Older seasons are box scores only (fetch/fetch_history.py).
PBP_SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(1996, 2026)]  # 1996-97 to 2025-26
SEASON_TYPES = ["Regular Season", "Playoffs"]


def find_games(season: str, season_type: str, retries: int = 4) -> pd.DataFrame:
    """LeagueGameFinder: one row per team per game played or under way (GAME_ID, GAME_DATE, WL, ...)."""
    for attempt in range(retries):
        try:
            return leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                league_id_nullable="00",
                season_type_nullable=season_type,
            ).get_data_frames()[0]
        except Exception as e:  # a hiccup here would end a multi-hour run
            if attempt == retries - 1:
                raise
            print(f"  {season} {season_type} game list attempt {attempt + 1} failed: {e}")
            time.sleep(10 * (attempt + 1))


def get_game_ids(season: str, season_type: str, final_only: bool = False) -> list[str]:
    """Game IDs of a season; final_only keeps games with a result for both teams dated before today (US Eastern),
    so a game still being played (or finished minutes ago, with the feed still settling) waits for tomorrow."""
    df = find_games(season, season_type)
    if final_only and len(df):
        done = df["WL"].notna() & (pd.to_datetime(df["GAME_DATE"]).dt.date < today_et())
        ok = done.groupby(df["GAME_ID"]).agg(["all", "size"])
        return sorted(ok.index[ok["all"] & (ok["size"] == 2)])
    return sorted(df["GAME_ID"].unique())


def is_final(df: pd.DataFrame) -> bool:
    """The feed reaches the end of regulation or an overtime with the score not tied."""
    if not len(df):
        return False
    last = df.iloc[-1]
    home, away = (pd.to_numeric(df[c], errors="coerce").max() for c in ("scoreHome", "scoreAway"))
    return (str(last.get("actionType")) == "period" and str(last.get("subType")) == "end"
            and int(last.get("period", 0)) >= 4 and home != away)


def fetch_game(gid: str, season: str, path: Path, retries: int = 3, final_only: bool = False):
    """True if saved, False if every attempt failed, None if final_only and the feed isn't final yet (not saved)."""
    for attempt in range(retries):
        try:
            df = playbyplayv3.PlayByPlayV3(game_id=gid, timeout=30).get_data_frames()[0]
            if final_only and not is_final(df):
                return None
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
    ap.add_argument("--seasons", nargs="+", default=PBP_SEASONS)
    ap.add_argument("--season_types", nargs="+", default=SEASON_TYPES, help='"Regular Season", "Playoffs" or "PlayIn"')
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--sleep", type=float, default=0.6, help="seconds between requests")
    args = ap.parse_args()

    for season in args.seasons:
        fetch_season_pbp(season, args.season_types, args.out, args.sleep)


def fetch_season_pbp(season, season_types=SEASON_TYPES, out="data/raw", sleep=0.6):
    """Cache every game of a season not on disk yet; for the current season only finished games.
    Returns (games listed, newly saved, failed IDs, IDs whose feed wasn't final yet)."""
    final_only = season >= current_season()
    sdir = Path(out) / season
    sdir.mkdir(parents=True, exist_ok=True)
    listed, saved, failed, pending = 0, 0, [], []
    for season_type in season_types:
        ids = get_game_ids(season, season_type, final_only=final_only)
        listed += len(ids)
        print(f"{season} {season_type}: {len(ids)} {'finished ' if final_only else ''}games")
        for i, gid in enumerate(ids):
            path = sdir / f"{gid}.parquet"
            if path.exists():
                continue
            ok = fetch_game(gid, season, path, final_only=final_only)
            if ok:
                saved += 1
            elif ok is None:
                pending.append(gid)
            else:
                failed.append(gid)
            time.sleep(sleep)
            if i % 100 == 0:
                print(f"  {i}/{len(ids)}")
        print(f"{season} {season_type}: done, {len(failed)} failed {failed[:10]}"
              + (f", {len(pending)} not final yet {pending[:10]}" if pending else ""))
    return listed, saved, failed, pending


if __name__ == "__main__":
    main()
