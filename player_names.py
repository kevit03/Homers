"""One full name per NBA player ID, shared by every export that shows a player on the dashboard.

Play-by-play only carries surnames ("Brunson") and initials ("J. Brunson"), so names come from, in order:
NBA.com's CommonAllPlayers list (fetch_bbref.py, every player all time), nba_api's bundled player list,
and the season rosters (fetch_rosters.py). Callers pass their own fallback for anyone none of these know.
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd


@lru_cache(maxsize=None)
def full_names(context="data/context"):
    """personId -> "First Last" from every name source on disk (earlier sources win)."""
    ctx, names = Path(context), {}

    def add(ids, vals):
        for pid, name in zip(ids, vals):
            try:
                pid = int(pid)
            except (TypeError, ValueError):
                continue
            name = str(name or "").strip()
            if pid and name and name.lower() not in ("nan", "none"):
                names.setdefault(pid, name)

    f = ctx / "bbref" / "nba_all_players.parquet"
    if f.exists():
        df = pd.read_parquet(f, columns=["PERSON_ID", "DISPLAY_FIRST_LAST"])
        add(df["PERSON_ID"], df["DISPLAY_FIRST_LAST"])
    try:
        from nba_api.stats.static import players
        static = players.get_players()
        add([p["id"] for p in static], [p["full_name"] for p in static])
    except Exception:  # nba_api missing: the files on disk still cover everyone fetched
        pass
    f = ctx / "rosters.parquet"
    if f.exists():
        df = pd.read_parquet(f, columns=["PLAYER_ID", "PLAYER", "season"]).sort_values("season", ascending=False)
        add(df["PLAYER_ID"], df["PLAYER"])
    return names


def full_name(pid, fallback=None, context="data/context"):
    """Full name for one player; `fallback` (then the ID) when no source knows him."""
    try:
        name = full_names(str(context)).get(int(pid))
    except (TypeError, ValueError):
        name = None
    return name or fallback or str(pid)


if __name__ == "__main__":
    n = full_names()
    print(f"{len(n):,} players with full names")
    for pid in (1628973, 1629029, 203999, 2544):
        print(pid, n.get(pid))
