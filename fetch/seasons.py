"""Season names and dates the fetchers share; no network imports, so anything can use it.

NBA seasons are named for the year they start ("2026-27"). From August on, the coming season is the current one
(rosters are set for it); games are scheduled by US Eastern dates.
"""
from datetime import date, datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def season_name(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[2:]}"


def today_et() -> date:
    return datetime.now(ET).date()


def current_season(today=None) -> str:
    today = today or today_et()
    return season_name(today.year if today.month >= 8 else today.year - 1)
