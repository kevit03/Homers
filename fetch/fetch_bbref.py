"""Download current NBA team rosters from Basketball-Reference -> data/context/bbref_rosters.parquet.

The play-by-play says which team a player shot for in past seasons; this says where he is now
(number, position, height, weight, birth date, experience, college). Each player is matched to his
NBA.com personId by name so the dashboard can join the two.

Raw pages are cached in data/context/bbref/<season>/, and requests are spaced 3.5 s apart to stay
under Basketball-Reference's limit of 20 requests a minute.

    python -m fetch.fetch_bbref                    # current season (2026-27 from August 2026 on)
    python -m fetch.fetch_bbref --season 2025-26   # any other season
    python -m fetch.fetch_bbref --refresh          # re-download cached pages (rosters change all offseason)
"""
import argparse
import html
import re
import time
import unicodedata
import urllib.request
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from nba_api.stats.static import players, teams

BASE = "https://www.basketball-reference.com"
# Basketball-Reference abbreviations that differ from NBA.com tricodes
BBREF_ABBR = {"BKN": "BRK", "CHA": "CHO", "PHX": "PHO"}
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140 Safari/537.36"
COLS = {"number": "number", "player": "player", "pos": "pos", "height": "height", "weight": "weight",
        "birth_date": "birth_date", "years_experience": "exp", "college": "college"}


def current_season(today=None):
    """NBA seasons are named for the year they start; from August on, rosters are for the coming season."""
    today = today or date.today()
    start = today.year if today.month >= 8 else today.year - 1
    return f"{start}-{str(start + 1)[2:]}"


def fetch(url, path, refresh, sleep):
    if path.exists() and not refresh:
        return path.read_text()
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                text = r.read().decode("utf-8", "replace")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            time.sleep(sleep)
            return text
        except Exception as e:  # 429 rate limit or network hiccup
            print(f"  {url} attempt {attempt + 1} failed: {e}")
            time.sleep(30 * (attempt + 1))
    return None


def cell_text(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s)).replace("\xa0", " ").strip()


def parse_roster(page):
    """Rows of the id="roster" table as dicts, plus each player's Basketball-Reference id."""
    i = page.find('id="roster"')
    if i < 0:
        return []
    table = page[i:page.index("</table>", i)]
    out = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S):
        cells = dict(re.findall(r'data-stat="(\w+)"[^>]*>(.*?)</t[dh]>', row, re.S))
        if "player" not in cells or cell_text(cells["player"]) in ("", "Player"):  # repeated header rows
            continue
        rec = {COLS[k]: cell_text(v) for k, v in cells.items() if k in COLS}
        m = re.search(r"/players/\w/(\w+)\.html", cells["player"])
        rec["bbref_id"] = m.group(1) if m else ""
        rec["player"] = re.sub(r"\s*\((TW|TW-\w+)\)$", "", rec["player"])  # two-way contract marker
        out.append(rec)
    return out


def norm_name(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'’\-]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def nba_ids(cache):
    """Normalized name -> NBA.com personId, preferring the most recent player when two share a name.

    nba_api's bundled player list lags a season or two, so the live NBA.com list (commonallplayers,
    cached next to the rosters) is layered on top of it to pick up rookies.
    """
    out = {norm_name(p["full_name"]): p["id"] for p in sorted(players.get_players(), key=lambda p: p["is_active"])}
    path = Path(cache) / "nba_all_players.parquet"
    if not path.exists():
        try:
            from nba_api.stats.endpoints import commonallplayers
            df = commonallplayers.CommonAllPlayers(is_only_current_season=0, timeout=60).get_data_frames()[0]
            df[["PERSON_ID", "DISPLAY_FIRST_LAST", "TO_YEAR"]].to_parquet(path)
        except Exception as e:  # stats.nba.com unreachable: the bundled list still covers veterans
            print(f"  live NBA.com player list unavailable ({e}); using nba_api's bundled list")
            return out
    df = pd.read_parquet(path).sort_values("TO_YEAR")
    for pid, name in zip(df["PERSON_ID"], df["DISPLAY_FIRST_LAST"]):
        out[norm_name(name)] = int(pid)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=current_season(), help='e.g. "2026-27"')
    ap.add_argument("--out", default="data/context/bbref_rosters.parquet")
    ap.add_argument("--cache", default="data/context/bbref")
    ap.add_argument("--sleep", type=float, default=3.5)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    end_year = int(args.season[:4]) + 1
    cache = Path(args.cache) / args.season
    ids = nba_ids(args.cache)

    rows = []
    for t in sorted(teams.get_teams(), key=lambda t: t["abbreviation"]):
        abbr = BBREF_ABBR.get(t["abbreviation"], t["abbreviation"])
        url = f"{BASE}/teams/{abbr}/{end_year}.html"
        page = fetch(url, cache / f"{abbr}.html", args.refresh, args.sleep)
        if page is None:
            print(f"  {t['abbreviation']}: failed")
            continue
        roster = parse_roster(page)
        fetched = datetime.fromtimestamp((cache / f"{abbr}.html").stat().st_mtime).strftime("%Y-%m-%d")
        for r in roster:
            rows.append({**r, "season": args.season, "team": t["abbreviation"], "team_name": t["full_name"],
                         "teamId": t["id"], "bbref_team": abbr, "url": url, "fetched": fetched,
                         "personId": ids.get(norm_name(r["player"]))})
        print(f"  {t['abbreviation']}: {len(roster)} players")

    df = pd.DataFrame(rows)
    df["personId"] = df["personId"].astype("Int64")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.out).exists():  # keep other seasons already fetched
        old = pd.read_parquet(args.out)
        df = pd.concat([old[old["season"] != args.season], df], ignore_index=True)
    df.to_parquet(args.out)
    cur = df[df["season"] == args.season]
    miss = cur[cur["personId"].isna()]
    print(f"{args.season}: {len(cur)} roster spots on {cur['team'].nunique()} teams, "
          f"{len(miss)} not matched to an NBA.com id (mostly rookies): {', '.join(miss['player'].head(12))}")


if __name__ == "__main__":
    main()
