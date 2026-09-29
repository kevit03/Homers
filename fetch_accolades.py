"""Download player accolades and coach records from Basketball-Reference.

  - season stats pages (one per season)   -> data/context/bbref_players.parquet
    every player's Basketball-Reference id, matched to his NBA.com personId, and that season's award
    votes and selections (e.g. "MVP-1,AS,NBA1")
  - each player's page                     -> data/context/bbref_accolades.json
    his career accolades as Basketball-Reference lists them under his name ("4x MVP", "22x All Star")
  - season coaches pages (one per season)  -> data/context/bbref_coach_seasons.parquet
    every head coach of every team, with that season's record (mid-season changes included)
  - each coach's page                      -> data/context/bbref_coaches.json
    career record, championships, Coach of the Year and other awards, season by season, and photo

Requests are spaced 3.5 s apart to stay under Basketball-Reference's limit of 20 a minute. Player and
coach pages are parsed as they arrive and only the parsed result is kept, so a run can be stopped and
resumed. Player pages go in order of minutes played, so the most-watched players come first.

    python fetch_accolades.py                   # every season in fetch_data.SEASONS
    python fetch_accolades.py --skip_players    # coaches only (about 5 minutes)
    python fetch_accolades.py --refresh         # re-download pages already parsed
"""
import argparse
import html
import json
import re
import time
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

from fetch_bbref import BASE, BBREF_ABBR, UA, cell_text, current_season, norm_name
from fetch_data import SEASONS

NBA_ABBR = {v: k for k, v in BBREF_ABBR.items()}  # Basketball-Reference -> NBA.com tricodes
NBA_ABBR.update({"NOH": "NOP", "NJN": "BKN", "CHH": "CHA", "SEA": "OKC", "VAN": "MEM"})


class Fetcher:
    """GETs spaced `sleep` seconds apart; `until` stops reading once a marker has arrived (player pages are
    ~2 MB, the accolades sit in the first ~100 KB)."""

    def __init__(self, sleep):
        self.sleep, self.last = sleep, 0.0

    def get(self, url, until=None):
        for attempt in range(4):
            time.sleep(max(0.0, self.last + self.sleep - time.time()))
            self.last = time.time()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=30) as r:
                    if until is None:
                        return r.read().decode("utf-8", "replace")
                    buf = b""
                    while chunk := r.read(65536):
                        buf += chunk
                        if any(m in buf for m in until):
                            break
                    return buf.decode("utf-8", "replace")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                print(f"  {url} attempt {attempt + 1} failed: {e}")
                time.sleep(30 * (attempt + 1))
            except Exception as e:  # rate limit or network hiccup
                print(f"  {url} attempt {attempt + 1} failed: {e}")
                time.sleep(30 * (attempt + 1))
        return None


def table_rows(page, table_id):
    """Rows of one table as {data-stat: raw cell html}, header rows dropped."""
    i = page.find(f'id="{table_id}"')
    if i < 0:
        return []
    body = page[i:page.find("</table>", i)]
    rows = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S):
        rows.append(dict(re.findall(r'data-stat="(\w+)"[^>]*>(.*?)</t[dh]>', row, re.S)))
    return rows


def link_id(cell, kind):
    m = re.search(rf"/{kind}/(?:\w/)?(\w+)\.html", cell or "")
    return m.group(1) if m else ""


def num(s, cast=int):
    try:
        return cast(cell_text(s))
    except (TypeError, ValueError):
        return None


def season_end(season):
    return int(season[:4]) + 1


# ---------------------------------------------------------------- players

def fetch_player_seasons(f, seasons, cache):
    """Every player row of each season's per-game table (traded players: one row per team plus a total)."""
    rows = []
    for season in seasons:
        path = cache / "seasons" / f"NBA_{season_end(season)}_per_game.html"
        if path.exists():
            page = path.read_text()
        else:
            page = f.get(f"{BASE}/leagues/NBA_{season_end(season)}_per_game.html")
            if page is None:
                print(f"  {season}: per-game page failed")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(page)
        n = 0
        for r in table_rows(page, "per_game_stats"):
            pid = link_id(r.get("name_display") or r.get("player"), "players")
            if not pid:
                continue
            team = cell_text(r.get("team_name_abbr") or r.get("team_id") or "")
            rows.append({"season": season, "bbref_id": pid, "player": cell_text(r.get("name_display") or r.get("player")),
                         "team": NBA_ABBR.get(team, team), "g": num(r.get("games") or r.get("g")) or 0,
                         "mpg": num(r.get("mp_per_g"), float) or 0.0,
                         "awards": cell_text(r.get("awards", "")).replace(" ", "")})
            n += 1
        print(f"  {season}: {n} player rows")
    df = pd.DataFrame(rows)
    # a traded player's "2TM" total row carries his season awards; keep one row per player-season
    df["min"] = df["g"] * df["mpg"]
    tot = df[df["team"].str.fullmatch(r"\dTM")]
    one = df[~df["team"].str.fullmatch(r"\dTM")]
    agg = (one.groupby(["season", "bbref_id"], as_index=False)
           .agg(player=("player", "first"), teams=("team", lambda t: ",".join(dict.fromkeys(t))),
                g=("g", "sum"), min=("min", "sum"), awards=("awards", "max")))
    if len(tot):
        agg = agg.merge(tot[["season", "bbref_id", "awards"]].rename(columns={"awards": "awards_tot"}), how="left")
        agg["awards"] = agg["awards_tot"].where(agg["awards_tot"].fillna("") != "", agg["awards"])
        agg = agg.drop(columns="awards_tot")
    return agg


def match_person_ids(df, context):
    """NBA.com personId for each Basketball-Reference player: exact name first, then the NBA.com season
    roster of the same team and season (catches "Nicolas" vs "Nic", suffixes, accents)."""
    ctx = Path(context)
    cands = {}
    allp = ctx / "bbref" / "nba_all_players.parquet"
    if allp.exists():
        a = pd.read_parquet(allp)
        a = a[pd.to_numeric(a["TO_YEAR"], errors="coerce").fillna(0) >= int(SEASONS[0][:4])]
        for pid, name in zip(a["PERSON_ID"], a["DISPLAY_FIRST_LAST"]):
            cands.setdefault(norm_name(name), set()).add(int(pid))
    ros = pd.read_parquet(ctx / "rosters.parquet") if (ctx / "rosters.parquet").exists() else pd.DataFrame()
    by_team = {}
    if len(ros):
        from nba_api.stats.static import teams
        tri = {t["id"]: t["abbreviation"] for t in teams.get_teams()}
        for season, tid, name, pid in zip(ros["season"], ros["TeamID"], ros["PLAYER"], ros["PLAYER_ID"]):
            by_team.setdefault((season, tri.get(int(tid), "")), []).append((norm_name(name), int(pid)))
            cands.setdefault(norm_name(name), set()).add(int(pid))
    cur = ctx / "bbref_rosters.parquet"
    direct = {}
    if cur.exists():
        c = pd.read_parquet(cur).dropna(subset=["personId"])
        direct = dict(zip(c["bbref_id"], c["personId"].astype(int)))

    def last(n):
        return n.split()[-1] if n.split() else n

    out = {}
    for bid, g in df.groupby("bbref_id"):
        if bid in direct:
            out[bid] = direct[bid]
            continue
        n = norm_name(g["player"].iloc[0])
        ids = cands.get(n, set())
        if len(ids) == 1:
            out[bid] = next(iter(ids))
            continue
        # same team and season on the NBA.com roster, same surname
        hits = set()
        for season, teams_ in zip(g["season"], g["teams"]):
            for t in teams_.split(","):
                hits |= {pid for rn, pid in by_team.get((season, t), []) if last(rn) == last(n) and (not ids or pid in ids)}
        if len(hits) == 1:
            out[bid] = next(iter(hits))
    return out


def parse_bling(page):
    i = page.find('id="bling"')
    if i < 0:
        return []
    ul = page[i:page.find("</ul>", i)]
    return [cell_text(x) for x in re.findall(r"<li[^>]*>(.*?)</li>", ul, re.S) if cell_text(x)]


def fetch_player_pages(f, ids, out_path, refresh, limit):
    have = json.loads(out_path.read_text()) if out_path.exists() else {}
    todo = [b for b in ids if refresh or b not in have][:limit]
    print(f"player pages: {len(have)} parsed, {len(todo)} to fetch (~{len(todo) * f.sleep / 60:.0f} min)")
    for k, bid in enumerate(todo):
        page = f.get(f"{BASE}/players/{bid[0]}/{bid}.html", until=(b'class="uni_holder', b'id="all_', b'id="inner_nav"'))
        if page is None:
            continue
        m = re.search(r"<img[^>]+src=['\"]([^'\"]+/headshots/[^'\"]+)['\"]", page)
        have[bid] = {"acc": parse_bling(page), "photo": m.group(1) if m else "", "fetched": date.today().isoformat()}
        if k % 25 == 0 or k == len(todo) - 1:
            out_path.write_text(json.dumps(have, ensure_ascii=False, separators=(",", ":")))
            print(f"  {k + 1}/{len(todo)}  {bid}: {', '.join(have[bid]['acc'][:4]) or '-'}")
    out_path.write_text(json.dumps(have, ensure_ascii=False, separators=(",", ":")))
    return have


# ---------------------------------------------------------------- coaches

def fetch_coach_seasons(f, seasons, cache):
    rows = []
    for season in seasons:
        path = cache / "seasons" / f"NBA_{season_end(season)}_coaches.html"
        if path.exists():
            page = path.read_text()
        else:
            page = f.get(f"{BASE}/leagues/NBA_{season_end(season)}_coaches.html")
            if page is None:
                print(f"  {season}: coaches page failed")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(page)
        n = 0
        for r in table_rows(page, "NBA_coaches"):
            cid = link_id(r.get("coach"), "coaches")
            if not cid:
                continue
            team = cell_text(r.get("team", ""))
            rows.append({"season": season, "bbref_coach": cid, "coach": cell_text(r["coach"]), "team": NBA_ABBR.get(team, team),
                         "g": num(r.get("cur_g")) or 0, "w": num(r.get("cur_w")) or 0, "l": num(r.get("cur_l")) or 0,
                         "pg": num(r.get("cur_g_p")) or 0, "pw": num(r.get("cur_w_p")) or 0, "pl": num(r.get("cur_l_p")) or 0})
            n += 1
        print(f"  {season}: {n} coaches")
    return pd.DataFrame(rows)


def current_coaches(context):
    """This season's head coaches from the team pages fetch_bbref.py cached."""
    season = current_season()
    out = []
    for page_path in sorted((Path(context) / "bbref" / season).glob("*.html")):
        m = re.search(r'Coach:</strong>\s*<a href="/coaches/(\w+)\.html">([^<]+)</a>', page_path.read_text())
        if m:
            abbr = page_path.stem
            out.append({"season": season, "bbref_coach": m.group(1), "coach": html.unescape(m.group(2)), "team": NBA_ABBR.get(abbr, abbr)})
    return pd.DataFrame(out)


def parse_coach(page):
    m = re.search(r"<img[^>]+src=['\"]([^'\"]+/headshots/[^'\"]+)['\"]", page)
    seasons, career = [], {}
    for r in table_rows(page, "coach-stats"):
        s = cell_text(r.get("season", ""))
        rec = {k: num(r.get(src)) for k, src in (("g", "g"), ("w", "wins"), ("l", "losses"), ("pg", "g_playoffs"),
                                                  ("pw", "wins_playoffs"), ("pl", "losses_playoffs"))}
        if re.fullmatch(r"\d{4}-\d{2}", s):
            seasons.append({"season": s, "team": NBA_ABBR.get(cell_text(r.get("team_id", "")), cell_text(r.get("team_id", ""))),
                            **rec, "finish": cell_text(r.get("rank_team", "")), "note": cell_text(r.get("coach_remarks", ""))})
        elif s == "Career" and not career:
            career = rec
    awards = [{"season": cell_text(r.get("season", "")), "award": cell_text(r.get("award_id", ""))}
              for r in table_rows(page, "coach-awards") if re.fullmatch(r"\d{4}-\d{2}", cell_text(r.get("season", "")))]
    hof = bool(re.search(r"Hall of Fame", page[:page.find('id="coach-stats"')] if 'id="coach-stats"' in page else ""))
    return {"photo": m.group(1) if m else "", "career": career, "seasons": seasons, "awards": awards,
            "titles": sum(1 for s in seasons if s["note"] == "NBA Champions"),
            "coy": sum(1 for a in awards if a["award"] == "Coach of the Year"), "hof": hof}


def fetch_coach_pages(f, ids, out_path, names, refresh):
    have = json.loads(out_path.read_text()) if out_path.exists() else {}
    todo = [c for c in ids if refresh or c not in have]
    print(f"coach pages: {len(have)} parsed, {len(todo)} to fetch (~{len(todo) * f.sleep / 60:.0f} min)")
    for k, cid in enumerate(todo):
        page = f.get(f"{BASE}/coaches/{cid}.html")
        if page is None:
            continue
        have[cid] = {"name": names.get(cid, cid), **parse_coach(page), "fetched": date.today().isoformat()}
        if k % 10 == 0 or k == len(todo) - 1:
            out_path.write_text(json.dumps(have, ensure_ascii=False, separators=(",", ":")))
            c = have[cid]
            print(f"  {k + 1}/{len(todo)}  {c['name']}: {c['career'].get('w')}-{c['career'].get('l')}, {c['titles']} titles, {c['coy']} COY")
    out_path.write_text(json.dumps(have, ensure_ascii=False, separators=(",", ":")))
    return have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=SEASONS)
    ap.add_argument("--context", default="data/context")
    ap.add_argument("--sleep", type=float, default=3.5)
    ap.add_argument("--refresh", action="store_true", help="re-download player and coach pages already parsed")
    ap.add_argument("--skip_players", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="fetch at most this many player pages this run")
    args = ap.parse_args()
    ctx = Path(args.context)
    cache = ctx / "bbref"
    f = Fetcher(args.sleep)

    print("coaches")
    cs = fetch_coach_seasons(f, args.seasons, cache)
    cur = current_coaches(args.context)
    allc = pd.concat([cs, cur], ignore_index=True)
    allc.to_parquet(ctx / "bbref_coach_seasons.parquet")
    names = dict(zip(allc["bbref_coach"], allc["coach"]))
    order = allc.groupby("bbref_coach")["season"].max().sort_values(ascending=False).index.tolist()
    fetch_coach_pages(f, order, ctx / "bbref_coaches.json", names, args.refresh)

    print("players")
    ps = fetch_player_seasons(f, args.seasons, cache)
    ids = match_person_ids(ps, args.context)
    ps["personId"] = ps["bbref_id"].map(ids).astype("Int64")
    ps.to_parquet(ctx / "bbref_players.parquet")
    n_ids = ps["bbref_id"].nunique()
    print(f"{n_ids} players, {len(ids)} matched to an NBA.com id; unmatched: "
          f"{', '.join(ps.loc[ps['personId'].isna(), 'player'].drop_duplicates().head(12))}")
    if not args.skip_players:
        by_min = ps.groupby("bbref_id")["min"].sum().sort_values(ascending=False).index.tolist()
        fetch_player_pages(f, by_min, ctx / "bbref_accolades.json", args.refresh, args.limit)


if __name__ == "__main__":
    main()
