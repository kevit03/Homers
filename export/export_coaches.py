"""Coaches for the dashboard's Coaches tab: every head coach since the first season in the data, with a photo,
Basketball-Reference's record, titles and awards, and each team season's offensive and defensive play data.

Sources (all in data/context):
  bbref_coach_seasons.parquet, bbref_coaches.json   fetch_accolades.py (Basketball-Reference)
  nba_coaches.parquet                               fetch_accolades.py (NBA.com coach ids and headshots)
  tracking.parquet, playtypes.parquet               fetch_context.py (see export_tracking.py)

Used by export_dashboard.py.
"""
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

from export_tracking import labels, team_units

NBA_HEAD = "https://cdn.nba.com/headshots/nba/latest/1040x760/{}.png"
MINOR_AWARD = re.compile(r"Coach of the Month", re.I)


def _norm(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", "", s).split()


def _same_person(a, b):
    """Basketball-Reference and NBA.com spell some coaches differently ("Jordi Fernández Torres" vs
    "Jordi Fernandez", "J.B." vs "JB"): same first word and a shared later word is enough within one team season."""
    x, y = _norm(a), _norm(b)
    return bool(x and y) and x[0][:2] == y[0][:2] and bool(set(x[1:]) & set(y[1:]) or x[-1] == y[-1])


def _badges(page):
    """Basketball-Reference style accolades: "4x NBA Champ", "2015-16 COY", "12x Coach of the Month"."""
    out = []
    if page.get("titles"):
        out.append(f"{page['titles']}x NBA Champ" if page["titles"] > 1 else
                   next(f"{s['season']} NBA Champ" for s in page["seasons"] if s["note"] == "NBA Champions"))
    by = Counter(a["award"] for a in page.get("awards", []) if not MINOR_AWARD.search(a["award"]))
    order = ["Coach of the Year"] + sorted(k for k in by if k != "Coach of the Year")
    for award in order:
        if not by.get(award):
            continue
        name = "COY" if award == "Coach of the Year" else award
        seasons = [a["season"] for a in page["awards"] if a["award"] == award]
        out.append(f"{by[award]}x {name}" if by[award] > 1 else f"{seasons[0]} {name}")
    months = [a["season"] for a in page.get("awards", []) if MINOR_AWARD.search(a["award"])]
    if months:
        out.append(f"{len(months)}x Coach of the Month" if len(months) > 1 else f"{months[0]} Coach of the Month")
    if page.get("hof"):
        out.append("Hall of Fame")
    return out


def build_coaches_payload(context="data/context", shot_coaches=None):
    ctx = Path(context)
    cs_path, pages_path = ctx / "bbref_coach_seasons.parquet", ctx / "bbref_coaches.json"
    if not cs_path.exists():
        return None
    cs = pd.read_parquet(cs_path)
    pages = json.loads(pages_path.read_text()) if pages_path.exists() else {}
    wiki = json.loads((ctx / "coach_photos_wiki.json").read_text()) if (ctx / "coach_photos_wiki.json").exists() else {}
    nba = pd.read_parquet(ctx / "nba_coaches.parquet") if (ctx / "nba_coaches.parquet").exists() else pd.DataFrame(columns=["season", "team", "coachId", "coach", "photo"])
    current = cs.loc[cs["g"].isna(), "season"].max() if cs["g"].isna().any() else None  # the current season has no record yet

    # Basketball-Reference coach -> NBA.com coach id by name. Not through team seasons: NBA.com's roster feed
    # lists the wrong head coach for some of them (Indiana 2019-20 shows Nate Bjorkgren, who started in 2020-21).
    nba_names = nba.drop_duplicates("coachId", keep="last")
    nba_photo = dict(zip(nba_names["coachId"].astype(int), nba_names["photo"].astype(bool)))
    nba_id = {}
    for bid, name in cs.drop_duplicates("bbref_coach")[["bbref_coach", "coach"]].itertuples(index=False):
        hits = [int(i) for i, n in zip(nba_names["coachId"], nba_names["coach"]) if _same_person(name, n)]
        if len(hits) == 1:
            nba_id[bid] = hits[0]
    shot_idx = {c["id"]: i for i, c in enumerate(shot_coaches or [])}

    teams = team_units(context)
    data_seasons = set(teams["league"]) if teams else set()
    out = []
    for bid, g in cs.groupby("bbref_coach"):
        g = g.sort_values("season")
        page = pages.get(bid, {})
        nid = nba_id.get(bid)
        photos = ([NBA_HEAD.format(nid)] if nid and nba_photo.get(nid) else []) + [u for u in (page.get("photo"), (wiki.get(bid) or {}).get("photo")) if u]
        stints = [{"season": s, "team": t, "g": int(gg) if pd.notna(gg) else 0, "w": int(w) if pd.notna(w) else 0, "l": int(l) if pd.notna(l) else 0,
                   "pw": int(pw) if pd.notna(pw) else 0, "pl": int(pl) if pd.notna(pl) else 0, "data": f"{s}|{t}" in (teams or {}).get("units", {})}
                  for s, t, gg, w, l, pw, pl in zip(g["season"], g["team"], g["g"], g["w"], g["l"], g["pw"], g["pl"]) if s != current]
        now = g.loc[g["season"] == current, "team"].tolist() if current else []
        out.append({
            "id": bid, "nba": nid, "name": page.get("name") or g["coach"].iloc[-1], "photos": photos,
            "now": now[0] if now else None, "last": g["team"].iloc[-1],
            "career": page.get("career") or {}, "titles": page.get("titles", 0), "coy": page.get("coy", 0),
            "acc": _badges(page) if page else [],
            "history": [[s["season"], s["team"], s["g"], s["w"], s["l"], s["pg"], s["pw"], s["pl"], s["finish"], s["note"]]
                        for s in page.get("seasons", []) if s["g"]],
            "stints": stints,
            "shot": shot_idx.get(nid, -1),
        })
    out.sort(key=lambda c: (c["now"] is None, c["now"] or "", _norm(c["name"])[-1:] or [""]))
    units = {k: u for k, u in teams["units"].items()} if teams else {}
    return {"current": current, "seasons": sorted(data_seasons), "coaches": out, "units": units,
            "league": teams["league"] if teams else {}, **labels(),
            "fetched": max((p.get("fetched", "") for p in pages.values()), default=""),
            "credits": [{"title": w["name"], **{k: w[k] for k in ("author", "license", "license_url", "source")}}
                        for bid, w in wiki.items() if w and any(c["id"] == bid and c["photos"] and c["photos"][0] == w["photo"] for c in out)]}


if __name__ == "__main__":
    p = build_coaches_payload()
    print(f"{len(p['coaches'])} coaches ({sum(1 for c in p['coaches'] if c['now'])} current), "
          f"{sum(1 for c in p['coaches'] if c['photos'])} with photos, {sum(1 for c in p['coaches'] if c['nba'])} with NBA.com ids")
    for c in p["coaches"][:6]:
        print(c["name"], c["now"], c["nba"], c["career"], c["acc"], len(c["stints"]), c["photos"][:1])
    print("no NBA id:", [c["name"] for c in p["coaches"] if not c["nba"]])
    print(f"{len(json.dumps(p)) / 1e3:.0f} KB")
