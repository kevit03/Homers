"""Player profiles for the dashboard: season box scores, fantasy points, game logs, and, for every
replayed game, who made each play and who was on the floor.

Stats are grouped by tokenize_pbp.season_key: "2024-25" is the regular season and "2024-25 Playoffs"
its playoffs, so the two never mix. A player's game log covers his latest season, playoffs included.

Everything comes from the raw play-by-play (plus data/context/rosters.parquet for bios, if fetched),
so it works without the shot model. Per-game results are cached in data/processed/profiles_cache.pkl.

    python -m export.export_profiles                 # build/refresh the cache and print a summary
"""
import argparse
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from models.build_shots import TEAM_ID_MIN, load_matchups, name_map, norm
from models.player_names import full_name
from models.tokenize_pbp import STOI, classify, game_order, parse_clock, season_key

STATS = ["min", "pts", "fgm", "fga", "tpm", "tpa", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf"]
FP_WEIGHTS = {"pts": 1, "reb": 1.2, "ast": 1.5, "stl": 3, "blk": 3, "tov": -1}  # NBA.com / FanDuel scoring
LOG_STATS = ["ast", "stl", "blk", "tov", "fgm", "fga", "tpm", "tpa", "ftm", "fta"]
EXTRA = {"AST": 0, "STL": 1, "BLK": 2}
AST_RE = re.compile(r"\(([^()]+?) \d+ AST\)")
SUB_RE = re.compile(r"SUB:\s*(.+?)\s+FOR\s+(.+)")
CACHE_VERSION = 4  # 4: rows in game-clock order (tokenize_pbp.game_order)
_ROSTER_NAMES = {}


def is_player(pid):
    return 0 < pid < TEAM_ID_MIN


def prep(df):
    """Same row order as tokenize_pbp.process_game, so token positions line up."""
    df = game_order(df)
    df["personId"] = pd.to_numeric(df["personId"], errors="coerce").fillna(0).astype(int)
    df["teamId"] = pd.to_numeric(df["teamId"], errors="coerce").fillna(0).astype(int)
    df["period"] = pd.to_numeric(df["period"], errors="coerce").fillna(1).astype(int)
    df["teamTricode"] = df["teamTricode"].fillna("").astype(str).replace("None", "")
    return df


def roster_names(ctx):
    """(season, teamId) -> {name form: personId} from data/context/rosters.parquet, to resolve names in SUB: lines."""
    key = str(ctx)
    if key not in _ROSTER_NAMES:
        out = defaultdict(dict)
        path = Path(ctx) / "rosters.parquet" if ctx else None
        if path and path.exists():
            r = pd.read_parquet(path, columns=["season", "TeamID", "PLAYER", "PLAYER_ID"]).dropna()
            for season, tid, name, pid in zip(r["season"], r["TeamID"], r["PLAYER"], r["PLAYER_ID"]):
                parts = str(name).split()
                if len(parts) < 2:
                    continue
                first, last = parts[0], " ".join(parts[1:])
                for form in (name, last, parts[-1], f"{first[0]}. {last}", f"{first[:2]}. {last}"):
                    out[(season, int(tid))].setdefault(norm(form), int(pid))
        _ROSTER_NAMES[key] = out
    return _ROSTER_NAMES[key]


def floor_by_clock(df, names, teams_):
    """On-floor players after every row, plus minutes played.

    Rows are walked in game-clock order, not actionNumber order: late scorekeeper entries get high
    actionNumbers, so substitutions can otherwise land after plays that happened later.
    Each period starts with the players whose first appearance isn't a sub-in (topped up from the
    previous period's lineup); a player who makes a play while we think he's on the bench is put back on.
    If that leaves six on the floor (a missed sub-out), the one who has been idle longest comes off.
    """
    clock = df["clock"].map(parse_clock).to_numpy()
    recs = df.to_dict("records")
    on_floor, minutes = {}, defaultdict(float)
    prev_end = {t: [] for t in teams_}
    for period, pdf in df.groupby("period", sort=True):
        order = sorted(pdf.index, key=lambda i: (-clock[i], i))
        events = []  # (row, team, player, kind)
        for i in order:
            r = recs[i]
            t, pid = r["teamTricode"], int(r["personId"])
            if t not in prev_end:
                continue
            if r["actionType"] == "Substitution":
                m = SUB_RE.match(str(r["description"]))
                inc = names[t].get(norm(m.group(1))) if m else None
                if is_player(pid):
                    events.append((i, t, pid, "out"))
                if inc and is_player(int(inc)):
                    events.append((i, t, int(inc), "in"))
            elif is_player(pid) and r["actionType"] != "Ejection" and not (
                    r["actionType"] == "Foul" and "technical" in str(r["subType"]).lower()):  # bench techs don't mean he's on
                events.append((i, t, pid, "act"))
        first = {}
        for _, t, pid, kind in events:
            first.setdefault((t, pid), kind)
        cur = {}
        for t in prev_end:
            start = [pid for (tt, pid), k in first.items() if tt == t and k != "in"][:5]
            start += [pid for pid in prev_end[t] if len(start) < 5 and pid not in start and (t, pid) not in first]
            cur[t] = set(start)
        by_row = defaultdict(list)
        for i, t, pid, kind in events:
            by_row[i].append((t, pid, kind))
        last_c = 720.0 if period <= 4 else 300.0
        seen = {}
        for k, i in enumerate(order):
            dt = last_c - clock[i]
            if dt > 0:
                for five in cur.values():
                    for q in five:
                        minutes[q] += dt / 60
                last_c = clock[i]
            for t, pid, kind in by_row.get(i, ()):
                seen[pid] = k
                if kind == "out":
                    cur[t].discard(pid)
                else:
                    cur[t].add(pid)
                    if len(cur[t]) > 5:
                        cur[t].discard(min((q for q in cur[t] if q != pid), key=lambda q: seen.get(q, -1)))
            on_floor[i] = {t: frozenset(v) for t, v in cur.items()}
        prev_end = {t: sorted(v) for t, v in cur.items()}
    return on_floor, minutes


def process_game(df, ctx=None):
    df = prep(df)
    gid, season = str(df["gameId"].iloc[0]), str(df["season"].iloc[0])
    side = df[df["location"].isin(["h", "v"]) & (df["teamId"] > 0) & (df["teamTricode"] != "")].drop_duplicates("location")
    team_of = {r.location: (int(r.teamId), r.teamTricode) for r in side.itertuples()}
    if set(team_of) != {"h", "v"}:
        return None
    H, A = team_of["h"][1], team_of["v"][1]
    m = load_matchups(ctx, season, gid) if ctx else None
    names = name_map(df, m)
    for tid, tri in team_of.values():  # names seen in this game win; rosters fill the gaps
        for form, pid in roster_names(ctx).get((season, tid), {}).items():
            names[tri].setdefault(form, pid)
    floor, minutes = floor_by_clock(df, names, [H, A])

    full = {}
    if m is not None:
        for sfx in ("Off", "Def"):
            cols = [f"personId{sfx}", f"firstName{sfx}", f"familyName{sfx}"]
            if all(c in m for c in cols):
                for pid, fn, ln in m[cols].dropna().drop_duplicates(cols[0]).itertuples(index=False):
                    full[int(pid)] = f"{fn} {ln}".strip()

    box = defaultdict(lambda: dict.fromkeys(STATS, 0.0))
    team, short = {}, {}
    toks, actors, rows, extras = [], [], [], []
    last_shot_side = None
    for i, row in enumerate(df.to_dict("records")):
        pid, tri = int(row["personId"]), row["teamTricode"]
        a = str(row["actionType"]).strip().lower()
        desc = str(row["description"])
        if is_player(pid) and tri:
            team.setdefault(pid, tri)
            short.setdefault(pid, str(row.get("playerNameI") or row.get("playerName") or pid))

        tok = classify(row, last_shot_side)
        if tok is not None:
            toks.append(STOI[tok]); rows.append(i)
            actors.append(pid if is_player(pid) else 0)
            if tok[2:] in ("2PT_MISS", "3PT_MISS", "FT_MISS"):
                last_shot_side = tok[0]
        step = len(toks)  # position in the game's token array (0 is <bos>)
        if not is_player(pid):
            continue
        b = box[pid]
        if a in ("made shot", "missed shot"):
            three = str(row.get("shotValue", "")).startswith("3") or "3PT" in desc.upper()
            b["fga"] += 1; b["tpa"] += three
            if a == "made shot":
                b["fgm"] += 1; b["tpm"] += three; b["pts"] += 3 if three else 2
                mm = AST_RE.search(desc)
                ap = names[tri].get(norm(mm.group(1))) if mm else None
                if ap and is_player(int(ap)):
                    box[int(ap)]["ast"] += 1
                    extras.append((step, int(ap), EXTRA["AST"]))
        elif a == "free throw":
            b["fta"] += 1
            if "MISS" not in desc.upper():
                b["ftm"] += 1; b["pts"] += 1
        elif a == "rebound" and tok is not None:
            b["oreb" if tok.endswith("OREB") else "dreb"] += 1
        elif a == "turnover":
            b["tov"] += 1
        elif a == "foul":
            b["pf"] += 1
        elif a == "" and "STEAL" in desc.upper():
            b["stl"] += 1; extras.append((step, pid, EXTRA["STL"]))
        elif a == "" and "BLOCK" in desc.upper():
            b["blk"] += 1; extras.append((step, pid, EXTRA["BLK"]))

    # who is on the floor at step s = the lineup after any substitutions that follow play s
    def five(i):
        lu = floor.get(i, {})
        return sorted(lu.get(H, ())), sorted(lu.get(A, ()))
    for q, mins in minutes.items():
        box[q]["min"] = mins
    # who is on the floor at step s = the lineup for the next play (subs happen at dead balls, so they're known)
    ends = rows + rows[-1:]
    lu, last = [], None
    for s, i in enumerate(ends):
        cur = five(i)
        if cur != last:
            lu.append((s, *cur)); last = cur
    for (h5, a5) in [(x[1], x[2]) for x in lu]:
        for q in h5: team.setdefault(q, H)
        for q in a5: team.setdefault(q, A)

    return {"gameId": gid, "season": season, "home": team_of["h"], "away": team_of["v"],
            "tokens": toks, "actors": actors, "lu": lu, "extras": extras,
            "box": {int(k): v for k, v in box.items() if v["min"] > 0 or any(v[s] for s in STATS[1:])},
            "team": team, "short": short, "full": full}


def load_cache(raw, ctx, cache_path):
    cache_path = Path(cache_path)
    cache = {}
    if cache_path.exists():
        c = pickle.load(open(cache_path, "rb"))
        if c.get("version") == CACHE_VERSION:
            cache = c["games"]
    files = sorted(Path(raw).rglob("*.parquet"))
    todo = [f for f in files if f.stem not in cache]
    for k, f in enumerate(todo):
        try:
            g = process_game(pd.read_parquet(f), ctx)
        except Exception as e:  # malformed game: skip, but keep going
            print(f"  profiles: skip {f.name}: {e}")  # not cached, so it's retried next time
            continue
        if g is not None:
            cache[f.stem] = g
        if k % 500 == 0:
            print(f"  profiles: {k}/{len(todo)} new games")
    if todo:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump({"version": CACHE_VERSION, "games": cache}, open(cache_path, "wb"))
    return cache


def fantasy(b):
    return (b["pts"] + FP_WEIGHTS["reb"] * (b["oreb"] + b["dreb"]) + FP_WEIGHTS["ast"] * b["ast"]
            + FP_WEIGHTS["stl"] * b["stl"] + FP_WEIGHTS["blk"] * b["blk"] + FP_WEIGHTS["tov"] * b["tov"])


def load_rosters(ctx):
    p = Path(ctx) / "rosters.parquet" if ctx else None
    if not p or not p.exists():
        return {}
    r = pd.read_parquet(p).sort_values("season")
    bio = {}
    for row in r.to_dict("records"):  # later seasons overwrite earlier ones
        bio[int(row["PLAYER_ID"])] = {
            "name": row.get("PLAYER"), "num": str(row.get("NUM") or ""), "pos": str(row.get("POSITION") or ""),
            "ht": str(row.get("HEIGHT") or ""), "wt": str(row.get("WEIGHT") or ""), "born": str(row.get("BIRTH_DATE") or ""),
            "school": str(row.get("SCHOOL") or ""), "season": row.get("season"),
        }
        exp, start = str(row.get("EXP") or ""), int(str(row.get("season"))[:4])
        if exp == "R" or exp.isdigit():
            bio[int(row["PLAYER_ID"])]["since"] = start - (0 if exp == "R" else int(exp))
    return bio


def build_profiles_payload(raw="data/raw", games=(), context="data/context", cache="data/processed/profiles_cache.pkl"):
    """games: the dashboard's game dicts (id + tok), for per-play actors and lineups."""
    ctx = Path(context) if context and Path(context).exists() else None
    per_game = load_cache(raw, ctx, cache)
    if not per_game:
        return None
    bio = load_rosters(ctx)

    # season totals, team ids and game logs, in time order (game IDs alone sort every playoff game last)
    seasons = sorted({season_key(g["season"], gid) for gid, g in per_game.items()})
    tot = defaultdict(lambda: defaultdict(lambda: dict.fromkeys(STATS + ["gp"], 0.0)))
    logs = defaultdict(list)
    team_ids, last_team, names = {}, {}, {}
    for gid in sorted(per_game, key=lambda k: (per_game[k]["season"], k)):
        g = per_game[gid]
        team_ids[g["home"][1]] = g["home"][0]; team_ids[g["away"][1]] = g["away"][0]
        for pid, b in g["box"].items():
            t = tot[pid][season_key(g["season"], gid)]
            for s in STATS:
                t[s] += b[s]
            t["gp"] += 1
            tri = g["team"].get(pid, "")
            t["team"] = tri
            last_team[pid] = (g["season"], tri)
            names.setdefault(pid, g["full"].get(pid) or g["short"].get(pid))
            if g["full"].get(pid):
                names[pid] = g["full"][pid]
            home = tri == g["home"][1]
            logs[(pid, g["season"])].append([gid, g["away"][1] if home else g["home"][1], int(home), round(b["min"], 1),
                                             int(b["pts"]), int(b["oreb"] + b["dreb"])]
                                            + [int(b[k]) for k in LOG_STATS] + [round(fantasy(b), 1)])

    # league per-minute rates of each attributable play, per season (the prior for low-minute players)
    league = {}
    for s in seasons:
        agg = dict.fromkeys(STATS, 0.0)
        for pid in tot:
            if s in tot[pid]:
                for k in STATS:
                    agg[k] += tot[pid][s][k]
        league[s] = {k: round(agg[k], 1) for k in STATS}

    players = {}
    for pid, by_season in tot.items():
        if sum(v["min"] for v in by_season.values()) < 1:
            continue
        last_season, tri = last_team[pid]
        b = bio.get(pid, {})
        players[str(pid)] = {
            "name": full_name(pid, b.get("name") or names.get(pid), context),
            "team": tri, "teamId": team_ids.get(tri, 0),
            "bio": {k: v for k, v in b.items() if k not in ("name", "season") and v not in (None, "", "nan", "None")},
            "seasons": {s: {**{k: round(v[k], 1) if k == "min" else int(v[k]) for k in STATS + ["gp"]}, "team": v["team"]}
                        for s, v in sorted(by_season.items())},
            "log": logs[(pid, last_season)],
        }

    # every game's result for everyone who played in it, for head-to-head records. Games run in season-key order
    # (key k owns games start[k] to start[k+1]); a player's list holds (game index gap) * 2 + won, so two players
    # were opponents in a game exactly when one of them won it. The winner is the side whose box scores add up to more.
    order = sorted(per_game, key=lambda k: (season_key(per_game[k]["season"], k), k))
    keys = sorted({season_key(per_game[k]["season"], k) for k in order})
    start, res, last = [], defaultdict(list), {}
    for i, gid in enumerate(order):
        g = per_game[gid]
        sk = season_key(g["season"], gid)
        if len(start) < len(keys) and keys[len(start)] == sk:
            start.append(i)
        pts = defaultdict(float)
        for pid, b in g["box"].items():
            pts[g["team"].get(pid, "")] += b["pts"]
        won = pts[g["home"][1]] > pts[g["away"][1]]
        for pid in g["box"]:
            side = g["team"].get(pid, "")
            if side not in (g["home"][1], g["away"][1]):
                continue
            res[pid].append((i - last.get(pid, -1)) * 2 + int(won == (side == g["home"][1])))
            last[pid] = i
    meet = {"keys": keys, "start": start, "res": {str(p): v for p, v in res.items() if str(p) in players}}

    # per-play actors, extras and lineups for the replayed games
    out_games = {}
    for dg in games:
        g = per_game.get(dg["id"])
        if g is None:
            continue
        L = len(dg["tok"])
        if list(g["tokens"]) != list(dg["tok"][1:1 + len(g["tokens"])]) or len(g["tokens"]) < L - 2:
            continue  # token streams disagree (different raw data); skip rather than mislabel plays
        roster = sorted({p for p in g["actors"] if p} | {p for x in g["lu"] for p in x[1] + x[2]} | {x[1] for x in g["extras"]})
        ix = {p: k for k, p in enumerate(roster)}
        act = ([0] + [ix[p] + 1 if p else 0 for p in g["actors"]] + [0])[:L]
        out_games[dg["id"]] = {
            "h": g["home"][1], "a": g["away"][1], "hid": g["home"][0], "aid": g["away"][0],
            "r": roster, "act": act,
            "lu": [[s, [ix[p] for p in h5], [ix[p] for p in a5]] for s, h5, a5 in g["lu"] if s < L],
            "x": [[s, ix[p], c] for s, p, c in g["extras"] if s < L],
        }
        for p in roster:  # make sure everyone in a replayed game has a profile, even with no stats
            players.setdefault(str(p), {"name": full_name(p, g["full"].get(p) or g["short"].get(p), context), "team": g["team"].get(p, ""),
                                        "teamId": team_ids.get(g["team"].get(p, ""), 0), "bio": {}, "seasons": {}, "log": []})

    return {"stats": STATS, "fp": FP_WEIGHTS, "seasons": seasons, "league": league,
            "log_cols": ["gameId", "opp", "home", "min", "pts", "reb"] + LOG_STATS + ["fp"],
            "teams": {t: i for t, i in team_ids.items()}, "players": players, "games": out_games, "meet": meet}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--context", default="data/context")
    ap.add_argument("--cache", default="data/processed/profiles_cache.pkl")
    args = ap.parse_args()
    p = build_profiles_payload(args.raw, (), args.context, args.cache)
    if p is None:
        print("no games with player data"); return
    rows, latest = [], [k for k in p["seasons"] if " " not in k][-1]  # latest regular season, not "... Playoffs"
    for pid, pl in p["players"].items():
        s = pl["seasons"].get(latest)
        if s and s["gp"] >= 20:
            fp = (s["pts"] + 1.2 * (s["oreb"] + s["dreb"]) + 1.5 * s["ast"] + 3 * (s["stl"] + s["blk"]) - s["tov"]) / s["gp"]
            rows.append((fp, pl["name"], pl["team"], s["gp"], s["min"] / s["gp"], s["pts"] / s["gp"]))
    print(f"{len(p['players'])} players, seasons {p['seasons']}")
    for fp, n, t, gp, mn, pts in sorted(rows, reverse=True)[:15]:
        print(f"  {n:<26} {t}  gp {gp:>2}  min {mn:4.1f}  pts {pts:4.1f}  fantasy {fp:5.1f}")


if __name__ == "__main__":
    main()
