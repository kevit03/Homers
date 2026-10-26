"""Bring the processed data up to date with the season in progress: the daily step after fetching new games.

Takes every finished game in data/raw/<season>/ that isn't processed yet and adds it everywhere the site reads from,
without touching any earlier game (each step checks that):

  games.pkl, games_30.pkl   tokenized, under the split "current" (never train/val/test); games_30 also gets the
                            Elo and team IDs games.pkl has, by game ID
  Elo                       the saved recipe (elo.json: K, home edge, carry, picked on training games) run on through
                            the new games; elo_current.json holds the ratings after the last one, for games not
                            played yet (live win chances)
  game_context.pkl          lineups for the new games only (lineups_cache.pkl), player ratings refit only for the
                            new blocks (ratings_cache.pkl), then starting fives / missing players / the rest rebuilt
  shots.parquet, shots_30   the new games' shots, appended

Games processed before NBA.com posted their defensive matchups (who guarded whom; it can lag a day) are listed in
season_state.json and redone once the matchups show up. Nothing new and nothing pending: nothing is written.

    python -m models.update_season                     # 2026-27
    python -m models.update_season --season 2026-27 --workers 8
"""
import argparse
import json
import math
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from models import game_context as gc
from models.build_shots import build_season, fill_jumpers
from models.team_strength import chrono_order, game_teams, log_loss, run_elo
from models.tokenize_pbp import process_game

SPLIT = "current"
FIRST_MATCHUPS = "2017-18"  # NBA.com's matchup feed starts here


def is_final(df):
    """A finished game: the feed ends on the end of the 4th period or later, with a winner."""
    last = df.iloc[-1]
    h = pd.to_numeric(df["scoreHome"], errors="coerce").ffill().fillna(0).max()
    a = pd.to_numeric(df["scoreAway"], errors="coerce").ffill().fillna(0).max()
    return (str(last["actionType"]).lower() in ("period", "game") and str(last["subType"]).lower() == "end"
            and int(pd.to_numeric(last["period"], errors="coerce") or 0) >= 4 and h != a)


def tokenize(f):
    try:
        df = pd.read_parquet(f)
        return f.stem, (process_game(df) if is_final(df) else None), None
    except Exception as e:
        return f.stem, None, str(e)


def load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def dump(obj, path):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
    tmp.replace(path)  # a training or export run reading it never sees half a file


def same_game(a, b):
    return a.keys() == b.keys() and all(
        np.array_equal(a[k], b[k]) if isinstance(a[k], np.ndarray) else a[k] == b[k] for k in a)


def add_games(d, new, season):
    """Put the season's games (old ones kept, new ones added) at the end, in game ID order, as split SPLIT.
    Earlier seasons keep their indices and splits."""
    keep = [g for g in d["games"] if g["season"] != season]
    cur = {g["gameId"]: g for g in d["games"] if g["season"] == season}
    cur.update(new)
    n = len(keep)
    if any(i >= n for k, idx in d["splits"].items() if k != SPLIT for i in idx):
        raise SystemExit(f"{season} games sit in a train/val/test split; refusing to move them")
    d["games"] = keep + [cur[k] for k in sorted(cur)]
    d["splits"] = {**{k: v for k, v in d["splits"].items() if k != SPLIT}, SPLIT: list(range(n, len(d["games"])))}


def update_elo(d, data_path, raw, season):
    """Elo through the new games with the saved recipe; old games must come out exactly as stored."""
    rec = json.load(open(Path(data_path).with_name("elo.json")))
    games = d["games"]
    teams = game_teams(raw, games, Path(data_path).with_name(Path(data_path).stem + "_teams.parquet"))
    order = chrono_order(games)
    edge, R = run_elo(games, order, teams, rec["k"], rec["hca"], rec["carry"], final=True)
    changed = 0
    for g, e in zip(games, edge):
        v = 0.0 if np.isnan(e) else float(e * math.log(10) / 400)
        if g["season"] == season:
            g["elo"] = v
            if g["gameId"] in teams:
                g["home"], g["away"] = teams[g["gameId"]]
        elif abs(g.get("elo", 0.0) - v) > 1e-12:
            changed += 1
    if changed:
        raise SystemExit(f"Elo recomputed differently for {changed} earlier games; not writing")
    idx = np.array([i for i in d["splits"][SPLIT] if not np.isnan(edge[i])])
    if len(idx):
        y = np.array([games[i]["home_win"] for i in idx], float)
        p = 1 / (1 + 10 ** (-edge[idx] / 400))
        rec[SPLIT] = {"season": season, "games": int(len(idx)), "log_loss": log_loss(edge[idx], y),
                      "brier": float(np.mean((p - y) ** 2)), "acc": float(np.mean((p > 0.5) == y))}
        json.dump(rec, open(Path(data_path).with_name("elo.json"), "w"), indent=1)
    last = games[order[-1]]
    json.dump({"season": last["season"], "after_game": last["gameId"], "hca": rec["hca"], "k": rec["k"],
               "note": "P(home win) = 1 / (1 + 10 ** (-(R[home] + hca - R[away]) / 400)); elo logit = that edge * ln10/400",
               "ratings": {str(t): round(r, 2) for t, r in sorted(R.items())}},
              open(Path(data_path).with_name("elo_current.json"), "w"), indent=1)


def sync_games_30(path, d, season):
    """games_30.pkl: the season's games from games.pkl (with Elo and team IDs), and Elo copied onto 2016-17 on."""
    d30 = load(path)
    before = {g["gameId"]: g for g in d30["games"] if g["season"] != season}
    splits_before = {k: list(v) for k, v in d30["splits"].items() if k != SPLIT}
    add_games(d30, {g["gameId"]: g for g in d["games"] if g["season"] == season}, season)
    elo = {g["gameId"]: g["elo"] for g in d["games"]}
    for g in d30["games"]:
        if g["gameId"] in elo and g["season"] != season:
            g["elo"] = elo[g["gameId"]]
    for g in d30["games"]:
        if g["season"] != season:
            b = before[g["gameId"]]
            if not all(np.array_equal(g[k], b[k]) if isinstance(b[k], np.ndarray) else g[k] == b[k] for k in b):
                raise SystemExit(f"games_30.pkl: earlier game {g['gameId']} changed; not writing")
    if any(d30["splits"][k] != v for k, v in splits_before.items()):
        raise SystemExit("games_30.pkl: earlier splits changed; not writing")
    dump(d30, path)
    return len(d30["splits"][SPLIT])


def has_matchups(ctx, season, gid):
    return season < FIRST_MATCHUPS or (Path(ctx) / "matchups" / season / f"{gid}.parquet").exists()


def update_context(data_path, raw, ctx, lineups_path, out, shots_path, todo, season, workers):
    d = load(data_path)
    lu = load(lineups_path)
    files = [Path(raw) / season / f"{gid}.parquet" for gid in todo]
    with ProcessPoolExecutor(workers) as ex:
        for gid, r in ex.map(gc.extract, files, [ctx] * len(files)):
            lu[gid] = r
    dump(lu, lineups_path)
    old = load(out) if Path(out).exists() else None
    feats, coach = gc.build(d["games"], d["splits"], lu, ctx, shots_path,
                            ratings_cache=Path(lineups_path).with_name("ratings_cache.pkl"))
    if old is not None and old["features"] == gc.FEATURES:
        seasons = {g["gameId"]: g["season"] for g in d["games"]}
        bad = [gid for gid, v in old["values"].items() if seasons.get(gid) != season
               and (gid not in feats or not np.array_equal(v, feats[gid], equal_nan=True))]
        if bad:
            raise SystemExit(f"game_context: {len(bad)} earlier games changed (e.g. {bad[:3]}); not writing")
    gc.save(out, feats, coach, lu)
    n = sum(1 for g in d["games"] if g["season"] == season and g["gameId"] in feats
            and np.isfinite(feats[g["gameId"]][0, gc.FEATURES.index("start_net")]))
    return n


def update_shots(raw, ctx, todo, season, paths, workers):
    """Shots for the todo games, replacing any earlier rows for them, appended to each shots parquet."""
    coaches = {}
    cpath = Path(ctx) / "coaches.parquet"
    if cpath.exists():
        c = pd.read_parquet(cpath)
        coaches = {(s, int(t)): int(cid) for s, t, cid in zip(c["season"], c["teamId"], c["coachId"])}
    files = sorted(Path(raw) / season / f"{gid}.parquet" for gid in todo)
    chunks = [files[k::workers] for k in range(workers) if files[k::workers]]
    with ProcessPoolExecutor(len(chunks)) as ex:
        results = list(ex.map(build_season, [(season, fs, Path(ctx), coaches) for fs in chunks]))
    rows = [r for rs, _ in results for r in rs]
    for _, sk in results:
        for m in sk:
            print(f"  shots: skip {m}")
    new = pd.DataFrame(rows)
    counts = {}
    for path in paths:
        s = pd.read_parquet(path)
        n_before = len(s)
        cur = s[s["season"] == season]
        cur = cur[~cur["gameId"].isin(set(todo))]
        add = new.reindex(columns=s.columns) if len(new) else new
        for c in ("coord_imputed", "sc_match"):
            if c in s.columns:
                add[c] = add[c].fillna(0).astype("int8")
        cur = pd.concat([cur, add], ignore_index=True) if len(add) else cur
        if "coord_imputed" in cur.columns:
            cur = fill_jumpers(cur)
        cur = cur.sort_values(["gameId"], kind="mergesort")
        rest = s[s["season"] != season]
        if len(rest) != n_before - int((s["season"] == season).sum()):
            raise SystemExit("shots: row count mismatch")
        out = pd.concat([rest, cur], ignore_index=True).sort_values(["season", "gameId"], kind="mergesort")
        out = out.reset_index(drop=True)
        tmp = Path(str(path) + ".tmp.parquet")
        out.to_parquet(tmp)
        tmp.replace(path)
        counts[str(path)] = int((out["season"] == season).sum())
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2026-27")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--context", default="data/context")
    ap.add_argument("--data", default="data/processed/games.pkl")
    ap.add_argument("--data30", default="data/processed/games_30.pkl")
    ap.add_argument("--lineups", default="data/processed/lineups_cache.pkl")
    ap.add_argument("--game_context", default="data/processed/game_context.pkl")
    ap.add_argument("--shots", nargs="+", default=["data/processed/shots.parquet", "data/processed/shots_30.parquet"])
    ap.add_argument("--state", default="data/processed/season_state.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip_context", action="store_true")
    ap.add_argument("--skip_shots", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    season = args.season

    state = json.load(open(args.state)) if Path(args.state).exists() else {}
    if state.get("season") != season:
        state = {"season": season, "pending_matchups": []}
    d = load(args.data)
    have = {g["gameId"] for g in d["games"] if g["season"] == season}
    raw_files = sorted((Path(args.raw) / season).glob("*.parquet"))
    fresh = [f for f in raw_files if f.stem not in have]
    new, not_final = {}, []
    if fresh:
        with ProcessPoolExecutor(args.workers) as ex:
            for gid, g, err in ex.map(tokenize, fresh, chunksize=8):
                if err:
                    print(f"  skip {gid}: {err}")
                elif g is None:
                    not_final.append(gid)
                else:
                    new[gid] = g
    ready = [gid for gid in state["pending_matchups"] if has_matchups(args.context, season, gid)]
    print(f"{season}: {len(raw_files)} raw games, {len(have)} already processed, {len(new)} new, "
          f"{len(not_final)} not final, {len(ready)} with matchups now in")
    if not new and not ready:
        print(f"nothing to do ({time.time() - t0:.0f}s)")
        return

    if new:
        before = {g["gameId"]: g for g in d["games"] if g["season"] != season}
        splits_before = {k: list(v) for k, v in d["splits"].items() if k != SPLIT}
        add_games(d, new, season)
        update_elo(d, args.data, args.raw, season)
        for g in d["games"]:
            if g["season"] != season and not same_game(g, before[g["gameId"]]):
                raise SystemExit(f"games.pkl: earlier game {g['gameId']} changed; not writing")
        if any(d["splits"][k] != v for k, v in splits_before.items()):
            raise SystemExit("games.pkl: earlier splits changed; not writing")
        dump(d, args.data)
        n30 = sync_games_30(args.data30, d, season)
        print(f"  games.pkl and games_30.pkl: {len(d['splits'][SPLIT])} / {n30} {season} games ({time.time() - t0:.0f}s)")

    todo = sorted(set(new) | set(ready))
    if not args.skip_context:
        n = update_context(args.data, args.raw, args.context, args.lineups, args.game_context, args.shots[-1],
                           todo, season, args.workers)
        print(f"  game_context.pkl: lineups for {n} {season} games ({time.time() - t0:.0f}s)")
    if not args.skip_shots:
        counts = update_shots(args.raw, args.context, todo, season, args.shots, args.workers)
        print(f"  shots: {counts} ({time.time() - t0:.0f}s)")

    state["pending_matchups"] = sorted((set(state["pending_matchups"]) - set(ready))
                                       | {gid for gid in new if not has_matchups(args.context, season, gid)})
    state.update({"games": len(d["splits"][SPLIT]), "last_game": max(have | set(new)),
                  "updated": time.strftime("%Y-%m-%d %H:%M:%S")})
    json.dump(state, open(args.state, "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s; {len(state['pending_matchups'])} games waiting on matchups")


if __name__ == "__main__":
    main()
