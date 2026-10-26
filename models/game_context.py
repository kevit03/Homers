"""Leak-free game context for win probability: who is on the floor, what kind of players they are, the coaches,
and how often each team and coach has come back from 10 and 20 down.

Every value at a play uses only games played before that game (plus who is on the floor right now), so it can
feed a model scored on later games:

  lineups      the 5 players per side at every token of games.pkl, rebuilt from substitutions in the raw feed
               (build_shots.lineups on the tokenizer's own row order, checked token for token against games.pkl)
  ratings      offensive and defensive adjusted plus-minus per player (ridge on stints, points per 48 minutes),
               refit every BLOCK games on earlier games only; older seasons count less (SEASON_DECAY)
  matchup      home attack = home offense + away defense, away attack = the reverse, for the 5 on 5 on the floor;
               net = their difference; start_net = the same for the starting fives (who is playing tonight)
  types        bigs on the floor (roster position includes C) and spacing (prior seasons' 3PA share of FGA),
               home minus away
  missing      rotation players (MISS_MIN+ usual minutes, played in one of the team's last MISS_WINDOW games) who don't
               play tonight: their usual minutes / 48 times their net rating (miss_diff) and the minutes alone
               (nmiss_diff), home minus away. Who suits up is known before tip-off (injury report)
  coach, team  share of games won after trailing by 10+ / 20+ (shrunk toward the league), and the same as wins
               above what score, clock and Elo expected at the moment the deficit was reached; coaches from
               Basketball-Reference stints (bbref_coach_seasons), not coaches.parquet

    python -m models.game_context                     # writes data/processed/game_context.pkl (~5 min, 8 workers)
    python -m models.game_context --skip_extract      # reuse the cached lineups, recompute features only
"""
import argparse
import pickle
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from models.baseline import elo_features, fit_elo_baseline
from models.build_shots import TEAM_ID_MIN, lineups, load_matchups, name_map
from models.team_strength import chrono_order
from models.tokenize_pbp import STOI, BOS, EOS, classify, game_order, parse_clock, seconds_remaining

WARMUP_SEASONS = ["2014-15", "2015-16"]  # stints only: ratings start from something in 2016-17
BLOCK = 100            # games between rating refits
SEASON_DECAY = 0.5     # a season ago counts half; three or more seasons ago not at all
RIDGE = 1000.0         # in stint-minutes; --tune_ridge: 1000 and 3000 tie on held-out blocks, 300 and 10000 worse
TRAIL = (10, 20)
PRIOR_GAMES = {"team": 8.0, "coach": 15.0}  # shrinkage toward the league rate, in trailing games
MISS_MIN, MISS_WINDOW, MISS_LAST = 15.0, 6, 10  # rotation: 15+ minutes over his last 10 games, seen in the last 6
FEATURES = ["net", "home_att", "away_att", "start_net", "bigs_diff", "spacing_diff",
            "team_cb10_h", "team_cb10_a", "team_cb20_h", "team_cb20_a",
            "team_cbx10_h", "team_cbx10_a", "team_cbx20_h", "team_cbx20_a",
            "coach_cb10_h", "coach_cb10_a", "coach_cb20_h", "coach_cb20_a",
            "coach_cbx10_h", "coach_cbx10_a", "coach_cbx20_h", "coach_cbx20_a", "coach_wax_h", "coach_wax_a",
            "miss_diff", "nmiss_diff"]
GROUPS = {"lineup": ["net", "start_net"], "matchup": ["home_att", "away_att", "start_net"],
          "types": ["bigs_diff", "spacing_diff"],
          "team_comeback": ["team_cb10_h", "team_cb10_a", "team_cb20_h", "team_cb20_a"],
          "team_comeback_x": ["team_cbx10_h", "team_cbx10_a", "team_cbx20_h", "team_cbx20_a"],
          "coach_comeback": ["coach_cb10_h", "coach_cb10_a", "coach_cb20_h", "coach_cb20_a"],
          "coach_comeback_x": ["coach_cbx10_h", "coach_cbx10_a", "coach_cbx20_h", "coach_cbx20_a",
                               "coach_wax_h", "coach_wax_a"],
          "missing": ["miss_diff", "nmiss_diff"]}
# +1: higher means the home team is likelier to win; -1 the reverse; 0 no constraint
MONOTONE = {"net": 1, "home_att": 1, "away_att": -1, "start_net": 1, "bigs_diff": 0, "spacing_diff": 0,
            **{f: (1 if f.endswith("_h") else -1) for f in FEATURES[6:24]}, "miss_diff": -1, "nmiss_diff": -1}
PREGAME = ["start_net", "miss_diff", "nmiss_diff"]  # the context Tempo uses: one value per game, known before tip-off


def attach_pregame(games, path="data/processed/game_context.pkl"):
    """Set g[name] for the PREGAME values Tempo uses (starting fives, missing rotation players) from a game_context.pkl.
    Returns how many games got them; the rest (no lineups, e.g. seasons before 2016-17) stay at 0."""
    with open(path, "rb") as fh:
        ctx = pickle.load(fh)
    js, n = [ctx["features"].index(f) for f in PREGAME], 0
    for g in games:
        v = ctx["values"].get(g["gameId"])
        if v is not None and np.isfinite(v[0, js[0]]):
            n += 1
            for f, j in zip(PREGAME, js):
                g[f] = float(np.nan_to_num(v[0, j]))
    return n


# ---------------------------------------------------------------- lineups per token

def prep(df):
    """The tokenizer's row order, with build_shots' column cleanup (lineups() needs ints and clean strings)."""
    df = game_order(df)
    for col in ("scoreHome", "scoreAway"):
        df[col] = pd.to_numeric(df[col], errors="coerce").ffill().fillna(0).cummax()
    df["period"] = pd.to_numeric(df["period"], errors="coerce").fillna(1).astype(int)
    df["personId"] = pd.to_numeric(df["personId"], errors="coerce").fillna(0).astype(int)
    df["teamId"] = pd.to_numeric(df["teamId"], errors="coerce").fillna(0).astype(int)
    df["teamTricode"] = df["teamTricode"].fillna("").astype(str).replace("None", "")
    for col in ("actionType", "subType", "location"):
        if col in df:
            df[col] = df[col].fillna("").astype(str).str.strip()
    return df


def extract(f, ctx="data/context"):
    """One raw game -> tokens (to check against games.pkl), home/away fives per token, and stints for the ratings."""
    try:
        df = prep(pd.read_parquet(f))
        gid, season = str(df["gameId"].iloc[0]), str(df["season"].iloc[0])
        side = df[df["location"].isin(["h", "v"]) & (df["teamId"] > 0) & (df["teamTricode"] != "")]
        tri = side.groupby("location")["teamTricode"].agg(lambda s: s.mode().iloc[0]).to_dict()
        if set(tri) != {"h", "v"}:
            return gid, None
        floor = lineups(df, name_map(df, load_matchups(Path(ctx), season, gid)))
        h_t, a_t = tri["h"], tri["v"]

        def five(idx, t):
            s = floor.get(idx, {}).get(t, frozenset())
            return tuple(sorted(s)) if len(s) == 5 else None

        tokens, h5, a5 = [BOS], [], []
        last_shot_side = None
        stints = defaultdict(lambda: [0.0, 0.0, 0.0])  # (home5, away5) -> [seconds, home pts, away pts]
        prev = None  # (period, clock, home score, away score, home5, away5) of the previous row
        for row in df.reset_index().to_dict("records"):
            i, p, clk = row["index"], int(row["period"]), parse_clock(row["clock"])
            hf, af = five(i, h_t), five(i, a_t)
            if prev is not None and prev[0] == p and prev[4] and prev[5]:
                stints[(prev[4], prev[5])][0] += max(prev[1] - clk, 0.0)  # time since the last row: the old lineup
            if hf and af:
                st = stints[(hf, af)]
                if prev is not None:
                    st[1] += row["scoreHome"] - prev[2]
                    st[2] += row["scoreAway"] - prev[3]
            prev = (p, clk, row["scoreHome"], row["scoreAway"], hf, af)
            tok = classify(row, last_shot_side)
            if tok is None:
                continue
            if tok[2:] in ("2PT_MISS", "3PT_MISS", "FT_MISS"):
                last_shot_side = tok[0]
            tokens.append(STOI[tok])
            h5.append(hf or (0,) * 5); a5.append(af or (0,) * 5)
        if not h5:
            return gid, None
        tokens.append(EOS)
        h5 = [h5[0]] + h5 + [h5[-1]]  # <bos> gets the starters, <eos> the last five
        a5 = [a5[0]] + a5 + [a5[-1]]
        st = [(k[0], k[1], v[0], v[1], v[2]) for k, v in stints.items() if v[0] > 0 or v[1] or v[2]]
        return gid, {"season": season, "tokens": np.array(tokens, dtype=np.int16),
                     "home5": np.array(h5, dtype=np.int32), "away5": np.array(a5, dtype=np.int32),
                     "home_tri": h_t, "away_tri": a_t, "stints": st}
    except Exception as e:  # a broken feed: no context for that game
        return Path(f).stem, {"error": str(e)}


def extract_all(raw, games, workers):
    want = {g["gameId"]: g["season"] for g in games}
    files = [Path(raw) / s / f"{gid}.parquet" for gid, s in want.items()]
    for s in WARMUP_SEASONS:
        files += sorted((Path(raw) / s).glob("002*.parquet"))
    files = [f for f in files if f.exists()]
    out = {}
    with ProcessPoolExecutor(workers) as ex:
        for k, (gid, r) in enumerate(ex.map(extract, files, chunksize=16)):
            out[gid] = r
            if (k + 1) % 2000 == 0:
                print(f"  {k + 1}/{len(files)} games", flush=True)
    return out


# ---------------------------------------------------------------- player ratings

def stint_arrays(lu, order_ids):
    """Stack stints in game order: (game rank, home5[n,5], away5[n,5], minutes, home pts, away pts)."""
    rank = {gid: r for r, gid in enumerate(order_ids)}
    G, H, A, M, PH, PA = [], [], [], [], [], []
    for gid in order_ids:
        r = lu.get(gid)
        if not r or "stints" not in r:
            continue
        for h5, a5, secs, ph, pa in r["stints"]:
            G.append(rank[gid]); H.append(h5); A.append(a5); M.append(secs / 60.0); PH.append(ph); PA.append(pa)
    return (np.array(G), np.array(H, dtype=np.int64).reshape(-1, 5), np.array(A, dtype=np.int64).reshape(-1, 5),
            np.array(M), np.array(PH, dtype=float), np.array(PA, dtype=float))


def fit_ratings(H, A, M, PH, PA, w, ridge):
    """Ridge on stint rows: points per minute = mean + home edge + sum offense(5) + sum defense(other 5).
    Returns {player: (offense, defense)} in points per 48 minutes; defense > 0 means he gives up more."""
    from scipy import sparse
    keep = (M > 0.05) & (w > 0)
    H, A, M, PH, PA, w = H[keep], A[keep], M[keep], PH[keep], PA[keep], w[keep]
    ids, inv = np.unique(np.concatenate([H.ravel(), A.ravel()]), return_inverse=True)
    P, n = len(ids), len(M)
    hi, ai = inv[:5 * n].reshape(n, 5), inv[5 * n:].reshape(n, 5)
    # row 2k: home offense vs away defense; row 2k+1: away offense vs home defense
    off = np.empty((2 * n, 5), dtype=np.int64); dfn = np.empty((2 * n, 5), dtype=np.int64)
    off[0::2], off[1::2], dfn[0::2], dfn[1::2] = hi, ai, P + ai, P + hi
    rows = np.repeat(np.arange(2 * n), 10)
    cols = np.concatenate([off, dfn], 1).ravel()
    home = np.zeros(2 * n); home[0::2] = 1
    X = sparse.hstack([sparse.csr_matrix((np.ones(20 * n), (rows, cols)), shape=(2 * n, 2 * P)),
                       sparse.csr_matrix(np.column_stack([np.ones(2 * n), home]))]).tocsr()
    y = np.empty(2 * n); y[0::2], y[1::2] = PH / M, PA / M
    wt = np.repeat(M * w, 2)
    XtW = X.T.multiply(wt).tocsr()
    lhs = (XtW @ X).toarray()
    lhs[np.arange(2 * P), np.arange(2 * P)] += ridge
    beta = np.linalg.solve(lhs, XtW @ y)
    return {int(p): (48 * beta[i], 48 * beta[P + i]) for i, p in enumerate(ids)}


def season_index(s):
    return int(s[:4])


def rolling_ratings(lu, order_ids, seasons_of, ridge=RIDGE, block=BLOCK):
    """{gameId: ratings fitted on earlier games only}, refit every `block` games (warm-up seasons included)."""
    Gr, H, A, M, PH, PA = stint_arrays(lu, order_ids)
    sidx = np.array([season_index(seasons_of[order_ids[g]]) for g in Gr]) if len(Gr) else np.array([])
    out, cur = {}, {}
    for start in range(0, len(order_ids), block):
        ids_blk = order_ids[start:start + block]
        s_now = season_index(seasons_of[ids_blk[0]])
        past = Gr < start
        if past.any():
            age = s_now - sidx[past]
            w = np.where(age <= 2, SEASON_DECAY ** age, 0.0)
            cur = fit_ratings(H[past], A[past], M[past], PH[past], PA[past], w, ridge)
        for gid in ids_blk:
            out[gid] = cur
    return out


def tune_ridge(lu, order_ids, seasons_of, grid=(300, 1000, 3000, 10000)):
    """Weighted squared error on the next block's stints, for a few blocks of 2022-23..2023-24 (training seasons)."""
    Gr, H, A, M, PH, PA = stint_arrays(lu, order_ids)
    sidx = np.array([season_index(seasons_of[order_ids[g]]) for g in Gr])
    starts = [i for i, gid in enumerate(order_ids) if seasons_of[gid] in ("2022-23", "2023-24") and i % 400 == 0]
    for ridge in grid:
        err = wsum = 0.0
        for start in starts:
            past, nxt = Gr < start, (Gr >= start) & (Gr < start + BLOCK)
            age = season_index(seasons_of[order_ids[start]]) - sidx[past]
            r = fit_ratings(H[past], A[past], M[past], PH[past], PA[past],
                            np.where(age <= 2, SEASON_DECAY ** age, 0.0), ridge)
            o = lambda five: np.array([[r.get(int(p), (0.0, 0.0))[0] for p in f] for f in five]).sum(1)
            d = lambda five: np.array([[r.get(int(p), (0.0, 0.0))[1] for p in f] for f in five]).sum(1)
            pred = (o(H[nxt]) + d(A[nxt]) - o(A[nxt]) - d(H[nxt])) / 48 * M[nxt]  # home margin over the stint
            act = PH[nxt] - PA[nxt]
            err += np.sum((act - pred) ** 2); wsum += len(act)
        print(f"  ridge {ridge:>6}: held-out stint margin MSE {err / wsum:.4f}")


# ---------------------------------------------------------------- player types

def player_types(ctx, shots_path):
    """(bigs: set of (season, player) with C in the roster position, spacing: {(season, player): 3PA share of FGA}).
    Spacing for a season uses the two seasons before it (the latest counts double), shrunk toward the league."""
    r = pd.read_parquet(Path(ctx) / "rosters.parquet", columns=["season", "PLAYER_ID", "POSITION"]).dropna()
    big_any = set(r.loc[r["POSITION"].str.contains("C"), "PLAYER_ID"].astype(int))
    pos = {(s, int(p)): "C" in str(x) for s, p, x in zip(r["season"], r["PLAYER_ID"], r["POSITION"])}
    s = pd.read_parquet(shots_path, columns=["season", "shooterId", "is3"])
    agg = s.groupby(["season", "shooterId"])["is3"].agg(["sum", "size"])
    league = s.groupby("season")["is3"].mean()
    return pos, big_any, agg, league


def spacing_lookup(agg, league, season, k=100.0):
    y = season_index(season)
    prev = [f"{y - 1}-{str(y)[-2:]}", f"{y - 2}-{str(y - 1)[-2:]}"]
    lg = float(league.get(prev[0], league.iloc[-1]))
    tot = defaultdict(lambda: [0.0, 0.0])
    for wgt, s in zip((2.0, 1.0), prev):
        if s in agg.index.get_level_values(0):
            a = agg.loc[s]
            for pid, (m3, n) in zip(a.index, a.to_numpy()):
                t = tot[int(pid)]; t[0] += wgt * m3; t[1] += wgt * n
    return {p: (m3 + k * lg) / (n + k) for p, (m3, n) in tot.items()}, lg


# ---------------------------------------------------------------- coaches and comebacks

def coach_of_games(games, order, ctx):
    """{(gameId, side): bbref coach id}. A team-season's stints (bbref_coach_seasons, in table order) take its
    regular-season games in date order by their game counts; playoff games go to the coach with playoff games."""
    cs = pd.read_parquet(Path(ctx) / "bbref_coach_seasons.parquet")
    cs[["g", "pg"]] = cs[["g", "pg"]].fillna(0)
    stints = defaultdict(list)
    for r in cs.itertuples():
        stints[(r.season, r.team)].append((r.bbref_coach, int(r.g or 0), int(r.pg or 0)))
    seen = defaultdict(int)
    out = {}
    for i in order:
        g = games[i]
        for side in ("home_tri", "away_tri"):
            t = g.get(side)
            st = stints.get((g["season"], t))
            if not t or not st:
                continue
            if g["gameId"].startswith("004"):
                po = [c for c, _, pg in st if pg > 0]
                out[(g["gameId"], side[:4])] = po[-1] if po else st[-1][0]
                continue
            k = seen[(g["season"], t)]; seen[(g["season"], t)] += 1
            acc = 0
            for c, n, _ in st:
                acc += n
                if k < acc:
                    out[(g["gameId"], side[:4])] = c
                    break
            else:
                out[(g["gameId"], side[:4])] = st[-1][0]
    return out


def comeback_events(g, p_home):
    """For each side and threshold: (trailed by that much?, won?, P(win) when the deficit was first reached)."""
    d = g["diff"]
    ev = {}
    for side, sgn in (("home", 1), ("away", -1)):
        won = g["home_win"] if side == "home" else 1 - g["home_win"]
        for k in TRAIL:
            hit = np.nonzero(sgn * d <= -k)[0]
            if len(hit):
                p = p_home[hit[0]] if side == "home" else 1 - p_home[hit[0]]
                ev[(side, k)] = (won, float(p))
    return ev


class Tally:
    """Running comeback record per key (team or coach), read before each game and updated after it."""

    def __init__(self, prior):
        self.prior = prior
        self.n = defaultdict(float); self.w = defaultdict(float); self.x = defaultdict(float)
        self.league_n = defaultdict(float); self.league_w = defaultdict(float)
        self.games = defaultdict(float); self.wax = defaultdict(float)

    def rate(self, key, k):
        lg = (self.league_w[k] + 1) / (self.league_n[k] + 6)  # league comeback rate, itself started near 1 in 6
        n = self.n[(key, k)]
        return (self.w[(key, k)] + self.prior * lg) / (n + self.prior), self.x[(key, k)] / (n + self.prior)

    def wins_above(self, key):
        return self.wax[key] / (self.games[key] + 40.0)

    def add(self, key, k, won, p):
        self.n[(key, k)] += 1; self.w[(key, k)] += won; self.x[(key, k)] += won - p
        self.league_n[k] += 1; self.league_w[k] += won

    def add_game(self, key, won, p_pre):
        self.games[key] += 1; self.wax[key] += won - p_pre

    def decay(self, f):
        for dct in (self.n, self.w, self.x, self.games, self.wax):
            for key in dct:
                dct[key] *= f


# ---------------------------------------------------------------- features

def player_minutes(stints):
    """{"h": {player: minutes}, "a": {...}} for one game, from its stints."""
    out = {"h": defaultdict(float), "a": defaultdict(float)}
    for h5, a5, secs, *_ in stints:
        for p in h5:
            out["h"][p] += secs / 60
        for p in a5:
            out["a"][p] += secs / 60
    return out


def five_sum(value):
    """fives [n, 5] of player IDs -> [n] sums of value(player), looking each distinct player up once."""
    def f(fives):
        ids, inv = np.unique(fives, return_inverse=True)
        v = np.array([value(int(p)) for p in ids], dtype=float)
        return v[inv.reshape(fives.shape)].sum(1)
    return f


def build(games, splits, lu, ctx, shots_path, ridge=RIDGE, ratings_cache=None):
    seasons_of = {g["gameId"]: g["season"] for g in games}
    for gid, r in lu.items():
        if r and "season" in r:
            seasons_of.setdefault(gid, r["season"])
    for g in games:
        r = lu.get(g["gameId"]) or {}
        g["home_tri"], g["away_tri"] = r.get("home_tri"), r.get("away_tri")
    order = chrono_order(games)
    warm = sorted([gid for gid, r in lu.items() if gid not in {g["gameId"] for g in games} and r and "stints" in r],
                  key=lambda gid: (seasons_of[gid], gid))
    order_ids = warm + [games[i]["gameId"] for i in order]
    print(f"fitting ratings every {BLOCK} games on {len(order_ids)} games ({len(warm)} warm-up)", flush=True)
    key = (ridge, BLOCK, SEASON_DECAY, len(order_ids))
    cached = pickle.load(open(ratings_cache, "rb")) if ratings_cache and Path(ratings_cache).exists() else None
    if cached and cached["key"] == key:
        ratings = cached["ratings"]
    else:
        ratings = rolling_ratings(lu, order_ids, seasons_of, ridge)
        if ratings_cache:
            with open(ratings_cache, "wb") as fh:
                pickle.dump({"key": key, "ratings": ratings}, fh)

    pos, big_any, agg, league = player_types(ctx, shots_path)
    spacing = {s: spacing_lookup(agg, league, s) for s in sorted({g["season"] for g in games})}
    coach = coach_of_games(games, order, ctx)

    clf = fit_elo_baseline([games[i] for i in splits["train"]])
    team_t, coach_t = Tally(PRIOR_GAMES["team"]), Tally(PRIOR_GAMES["coach"])
    usual = defaultdict(lambda: deque(maxlen=MISS_LAST))  # player -> minutes in his last games played
    seen, team_n = defaultdict(dict), defaultdict(int)    # team -> {player: team game he last played}, games so far
    out, cur_season, ok = {}, None, 0
    for i in order:
        g = games[i]
        L = len(g["sec"])
        if g["season"] != cur_season:
            if cur_season is not None:
                team_t.decay(0.5)  # last season's comebacks count half for the team (rosters change)
            cur_season = g["season"]
        r = lu.get(g["gameId"]) or {}
        F = np.full((L, len(FEATURES)), np.nan, dtype=np.float32)
        col = {f: j for j, f in enumerate(FEATURES)}
        good = "home5" in r and len(r["tokens"]) == L and np.array_equal(r["tokens"], g["tokens"])
        if good:
            H, A = r["home5"], r["away5"]
            full = (H > 0).all(1) & (A > 0).all(1)
        if good and full.any():
            ok += 1
            rt = ratings.get(g["gameId"], {})
            sp, lg = spacing[g["season"]]
            o = five_sum(lambda p: rt.get(p, (0.0, 0.0))[0])
            dd = five_sum(lambda p: rt.get(p, (0.0, 0.0))[1])
            ha, aa = o(H) + dd(A), o(A) + dd(H)
            F[:, col["home_att"]], F[:, col["away_att"]], F[:, col["net"]] = ha, aa, ha - aa
            F[:, col["start_net"]] = (ha - aa)[np.argmax(full)]  # the first full fives: the starters
            big = five_sum(lambda p: float(pos.get((g["season"], p), p in big_any)))
            spc = five_sum(lambda p: sp.get(p, lg))
            F[:, col["bigs_diff"]] = big(H) - big(A)
            F[:, col["spacing_diff"]] = spc(H) - spc(A)
            F[~full, :6] = np.nan
            F[~full, col["start_net"]] = (ha - aa)[np.argmax(full)]
        keys = {"team": {"h": g["home"], "a": g["away"]},
                "coach": {"h": coach.get((g["gameId"], "home")), "a": coach.get((g["gameId"], "away"))}}
        for kind, tally in (("team", team_t), ("coach", coach_t)):
            for s in ("h", "a"):
                key = keys[kind][s]
                if key is None:
                    continue
                for k in TRAIL:
                    raw, above = tally.rate(key, k)
                    F[:, col[f"{kind}_cb{k}_{s}"]] = raw
                    F[:, col[f"{kind}_cbx{k}_{s}"]] = above
                if kind == "coach":
                    F[:, col[f"coach_wax_{s}"]] = tally.wins_above(key)
        mins = player_minutes(r["stints"]) if "stints" in r else None
        if mins:
            rt = ratings.get(g["gameId"], {})
            net = lambda p: rt.get(p, (0.0, 0.0))[0] - rt.get(p, (0.0, 0.0))[1]
            lost = {}
            for s, t in (("h", g["home"]), ("a", g["away"])):
                gone = [(np.mean(usual[p]) / 48, p) for p, last in seen[t].items()
                        if team_n[t] - last <= MISS_WINDOW and np.mean(usual[p]) >= MISS_MIN and p not in mins[s]]
                lost[s] = (sum(m * net(p) for m, p in gone), sum(m for m, _ in gone))
            F[:, col["miss_diff"]] = lost["h"][0] - lost["a"][0]
            F[:, col["nmiss_diff"]] = lost["h"][1] - lost["a"][1]
            for s, t in (("h", g["home"]), ("a", g["away"])):
                for p, m in mins[s].items():
                    usual[p].append(m); seen[t][p] = team_n[t]
                team_n[t] += 1
        out[g["gameId"]] = F
        # after the game: update the tallies with what happened in it
        p_home = clf.predict_proba(elo_features(g, L))[:, 1]
        ev = comeback_events(g, p_home)
        p_pre = 1 / (1 + np.exp(-g.get("elo", 0.0)))
        for kind, tally in (("team", team_t), ("coach", coach_t)):
            for s, side in (("h", "home"), ("a", "away")):
                key = keys[kind][s]
                if key is None:
                    continue
                for k in TRAIL:
                    if (side, k) in ev:
                        tally.add(key, k, *ev[(side, k)])
                won = g["home_win"] if side == "home" else 1 - g["home_win"]
                tally.add_game(key, won, p_pre if side == "home" else 1 - p_pre)
    print(f"lineups matched the tokens in {ok}/{len(games)} games")
    return out, coach


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/games.pkl")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--context", default="data/context")
    ap.add_argument("--shots", default="data/processed/shots_30.parquet", help="for spacing: needs the season before the first")
    ap.add_argument("--lineups", default="data/processed/lineups_cache.pkl", help="cache of the raw extraction")
    ap.add_argument("--out", default="data/processed/game_context.pkl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip_extract", action="store_true")
    ap.add_argument("--tune_ridge", action="store_true")
    args = ap.parse_args()

    with open(args.data, "rb") as fh:
        d = pickle.load(fh)
    games, splits = d["games"], d["splits"]
    if args.skip_extract and Path(args.lineups).exists():
        lu = pickle.load(open(args.lineups, "rb"))
    else:
        print(f"extracting lineups from {args.raw}", flush=True)
        lu = extract_all(args.raw, games, args.workers)
        with open(args.lineups, "wb") as fh:
            pickle.dump(lu, fh)
    bad = [k for k, v in lu.items() if not v or "error" in v]
    print(f"{len(lu)} games extracted, {len(bad)} without lineups")
    if args.tune_ridge:
        seasons_of = {gid: r["season"] for gid, r in lu.items() if r and "season" in r}
        order_ids = sorted(seasons_of, key=lambda gid: (seasons_of[gid], gid[:3] != "002", gid))
        tune_ridge(lu, order_ids, seasons_of)
        return
    feats, coach = build(games, splits, lu, args.context, args.shots,
                         ratings_cache=Path(args.lineups).with_name("ratings_cache.pkl"))
    lineup = {gid: (r["home5"], r["away5"]) for gid, r in lu.items() if r and "home5" in r}
    with open(args.out, "wb") as fh:
        pickle.dump({"features": FEATURES, "groups": GROUPS, "monotone": MONOTONE, "values": feats,
                     "lineups": lineup, "coach": coach}, fh)
    print(f"wrote {args.out}: {len(feats)} games x {len(FEATURES)} features")


if __name__ == "__main__":
    main()
