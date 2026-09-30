"""Pre-game team strength: a leak-free Elo rating for every game, written into games.pkl.

Tempo sees only the plays, so before tip-off it cannot tell a 60-win team from a 20-win one and its
first-quarter win chance sits near a coin flip. This gives each game the home team's pre-game edge

    elo = (R_home + HCA - R_away) * ln(10) / 400      (a logit: P(home win) = sigmoid(elo))

where every rating is built only from games played before that one, so nothing leaks from the future.
Ratings follow the FiveThirtyEight NBA recipe:
  - update R += K * mov_mult * (result - expected) after each game
  - mov_mult = (|margin| + 3)^0.8 / (7.5 + 0.006 * winner's pre-game Elo edge), so blowouts count more
    but a favourite running up the score counts less (the autocorrelation fix)
  - between seasons each rating moves back toward the mean: R = carry * R + (1 - carry) * 1505
K, HCA and carry are chosen by grid search on the pre-game log loss of the training games only.

    python team_strength.py --data data/processed/games.pkl
"""
import argparse
import itertools
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

MEAN = 1505.0
TYPE_RANK = {"002": 0, "005": 1, "004": 2}  # regular season, then play-in, then playoffs
GRID = {"k": [10, 15, 20, 25, 30], "hca": [40, 60, 80, 100], "carry": [0.5, 0.65, 0.75, 0.85]}


def chrono_order(games):
    """Indices of games in time order: season, then game type, then game ID (the league numbers games by schedule)."""
    return sorted(range(len(games)), key=lambda i: (games[i]["season"], TYPE_RANK.get(games[i]["gameId"][:3], 3),
                                                   games[i]["gameId"]))


def game_teams(raw, games, cache):
    """gameId -> (home teamId, away teamId), from the game dicts when tokenize_pbp stored them, else the raw feed."""
    known = pd.read_parquet(cache).set_index("gameId") if cache.exists() else pd.DataFrame(columns=["home", "away"])
    out, fresh = {}, []
    for g in games:
        gid = g["gameId"]
        if g.get("home"):
            out[gid] = (g["home"], g["away"])
        elif gid in known.index:
            out[gid] = tuple(int(v) for v in known.loc[gid, ["home", "away"]])
        else:
            f = Path(raw) / g["season"] / f"{gid}.parquet"
            if not f.exists():
                continue
            try:
                df = pd.read_parquet(f, columns=["teamId", "location"])
            except Exception:  # synthetic games have no team IDs
                continue
            df = df[pd.to_numeric(df["teamId"], errors="coerce").fillna(0) > 0]
            side = df.groupby(df["location"].str.lower())["teamId"].agg(lambda s: int(s.mode().iloc[0]))
            if "h" in side and "v" in side:
                out[gid] = (int(side["h"]), int(side["v"]))
                fresh.append({"gameId": gid, "home": out[gid][0], "away": out[gid][1]})
    if fresh:
        cache.parent.mkdir(parents=True, exist_ok=True)
        rows = pd.concat([known.reset_index(), pd.DataFrame(fresh)], ignore_index=True).drop_duplicates("gameId")
        rows.to_parquet(cache, index=False)
    return out


def final_margin(g):
    return float(g["diff"][-1])


def run_elo(games, order, teams, k, hca, carry):
    """Pre-game home edge in Elo points for every game (NaN if the teams are unknown)."""
    R, season, edge = {}, None, np.full(len(games), np.nan)
    for i in order:
        g = games[i]
        if g["season"] != season:
            season = g["season"]
            R = {t: carry * r + (1 - carry) * MEAN for t, r in R.items()}
        if g["gameId"] not in teams:
            continue
        h, a = teams[g["gameId"]]
        rh, ra = R.get(h, MEAN), R.get(a, MEAN)
        d = rh + hca - ra
        edge[i] = d
        expected = 1 / (1 + 10 ** (-d / 400))
        margin = final_margin(g)
        win_edge = d if margin > 0 else -d
        mult = (abs(margin) + 3) ** 0.8 / (7.5 + 0.006 * win_edge)
        shift = k * mult * (g["home_win"] - expected)
        R[h], R[a] = rh + shift, ra - shift
    return edge


def log_loss(edge, y):
    p = np.clip(1 / (1 + 10 ** (-edge / 400)), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/games.pkl")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--warmup", type=int, default=400, help="skip this many earliest games when scoring the grid (ratings start flat)")
    args = ap.parse_args()

    with open(args.data, "rb") as fh:
        d = pickle.load(fh)
    games, splits = d["games"], d["splits"]
    teams = game_teams(args.raw, games, Path(args.data).with_name(Path(args.data).stem + "_teams.parquet"))
    order = chrono_order(games)
    print(f"teams known for {len(teams)}/{len(games)} games")
    if not teams:
        for g in games:
            g["elo"] = 0.0
        print("no team IDs (synthetic data?): elo set to 0 for every game")
    else:
        y = np.array([g["home_win"] for g in games], dtype=float)
        rank = np.empty(len(games), int); rank[order] = np.arange(len(games))
        fit = np.array([i for i in splits["train"] if rank[i] >= args.warmup and games[i]["gameId"] in teams])
        best = None
        for k, hca, carry in itertools.product(GRID["k"], GRID["hca"], GRID["carry"]):
            ll = log_loss(run_elo(games, order, teams, k, hca, carry)[fit], y[fit])
            if best is None or ll < best[0]:
                best = (ll, k, hca, carry)
        ll, k, hca, carry = best
        edge = run_elo(games, order, teams, k, hca, carry)
        for g, e in zip(games, edge):
            g["elo"] = 0.0 if np.isnan(e) else float(e * math.log(10) / 400)
            if g["gameId"] in teams:
                g["home"], g["away"] = teams[g["gameId"]]

        report = {"k": k, "hca": hca, "carry": carry, "train_log_loss": ll}
        for split in ("val", "test"):
            idx = np.array([i for i in splits[split] if not np.isnan(edge[i])])
            p = 1 / (1 + 10 ** (-edge[idx] / 400))
            report[split] = {"log_loss": log_loss(edge[idx], y[idx]), "brier": float(np.mean((p - y[idx]) ** 2)),
                             "acc": float(np.mean((p > 0.5) == y[idx])), "home_rate": float(y[idx].mean())}
        print(json.dumps(report, indent=1))
        json.dump(report, open(Path(args.data).with_name("elo.json"), "w"), indent=1)

    with open(args.data, "wb") as fh:
        pickle.dump(d, fh)
    print(f"wrote pre-game elo into {args.data}")


if __name__ == "__main__":
    main()
