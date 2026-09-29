"""Turn raw play-by-play into token sequences + game-state features.

Each event becomes one token such as H_3PT_MAKE or A_DREB (H = home, A = away).
Alongside each token we store the game state *after* that event:
seconds remaining, score differential (home - away), and period.

    python tokenize_pbp.py --raw data/raw --out data/processed/games.pkl
"""
import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

SPECIAL = ["<pad>", "<bos>", "<eos>"]
EVENTS = ["2PT_MAKE", "2PT_MISS", "3PT_MAKE", "3PT_MISS", "FT_MAKE", "FT_MISS",
          "OREB", "DREB", "TOV", "FOUL", "VIOLATION", "TIMEOUT", "JUMPBALL"]
VOCAB = SPECIAL + [f"{s}_{e}" for s in ("H", "A") for e in EVENTS] + ["PERIOD_START", "PERIOD_END"]
STOI = {t: i for i, t in enumerate(VOCAB)}
PAD, BOS, EOS = 0, 1, 2
SIMPLE = {"turnover": "TOV", "foul": "FOUL", "violation": "VIOLATION",
          "timeout": "TIMEOUT", "jump ball": "JUMPBALL"}
GAME_TYPES = {"002": "Regular Season", "004": "Playoffs", "005": "Play-In"}  # game ID prefix -> type


def game_type(gid) -> str:
    return GAME_TYPES.get(str(gid)[:3], "Other")


def season_key(season, gid) -> str:
    """How stats are grouped: '2024-25' for the regular season, '2024-25 Playoffs' for its playoffs."""
    t = game_type(gid)
    return season if t == "Regular Season" else f"{season} {t}"


def chrono_games(df) -> list:
    """Game IDs in time order. Sorting IDs alone puts every regular-season game (002...) before any playoff game (004...)."""
    return df.drop_duplicates("gameId").sort_values(["season", "gameId"])["gameId"].tolist()


def parse_clock(c) -> float:
    m = re.match(r"PT(\d+)M([\d.]+)S", str(c))
    return int(m.group(1)) * 60 + float(m.group(2)) if m else 0.0


def seconds_remaining(period: int, clock: float) -> float:
    """Regulation time left; in overtime, time left in the OT period."""
    return (4 - period) * 720 + clock if period <= 4 else clock


def game_order(df: pd.DataFrame) -> pd.DataFrame:
    """Rows in game-clock order: period, then clock counting down, then the feed's own order.

    Late scorekeeper entries get high actionNumbers, so sorting by actionNumber put about 1% of plays
    minutes out of place (a block at 3:58 of 2OT landing after the end of the period, say).
    The feed's row order already has them in the right place, so it breaks ties within a second.
    """
    df = df.copy()
    df["actionNumber"] = pd.to_numeric(df["actionNumber"], errors="coerce")
    df["period"] = pd.to_numeric(df["period"], errors="coerce").fillna(1).astype(int)
    df["_clock"] = df["clock"].map(parse_clock)
    df["_row"] = np.arange(len(df))
    df = df.sort_values(["period", "_clock", "_row"], ascending=[True, False, True], kind="mergesort")
    return df.drop(columns=["_clock", "_row"]).reset_index(drop=True)


def classify(row, last_shot_side):
    a = str(row.get("actionType", "")).strip().lower()
    desc = str(row.get("description", "")).upper()
    if a == "period":
        return "PERIOD_START" if "start" in str(row.get("subType", "")).lower() else "PERIOD_END"
    side = {"h": "H", "v": "A"}.get(str(row.get("location", "")).strip().lower())
    if side is None:
        return None
    if a in ("made shot", "missed shot"):
        three = str(row.get("shotValue", "")).startswith("3") or "3PT" in desc
        return f"{side}_{'3PT' if three else '2PT'}_{'MAKE' if a == 'made shot' else 'MISS'}"
    if a == "free throw":
        return f"{side}_FT_{'MISS' if 'MISS' in desc else 'MAKE'}"
    if a == "rebound":
        if last_shot_side is None:
            return None
        return f"{side}_{'OREB' if side == last_shot_side else 'DREB'}"
    if a in SIMPLE:
        return f"{side}_{SIMPLE[a]}"
    return None  # substitutions, replays, stoppages, etc.


def process_game(df: pd.DataFrame):
    df = game_order(df)
    # A team's score never goes down, so carry its running max. Older feeds zero-fill non-scoring rows
    # (a miss reads 0-0), and period rows can repeat a stale score: 0022500232's "End of 4th Period"
    # reads 60-55, the halftime score, which made MIN the winner of a game DEN won 123-112.
    for col in ("scoreHome", "scoreAway"):
        df[col] = pd.to_numeric(df[col], errors="coerce").ffill().fillna(0).cummax()
    df["period"] = pd.to_numeric(df["period"], errors="coerce").fillna(1).astype(int)

    final_h, final_a = df["scoreHome"].iloc[-1], df["scoreAway"].iloc[-1]
    if final_h == final_a:
        return None

    tokens, sec, diff, period, total = [BOS], [2880.0], [0.0], [1], [0]
    last_shot_side = None
    for row in df.to_dict("records"):
        tok = classify(row, last_shot_side)
        if tok is None:
            continue
        if tok[2:] in ("2PT_MISS", "3PT_MISS", "FT_MISS"):
            last_shot_side = tok[0]
        p = int(row["period"])
        tokens.append(STOI[tok])
        sec.append(seconds_remaining(p, parse_clock(row["clock"])))
        diff.append(float(row["scoreHome"] - row["scoreAway"]))
        total.append(int(row["scoreHome"] + row["scoreAway"]))
        period.append(p)
    tokens.append(EOS); sec.append(0.0); diff.append(float(final_h - final_a)); period.append(period[-1]); total.append(int(final_h + final_a))

    return {
        "gameId": str(df["gameId"].iloc[0]),
        "season": str(df["season"].iloc[0]),
        "tokens": np.array(tokens, dtype=np.int16),
        "sec": np.array(sec, dtype=np.float32),
        "diff": np.array(diff, dtype=np.float32),
        "period": np.array(period, dtype=np.int8),
        "total": np.array(total, dtype=np.int16),  # home + away points after each event, from the feed's own score
        "home_win": int(final_h > final_a),
    }


def split(games, val_frac=0.1):
    """Latest season = test. Otherwise last chunk (chronological) = test."""
    seasons = sorted({g["season"] for g in games})
    idx = list(range(len(games)))
    if len(seasons) > 1:
        test = [i for i in idx if games[i]["season"] == seasons[-1]]
        rest = [i for i in idx if games[i]["season"] != seasons[-1]]
    else:
        n = int(len(idx) * 0.1)
        test, rest = idx[-n:], idx[:-n]
    n_val = max(1, int(len(rest) * val_frac))
    return {"train": rest[:-n_val], "val": rest[-n_val:], "test": test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed/games.pkl")
    ap.add_argument("--seasons", nargs="*", help="only these season folders, e.g. 2021-22 2022-23 (default: all)")
    args = ap.parse_args()

    files = sorted(f for f in Path(args.raw).rglob("*.parquet") if not args.seasons or f.parent.name in args.seasons)
    print(f"{len(files)} raw games")
    games = []
    for f in files:
        try:
            g = process_game(pd.read_parquet(f))
        except Exception as e:
            print(f"skip {f.name}: {e}")
            continue
        if g is not None:
            games.append(g)
    games.sort(key=lambda g: (g["season"], g["gameId"]))
    splits = split(games)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump({"vocab": VOCAB, "games": games, "splits": splits}, fh)
    json.dump(VOCAB, open(out.parent / "vocab.json", "w"), indent=1)

    lens = [len(g["tokens"]) for g in games]
    print(f"{len(games)} games | tokens: {sum(lens):,} | mean len {np.mean(lens):.0f}, max {max(lens)}")
    print({k: len(v) for k, v in splits.items()})
    print(pd.Series([game_type(g["gameId"]) for g in games]).value_counts().to_dict())
    print(f"home win rate: {np.mean([g['home_win'] for g in games]):.3f}")


if __name__ == "__main__":
    main()
