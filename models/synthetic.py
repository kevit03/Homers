"""Generate fake play-by-play in the PlayByPlayV3 schema, for testing the pipeline
without hitting the NBA API.

    python -m models.synthetic --games 600 --out data/raw_synthetic
    python -m models.tokenize_pbp --raw data/raw_synthetic --out data/processed/synthetic.pkl
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def sim_game(gid, season, rng):
    edge = rng.normal(0, 0.04)  # home team strength for this game
    rows, sh, sa = [], 0, 0

    def add(period, clock, atype, loc="", sub="", desc="", shot_value="", result=""):
        rows.append(dict(actionNumber=len(rows) + 1, period=period,
                         clock=f"PT{int(clock // 60):02d}M{clock % 60:05.2f}S", location=loc,
                         actionType=atype, subType=sub, description=desc, shotValue=str(shot_value),
                         shotResult=result, scoreHome=str(sh), scoreAway=str(sa), gameId=gid, season=season))

    period, poss = 0, rng.integers(2)
    while period < 4 or sh == sa:
        period += 1
        clock = 720.0 if period <= 4 else 300.0
        add(period, clock, "period", sub="start")
        while True:
            clock -= rng.uniform(6, 20)
            if clock <= 0:
                break
            loc, other = ("h", "v") if poss == 0 else ("v", "h")
            bonus = edge if loc == "h" else -edge
            r = rng.random()
            if r < 0.13:
                add(period, clock, "Turnover", loc); poss ^= 1
            elif r < 0.16:
                add(period, clock, "Timeout", loc)
            elif r < 0.24:
                add(period, clock, "Foul", other, sub="Shooting")
                for k in range(2):
                    made = rng.random() < 0.77 + bonus
                    if made:
                        sh, sa = (sh + 1, sa) if loc == "h" else (sh, sa + 1)
                    add(period, clock, "Free Throw", loc, desc="" if made else "MISS free throw")
                poss ^= 1
            else:
                three = rng.random() < 0.38
                made = rng.random() < (0.36 if three else 0.53) + bonus
                if made:
                    pts = 3 if three else 2
                    sh, sa = (sh + pts, sa) if loc == "h" else (sh, sa + pts)
                    add(period, clock, "Made Shot", loc, shot_value=pts, result="Made"); poss ^= 1
                else:
                    add(period, clock, "Missed Shot", loc, shot_value=3 if three else 2, result="Missed")
                    oreb = rng.random() < 0.25
                    add(period, clock, "Rebound", loc if oreb else other)
                    if not oreb:
                        poss ^= 1
        add(period, 0.0, "period", sub="end")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--out", default="data/raw_synthetic")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    for i in range(args.games):
        season = "2098-99" if i < args.games * 0.8 else "2099-00"
        d = Path(args.out) / season
        d.mkdir(parents=True, exist_ok=True)
        gid = f"00{9900000 + i}"
        sim_game(gid, season, rng).to_parquet(d / f"{gid}.parquet")
    print(f"wrote {args.games} synthetic games to {args.out}")


if __name__ == "__main__":
    main()
