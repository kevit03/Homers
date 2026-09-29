"""Man-to-man matchup data for the dashboard: the trained model's weights (so predictions run in the
browser for any scorer / defender pair), every pair's actual head-to-head totals, current teams, and
the held-out scores. Used by export_dashboard.py when runs/matchups/best.pt exists.

    python matchup_model.py            # train first
    python export_matchups.py          # size check
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def r(a, nd=4):
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def build_matchups_payload(run_dir="runs/matchups", rosters_path="data/context/bbref_rosters.parquet", min_pair_poss=25):
    run = Path(run_dir)
    meta = json.load(open(run / "meta.json"))
    sd = torch.load(run / "best.pt", map_location="cpu")
    df = pd.read_parquet(run / "matchups.parquet")
    vo = {int(k): v for k, v in meta["vocab_off"].items()}
    vd = {int(k): v for k, v in meta["vocab_def"].items()}
    names = {int(k): v for k, v in meta["names"].items()}

    now = {}
    if rosters_path and Path(rosters_path).exists():
        ro = pd.read_parquet(rosters_path)
        ro = ro[ro["season"] == sorted(ro["season"].unique())[-1]].dropna(subset=["personId"])
        now = {int(p): t for p, t in zip(ro["personId"], ro["team"])}
        roster_season = sorted(ro["season"].unique())[-1]
    else:
        roster_season = None

    # recent volume decides who shows up first in team-vs-team grids
    recent = sorted(df["season"].unique())[-2:]
    rec = df[df["season"].isin(recent)]
    off_poss = rec.groupby("personIdOff")["poss"].sum()
    def_poss = rec.groupby("personIdDef")["poss"].sum()
    tot_off = df.groupby("personIdOff")["poss"].sum()
    tot_def = df.groupby("personIdDef")["poss"].sum()

    W = lambda k: sd[k].numpy()
    players = []
    for pid in sorted(set(vo) | set(vd)):
        players.append({
            "id": pid, "n": names.get(pid, str(pid)), "now": now.get(pid, ""),
            "o": vo.get(pid, 0), "d": vd.get(pid, 0),  # row in the weight tables (0 = shared "other" row)
            "op": int(round(off_poss.get(pid, 0))), "dp": int(round(def_poss.get(pid, 0))),
            "top": int(round(tot_off.get(pid, 0))), "tdp": int(round(tot_def.get(pid, 0))),
        })

    # actual head-to-head totals for pairs that met often enough to be worth showing
    pair = df.groupby(["personIdOff", "personIdDef"])[["poss", "pts", "fgm", "fga", "fg3m", "fg3a", "tov"]].sum()
    pair = pair[pair["poss"] >= min_pair_poss].reset_index()
    seasons_met = df.groupby(["personIdOff", "personIdDef"])["season"].nunique()
    h2h = []
    for row in pair.itertuples(index=False):
        h2h += [int(row.personIdOff), int(row.personIdDef), round(float(row.poss), 1), int(row.pts), int(row.fgm), int(row.fga),
                int(row.fg3m), int(row.fg3a), int(row.tov), int(seasons_met[(row.personIdOff, row.personIdDef)])]

    last_complete = meta["test_season"]
    si = meta["seasons"].index(last_complete)
    return {
        "targets": meta["targets"], "seasons": meta["seasons"], "test_season": meta["test_season"],
        "season_for_predictions": last_complete, "roster_season": roster_season,
        "b": r(W("b") + W("season.weight")[si]),  # league rates in the latest complete season
        "off": r(W("off.weight")), "def": r(W("dfn.weight")), "U": r(W("U.weight")), "V": r(W("V.weight")), "w": r(W("w")),
        "players": players, "h2h": h2h, "h2h_cols": ["off", "def", "poss", "pts", "fgm", "fga", "fg3m", "fg3a", "tov", "seasons"],
        "held_out": meta["held_out"], "calibration": meta.get("calibration", []), "rows": meta["rows"], "games": meta["games"], "args": meta["args"], "epochs": meta["epochs"],
        "min_pair_poss": min_pair_poss,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/matchups")
    args = ap.parse_args()
    p = build_matchups_payload(args.run)
    size = len(json.dumps(p, separators=(",", ":")))
    print(f"{len(p['players'])} players, {len(p['h2h']) // len(p['h2h_cols']):,} head-to-head pairs, "
          f"{sum(1 for x in p['players'] if x['now'])} on a {p['roster_season']} roster -> {size / 1e6:.2f} MB")
