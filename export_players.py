"""Player / defender / coach data for the dashboard, including shot-model weights for in-browser inference.

Used by export_dashboard.py when --shot_run is given.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from build_shots import STYLES, ZONES
from fetch_context import PLAY_TYPES
from player_names import full_name
from shot_model import ShotNet, encode, player_priors, split_shots, team_profiles

ZONE_VALUE = np.array([3 if z.endswith("_3") else 2 for z in ZONES], dtype=np.float32)
RIM, MID, THREE = [0, 1], [2, 3, 4, 5], [6, 7]


def r(a, nd=4):
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def zone_table(df):
    g = df.groupby("zone")["made"].agg(["count", "sum"]).reindex(ZONES).fillna(0).astype(int)
    return g.to_numpy().tolist()  # [[fga, fgm] per zone]


def style_table(df):
    g = df.groupby("style")["made"].agg(["count", "sum"]).reindex(STYLES).fillna(0).astype(int)
    return g.to_numpy().tolist()


def share(df, col, keys):
    return r(df[col].value_counts(normalize=True).reindex(keys).fillna(0), 4)


@torch.no_grad()
def predict_with(model, b, defender_vec=None, coach_def_vec=None, coach_off_vec=None):
    """Zone probs, make probs for every zone, expected points per shot; optionally overriding context vectors."""
    n = len(b["shooter"])
    parts = [model.off_emb(b["shooter"]), model.masked_mean(model.off_emb, b["mates"]),
             model.def_emb(b["defender"]) if defender_vec is None else defender_vec.expand(n, -1),
             model.masked_mean(model.def_emb, b["defs"]),
             model.coach_off_emb(b["coach_off"]) if coach_off_vec is None else coach_off_vec.expand(n, -1),
             model.coach_def_emb(b["coach_def"]) if coach_def_vec is None else coach_def_vec.expand(n, -1),
             b["prof_off"], b["prof_def"], b["state"]]
    h = model.mlp(torch.cat(parts, -1))
    pz = torch.softmax(model.prior_zone[b["shooter"]] + model.zone_head(h), -1)
    Z = len(ZONES)
    pm = torch.stack([torch.sigmoid(model.prior_make[b["shooter"], z] + model.make_head(
        torch.cat([h, torch.nn.functional.one_hot(torch.full((n,), z), Z).float()], -1)).squeeze(-1)) for z in range(Z)], -1)
    pps = (pz * pm * torch.from_numpy(ZONE_VALUE)).sum(-1)
    return pz, pm, pps


def effect(pz, pps, pz0, pps0):
    d = (pz - pz0).mean(0)
    return {"rim": round(float(d[RIM].sum()), 4), "mid": round(float(d[MID].sum()), 4),
            "three": round(float(d[THREE].sum()), 4), "pps": round(float((pps - pps0).mean()), 4)}


@torch.no_grad()
def build_players_payload(shots_path, run_dir, context="data/context", n_players=250, max_dots=350, seed=0):
    run_dir = Path(run_dir)
    meta = json.load(open(run_dir / "meta.json"))
    shots = pd.read_parquet(shots_path)
    train, _, test = split_shots(shots)
    vocabs = {k: {int(p): i for p, i in v.items()} for k, v in meta["vocabs"].items()}
    prior_zone, prior_make = player_priors(train, vocabs["off"])
    d = meta["dims"]
    model = ShotNet(d["n_off"], d["n_def"], d["n_coach"], prior_zone, prior_make)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    model.eval()
    prof_off, prof_def, _ = team_profiles(context)
    rng = np.random.default_rng(seed)

    # names: full names from player_names.py; the play-by-play surname only if no source knows him
    shooter_names = shots.drop_duplicates("shooterId").set_index("shooterId")["shooter"].to_dict()
    name = lambda pid: full_name(pid, shooter_names.get(int(pid)), context)

    coaches = pd.read_parquet(Path(context) / "coaches.parquet") if (Path(context) / "coaches.parquet").exists() else pd.DataFrame()
    coach_name = dict(zip(coaches.get("coachId", []), coaches.get("coach", [])))

    # reference sample for effect sizes
    ref = shots.sample(min(20000, len(shots)), random_state=seed).reset_index(drop=True)
    eref = encode(ref, vocabs, prof_off, prof_def)
    def_ids = [p for p in vocabs["def"]]
    def_counts = pd.Series(np.concatenate(train["def5"].to_numpy())).value_counts()
    w = torch.tensor([def_counts.get(p, 0) for p in def_ids], dtype=torch.float32)
    def_mean = (model.def_emb.weight[[vocabs["def"][p] for p in def_ids]] * w[:, None]).sum(0) / w.sum()
    coach_ids = list(vocabs["coach"])
    coff_mean = model.coach_off_emb.weight[[vocabs["coach"][c] for c in coach_ids]].mean(0)
    cdef_mean = model.coach_def_emb.weight[[vocabs["coach"][c] for c in coach_ids]].mean(0)
    pz0, _, pps0 = predict_with(model, eref, defender_vec=def_mean, coach_def_vec=cdef_mean, coach_off_vec=coff_mean)

    # defenders: model effect + what actually happened when he was the primary defender
    defenders = []
    for pid in def_ids:
        vec = model.def_emb.weight[vocabs["def"][pid]]
        pz, _, pps = predict_with(model, eref, defender_vec=vec, coach_def_vec=cdef_mean, coach_off_vec=coff_mean)
        faced = shots[shots["defenderId"] == pid]
        defenders.append({"id": int(pid), "name": name(pid), "idx": vocabs["def"][pid], "n": int(len(faced)),
                          "fg": round(float(faced["made"].mean()), 4) if len(faced) else None,
                          "zones": share(faced, "zone", ZONES) if len(faced) else None,
                          "effect": effect(pz, pps, pz0, pps0)})
    defenders.sort(key=lambda x: -x["n"])

    # coaches
    coach_rows = []
    for cid in coach_ids:
        if cid not in coach_name:
            continue
        pz, _, pps = predict_with(model, eref, defender_vec=def_mean, coach_def_vec=model.coach_def_emb.weight[vocabs["coach"][cid]], coach_off_vec=coff_mean)
        pzo, _, ppso = predict_with(model, eref, defender_vec=def_mean, coach_def_vec=cdef_mean, coach_off_vec=model.coach_off_emb.weight[vocabs["coach"][cid]])
        stints = coaches[coaches["coachId"] == cid].sort_values("season")
        last = stints.iloc[-1]
        off = shots[shots["offCoachId"] == cid]
        dfn = shots[shots["defCoachId"] == cid]
        key = (last["season"], int(last["teamId"]))
        coach_rows.append({
            "id": int(cid), "name": coach_name[cid], "idx": vocabs["coach"][cid],
            "stints": [f"{t} {s}" for t, s in zip(stints["team"], stints["season"])],
            "team": last["team"], "season": last["season"],
            "prof_off": r(prof_off.get(key, np.zeros(len(PLAY_TYPES)))), "prof_def": r(prof_def.get(key, np.zeros(len(PLAY_TYPES)))),
            "off_zones": share(off, "zone", ZONES), "def_zones": share(dfn, "zone", ZONES),
            "off_fg": round(float(off["made"].mean()), 4) if len(off) else None,
            "def_fg": round(float(dfn["made"].mean()), 4) if len(dfn) else None,
            "effect_off": effect(pzo, ppso, pz0, pps0), "effect_def": effect(pz, pps, pz0, pps0),
        })
    coach_rows.sort(key=lambda c: c["name"].split()[-1])

    # Synergy play types (team offense for coaches, players for players)
    pt_team, pt_player = {}, {}
    ppath = Path(context) / "playtypes.parquet"
    if ppath.exists():
        pt = pd.read_parquet(ppath)
        t = pt[(pt["level"] == "T") & (pt["grouping"] == "offensive")]
        for (s, tid), g in t.groupby(["season", "TEAM_ID"]):
            g = g.set_index("PLAY_TYPE").reindex(PLAY_TYPES)
            pt_team[(s, int(tid))] = {"freq": r(g["POSS_PCT"].fillna(0), 3), "ppp": r(g["PPP"].fillna(0), 3)}
        p = pt[pt["level"] == "P"]
        for pid, g in p.groupby("PLAYER_ID"):
            agg = g.groupby("PLAY_TYPE").agg(poss=("POSS", "sum"), pts=("PTS", "sum")).reindex(PLAY_TYPES).fillna(0)
            tot = agg["poss"].sum()
            if tot > 0:
                pt_player[int(pid)] = {"freq": r(agg["poss"] / tot, 3),
                                       "ppp": r(np.where(agg["poss"] > 0, agg["pts"] / agg["poss"].clip(lower=1), 0), 3),
                                       "poss": int(tot)}
    league_pt = None
    if pt_team:
        league_pt = {"freq": r(np.mean([v["freq"] for v in pt_team.values()], 0), 3),
                     "ppp": r(np.mean([v["ppp"] for v in pt_team.values()], 0), 3)}
    for c in coach_rows:
        c["playtypes"] = pt_team.get((c["season"], int(coaches[(coaches["coachId"] == c["id"])].iloc[-1]["teamId"])))

    # players
    top = shots["shooterId"].value_counts()
    top = [int(p) for p in top.index if int(p) in vocabs["off"]][:n_players]
    players = []
    for pid in top:
        ps = shots[shots["shooterId"] == pid]
        last = ps.sort_values(["season", "gameId"]).iloc[-1]
        enc = encode(ps.head(2000), vocabs, prof_off, prof_def)
        mates_vec = model.masked_mean(model.off_emb, enc["mates"]).mean(0)
        dots = ps.sample(min(max_dots, len(ps)), random_state=seed)
        faced = (ps[ps["defenderId"] > 0].groupby("defenderId")
                 .agg(n=("made", "size"), fgm=("made", "sum")).query("n >= 12").sort_values("n", ascending=False).head(12))
        players.append({
            "id": pid, "name": name(pid), "idx": vocabs["off"][pid], "team": last["offTeam"],
            "coachId": int(last["offCoachId"]), "season": last["season"], "n": int(len(ps)),
            "zones": zone_table(ps), "styles": style_table(ps),
            "assisted": round(float(ps["assisted"][ps["made"] == 1].mean()), 3) if ps["made"].sum() else 0,
            "dots": [[int(x), int(y), int(m), ZONES.index(z)] for x, y, m, z in zip(dots["x"], dots["y"], dots["made"], dots["zone"])],
            "faced": [{"id": int(i), "name": name(i), "n": int(row.n), "fg": round(float(row.fgm / row.n), 3)} for i, row in faced.iterrows()],
            "mates": r(mates_vec.numpy()),
            "prof_off": r(prof_off.get((last["season"], int(last["offTeamId"])), np.zeros(len(PLAY_TYPES)))),
            "playtypes": pt_player.get(pid),
        })

    sd = model.state_dict()
    W = lambda k: r(sd[k].numpy())
    weights = {k: W(k) for k in ("off_emb.weight", "def_emb.weight", "coach_off_emb.weight", "coach_def_emb.weight",
                                 "prior_zone", "prior_make", "mlp.0.weight", "mlp.0.bias", "mlp.3.weight", "mlp.3.bias",
                                 "zone_head.weight", "zone_head.bias", "style_head.weight", "style_head.bias",
                                 "make_head.weight", "make_head.bias")}
    return {
        "zones": ZONES, "styles": STYLES, "play_types": PLAY_TYPES, "zone_value": ZONE_VALUE.tolist(),
        "league": {"zones": zone_table(shots), "styles": style_table(shots), "playtypes": league_pt},
        "metrics": meta["metrics"], "seasons": sorted(shots["season"].unique().tolist()),
        "n_shots": int(len(shots)), "defender_coverage": round(float((shots["defenderId"] > 0).mean()), 3),
        "players": players, "defenders": defenders, "coaches": coach_rows,
        "avg": {"def": r(def_mean.numpy()), "coach_off": r(coff_mean.numpy()), "coach_def": r(cdef_mean.numpy()),
                "defs": r(model.masked_mean(model.def_emb, eref["defs"]).mean(0).numpy()),
                "conf": round(float(shots.loc[shots["defenderId"] > 0, "defenderConf"].mean()), 3) if (shots["defenderId"] > 0).any() else 0.0},
        "weights": weights,
    }
