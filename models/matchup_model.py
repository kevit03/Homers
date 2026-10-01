"""Man-to-man matchup model: what happens when scorer X is guarded by defender Y.

Data is the NBA's matchup feed (BoxScoreMatchupsV3, fetched by fetch/fetch_context.py): for every offensive
player / defender pair in every game, the partial possessions they spent matched up and what the
scorer did in them (points, field goals, threes, turnovers).

For each outcome t the model is a generalised linear model with a low-rank interaction,

    eta_t = b_t + season_t + off_t[X] + def_t[Y] + sum_k w_tk * U[X, k] * V[Y, k]

  - points, shot attempts and turnovers per possession: Poisson regression with the partial
    possessions as exposure (log link), so a 3-possession matchup counts 3x less than a 9-possession one
  - FG%: binomial (logistic) regression on the attempts in the matchup
  - off / def are each player's main effects; U, V are shared k-dim "style" vectors whose products
    capture specific pairings (a big guarding a guard, a length defender on a jump shooter, ...)
  - ridge (L2) penalties shrink everything toward the league average, harder on the interaction

Evaluation is out of time: train on every season but the last, early-stop on the last 10% of those
games, then score the held-out season against league-average, scorer-only and additive
(scorer + defender, no interaction) versions of the same model. The final model is refit on all
seasons with the chosen number of epochs, and that is what the dashboard uses.

    python -m models.matchup_model --context data/context --out runs/matchups
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.tokenize_pbp import chrono_games

TARGETS = ["pts", "fga", "tov"]        # Poisson counts per possession
COLS = {"matchupFieldGoalsMade": "fgm", "matchupFieldGoalsAttempted": "fga", "playerPoints": "pts",
        "matchupTurnovers": "tov", "matchupThreePointersAttempted": "fg3a", "matchupThreePointersMade": "fg3m",
        "partialPossessions": "poss", "matchupFreeThrowsAttempted": "fta"}


def load_matchups(context):
    frames = []
    for f in sorted(Path(context, "matchups").rglob("*.parquet")):
        m = pd.read_parquet(f, columns=["gameId", "personIdOff", "personIdDef", "firstNameOff", "familyNameOff",
                                        "firstNameDef", "familyNameDef", *COLS])
        m["season"] = f.parent.name
        frames.append(m)
    df = pd.concat(frames, ignore_index=True).rename(columns=COLS)
    for c in ["personIdOff", "personIdDef", *COLS.values()]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["personIdOff", "personIdDef", "poss"])
    df = df[df["poss"] > 0].copy()
    df[["personIdOff", "personIdDef"]] = df[["personIdOff", "personIdDef"]].astype(int)
    df[list(COLS.values())] = df[list(COLS.values())].fillna(0)
    return df.reset_index(drop=True)


def vocab(ids, poss, min_poss):
    tot = pd.Series(poss).groupby(np.asarray(ids)).sum()
    keep = sorted(int(i) for i in tot[tot >= min_poss].index)
    return {pid: k + 1 for k, pid in enumerate(keep)}  # 0 = everyone below the cutoff


class MatchupNet(nn.Module):
    def __init__(self, n_off, n_def, n_season, k, base, interaction=True, defense=True):
        super().__init__()
        T = len(TARGETS) + 1  # + FG%
        self.b = nn.Parameter(base.clone())  # start at league rates, so every effect is a shift from average
        self.season = nn.Embedding(n_season, T)
        self.off = nn.Embedding(n_off, T)
        self.dfn = nn.Embedding(n_def, T)
        self.U = nn.Embedding(n_off, k)
        self.V = nn.Embedding(n_def, k)
        self.w = nn.Parameter(torch.ones(T, k) * 0.3)
        for e in (self.season, self.off, self.dfn):
            nn.init.zeros_(e.weight)
        for e in (self.U, self.V):
            nn.init.normal_(e.weight, std=0.3)
        self.interaction, self.defense = interaction, defense

    def forward(self, o, d, s):
        eta = self.b + self.season(s) + self.off(o)
        if self.defense:
            eta = eta + self.dfn(d)
        if self.interaction:
            eta = eta + (self.U(o) * self.V(d)) @ self.w.T
        return eta  # [n, T]: log-rates for pts, fga, tov, then logit FG%


def nll(eta, y):
    """Poisson NLL for the per-possession counts (exposure = poss) + binomial NLL for makes given attempts."""
    poss = y[:, 0]
    loss = 0.0
    for t in range(len(TARGETS)):
        mu = poss * torch.exp(eta[:, t])
        loss = loss + (mu - y[:, 1 + t] * torch.log(mu + 1e-9)).sum()
    fga, fgm = y[:, 2], y[:, 4]
    loss = loss + (fga * torch.nn.functional.softplus(eta[:, 3]) - fgm * eta[:, 3]).sum()
    return loss


def deviances(eta, y):
    """Per-target mean deviance (lower is better), comparable across models."""
    poss, out = y[:, 0], {}
    for t, name in enumerate(TARGETS):
        mu, k = poss * torch.exp(eta[:, t]), y[:, 1 + t]
        dev = 2 * (torch.where(k > 0, k * torch.log(k / mu), torch.zeros_like(k)) - (k - mu))
        out[name] = float(dev.sum() / poss.sum())
    fga, fgm = y[:, 2], y[:, 4]
    p = torch.sigmoid(eta[:, 3]).clamp(1e-6, 1 - 1e-6)
    out["fg"] = float(-(fgm * torch.log(p) + (fga - fgm) * torch.log(1 - p)).sum() / fga.sum())
    return out


def league_rates(y):
    """log points / shots / turnovers per possession and logit FG% over a set of rows."""
    poss, pts, fga, tov, fgm = (float(y[:, i].sum()) for i in range(5))
    return torch.tensor([math.log(pts / poss), math.log(fga / poss), math.log(tov / poss), math.log(fgm / (fga - fgm))])


def tensors(df, voff, vdef, vseason):
    o = torch.tensor([voff.get(p, 0) for p in df["personIdOff"]])
    d = torch.tensor([vdef.get(p, 0) for p in df["personIdDef"]])
    s = torch.tensor([vseason[x] for x in df["season"]])
    y = torch.tensor(df[["poss", "pts", "fga", "tov", "fgm"]].to_numpy(np.float32))
    return o, d, s, y


def fit(model, data, val=None, epochs=40, lr=0.03, l2_main=2.0, l2_int=8.0, bs=65536, patience=4, seed=0, log=True):
    """Adam on minibatches; ridge penalties are scaled per row so their strength doesn't depend on data size."""
    torch.manual_seed(seed)
    o, d, s, y = data
    n = len(y)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_ep, bad, state, hist = float("inf"), 0, 0, None, []
    total_poss = float(y[:, 0].sum())
    for ep in range(epochs):
        perm = torch.randperm(n)
        run = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            eta = model(o[idx], d[idx], s[idx])
            frac = len(idx) / n
            pen = (l2_main * (model.off.weight.pow(2).sum() + model.dfn.weight.pow(2).sum() + model.season.weight.pow(2).sum())
                   + l2_int * (model.U.weight.pow(2).sum() + model.V.weight.pow(2).sum() + model.w.pow(2).sum()))
            loss = (nll(eta, y[idx]) / total_poss) + frac * pen / total_poss * 1000
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item()
        rec = {"epoch": ep + 1, "train": run}
        if val is not None:
            with torch.no_grad():
                dv = deviances(model(*val[:3]), val[3])
            rec.update({f"val_{k}": v for k, v in dv.items()})
            score = dv["pts"] + dv["fg"]
            if score < best - 1e-5:
                best, best_ep, bad = score, ep + 1, 0
                state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            if log and (ep < 3 or ep % 5 == 4):
                print(f"  ep {ep + 1:2d} | val dev pts {dv['pts']:.4f} fga {dv['fga']:.4f} tov {dv['tov']:.4f} fg {dv['fg']:.4f}")
            if bad >= patience:
                break
        hist.append(rec)
    if state is not None:
        model.load_state_dict(state)
    return best_ep or epochs, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", default="data/context")
    ap.add_argument("--out", default="runs/matchups")
    ap.add_argument("--k", type=int, default=6, help="rank of the scorer x defender interaction")
    ap.add_argument("--min_poss", type=float, default=300, help="partial possessions for a player to get his own parameters")
    ap.add_argument("--l2_main", type=float, default=0.3, help="ridge on main effects, in units of 1,000 possessions")
    ap.add_argument("--l2_int", type=float, default=1.0, help="ridge on the interaction (overwritten by the grid)")
    ap.add_argument("--l2_int_grid", type=float, nargs="+", default=[0.003, 0.03, 0.3], help="interaction ridge values to try on validation")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_matchups(args.context)
    seasons = sorted(df["season"].unique())
    print(f"{len(df):,} matchup rows, {df['gameId'].nunique():,} games, seasons {seasons}")
    per_season = df.groupby("season")["gameId"].nunique()
    complete = [s for s in seasons if per_season[s] >= 0.9 * per_season.max()]
    test_season = complete[-1]
    rest = df[df["season"] < test_season]  # nothing from the test season or later leaks into training
    games = chrono_games(rest)
    val_games = set(games[-max(1, len(games) // 10):])
    train, val, test = rest[~rest["gameId"].isin(val_games)], rest[rest["gameId"].isin(val_games)], df[df["season"] == test_season]

    voff = vocab(train["personIdOff"], train["poss"], args.min_poss)
    vdef = vocab(train["personIdDef"], train["poss"], args.min_poss)
    vseason = {s: i for i, s in enumerate(seasons)}
    T = lambda x: tensors(x, voff, vdef, vseason)
    tr, va, te = T(train), T(val), T(test)
    print(f"train {len(train):,} / val {len(val):,} / test {test_season}: {len(test):,} rows | {len(voff)} scorers, {len(vdef)} defenders with own parameters")
    base = league_rates(tr[3])

    # held-out comparison: the same model with parts switched off
    results = {}
    with torch.no_grad():
        y = te[3]
        results["League average"] = deviances(base.expand(len(y), -1), y)
    epochs_used, val_scores, history = {}, {}, []
    runs = [("Scorer only", dict(interaction=False, defense=False), args.l2_int),
            ("Scorer + defender", dict(interaction=False, defense=True), args.l2_int)]
    runs += [(f"Interaction, l2 {l2}", dict(interaction=True, defense=True), l2) for l2 in args.l2_int_grid]
    for name, kw, l2i in runs:
        print(f"{name}:")
        m = MatchupNet(len(voff) + 1, len(vdef) + 1, len(seasons), args.k, base, **kw)
        ep, hist = fit(m, tr, va, epochs=args.epochs, l2_main=args.l2_main, l2_int=l2i, log=False)
        m.eval()
        with torch.no_grad():
            v = deviances(m(*va[:3]), va[3])
            results[name] = deviances(m(*te[:3]), te[3])
        epochs_used[name], val_scores[name] = ep, v["pts"] + v["fg"]
        print(f"  stopped at epoch {ep}; val pts {v['pts']:.4f} fg {v['fg']:.4f}")
        if kw["interaction"] and val_scores[name] <= min(s for n, s in val_scores.items() if n.startswith("Interaction")):
            history, best_int, best_model = hist, (name, l2i), m
    # keep only the interaction strength validation picked, under a plain name
    for name, _, _ in runs:
        if name.startswith("Interaction") and name != best_int[0]:
            results.pop(name); epochs_used.pop(name)
    results["Scorer + defender + interaction"] = results.pop(best_int[0])
    epochs_used["Scorer + defender + interaction"] = epochs_used.pop(best_int[0])
    args.l2_int = best_int[1]
    print(f"validation picked interaction ridge {best_int[1]}")

    # calibration on the held-out season: pairs grouped by predicted points per 100, predicted vs actual
    with torch.no_grad():
        pred = torch.exp(best_model(*te[:3])[:, 0]).numpy()
    g = test.assign(ep=pred * test["poss"].to_numpy()).groupby(["personIdOff", "personIdDef"]).agg(
        poss=("poss", "sum"), pts=("pts", "sum"), ep=("ep", "sum"))
    edges = [0, 15, 20, 25, 30, 40, 60, 1000]
    band = pd.cut(g["ep"] / g["poss"] * 100, edges)
    cal = g.groupby(band, observed=True).agg(pairs=("poss", "size"), poss=("poss", "sum"), ep=("ep", "sum"), pts=("pts", "sum"))
    calibration = [{"band": f"{int(iv.left)}+" if iv.right >= 1000 else f"{int(iv.left)}-{int(iv.right)}", "pairs": int(r.pairs),
                    "poss": round(float(r.poss)), "pred": round(float(r.ep / r.poss * 100), 1), "actual": round(float(r.pts / r.poss * 100), 1)}
                   for iv, r in cal.iterrows()]
    print("held-out calibration (pts per 100):", [(c["band"], c["pred"], c["actual"]) for c in calibration])
    print(f"\nheld-out {test_season} mean deviance (lower is better):")
    for name, r in results.items():
        print(f"  {name:34s} pts {r['pts']:.4f}  fga {r['fga']:.4f}  tov {r['tov']:.4f}  fg {r['fg']:.4f}")

    # final model on every season, same settings and the epoch count validation picked
    voff_all = vocab(df["personIdOff"], df["poss"], args.min_poss)
    vdef_all = vocab(df["personIdDef"], df["poss"], args.min_poss)
    full = tensors(df, voff_all, vdef_all, vseason)
    final = MatchupNet(len(voff_all) + 1, len(vdef_all) + 1, len(seasons), args.k, league_rates(full[3]))
    fit(final, full, None, epochs=epochs_used["Scorer + defender + interaction"], l2_main=args.l2_main, l2_int=args.l2_int, log=False)
    final.eval()
    torch.save(final.state_dict(), out / "best.pt")

    names = {}
    for side in ("Off", "Def"):
        n = df.drop_duplicates(f"personId{side}")
        names.update({int(p): f"{f} {l}".strip() for p, f, l in zip(n[f"personId{side}"], n[f"firstName{side}"], n[f"familyName{side}"])})
    meta = {"args": vars(args), "targets": TARGETS + ["fg"], "seasons": seasons, "test_season": test_season,
            "vocab_off": {str(p): i for p, i in voff_all.items()}, "vocab_def": {str(p): i for p, i in vdef_all.items()},
            "held_out": results, "epochs": epochs_used, "history": history, "calibration": calibration,
            "rows": {"train": len(train), "val": len(val), "test": len(test), "all": len(df)},
            "games": int(df["gameId"].nunique()), "names": {str(k): v for k, v in names.items()}}
    json.dump(meta, open(out / "meta.json", "w"))
    df.to_parquet(out / "matchups.parquet")
    print(f"saved {out}/best.pt, meta.json and matchups.parquet ({len(voff_all)} scorers, {len(vdef_all)} defenders)")


if __name__ == "__main__":
    main()
