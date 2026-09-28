"""Build a self-contained HTML dashboard (dashboard.html) from a trained model.

Runs the model once over the chosen split and bakes every prediction into the page,
so the result is a single file you can open locally or host anywhere static.

    python export_dashboard.py --ckpt runs/base/best.pt --data data/processed/games.pkl
    python export_dashboard.py --ckpt runs/syn/best.pt --data data/processed/synthetic.pkl
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from baseline import baseline_features
from evaluate import brier, calibration
from model import Config, NBAGPT, load_data, make_features

ROOT = Path(__file__).resolve().parent
TOP_K = 5


@torch.no_grad()
def run_game(model, g):
    T = min(len(g["tokens"]), model.config.block_size)
    tokens = torch.from_numpy(g["tokens"][:T].astype(np.int64))[None]
    feats = torch.from_numpy(make_features(g["sec"][:T], g["diff"][:T], g["period"][:T]))[None]
    logits, wp = model(tokens, feats)
    probs = torch.softmax(logits[0], -1)
    top_p, top_i = probs.topk(TOP_K, -1)
    nxt = tokens[0, 1:]
    return (torch.sigmoid(wp[0]).numpy(), top_i.numpy(), top_p.numpy(),
            probs[:-1].gather(-1, nxt[:, None]).squeeze(-1).numpy(), (probs[:-1].argmax(-1) == nxt).numpy())


def r(a, nd=3):
    return [round(float(v), nd) for v in a]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base/best.pt")
    ap.add_argument("--data", default="data/processed/games.pkl")
    ap.add_argument("--baseline", default="runs/baseline.joblib")
    ap.add_argument("--split", default="test")
    ap.add_argument("--template", default=str(ROOT / "dashboard_template.html"))
    ap.add_argument("--out", default="dashboard.html")
    ap.add_argument("--shots", default="data/processed/shots.parquet")
    ap.add_argument("--shot_run", default="runs/shots", help="shot model run dir; skipped if missing")
    ap.add_argument("--context", default="data/context")
    ap.add_argument("--raw", default="data/raw", help="raw play-by-play for player profiles; skipped for synthetic data")
    ap.add_argument("--profile_cache", default="data/processed/profiles_cache.pkl")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    model = NBAGPT(Config(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.eval()
    d = load_data(args.data)
    clf = joblib.load(args.baseline)
    games = [d["games"][i] for i in d["splits"][args.split]]

    out_games, p_m, p_b, y, sec_all, n_correct, nll, n_tok = [], [], [], [], [], 0, 0.0, 0
    for g in games:
        wm, top_i, top_p, p_next, correct = run_game(model, g)
        L = len(wm)
        wb = clf.predict_proba(baseline_features(g["sec"][:L], g["diff"][:L]))[:, 1]
        p_m.append(wm); p_b.append(wb); y.append(np.full(L, g["home_win"]))
        sec_all.append(np.where(g["period"][:L] <= 4, g["sec"][:L], 0))
        n_correct += int(correct.sum()); nll += float(-np.log(p_next).sum()); n_tok += len(p_next)
        out_games.append({
            "id": g["gameId"], "season": g["season"], "home_win": int(g["home_win"]),
            "tok": g["tokens"][:L].tolist(), "sec": r(g["sec"][:L], 1), "diff": g["diff"][:L].astype(int).tolist(),
            "per": g["period"][:L].astype(int).tolist(), "pm": r(wm), "pb": r(wb),
            "top_i": top_i.tolist(), "top_p": [r(row) for row in top_p],
        })
    p_m, p_b, y, sec_all = map(np.concatenate, (p_m, p_b, y, sec_all))
    phase = np.clip(4 - (sec_all // 720).astype(int), 1, 4)

    log_path = Path(args.ckpt).parent / "log.json"
    history = json.load(open(log_path))["history"] if log_path.exists() else []
    payload = {
        "meta": {"data": args.data, "ckpt": args.ckpt, "split": args.split,
                 "synthetic": "synthetic" in Path(args.data).name,
                 "n_games": len(d["games"]), "splits": {k: len(v) for k, v in d["splits"].items()},
                 "config": ck["config"], "n_params": model.n_params()},
        "vocab": d["vocab"],
        "metrics": {
            "brier_model": brier(p_m, y), "brier_baseline": brier(p_b, y),
            "brier_constant": brier(np.full_like(y, y.mean(), dtype=float), y),
            "acc": n_correct / n_tok, "ppl": float(np.exp(nll / n_tok)),
            "by_quarter": [{"q": q, "model": brier(p_m[phase == q], y[phase == q]),
                            "baseline": brier(p_b[phase == q], y[phase == q])} for q in range(1, 5)],
            "cal_model": calibration(p_m, y), "cal_baseline": calibration(p_b, y),
        },
        "history": history,
        # logistic baseline: p = sigmoid(b + w · [diff, diff/sqrt(min+1), min]) for the what-if calculator
        "baseline": {"w": clf.coef_[0].tolist(), "b": float(clf.intercept_[0])},
        "games": out_games,
        "shots": None,
    }
    if (Path(args.shot_run) / "best.pt").exists() and Path(args.shots).exists():
        from export_players import build_players_payload
        payload["shots"] = build_players_payload(args.shots, args.shot_run, args.context)
        print(f"added {len(payload['shots']['players'])} players, {len(payload['shots']['defenders'])} defenders, "
              f"{len(payload['shots']['coaches'])} coaches")
    payload["profiles"] = None
    if not payload["meta"]["synthetic"] and Path(args.raw).exists():
        from export_profiles import build_profiles_payload
        payload["profiles"] = build_profiles_payload(args.raw, out_games, args.context, args.profile_cache)
        if payload["profiles"]:
            print(f"added {len(payload['profiles']['players'])} player profiles, "
                  f"play-by-play actors for {len(payload['profiles']['games'])}/{len(out_games)} games")

    html = Path(args.template).read_text()
    html = html.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")).replace("</", "<\\/"))
    Path(args.out).write_text(html)
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB, {len(out_games)} games)")


if __name__ == "__main__":
    main()
