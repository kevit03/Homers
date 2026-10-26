"""Write Tempo's weights for the Live tab, which runs the transformer in JavaScript (live/tempo.mjs) on games in progress.

    /usr/bin/python3 -m export.export_live                     # runs/base/best.pt -> live/tempo.json + live/tempo.bin
    /usr/bin/python3 -m export.export_live --check 20          # and compare the JS win chance with PyTorch on 20 test games

tempo.bin is every tensor the win-probability path needs, float16, back to back in the order tempo.json lists them;
the next-play head (tied to tok_emb) isn't needed live, so nothing extra ships for it. --check needs node.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def export(ckpt, out):
    ck = torch.load(ckpt, map_location="cpu")
    cfg = ck["config"]
    if cfg.get("n_feat", 4) != 4 or cfg.get("wp_residual"):
        # the live tokenizer has no pre-game Elo or starting fives for games in progress
        raise SystemExit(f"{ckpt}: the Live tab only runs score-and-clock checkpoints (n_feat 4, no residual), got {cfg}")
    sd = {k: v.float().numpy() for k, v in ck["model"].items() if k != "lm_head.weight"}
    out.mkdir(parents=True, exist_ok=True)
    tensors, blobs, off = [], [], 0
    for name, a in sd.items():
        h = a.astype(np.float16)
        tensors.append({"name": name, "shape": list(a.shape), "offset": off})
        blobs.append(h.tobytes())
        off += h.size
    (out / "tempo.bin").write_bytes(b"".join(blobs))
    meta = {"config": {k: cfg[k] for k in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd")},
            "dtype": "float16", "tensors": tensors, "ckpt": str(ckpt)}
    from models.tokenize_pbp import VOCAB
    meta["vocab"] = VOCAB
    (out / "tempo.json").write_text(json.dumps(meta, separators=(",", ":")) + "\n")
    print(f"wrote {out}/tempo.bin ({(out / 'tempo.bin').stat().st_size / 1e6:.2f} MB, {off:,} weights) and tempo.json")


def check(ckpt, data, n, out):
    """PyTorch (float32 weights) vs live/tempo.mjs (float16 weights) on n test games, every position."""
    from models.model import Config, Tempo, load_data, make_features
    ck = torch.load(ckpt, map_location="cpu")
    model = Tempo(Config(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.eval()
    d = load_data(data)
    games = [d["games"][i] for i in d["splits"]["test"][:: max(1, len(d["splits"]["test"]) // n)][:n]]
    cases = []
    for g in games:
        L = min(len(g["tokens"]), model.config.block_size)
        with torch.no_grad():  # what export_dashboard.run_game does, without its imports
            feats = torch.from_numpy(make_features(g["sec"][:L], g["diff"][:L], g["period"][:L]))[None]
            wp = torch.sigmoid(model(torch.from_numpy(g["tokens"][:L].astype(np.int64))[None], feats)[1][0]).numpy()
        cases.append({"tok": g["tokens"][:L].tolist(), "sec": g["sec"][:L].tolist(), "diff": g["diff"][:L].tolist(),
                      "per": g["period"][:L].tolist(), "wp": wp.tolist()})
    f = out / ".check.json"
    f.write_text(json.dumps(cases))
    script = f"""
import {{ readFileSync }} from "node:fs";
import {{ loadTempo }} from "{(out / 'tempo.mjs').as_posix()}";
const T = loadTempo(JSON.parse(readFileSync("{(out / 'tempo.json').as_posix()}")), readFileSync("{(out / 'tempo.bin').as_posix()}"));
let worst = 0, n = 0, t0 = Date.now();
for (const c of JSON.parse(readFileSync("{f.as_posix()}"))) {{
  const s = T.session();
  // feed in uneven chunks, as polling would, to exercise the cache
  for (let i = 0; i < c.tok.length; i += 1 + (i % 37)) s.push(c.tok.slice(i, i + 1 + (i % 37)).map((t, j) => ({{ tok: t, sec: c.sec[i + j], diff: c.diff[i + j], per: c.per[i + j] }})));
  s.wp.forEach((p, i) => {{ worst = Math.max(worst, Math.abs(p - c.wp[i])); n++; }});
}}
console.log(JSON.stringify({{ worst, n, ms: Date.now() - t0 }}));
"""
    res = json.loads(subprocess.run(["node", "--input-type=module", "-e", script], check=True, capture_output=True, text=True).stdout)
    f.unlink()
    print(f"JS vs PyTorch over {res['n']:,} positions in {n} games: max |Δ win chance| = {res['worst']:.5f} ({res['ms']} ms in node)")
    if res["worst"] > 0.01:
        raise SystemExit("JS forward pass disagrees with PyTorch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base/best.pt")
    ap.add_argument("--data", default="data/processed/games.pkl")
    ap.add_argument("--out", default=str(ROOT / "live"))
    ap.add_argument("--check", type=int, default=0, help="also compare with PyTorch on this many test games")
    args = ap.parse_args()
    export(args.ckpt, Path(args.out))
    if args.check:
        check(args.ckpt, args.data, args.check, Path(args.out))


if __name__ == "__main__":
    main()
