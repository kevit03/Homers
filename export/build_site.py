"""Pack dashboard.html and games/ into site/, a static folder Vercel can host.

    python -m export.build_site            # then: npx vercel deploy --prod --cwd site   (or ./deploy.sh)

The page is one ~50 MB HTML file and every other season's replay games add ~15 MB each, more than the 100 MB a Hobby
account may upload in one deployment. So everything ships gzipped: site/index.html is a small loader that fetches
homers.html.gz, unzips it in the browser (DecompressionStream) and writes it in as the page; Game replay fetches
games/<season>.js.gz the same way (gpGz in dashboard_template.html). Opening dashboard.html from disk is unchanged.
"""
import argparse
import gzip
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LOADER = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HOMERs</title>
<meta name="description" content="NBA play-by-play, read one play at a time: win probability, shot charts, matchups and the league's best defenders.">
<style>
  :root { color-scheme: dark; --o: #eb6834; }
  body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: radial-gradient(120% 80% at 50% 100%, #1a1f26, #0f1216 60%);
         color: #f1f3f6; font: 500 15px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif; overflow: hidden; }
  main { display: grid; justify-items: center; gap: 10px; width: min(340px, calc(100vw - 32px)); text-align: center; }
  h1 { margin: 0; font: italic 800 52px/0.9 "Arial Narrow", "Helvetica Neue", sans-serif; letter-spacing: .01em; }
  h1 span { color: var(--o); }
  /* Homer himself (the poet the site is named after), dribbling while the page loads */
  svg { width: 190px; height: auto; overflow: visible; }
  .arm { transform-origin: 104px 78px; animation: arm .62s ease-in-out infinite; }
  .ball { transform-box: fill-box; transform-origin: 50% 100%; animation: ball .62s infinite; }
  .shadow { transform-box: fill-box; transform-origin: 50% 50%; animation: shadow .62s infinite; }
  .body { animation: bob .62s ease-in-out infinite; }
  @keyframes ball {
    0%   { transform: translateY(0); animation-timing-function: cubic-bezier(.45, 0, .9, .6); }
    46%  { transform: translateY(59px) scale(1, 1); }
    52%  { transform: translateY(59px) scale(1.18, .8); animation-timing-function: cubic-bezier(.1, .4, .55, 1); }
    100% { transform: translateY(0); }
  }
  @keyframes shadow { 0%, 100% { transform: scale(.45); opacity: .25; } 50% { transform: scale(1); opacity: .6; } }
  @keyframes arm { 0%, 100% { transform: rotate(0); } 22% { transform: rotate(7deg); } 62% { transform: rotate(-7deg); } }
  @keyframes bob { 0%, 100% { transform: translateY(0); } 45% { transform: translateY(2px); } }
  .meter { display: grid; grid-template-columns: auto 1fr; align-items: center; gap: 12px; width: 100%; }
  .meter b { font: italic 800 30px/1 "Arial Narrow", "Helvetica Neue", sans-serif; font-variant-numeric: tabular-nums; min-width: 2.6em; text-align: left; }
  .bar { height: 6px; border-radius: 99px; background: #262c34; overflow: hidden; }
  .bar i { display: block; height: 100%; width: 0; border-radius: 99px; transition: width .25s;
           background: repeating-linear-gradient(-45deg, var(--o) 0 8px, #f08a5d 8px 16px); background-size: 22px 22px; animation: stripes .6s linear infinite; }
  @keyframes stripes { to { background-position: 22px 0; } }
  p { margin: 0; color: #8b94a1; font-size: 14px; min-height: 1.5em; transition: opacity .25s; }
  p.err { color: #f08a7a; }
  @media (prefers-reduced-motion: reduce) { .arm, .ball, .shadow, .body, .bar i { animation: none; } }
</style>
</head>
<body>
<main>
  <h1>HOMER<span>s</span></h1>
  <svg viewBox="20 20 170 190" role="img" aria-label="Homer, the Greek poet, dribbling a basketball">
    <line x1="20" y1="202" x2="190" y2="202" stroke="#37404b" stroke-width="2" stroke-linecap="round"/>
    <ellipse class="shadow" cx="139" cy="202" rx="16" ry="3.5" fill="#000"/>
    <ellipse cx="88" cy="202" rx="34" ry="4" fill="#000" opacity=".35"/>
    <g class="body" stroke-linecap="round" stroke-linejoin="round" fill="none">
      <polyline points="80,150 72,174 76,199" stroke="#c9a27e" stroke-width="10"/>
      <polyline points="98,150 108,173 105,199" stroke="#c9a27e" stroke-width="10"/>
      <path d="M68 199h15M100 199h15M72 192l8-6M103 192l8-6" stroke="#8a5a32" stroke-width="4"/>
      <polyline points="72,80 58,108 64,128" stroke="#c9a27e" stroke-width="9"/>
      <path d="M66 74q22-12 44 0l6 86q-28 8-56 0z" fill="#ece5d8" stroke="none"/>
      <path d="M66 76l46 50M64 158q28 8 56 0" stroke="#eb6834" stroke-width="4"/>
      <path d="M70 92q14 14 10 50M98 96q6 22 2 54" stroke="#cfc6b6" stroke-width="2"/>
      <circle cx="88" cy="48" r="14" fill="#c9a27e" stroke="none"/>
      <path d="M74 50q2 22 14 24q12-2 14-24q-6 6-14 6t-14-6z" fill="#b8bec7" stroke="none"/>
      <path d="M74 44q2-14 14-14t14 14q-6-6-14-6t-14 6z" fill="#b8bec7" stroke="none"/>
      <path d="M73 42q15-12 30 0" stroke="#5fae5a" stroke-width="3" stroke-dasharray="3 2"/>
      <circle cx="83" cy="47" r="1.6" fill="#1b1f25" stroke="none"/><circle cx="93" cy="47" r="1.6" fill="#1b1f25" stroke="none"/>
      <path d="M80 43.5h5M91 43.5h5" stroke="#8e959e" stroke-width="1.6"/>
    </g>
    <g class="arm" stroke-linecap="round" stroke-linejoin="round" fill="none">
      <polyline points="104,78 124,100 136,118" stroke="#c9a27e" stroke-width="9"/>
    </g>
    <g class="ball">
      <circle cx="139" cy="131" r="12" fill="#eb6834"/>
      <path d="M127 131h24M139 119v24M130.5 122.5q6 8.5 0 17M147.5 122.5q-6 8.5 0 17" stroke="#7a2c0d" stroke-width="1.6" fill="none"/>
    </g>
  </svg>
  <div class="meter"><b id="pct">0%</b><div class="bar"><i id="bar"></i></div></div>
  <p id="msg" role="status">Lacing up the sandals…</p>
</main>
<script>
(async () => {
  const msg = document.getElementById("msg"), bar = document.getElementById("bar"), pct = document.getElementById("pct"), size = __BYTES__;
  const lines = ["Lacing up the sandals…", "Sing, O Muse, of the fourth quarter…", "Rewinding thirty years of tape…", "Charting every jump shot…",
                 "Drawing up the after-timeout play…", "Arguing with the refs…", "Checking the replay monitor…", "Tipping off any second…"];
  let k = 0;
  const say = t => { msg.style.opacity = 0; setTimeout(() => { msg.textContent = t; msg.style.opacity = 1; }, 220); };
  const spin = setInterval(() => say(lines[++k % lines.length]), 1900);
  try {
    if (typeof DecompressionStream === "undefined") throw new Error("This browser is too old to unpack the page. Try a current Chrome, Safari, Edge or Firefox.");
    const res = await fetch("__FILE__");
    if (!res.ok) throw new Error(`Couldn't load the page (${res.status}).`);
    const reader = res.body.getReader(), parts = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value); got += value.length;
      const f = Math.min(1, got / size);
      bar.style.width = (f * 100).toFixed(1) + "%"; pct.textContent = Math.floor(f * 100) + "%";
    }
    clearInterval(spin); pct.textContent = "100%"; say("Tip-off!");
    const blob = new Blob(parts), head = new Uint8Array(await blob.slice(0, 2).arrayBuffer());
    // gzip bytes, unless the server already undid the gzip in transit
    const html = head[0] === 0x1f && head[1] === 0x8b ? await new Response(blob.stream().pipeThrough(new DecompressionStream("gzip"))).text() : await blob.text();
    document.open(); document.write(html); document.close();
  } catch (e) {
    clearInterval(spin);
    msg.textContent = e.message || String(e); msg.className = "err"; msg.style.opacity = 1;
    bar.style.background = "#d1432f"; bar.style.animation = "none";
  }
})();
</script>
</body>
</html>
"""


def gz(src: Path, dst: Path) -> int:
    with open(src, "rb") as f, open(dst, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as g:
        shutil.copyfileobj(f, g)
    return dst.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default=str(ROOT / "dashboard.html"))
    ap.add_argument("--games", default=str(ROOT / "games"))
    ap.add_argument("--out", default=str(ROOT / "site"))
    ap.add_argument("--max_mb", type=float, default=100,
                    help="upload limit (Vercel Hobby: 100, Pro: 1000); the oldest replay seasons that don't fit are left out")
    ap.add_argument("--no_repack", action="store_true", help="gzip games/*.js as they are instead of repacking each season as v3")
    ap.add_argument("--jobs", type=int, default=0, help="seasons repacked in parallel (default: one per CPU)")
    args = ap.parse_args()

    page, games, out = Path(args.page), Path(args.games), Path(args.out)
    if not page.exists():
        raise SystemExit(f"{page} not found: build it first (python -m export.export_dashboard, or ./run.sh dashboard).")
    out.mkdir(exist_ok=True)
    # clear what an earlier run wrote, but keep .vercel (the project link)
    for p in out.iterdir():
        if p.name != ".vercel":
            shutil.rmtree(p) if p.is_dir() else p.unlink()

    name = "homers.html.gz"
    size = gz(page, out / name)
    (out / "index.html").write_text(LOADER.replace("__FILE__", name).replace("__BYTES__", str(size)))
    (out / "vercel.json").write_text(json.dumps({"$schema": "https://openapi.vercel.sh/vercel.json", "framework": None}, indent=2) + "\n")
    total = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    if total / 1e6 > args.max_mb:
        raise SystemExit(f"The page alone is {total / 1e6:.0f} MB gzipped, over the {args.max_mb:.0f} MB upload limit.")

    # Each season's replay games, repacked as one v3 object (export/replay_pack.py: about a third of the gzipped size
    # of the per-game files export_dashboard writes), newest first. The oldest seasons that don't fit are left out;
    # Game replay says so when one of them is picked (gpLoad in dashboard_template.html, on the 404).
    files = sorted(games.glob("*.js"), reverse=True) if games.is_dir() else []
    packed = []
    if files and args.no_repack:
        packed = [(f.stem, gzip.compress(f.read_bytes(), 9, mtime=0)) for f in files]
    elif files:
        import os
        from concurrent.futures import ProcessPoolExecutor
        from export.replay_pack import season_file_v3
        print(f"repacking {len(files)} seasons of replay games")
        with ProcessPoolExecutor(args.jobs or os.cpu_count()) as pool:
            packed = list(pool.map(season_file_v3, files))
    shipped, dropped, games_mb = [], [], 0.0
    for (season, data), f in zip(packed, files):
        if dropped or (total + len(data) + 4096) / 1e6 > args.max_mb:  # 4 KB for games/index.json
            dropped.append(season)
            continue
        (out / "games").mkdir(exist_ok=True)
        (out / "games" / (f.name + ".gz")).write_bytes(data)
        total += len(data)
        games_mb += len(data) / 1e6
        shipped.append(season)
        print(f"  {season}: {len(data) / 1e6:.2f} MB")
    if files:
        (out / "games").mkdir(exist_ok=True)
        index = out / "games" / "index.json"
        index.write_text(json.dumps({"shipped": sorted(shipped), "dropped": sorted(dropped)}) + "\n")
        total += index.stat().st_size

    print(f"{out}: page {size / 1e6:.1f} MB + {len(shipped)} seasons of games {games_mb:.1f} MB = {total / 1e6:.1f} MB gzipped "
          f"(from {(page.stat().st_size + sum(f.stat().st_size for f in files)) / 1e6:.0f} MB), limit {args.max_mb:.0f} MB")
    if dropped:
        print(f"left out {len(dropped)} season(s) to stay under {args.max_mb:.0f} MB: {', '.join(sorted(dropped))}. "
              "Game replay tells visitors they aren't on the hosted site yet; --max_mb 1000 fits them on Vercel Pro.")


if __name__ == "__main__":
    main()
