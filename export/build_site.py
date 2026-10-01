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
  :root { color-scheme: dark; }
  body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: #0f1216; color: #f1f3f6;
         font: 500 15px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif; }
  main { display: grid; gap: 14px; width: min(320px, calc(100vw - 32px)); }
  h1 { margin: 0; font: italic 800 52px/0.9 "Arial Narrow", "Helvetica Neue", sans-serif; letter-spacing: .01em; }
  h1 span { color: #eb6834; }
  .bar { height: 4px; border-radius: 99px; background: #262c34; overflow: hidden; }
  .bar i { display: block; height: 100%; width: 0; background: #eb6834; transition: width .2s; }
  p { margin: 0; color: #8b94a1; font-size: 13.5px; }
</style>
</head>
<body>
<main>
  <h1>HOMER<span>s</span></h1>
  <div class="bar"><i id="bar"></i></div>
  <p id="msg">Loading __MB__ MB of play-by-play…</p>
</main>
<script>
(async () => {
  const msg = document.getElementById("msg"), bar = document.getElementById("bar"), size = __BYTES__;
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
      bar.style.width = Math.min(100, got / size * 100).toFixed(1) + "%";
    }
    msg.textContent = "Unpacking…";
    const blob = new Blob(parts), head = new Uint8Array(await blob.slice(0, 2).arrayBuffer());
    // gzip bytes, unless the server already undid the gzip in transit
    const html = head[0] === 0x1f && head[1] === 0x8b ? await new Response(blob.stream().pipeThrough(new DecompressionStream("gzip"))).text() : await blob.text();
    document.open(); document.write(html); document.close();
  } catch (e) {
    msg.textContent = e.message || String(e);
    bar.style.background = "#d1432f";
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
    ap.add_argument("--max_mb", type=float, default=100, help="fail above this many MB (Vercel Hobby: 100, Pro: 1000)")
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
    (out / "index.html").write_text(LOADER.replace("__FILE__", name).replace("__BYTES__", str(size))
                                    .replace("__MB__", f"{size / 1e6:.0f}"))
    total = size
    seasons = sorted(games.glob("*.js")) if games.is_dir() else []
    if seasons:
        (out / "games").mkdir()
        for f in seasons:
            total += gz(f, out / "games" / (f.name + ".gz"))
    (out / "vercel.json").write_text(json.dumps({"$schema": "https://openapi.vercel.sh/vercel.json", "framework": None}, indent=2) + "\n")

    mb = total / 1e6
    print(f"{out}: page {size / 1e6:.1f} MB + {len(seasons)} seasons of games = {mb:.1f} MB gzipped "
          f"(from {(page.stat().st_size + sum(f.stat().st_size for f in seasons)) / 1e6:.0f} MB)")
    if mb > args.max_mb:
        raise SystemExit(f"{mb:.0f} MB is over the {args.max_mb:.0f} MB upload limit. Drop some seasons from games/, "
                         "or pass --max_mb 1000 on a Vercel Pro account.")


if __name__ == "__main__":
    main()
