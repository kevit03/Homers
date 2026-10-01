"""Photos for the dashboard, inlined as data URIs so dashboard.html stays one self-contained file.

Reads assets/credits.json (file, title, author, license, source for every photo) and the JPEGs in
assets/img. Missing files are skipped; the page falls back to plain gradients.
Used by export/export_dashboard.py.
"""
import base64
import json
from pathlib import Path


def build_assets_payload(root="assets"):
    root = Path(root)
    credits_path = root / "credits.json"
    if not credits_path.exists():
        return None
    credits = [c for c in json.load(open(credits_path)) if (root / "img" / c["file"]).exists()]
    img = {Path(c["file"]).stem: "data:image/jpeg;base64," + base64.b64encode((root / "img" / c["file"]).read_bytes()).decode()
           for c in credits}
    return {"img": img, "credits": credits}


if __name__ == "__main__":
    out = build_assets_payload()
    print(f"{len(out['img'])} images, {sum(len(v) for v in out['img'].values()) / 1e6:.2f} MB as base64")
