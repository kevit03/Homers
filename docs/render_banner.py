"""Render docs/banner.html into the README banners with headless Chrome.

    python docs/render_banner.py                  # writes docs/images/banner-light.png and banner-dark.png
    CHROME=/path/to/chrome python docs/render_banner.py
"""
import os
import subprocess
from pathlib import Path

DOCS = Path(__file__).resolve().parent
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def render(theme, out):
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--allow-file-access-from-files",
                    "--force-device-scale-factor=1", "--window-size=2560,800", "--virtual-time-budget=10000",
                    f"--screenshot={out}", (DOCS / "banner.html").as_uri() + "#" + theme],
                   check=True, capture_output=True)


if __name__ == "__main__":
    for theme in ("light", "dark"):
        out = DOCS / "images" / f"banner-{theme}.png"
        render(theme, out)
        print("wrote", out)
