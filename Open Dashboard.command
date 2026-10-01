#!/bin/bash
# Double-click in Finder: rebuilds dashboard.html from the newest model and opens it in your browser.
cd "$(dirname "$0")"
if [ -f runs/base/best.pt ] && [ -f data/processed/games.pkl ]; then
  python3 export_dashboard.py --ckpt runs/base/best.pt --data data/processed/games.pkl --all_seasons
else
  python3 export_dashboard.py --ckpt runs/syn/best.pt --data data/processed/synthetic.pkl
fi && open dashboard.html
