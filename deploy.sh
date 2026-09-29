#!/bin/bash
# Rebuilds dashboard.html from the newest model and publishes it to Vercel as the "homers" project.
# First time only: npx vercel login
set -e
cd "$(dirname "$0")"
if [ -f runs/base/best.pt ] && [ -f data/processed/games.pkl ]; then
  python3 export_dashboard.py --ckpt runs/base/best.pt --data data/processed/games.pkl
else
  python3 export_dashboard.py --ckpt runs/syn/best.pt --data data/processed/synthetic.pkl
fi
mkdir -p site
cp dashboard.html site/index.html
[ -d site/.vercel ] || npx -y vercel@latest link --yes --project homers --cwd site
npx -y vercel@latest deploy --prod --yes --cwd site
