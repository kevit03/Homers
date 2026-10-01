#!/bin/bash
# Publishes the dashboard to Vercel as the "homers" project: build_site.py packs dashboard.html and games/ into site/
# (gzipped, to fit the 100 MB Hobby upload limit), then the Vercel CLI uploads that folder as a static site.
#   ./deploy.sh              deploy the current dashboard.html to production
#   ./deploy.sh --preview    a preview URL instead of production
#   ./deploy.sh --rebuild    re-export dashboard.html (and games/) from the newest model first
# First time only: npx vercel login
set -e
cd "$(dirname "$0")"
PROD=--prod
for a in "$@"; do
  case "$a" in
    --preview) PROD= ;;
    --rebuild)
      if [ -f runs/base/best.pt ] && [ -f data/processed/games.pkl ]; then
        python3 export_dashboard.py --ckpt runs/base/best.pt --data data/processed/games.pkl --all_seasons
      else
        python3 export_dashboard.py --ckpt runs/syn/best.pt --data data/processed/synthetic.pkl
      fi ;;
    *) echo "unknown option: $a (use --preview or --rebuild)"; exit 1 ;;
  esac
done
python3 build_site.py
[ -d site/.vercel ] || npx -y vercel@latest link --yes --project homers --cwd site
npx -y vercel@latest deploy $PROD --yes --cwd site
