#!/bin/bash
# Rebuild the player / defender / coach model and the dashboard once downloads are finished.
#   ./refresh_players.sh
set -e
cd "$(dirname "$0")"
python3 fetch_context.py                      # resumes; picks up any missing matchups
python3 build_shots.py                        # -> data/processed/shots.parquet
python3 shot_model.py --batch_size 256        # -> runs/shots/
if [ -f runs/base/best.pt ] && [ -f data/processed/games.pkl ]; then
  python3 export_dashboard.py --ckpt runs/base/best.pt --data data/processed/games.pkl
else
  python3 export_dashboard.py --ckpt runs/syn/best.pt --data data/processed/synthetic.pkl
fi
echo "Open dashboard.html, or ask Claude to republish it."
