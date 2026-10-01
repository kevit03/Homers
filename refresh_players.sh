#!/bin/bash
# Rebuild the player / defender / coach model and the dashboard once downloads are finished.
#   ./refresh_players.sh
set -e
cd "$(dirname "$0")"
python3 -m fetch.fetch_context                      # resumes; picks up any missing matchups
python3 -m fetch.fetch_bbref --refresh || echo "Basketball-Reference unreachable; keeping the rosters already on disk"  # current teams
python3 -m models.build_shots                        # -> data/processed/shots.parquet
python3 -m models.shot_model --batch_size 256        # -> runs/shots/
if [ -f runs/base/best.pt ] && [ -f data/processed/games.pkl ]; then
  python3 -m export.export_dashboard --ckpt runs/base/best.pt --data data/processed/games.pkl
else
  python3 -m export.export_dashboard --ckpt runs/syn/best.pt --data data/processed/synthetic.pkl
fi
echo "Open dashboard.html, or ask Claude to republish it."
