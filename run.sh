#!/bin/bash
# One entry point for the whole project. Picks a Python that has torch, so it works
# no matter which python/python3 your terminal finds first.
#
#   ./run.sh              show commands
#   ./run.sh demo         fake data -> model -> dashboard, about a minute, no downloads
#   ./run.sh status       what's downloaded, trained, and running
#   ./run.sh all          the full real-data pipeline, start to finish
#
# Extra arguments are passed through, e.g.  ./run.sh train --epochs 5 --out runs/test
set -e
cd "$(dirname "$0")"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# --- find a Python with the project's packages ----------------------------------------
has_torch() { "$1" -c "import torch, pandas, sklearn" >/dev/null 2>&1; }
find_python() {
  if [ -n "$PYTHON" ]; then echo "$PYTHON"; return; fi
  for p in python3 /usr/bin/python3 python; do
    if command -v "$p" >/dev/null 2>&1 && has_torch "$p"; then echo "$p"; return; fi
  done
}
PY="$(find_python)"
need_python() {
  [ -n "$PY" ] && return
  die "no Python with torch found. Run:  ./run.sh setup"
}
py() { need_python; PYTHONWARNINGS=ignore "$PY" -u "$@"; }

# Python processes running a project script, printed as "script.py args"
jobs_matching() { ps -Ao command | grep -E "^[^ ]*[Pp]ython[0-9.]* (-u )?[^ ]*$1" | sed -E 's/^.*[Pp]ython[0-9.]* (-u )?//'; }
running() { [ -n "$(jobs_matching "$1")" ]; }
guard_training() {
  if running "train\\.py.*runs/base"; then
    die "a training run is already writing to runs/base. Wait for it (./run.sh status) or pass --out runs/other."
  fi
}

dashboard_inputs() {
  if [ -f runs/base/best.pt ] && [ -f data/processed/games.pkl ]; then
    echo "--ckpt runs/base/best.pt --data data/processed/games.pkl"
  else
    echo "--ckpt runs/syn/best.pt --data data/processed/synthetic.pkl"
  fi
}

# --- commands --------------------------------------------------------------------------
cmd_setup() {
  local p="${PYTHON:-python3}"
  step "Installing requirements into $(command -v "$p")"
  "$p" -m pip install -r requirements.txt
  has_torch "$p" && bold "Ready. Try: ./run.sh demo"
}

cmd_demo() {
  step "1/5 Generating 400 synthetic games";    py synthetic.py --games 400
  step "2/5 Tokenizing";                        py tokenize_pbp.py --raw data/raw_synthetic --out data/processed/synthetic.pkl
  step "3/5 Fitting the baseline";              py baseline.py --data data/processed/synthetic.pkl --out runs/baseline_syn.joblib
  step "4/5 Training a small model";            py train.py --data data/processed/synthetic.pkl --out runs/syn \
                                                  --n_layer 2 --n_embd 64 --n_head 2 --lr 1e-3 "$@"
  step "5/5 Building the dashboard";            py export_dashboard.py --ckpt runs/syn/best.pt \
                                                  --data data/processed/synthetic.pkl --baseline runs/baseline_syn.joblib
  open dashboard.html 2>/dev/null || bold "Open dashboard.html in your browser."
}

cmd_fetch() {
  step "Play-by-play (resumable)";              py fetch_data.py "$@"
  step "Coaches, play types, matchups (resumable)"; py fetch_context.py
  step "Rosters";                               py fetch_rosters.py
  step "Current rosters (Basketball-Reference)"; py fetch_bbref.py --refresh
}

cmd_tokenize() { step "Tokenizing play-by-play"; py tokenize_pbp.py "$@"; }

cmd_train() {
  case " $* " in *" --out "*) ;; *) guard_training ;; esac
  step "Fitting the baseline";                  py baseline.py
  step "Training the transformer";              py train.py --out runs/base "$@"
}

cmd_evaluate() {
  [ -f runs/base/best.pt ] || die "no trained model yet. Run: ./run.sh train"
  step "Evaluating on the test season";         py evaluate.py --ckpt runs/base/best.pt "$@"
  bold "Metrics and plots are in results/"
}

cmd_shots() {
  step "Building the shot table";               py build_shots.py
  step "Training the shot model";               py shot_model.py --batch_size 256 "$@"
}

cmd_dashboard() {
  step "Building the dashboard"
  # shellcheck disable=SC2046
  py export_dashboard.py $(dashboard_inputs) "$@"
  open dashboard.html 2>/dev/null || bold "Open dashboard.html in your browser."
}

cmd_all() {
  cmd_fetch; cmd_tokenize; cmd_train; cmd_evaluate; cmd_shots; cmd_dashboard
}

cmd_status() {
  bold "Python"
  if [ -n "$PY" ]; then echo "  $(command -v "$PY") (torch ok)"; else echo "  no Python with torch; run ./run.sh setup"; fi

  bold "Downloaded play-by-play"
  if [ -d data/raw ]; then
    for s in data/raw/*/; do
      [ -d "$s" ] && printf '  %s  %5d games (%d playoffs)\n' "$(basename "$s")" "$(ls "$s" | grep -c parquet)" "$(ls "$s" | grep -c '^004')"
    done
  else
    echo "  none yet (./run.sh fetch)"
  fi

  bold "Processed data"
  for f in data/processed/games.pkl data/processed/synthetic.pkl data/processed/shots.parquet; do
    [ -f "$f" ] && echo "  $f  ($(date -r "$f" '+%b %d %H:%M'))"
  done

  bold "Models"
  for f in runs/base/best.pt runs/syn/best.pt runs/shots; do
    [ -e "$f" ] && echo "  $f  ($(date -r "$f" '+%b %d %H:%M'))"
  done
  if [ -f runs/base/log.json ]; then
    "${PY:-python3}" - <<'EOF' 2>/dev/null || true
import json, math
log = json.load(open("runs/base/log.json"))
h, b = log["history"], log.get("best", {})
print(f"  runs/base: {len(h)} epochs, best val perplexity {math.exp(b['val_ce']):.2f}, win-prob BCE {b['val_wp_bce']:.4f}")
EOF
  fi

  bold "Running now"
  local any=0
  for j in fetch_data fetch_context fetch_rosters tokenize_pbp build_shots "train\\.py" evaluate shot_model export_dashboard; do
    if running "$j"; then jobs_matching "$j" | sed 's/^/  /'; any=1; fi
  done
  [ $any = 1 ] || echo "  nothing"
}

usage() {
  cat <<EOF
$(bold "HOMERs: ./run.sh <command> [extra args]")

  setup       install Python packages (only needed once)
  demo        try everything on fake games, about a minute, no downloads
  status      show downloads, models, and anything running

  fetch       download NBA play-by-play + context (hours, resumable)
  tokenize    turn play-by-play into model input
  train       fit the baseline and train the transformer into runs/base
  evaluate    score the model vs. the baseline -> results/
  shots       build and train the shot-selection model
  dashboard   build dashboard.html from the newest model and open it

  all         fetch -> tokenize -> train -> evaluate -> shots -> dashboard

Examples:
  ./run.sh demo
  ./run.sh train --epochs 5 --out runs/test
  ./run.sh fetch --seasons 2024-25
EOF
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  setup|demo|fetch|tokenize|train|evaluate|shots|dashboard|all|status) "cmd_$cmd" "$@" ;;
  help|-h|--help) usage ;;
  *) usage; echo; die "unknown command: $cmd" ;;
esac
