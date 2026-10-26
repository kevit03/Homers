#!/bin/bash
# Brings the site up to date with the current season: fetch new final games -> process them (tokens, Elo, lineups,
# shots) -> re-export dashboard.html and games/ -> pack site/ -> optionally deploy. Meant to run once a day.
#
#   ./update_season.sh                    fetch, process, export, build site/ (no deploy)
#   ./update_season.sh --deploy preview   ... and upload a preview build
#   ./update_season.sh --deploy prod      ... and publish to production (homers-nu.vercel.app)
#   ./update_season.sh --force            rebuild even if no new games came in
#   ./update_season.sh --dry-run          print the steps without running them
#   ./update_season.sh --season 2026-27   another season than the current one
#   ./update_season.sh --only fetch,process   just these steps (fetch, process, export, build, deploy)
#
# Each run logs to logs/update_season/<date>.log. A step that fails stops the run, so neither site/ nor production
# changes: site/ is built into site.next/ and swapped in only when the build succeeds; the build before it stays in
# site.prev/. Export, build and deploy are skipped when there are no new final games since the last good build.
#
# Daily at 9:00 (late games end ~1 AM ET; NBA.com's stats settle overnight). A Mac that is asleep at 9:00 runs it
# on wake. The scheduled run doesn't deploy: add <string>--deploy</string><string>prod</string> to the plist's
# ProgramArguments once production deploys are approved.
#   install:   cp scripts/com.homers.update-season.plist ~/Library/LaunchAgents/ &&
#              launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.homers.update-season.plist
#   run now:   launchctl kickstart gui/$(id -u)/com.homers.update-season
#   uninstall: launchctl bootout gui/$(id -u)/com.homers.update-season &&
#              rm ~/Library/LaunchAgents/com.homers.update-season.plist
# The repo lives in ~/Downloads, which macOS guards: give /bin/bash Full Disk Access (System Settings > Privacy &
# Security) or the scheduled run fails with "Operation not permitted".
#
# Environment overrides: PY (Python with nba_api and torch; default /usr/bin/python3), CKPT (Tempo checkpoint for the
# export; default runs/base/best.pt), DATA (default data/processed/games_30.pkl).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT" || exit 1
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin"
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore

PY="${PY:-/usr/bin/python3}"  # the one Python with both nba_api and torch (Anaconda's 3.14 has no nba_api)
CKPT="${CKPT:-runs/base/best.pt}"
DATA="${DATA:-data/processed/games_30.pkl}"
VERCEL_SCOPE="kevin-tangs-projects-7b84bab0"

y=$(date +%Y); m=$((10#$(date +%m)))
[ "$m" -ge 8 ] || y=$((y - 1))  # a season is named for the fall it starts in (2026-27 from August 2026)
SEASON="$y-$(printf %02d $(((y + 1) % 100)))"
FORCE=0 DRY=0 DEPLOY="" ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --season) SEASON="$2"; shift ;;
    --force) FORCE=1 ;;
    --dry-run) DRY=1 ;;
    --deploy) DEPLOY="$2"; shift
      [ "$DEPLOY" = prod ] || [ "$DEPLOY" = preview ] || { echo "--deploy takes prod or preview"; exit 2; } ;;
    --only) ONLY=",$2," ; shift ;;
    -h|--help) sed -n '2,/^set -uo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)"; exit 2 ;;
  esac
  shift
done

LOGDIR="$ROOT/logs/update_season"
mkdir -p "$LOGDIR"
STATE="$LOGDIR/last_built_$SEASON"  # raw game count at the last good build

# keep the Mac awake for the whole run (re-run this script under caffeinate once)
if [ "$DRY" = 0 ] && [ -z "${UPDATE_SEASON_AWAKE:-}" ]; then
  UPDATE_SEASON_AWAKE=1 exec /usr/bin/caffeinate -i "$0" --season "$SEASON" \
    $([ "$FORCE" = 1 ] && echo --force) ${DEPLOY:+--deploy "$DEPLOY"} ${ONLY:+--only "${ONLY:1:${#ONLY}-2}"}
fi

LOG="$LOGDIR/$(date +%F).log"
[ "$DRY" = 1 ] || exec > >(tee -a "$LOG") 2>&1
find "$LOGDIR" -name '20*.log' -mtime +30 -delete 2>/dev/null

notify() {  # a macOS notification; quietly nothing if there is no GUI session
  /usr/bin/osascript -e "display notification \"$2\" with title \"$1\"" >/dev/null 2>&1 || true
}

# one run at a time: mkdir is atomic; a lock left by a run that died is taken over
LOCK="$LOGDIR/.lock"
if [ "$DRY" = 0 ]; then
  if ! mkdir "$LOCK" 2>/dev/null; then
    pid=$(cat "$LOCK/pid" 2>/dev/null)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "$(date '+%F %T') another update is running (pid $pid); exiting"; exit 0
    fi
    echo "taking over a stale lock (pid ${pid:-?})"; rm -rf "$LOCK"; mkdir "$LOCK" || exit 1
  fi
  echo $$ > "$LOCK/pid"
  trap 'rm -rf "$LOCK"' EXIT
fi

want() { [ -z "$ONLY" ] || [[ "$ONLY" == *",$1,"* ]]; }
T0=$(date +%s)
step() {  # step <name> <command...>: run, time and log it; stop the whole run if it fails
  local name="$1"; shift
  if ! want "$name"; then return 0; fi
  echo; echo "=== $(date '+%F %T') $name: $*"
  [ "$DRY" = 1 ] && return 0
  local s=$(date +%s)
  "$@"
  local rc=$?
  echo "--- $name: exit $rc in $(( $(date +%s) - s ))s"
  if [ "$rc" != 0 ]; then
    echo "!!! $name failed; stopping before anything is published (site/ unchanged). Log: $LOG"
    notify "Homers update failed" "$SEASON: $name step failed (exit $rc). See logs/update_season."
    exit "$rc"
  fi
}
raw_games() { ls "data/raw/$SEASON" 2>/dev/null | grep -c '\.parquet$'; }

echo "=== $(date '+%F %T') update $SEASON${DEPLOY:+, deploy $DEPLOY}$([ "$FORCE" = 1 ] && echo ', forced')$([ "$DRY" = 1 ] && echo ' (dry run)')"
echo "    python $PY | checkpoint $CKPT | data $DATA"

step fetch "$PY" -m fetch.update_season --season "$SEASON"
NOW=$(raw_games); BUILT=$(cat "$STATE" 2>/dev/null || echo -1)
echo "    $SEASON final games on disk: $NOW (last good build: $BUILT)"
if [ "$DRY" = 0 ] && [ "$FORCE" = 0 ] && [ "$NOW" = "$BUILT" ]; then
  echo "=== no new games since the last build; nothing to publish ($(( $(date +%s) - T0 ))s)"; exit 0
fi

step process "$PY" -m models.update_season --season "$SEASON"
step export "$PY" -m export.export_dashboard --ckpt "$CKPT" --data "$DATA" --all_seasons

build_site() {  # into site.next/ (with the Vercel project link), then swap it in for site/
  rm -rf site.next && mkdir site.next
  [ -d site/.vercel ] && cp -R site/.vercel site.next/
  "$PY" -m export.build_site --out site.next || return 1
  rm -rf site.prev
  if [ -d site ]; then mv site site.prev || return 1; fi
  mv site.next site
}
step build build_site
want build && [ "$DRY" = 0 ] && echo "$NOW" > "$STATE"

deploy() {  # one retry: Vercel has failed once with a bare "error" / "Not authorized" that a retry fixed
  local args=(deploy --yes --cwd site --scope "$VERCEL_SCOPE")
  [ "$DEPLOY" = prod ] && args+=(--prod)
  npx -y vercel@latest "${args[@]}" || { sleep 20; npx -y vercel@latest "${args[@]}"; }
}
if [ -n "$DEPLOY" ]; then
  step deploy deploy
else
  echo; echo "=== not deploying (pass --deploy prod or --deploy preview)"
fi
echo; echo "=== $(date '+%F %T') done in $(( $(date +%s) - T0 ))s"
