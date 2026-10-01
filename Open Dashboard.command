#!/bin/bash
# Double-click in Finder: rebuilds dashboard.html from the newest model and opens it in your browser.
# run.sh picks a Python that has torch (Anaconda's python3 may not), so this works whichever python3 is first on PATH.
cd "$(dirname "$0")"
exec ./run.sh dashboard
