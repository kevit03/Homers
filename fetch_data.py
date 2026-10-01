# Temporary: the 1996-2016 retry pass (retry_pass.sh) still runs `python3 fetch_data.py` from the repo root once
# the first pass (PID 29388) ends. Delete this file after it finishes; the script now lives in fetch/fetch_data.py.
import runpy

runpy.run_module("fetch.fetch_data", run_name="__main__", alter_sys=True)
