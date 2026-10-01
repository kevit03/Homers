"""Basketball-Reference accolades for the dashboard's player profiles (fetch/fetch_accolades.py).

Adds to each profile player (by NBA.com personId):
  acc   career accolades as Basketball-Reference lists them under his name ("4x MVP", "22x All Star")
  aw    {season: award codes} for seasons in the data, e.g. {"2025-26": "MVP-1,CPOY-1,AS,NBA1"}
        (MVP-4 = fourth in MVP voting, AS = All-Star, NBA1 / DEF2 = All-NBA first team / All-Defensive second team)
  bref  his Basketball-Reference id, for a link to his page

Used by export/export_dashboard.py.
"""
import json
from pathlib import Path

import pandas as pd


def add_accolades(profiles, context="data/context"):
    """Mutates profiles["players"]; returns a summary for the page, or None when nothing was fetched."""
    ctx = Path(context)
    ids_path, acc_path = ctx / "bbref_players.parquet", ctx / "bbref_accolades.json"
    if not profiles or not ids_path.exists():
        return None
    ids = pd.read_parquet(ids_path).dropna(subset=["personId"])
    acc = json.loads(acc_path.read_text()) if acc_path.exists() else {}
    players = profiles["players"]
    n_acc = 0
    for bid, g in ids.groupby("bbref_id"):
        p = players.get(str(int(g["personId"].iloc[0])))
        if p is None:
            continue
        p["bref"] = bid
        aw = {s: a for s, a in zip(g["season"], g["awards"]) if isinstance(a, str) and a}
        if aw:
            p["aw"] = aw
        if acc.get(bid, {}).get("acc"):
            p["acc"] = acc[bid]["acc"]
            n_acc += 1
    return {"n_pages": len(acc), "n_players": int(ids["bbref_id"].nunique()), "n_with": n_acc,
            "fetched": max((v.get("fetched", "") for v in acc.values()), default="")}
