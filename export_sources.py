"""Data sources and methods for the dashboard's Sources tab.

Everything here is read from what is actually on disk (seasons, game counts, fetch dates, training
settings from the run logs), so the tab stays accurate after every re-fetch or re-train.
Used by export_dashboard.py.
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def _mtime(path):
    p = Path(path)
    if not p.exists():
        return None
    files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
    return datetime.fromtimestamp(max(f.stat().st_mtime for f in files)).strftime("%Y-%m-%d") if files else None


def _season_counts(root):
    root = Path(root)
    if not root.exists():
        return {}
    return {d.name: sum(1 for _ in d.glob("*.parquet")) for d in sorted(root.iterdir()) if d.is_dir()}


def _cov(counts, unit="games"):
    return ", ".join(f"{s} ({n:,} {unit})" for s, n in counts.items()) if counts else "not fetched yet"


def build_sources_payload(raw="data/raw", context="data/context", shots="data/processed/shots.parquet",
                          ckpt=None, shot_run="runs/shots", baseline="runs/baseline.joblib", synthetic=False):
    ctx = Path(context)
    pbp = _season_counts(raw)
    matchups = _season_counts(ctx / "matchups")
    seasons_of = lambda path, col="season": (sorted(pd.read_parquet(path, columns=[col])[col].unique())
                                             if Path(path).exists() else [])
    coach_seasons = seasons_of(ctx / "coaches.parquet")
    pt_seasons = seasons_of(ctx / "playtypes.parquet")
    roster_seasons = seasons_of(ctx / "rosters.parquet")
    bbref = ctx / "bbref_rosters.parquet"
    bb = pd.read_parquet(bbref) if bbref.exists() else None
    n_shots = len(pd.read_parquet(shots, columns=["made"])) if Path(shots).exists() else 0

    sources = [
        {"name": "NBA.com Stats: PlayByPlayV3", "org": "NBA.com via nba_api", "url": "https://www.nba.com/stats",
         "used": "Every event of every game: shots with court x/y location, distance, shot type, makes/misses, assists, substitutions, "
                 "score and clock. Feeds the game tokens for NBAGPT, the shot charts and the player profiles.",
         "coverage": _cov(pbp), "updated": _mtime(raw), "script": "fetch_data.py", "file": f"{raw}/<season>/<gameId>.parquet"},
        {"name": "NBA.com Stats: LeagueGameFinder", "org": "NBA.com via nba_api", "url": "https://www.nba.com/stats",
         "used": "The list of regular-season game IDs to download for each season.",
         "coverage": ", ".join(pbp) or "not fetched yet", "updated": _mtime(raw), "script": "fetch_data.py", "file": "(not stored)"},
        {"name": "NBA.com Stats: BoxScoreMatchupsV3", "org": "NBA.com via nba_api", "url": "https://www.nba.com/stats",
         "used": "Who guarded whom in each game (partial possessions, shots attempted). Used to estimate each shot's primary defender.",
         "coverage": _cov(matchups), "updated": _mtime(ctx / "matchups"), "script": "fetch_context.py", "file": f"{context}/matchups/<season>/<gameId>.parquet"},
        {"name": "NBA.com Stats: CommonTeamRoster", "org": "NBA.com via nba_api", "url": "https://www.nba.com/stats",
         "used": "Head coach of every team each season" + (", plus season rosters (number, position, height, weight, age, experience, school)" if roster_seasons else "") + ".",
         "coverage": ", ".join(sorted(set(coach_seasons) | set(roster_seasons))) or "not fetched yet",
         "updated": _mtime(ctx / "coaches.parquet"), "script": "fetch_context.py" + (", fetch_rosters.py" if roster_seasons else ""),
         "file": f"{context}/coaches.parquet" + (f", {context}/rosters.parquet" if roster_seasons else "")},
        {"name": "NBA.com Stats: SynergyPlayTypes", "org": "NBA.com / Synergy Sports via nba_api", "url": "https://www.nba.com/stats/players/isolation",
         "used": "How often each team and player uses each play type (isolation, pick-and-roll, spot-up, ...) and points per possession.",
         "coverage": ", ".join(pt_seasons) or "not fetched yet", "updated": _mtime(ctx / "playtypes.parquet"), "script": "fetch_context.py",
         "file": f"{context}/playtypes.parquet"},
        {"name": "NBA.com Stats: CommonAllPlayers", "org": "NBA.com via nba_api", "url": "https://www.nba.com/stats/players",
         "used": "Official player IDs and names, used to match Basketball-Reference rosters to NBA.com players (including this year's rookies).",
         "coverage": "All players, all time", "updated": _mtime(ctx / "bbref" / "nba_all_players.parquet"), "script": "fetch_bbref.py",
         "file": f"{context}/bbref/nba_all_players.parquet"},
        {"name": "Basketball-Reference team rosters", "org": "Sports Reference LLC", "url": "https://www.basketball-reference.com/teams/",
         "used": "Current team for every player, plus position, height, weight, birth date, years of experience and college.",
         "coverage": (", ".join(f"{s} ({(bb['season'] == s).sum()} players, {bb.loc[bb['season'] == s, 'team'].nunique()} teams)"
                                for s in sorted(bb["season"].unique())) if bb is not None else "not fetched yet"),
         "updated": bb["fetched"].max() if bb is not None else None, "script": "fetch_bbref.py", "file": str(bbref)},
        {"name": "nba_api static data", "org": "nba_api (open source)", "url": "https://github.com/swar/nba_api",
         "used": "Offline list of the 30 teams (IDs, names, abbreviations) and players' full names.",
         "coverage": "Bundled with the installed nba_api package", "updated": None, "script": "(library)", "file": "(library)"},
    ]
    if synthetic:
        sources.append({"name": "Synthetic games", "org": "This project (synthetic.py)", "url": None,
                        "used": "Simulated games that stand in for real ones in the game replay and model sections until NBAGPT is trained on real seasons.",
                        "coverage": "Simulated seasons 2098-99 and 2099-00", "updated": _mtime("data/raw_synthetic"), "script": "synthetic.py",
                        "file": "data/raw_synthetic/"})

    software = [
        {"name": "nba_api", "url": "https://github.com/swar/nba_api", "use": "Client for the NBA.com Stats endpoints"},
        {"name": "PyTorch", "url": "https://pytorch.org", "use": "NBAGPT transformer and the shot model"},
        {"name": "scikit-learn", "url": "https://scikit-learn.org", "use": "Logistic-regression baseline"},
        {"name": "pandas, NumPy, PyArrow", "url": "https://pandas.pydata.org", "use": "Data wrangling and Parquet storage"},
        {"name": "Chart.js 4.4.1", "url": "https://www.chartjs.org", "use": "Charts in this page (from cdnjs)"},
        {"name": "Barlow Condensed, Source Sans 3", "url": "https://fonts.google.com", "use": "Fonts (Google Fonts)"},
    ]

    # training settings straight from the run logs
    runs = {}
    if ckpt and (Path(ckpt).parent / "log.json").exists():
        log = json.load(open(Path(ckpt).parent / "log.json"))
        runs["nbagpt"] = {"args": log.get("args", {}), "config": log.get("config", {}), "n_params": log.get("n_params"),
                          "epochs_run": len(log.get("history", [])), "best_epoch": (log.get("best") or {}).get("epoch")}
    meta_path = Path(shot_run) / "meta.json"
    if meta_path.exists():
        meta = json.load(open(meta_path))
        runs["shots"] = {"args": meta.get("args", {}), "epochs_run": len(meta.get("history", [])),
                         "seasons": meta.get("seasons"), "metrics": meta.get("metrics")}
    runs["baseline_trained"] = Path(baseline).exists()
    return {"sources": sources, "software": software, "runs": runs, "n_shots": n_shots,
            "built": datetime.now().strftime("%Y-%m-%d %H:%M")}


if __name__ == "__main__":
    out = build_sources_payload(ckpt="runs/syn/best.pt", synthetic=True)
    for s in out["sources"]:
        print(f"- {s['name']}: {s['coverage']} (updated {s['updated']})")
    print(json.dumps(out["runs"], indent=1)[:600])
