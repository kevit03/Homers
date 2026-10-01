"""Data sources and methods for the dashboard's Sources tab.

Everything here is read from what is actually on disk (seasons, game counts, fetch dates, training
settings from the run logs), so the tab stays accurate after every re-fetch or re-train.
Used by export/export_dashboard.py.
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from models.tokenize_pbp import game_type


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


def _cov_by_type(root):
    """'2024-25 (1,230 regular season + 85 playoffs)' for each season folder."""
    root = Path(root)
    if not root.exists():
        return "not fetched yet"
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        n = pd.Series([game_type(f.stem) for f in d.glob("*.parquet")], dtype=object).value_counts()
        if len(n):
            out.append(f"{d.name} (" + " + ".join(f"{k:,} {t.lower()}" for t, k in n.items()) + ")")
    return ", ".join(out) or "not fetched yet"


def _finals(season_dir):
    """Champion, runner-up and series score from the Finals games (playoff IDs with round 4: 004YY004SG)."""
    wins, ids = {}, {}
    for f in sorted(Path(season_dir).glob("004*.parquet")):
        if f.stem[7] != "4":
            continue
        g = pd.read_parquet(f, columns=["location", "teamId", "teamTricode", "scoreHome", "scoreAway"])
        side = g[g["location"].isin(["h", "v"]) & (g["teamTricode"].fillna("").astype(str).str.len() == 3)].drop_duplicates("location")
        tri = dict(zip(side["location"], side["teamTricode"]))
        ids.update(zip(side["teamTricode"], pd.to_numeric(side["teamId"], errors="coerce").fillna(0).astype(int)))
        h, a = (pd.to_numeric(g[c], errors="coerce").ffill().iloc[-1] for c in ("scoreHome", "scoreAway"))
        if set(tri) == {"h", "v"} and h != a:
            for t in tri.values():
                wins.setdefault(t, 0)
            wins[tri["h" if h > a else "v"]] += 1
    if len(wins) != 2:
        return None
    (champ, w), (opp, l) = sorted(wins.items(), key=lambda kv: -kv[1])
    return {"champ": champ, "champ_id": int(ids.get(champ, 0)), "opp": opp, "opp_id": int(ids.get(opp, 0)), "w": w, "l": l}


def _history_seasons(history, before):
    """Seasons before `before` from fetch/fetch_history.py, flagged pbp=False: results and box scores only, no play-by-play.
    The dashboard marks them with an asterisk."""
    from fetch.fetch_history import finals, load_games, tracked
    g = load_games(history)
    if g.empty:
        return []
    g = g[g["season"] < before] if before else g
    stats = tracked(history)
    return [{"season": s, "rs": int((d["season_type"] == "Regular Season").sum()), "po": int((d["season_type"] == "Playoffs").sum()),
             "shots": 0, "split": {}, "finals": finals(d), "pbp": False, "tracked": stats.get(s, [])}
            for s, d in g.groupby("season")]


def build_seasons_payload(raw="data/raw", shots="data/processed/shots.parquet", data=None,
                          history="data/context/history_team_games.parquet"):
    """One record per season for the Home tab's orbit: games by type, shots, the Finals, and how Tempo used its games.
    Seasons from before the play-by-play (fetch/fetch_history.py) come first, with pbp=False."""
    root = Path(raw)
    if not root.exists():
        return None
    shots_per = (pd.read_parquet(shots, columns=["season"])["season"].value_counts().to_dict() if Path(shots).exists() else {})
    split_per = {}
    if data:
        for split, idx in data["splits"].items():
            for i in idx:
                s = split_per.setdefault(data["games"][i]["season"], {"train": 0, "val": 0, "test": 0})
                s[split] += 1
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        kinds = pd.Series([game_type(f.stem) for f in d.glob("*.parquet")], dtype=object).value_counts()
        if not len(kinds):
            continue
        out.append({"season": d.name, "rs": int(kinds.get("Regular Season", 0)), "po": int(kinds.get("Playoffs", 0)),
                    "shots": int(shots_per.get(d.name, 0)), "split": split_per.get(d.name, {}), "finals": _finals(d), "pbp": True})
    return _history_seasons(history, out[0]["season"] if out else None) + out


def build_sources_payload(raw="data/raw", context="data/context", shots="data/processed/shots.parquet",
                          ckpt=None, shot_run="runs/shots", baseline="runs/baseline.joblib", synthetic=False):
    ctx = Path(context)
    pbp = _season_counts(raw)
    from fetch.fetch_history import load_games
    hist = load_games(ctx / "history_team_games.parquet")
    hist = hist[hist["season"] < min(pbp)] if len(hist) and pbp else hist
    hist_counts = hist.groupby("season").size().astype(int).to_dict() if len(hist) else {}
    matchups = _season_counts(ctx / "matchups")
    seasons_of = lambda path, col="season": (sorted(pd.read_parquet(path, columns=[col])[col].unique())
                                             if Path(path).exists() else [])
    coach_seasons = seasons_of(ctx / "coaches.parquet")
    pt_seasons = seasons_of(ctx / "playtypes.parquet")
    roster_seasons = seasons_of(ctx / "rosters.parquet")
    bbref = ctx / "bbref_rosters.parquet"
    bb = pd.read_parquet(bbref) if bbref.exists() else None
    n_shots = len(pd.read_parquet(shots, columns=["made"])) if Path(shots).exists() else 0
    tr_seasons = seasons_of(ctx / "tracking.parquet")
    acc_seasons = seasons_of(ctx / "bbref_players.parquet")
    n_acc = len(json.loads((ctx / "bbref_accolades.json").read_text())) if (ctx / "bbref_accolades.json").exists() else 0
    n_coach = len(json.loads((ctx / "bbref_coaches.json").read_text())) if (ctx / "bbref_coaches.json").exists() else 0
    acc_cov = (f"{', '.join(acc_seasons)}; {n_acc:,} player pages, {n_coach} coaches" if acc_seasons else "not fetched yet")
    acc_updated = _mtime(ctx / "bbref_accolades.json") if n_acc else _mtime(ctx / "bbref_coaches.json")

    sources = [
        {"name": "NBA.com Stats: PlayByPlayV3", "org": "NBA.com via nba_api", "url": "https://www.nba.com/stats",
         "used": "Every event of every game: shots with court x/y location, distance, shot type, makes/misses, assists, substitutions, "
                 "score and clock. Feeds the game tokens for Tempo, the shot charts and the player profiles.",
         "coverage": _cov_by_type(raw), "updated": _mtime(raw), "script": "fetch/fetch_data.py", "file": f"{raw}/<season>/<gameId>.parquet"},
        {"name": "NBA.com Stats: LeagueGameFinder", "org": "NBA.com via nba_api", "url": "https://www.nba.com/stats",
         "used": "The list of regular-season and playoff game IDs to download for each season.",
         "coverage": ", ".join(pbp) or "not fetched yet", "updated": _mtime(raw), "script": "fetch/fetch_data.py", "file": "(not stored)"},
        {"name": "NBA.com Stats: LeagueGameLog (before play-by-play)*", "org": "NBA.com via nba_api", "url": "https://www.nba.com/stats",
         "used": "* Box scores only. NBA.com has no play-by-play before 1996-97, so these seasons are never tokenized, trained on "
                 "or replayed: each game's date, teams, final score and team totals, which thin out with age (before 1982-83 mostly "
                 "points, field goals and free throws; the full team box score from 1985-86). Used for champions in the seasons orbit "
                 "and, opt-in, to warm up the pre-game Elo ratings.",
         "coverage": (f"{min(hist_counts)} to {max(hist_counts)}: {sum(hist_counts.values()):,} games, "
                      f"{int((hist['season_type'] == 'Playoffs').sum()):,} of them playoffs") if hist_counts else "not fetched yet",
         "updated": _mtime(ctx / "history_team_games.parquet"), "script": "fetch/fetch_history.py", "file": f"{context}/history_team_games.parquet"},
        {"name": "NBA.com Stats: BoxScoreMatchupsV3", "org": "NBA.com via nba_api", "url": "https://www.nba.com/stats",
         "used": "Who guarded whom in each game: partial possessions, points, shots, turnovers. Used to estimate each shot's primary defender and to train the man-to-man matchup model. NBA.com has it from 2017-18 on.",
         "coverage": _cov_by_type(ctx / "matchups"), "updated": _mtime(ctx / "matchups"), "script": "fetch/fetch_context.py", "file": f"{context}/matchups/<season>/<gameId>.parquet"},
        {"name": "NBA.com Stats: CommonTeamRoster", "org": "NBA.com via nba_api", "url": "https://www.nba.com/stats",
         "used": "Head coach of every team each season" + (", plus season rosters (number, position, height, weight, age, experience, school)" if roster_seasons else "") + ".",
         "coverage": ", ".join(sorted(set(coach_seasons) | set(roster_seasons))) or "not fetched yet",
         "updated": _mtime(ctx / "coaches.parquet"), "script": "fetch/fetch_context.py" + (", fetch/fetch_rosters.py" if roster_seasons else ""),
         "file": f"{context}/coaches.parquet" + (f", {context}/rosters.parquet" if roster_seasons else "")},
        {"name": "NBA.com Stats: SynergyPlayTypes", "org": "NBA.com / Synergy Sports via nba_api", "url": "https://www.nba.com/stats/players/isolation",
         "used": "How often each team and player uses each play type (isolation, pick-and-roll, spot-up, ...) and points per possession, "
                 "on offense and on defense: what each team allows, and what each player allows as the defender.",
         "coverage": ", ".join(pt_seasons) or "not fetched yet", "updated": _mtime(ctx / "playtypes.parquet"), "script": "fetch/fetch_context.py",
         "file": f"{context}/playtypes.parquet"},
        {"name": "NBA.com Stats: tracking and hustle", "org": "NBA.com / Second Spectrum via nba_api", "url": "https://www.nba.com/stats/players/drives",
         "used": "Player-tracking actions for teams and players: drives, catch-and-shoot and pull-up shots, paint, post and elbow touches, "
                 "screen assists; hustle stats (deflections, contested shots, charges drawn, loose balls, box outs); and opponents' "
                 "FG% at the rim and from three against each defender and team (LeagueDashPtStats, LeagueHustleStats, LeagueDashPtDefend).",
         "coverage": ", ".join(tr_seasons) or "not fetched yet", "updated": _mtime(ctx / "tracking.parquet"), "script": "fetch/fetch_context.py",
         "file": f"{context}/tracking.parquet"},
        {"name": "Basketball-Reference awards and coaches", "org": "Sports Reference LLC", "url": "https://www.basketball-reference.com/awards/",
         "used": "Player accolades as Basketball-Reference lists them (All-Star, All-NBA, MVP, titles, ...) and each season's award votes; "
                 "every head coach's record, titles, Coach of the Year and other awards, season by season, and coach photos.",
         "coverage": acc_cov, "updated": acc_updated, "script": "fetch/fetch_accolades.py",
         "file": f"{context}/bbref_accolades.json, {context}/bbref_coaches.json"},
        {"name": "NBA.com Stats: CommonAllPlayers", "org": "NBA.com via nba_api", "url": "https://www.nba.com/stats/players",
         "used": "Official player IDs and names, used to match Basketball-Reference rosters to NBA.com players (including this year's rookies).",
         "coverage": "All players, all time", "updated": _mtime(ctx / "bbref" / "nba_all_players.parquet"), "script": "fetch/fetch_bbref.py",
         "file": f"{context}/bbref/nba_all_players.parquet"},
        {"name": "Basketball-Reference team rosters", "org": "Sports Reference LLC", "url": "https://www.basketball-reference.com/teams/",
         "used": "Current team for every player, plus position, height, weight, birth date, years of experience and college.",
         "coverage": (", ".join(f"{s} ({(bb['season'] == s).sum()} players, {bb.loc[bb['season'] == s, 'team'].nunique()} teams)"
                                for s in sorted(bb["season"].unique())) if bb is not None else "not fetched yet"),
         "updated": bb["fetched"].max() if bb is not None else None, "script": "fetch/fetch_bbref.py", "file": str(bbref)},
        {"name": "nba_api static data", "org": "nba_api (open source)", "url": "https://github.com/swar/nba_api",
         "used": "Offline list of the 30 teams (IDs, names, abbreviations) and players' full names.",
         "coverage": "Bundled with the installed nba_api package", "updated": None, "script": "(library)", "file": "(library)"},
    ]
    if synthetic:
        sources.append({"name": "Synthetic games", "org": "This project (models/synthetic.py)", "url": None,
                        "used": "Simulated games that stand in for real ones in the game replay and model sections until Tempo is trained on real seasons.",
                        "coverage": "Simulated seasons 2098-99 and 2099-00", "updated": _mtime("data/raw_synthetic"), "script": "models/synthetic.py",
                        "file": "data/raw_synthetic/"})

    # extras for the Sources tab cards: per-season counts (drawn as bars), season lists (drawn as pills),
    # which tabs each source feeds, and the badge style
    extras = {
        "NBA.com Stats: PlayByPlayV3": (pbp, [], ["games", "shots", "players", "model"]),
        "NBA.com Stats: LeagueGameFinder": ({}, list(pbp), ["games", "shots"]),
        "NBA.com Stats: LeagueGameLog (before play-by-play)*": (hist_counts, [], ["home"]),
        "NBA.com Stats: BoxScoreMatchupsV3": (matchups, [], ["matchups", "players"]),
        "NBA.com Stats: CommonTeamRoster": ({}, sorted(set(coach_seasons) | set(roster_seasons)), ["players"]),
        "NBA.com Stats: SynergyPlayTypes": ({}, pt_seasons, ["players", "coaches"]),
        "NBA.com Stats: tracking and hustle": ({}, tr_seasons, ["players", "coaches"]),
        "Basketball-Reference awards and coaches": ({}, acc_seasons, ["coaches", "fantasy", "players"]),
        "NBA.com Stats: CommonAllPlayers": ({}, [], ["shots", "players"]),
        "Basketball-Reference team rosters": ({}, sorted(bb["season"].unique()) if bb is not None else [], ["shots", "players"]),
        "nba_api static data": ({}, [], ["games", "players"]),
        "Synthetic games": ({}, [], ["games", "model"]),
    }
    for s in sources:
        counts, seasons, feeds = extras.get(s["name"], ({}, [], []))
        s.update(season_counts=counts, seasons=[str(x) for x in seasons], feeds=feeds,
                 kind="nba" if s["name"].startswith("NBA.com") else "sim" if s["url"] is None else "ref" if "Reference" in s["name"] else "oss")

    software = [
        {"name": "nba_api", "url": "https://github.com/swar/nba_api", "use": "Client for the NBA.com Stats endpoints"},
        {"name": "PyTorch", "url": "https://pytorch.org", "use": "Tempo transformer and the shot model"},
        {"name": "scikit-learn", "url": "https://scikit-learn.org", "use": "Logistic-regression baseline"},
        {"name": "pandas, NumPy, PyArrow", "url": "https://pandas.pydata.org", "use": "Data wrangling and Parquet storage"},
        {"name": "Chart.js 4.4.1", "url": "https://www.chartjs.org", "use": "Charts in this page (from cdnjs)"},
        {"name": "Barlow, Barlow Semi Condensed, Barlow Condensed", "url": "https://fonts.google.com/specimen/Barlow", "use": "Fonts (Google Fonts)"},
        {"name": "shadcn/ui NavigationMenu, radial orbital timeline", "url": "https://ui.shadcn.com/docs/components/navigation-menu",
         "use": "The menu and the seasons orbit, ported from React to plain JavaScript"},
        {"name": "Lucide", "url": "https://lucide.dev", "use": "Icons in the menu and the orbit (inlined SVG)"},
        {"name": "Wikimedia Commons", "url": "https://commons.wikimedia.org", "use": "Openly licensed photos (credits below)"},
    ]

    # training settings straight from the run logs
    runs = {}
    if ckpt and (Path(ckpt).parent / "log.json").exists():
        log = json.load(open(Path(ckpt).parent / "log.json"))
        runs["tempo"] = {"args": log.get("args", {}), "config": log.get("config", {}), "n_params": log.get("n_params"),
                         "epochs_run": len(log.get("history", [])), "best_epoch": (log.get("best") or {}).get("epoch")}
    meta_path = Path(shot_run) / "meta.json"
    if meta_path.exists():
        meta = json.load(open(meta_path))
        runs["shots"] = {"args": meta.get("args", {}), "epochs_run": len(meta.get("history", [])),
                         "seasons": meta.get("seasons"), "metrics": meta.get("metrics")}
    runs["baseline_trained"] = Path(baseline).exists()
    return {"sources": sources, "software": software, "runs": runs, "n_shots": n_shots,
            "n_games_raw": sum(pbp.values()), "seasons": list(pbp),
            "built": datetime.now().strftime("%Y-%m-%d %H:%M")}


if __name__ == "__main__":
    out = build_sources_payload(ckpt="runs/syn/best.pt", synthetic=True)
    for s in out["sources"]:
        print(f"- {s['name']}: {s['coverage']} (updated {s['updated']})")
    print(json.dumps(out["runs"], indent=1)[:600])
