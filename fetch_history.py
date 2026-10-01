"""Results of every game from before NBA.com's play-by-play: 1946-47 to 1995-96, regular season and playoffs.

These seasons are the asterisk tier ("box score only*"). NBA.com has no play-by-play for them, so they are
never tokenized, trained on or replayed. What it does have is each team's line for every game (LeagueGameLog):
date, opponent, home or away, the final score, and team box-score totals. NBA.com's team logs are thin for old
seasons: before 1982-83 mostly points, field goals and free throws made (threes from 1979-80, when the line came in);
rebounds, assists and attempts from 1982-83; the full team box score from 1985-86. tracked() lists each season's.
They are used for the seasons history on the dashboard and, opt-in, to warm up Elo (team_strength.py --history).

One call per season and type, cached, so the script can be stopped and re-run safely.

    python fetch_history.py                     # 1946-47 to 1995-96
    python fetch_history.py --seasons 1985-86   # just these
-> data/context/history_team_games.parquet  (one row per team per game, as NBA.com sends it)
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

HISTORY_SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(1946, 1996)]  # 1946-47 (the BAA) to 1995-96
SEASON_TYPES = ["Regular Season", "Playoffs"]
OUT = "data/context/history_team_games.parquet"
BOX = ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]


def fetch_season(season, season_type, retries=4):
    for attempt in range(retries):
        try:
            df = leaguegamelog.LeagueGameLog(season=season, season_type_all_star=season_type, timeout=60).get_data_frames()[0]
            return df.assign(season=season, season_type=season_type)
        except Exception as e:  # network hiccups / rate limiting
            print(f"  {season} {season_type} attempt {attempt + 1} failed: {e}")
            time.sleep(5 * (attempt + 1))
    return None


def load_games(path=OUT) -> pd.DataFrame:
    """One row per game: season, type, date, home and away team (ID, tricode, points), sorted by date.
    Home is the team whose MATCHUP reads "vs."; the few games NBA.com doesn't mark that way are dropped."""
    if not Path(path).exists():
        return pd.DataFrame()
    t = pd.read_parquet(path)
    t["home"] = t["MATCHUP"].str.contains(" vs. ", regex=False)
    cols = ["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "PTS"]
    h, a = t[t["home"]], t[~t["home"]]
    g = h[["season", "season_type", "GAME_DATE"] + cols].merge(a[cols], on="GAME_ID", suffixes=("_h", "_a"))
    g = g[g.groupby("GAME_ID")["GAME_ID"].transform("size") == 1]  # exactly one home and one away line
    g = g.rename(columns={"GAME_ID": "gameId", "GAME_DATE": "date", "TEAM_ID_h": "home", "TEAM_ID_a": "away",
                          "TEAM_ABBREVIATION_h": "home_tri", "TEAM_ABBREVIATION_a": "away_tri",
                          "TEAM_NAME_h": "home_name", "TEAM_NAME_a": "away_name",
                          "PTS_h": "home_pts", "PTS_a": "away_pts"})
    g = g.dropna(subset=["home_pts", "away_pts"])
    g[["home", "away"]] = g[["home", "away"]].astype(int)
    return g.sort_values(["date", "gameId"]).reset_index(drop=True)


def tracked(path=OUT, min_share=0.9) -> dict:
    """season -> the team box-score stats NBA.com has for (nearly) all of its regular-season games."""
    if not Path(path).exists():
        return {}
    t = pd.read_parquet(path)
    t = t[t["season_type"] == "Regular Season"]
    return {s: [c for c in BOX if c in d and d[c].notna().mean() >= min_share] for s, d in t.groupby("season")}


def finals(season_games: pd.DataFrame):
    """Champion and runner-up from a season's playoff games. Old game IDs don't encode the round, so the
    Finals are the series between the two teams in the season's last playoff game."""
    po = season_games[season_games["season_type"] == "Playoffs"]
    if po.empty:
        return None
    last = po.iloc[-1]
    pair = {last["home"], last["away"]}
    s = po[po["home"].isin(pair) & po["away"].isin(pair)]
    winner = s["home"].where(s["home_pts"] > s["away_pts"], s["away"])
    wins = winner.value_counts()
    tri = dict(zip(s["home"], s["home_tri"])) | dict(zip(s["away"], s["away_tri"]))
    name = dict(zip(s["home"], s["home_name"])) | dict(zip(s["away"], s["away_name"]))  # era names: Minneapolis Lakers
    champ = int(winner.iloc[-1])  # whoever won the last game won the series
    opp = (pair - {champ}).pop()
    return {"champ": tri[champ], "champ_id": champ, "champ_name": name[champ], "opp": tri[opp], "opp_id": int(opp), "opp_name": name[opp],
            "w": int(wins.get(champ, 0)), "l": int(wins.get(opp, 0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=HISTORY_SEASONS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between requests")
    args = ap.parse_args()

    out = Path(args.out)
    have = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["season", "season_type"])
    done = set(zip(have["season"], have["season_type"]))
    new = []
    for season in args.seasons:
        for season_type in SEASON_TYPES:
            if (season, season_type) in done:
                continue
            df = fetch_season(season, season_type)
            time.sleep(args.sleep)
            if df is None:
                print(f"{season} {season_type}: failed")
                continue
            print(f"{season} {season_type}: {df['GAME_ID'].nunique()} games")
            new.append(df)
            if len(new) % 10 == 0:  # save as we go
                have = pd.concat([have, *new], ignore_index=True); new = []
                out.parent.mkdir(parents=True, exist_ok=True); have.to_parquet(out, index=False)
    if new:
        have = pd.concat([have, *new], ignore_index=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        have.to_parquet(out, index=False)

    g = load_games(out)
    print(f"{len(g):,} games in {g['season'].nunique()} seasons ({(g['season_type'] == 'Playoffs').sum():,} playoffs) -> {out}")


if __name__ == "__main__":
    main()
