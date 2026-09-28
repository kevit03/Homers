"""Shot-chart data for every player who took a shot, built straight from data/processed/shots.parquet.

No model is needed, so it covers every shooter (not just the ones with a shot-model embedding).
For each player and season it stores counts only; the dashboard sums seasons and derives every
percentage in the browser:

  - location zones (restricted area, paint, mid-range L/C/R, corner 3 L/R, above-the-break 3 L/C/R, heaves)
  - shot types (layup, dunk, ...) and styles (pull-up, step-back, ...) from build_shots.py
  - a sparse hexbin grid of attempts / makes for the court chart
  - 3-pointers, assisted makes, total shot distance and games played

Current teams and bios come from data/context/bbref_rosters.parquet (fetch_bbref.py, Basketball-Reference).
Players on a current roster who have no shots in the data yet (mostly rookies) are included with an
empty shot history, so every player in the league can be looked up.

Used by export_dashboard.py when data/processed/shots.parquet exists. Run it alone for a size check:

    python export_shotcharts.py --shots data/processed/shots.parquet
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from nba_api.stats.static import teams as nba_teams

from build_shots import STYLES, ZONES

# Court units are tenths of a foot with the hoop at (0, 0); the baseline is y = -52.5.
LOC_ZONES = ["Restricted area", "Paint (non-RA)", "Mid-range left", "Mid-range center", "Mid-range right",
             "Corner 3 left", "Corner 3 right", "Above the break 3 left", "Above the break 3 center",
             "Above the break 3 right", "Heave (40+ ft)"]
HEAVE = len(LOC_ZONES) - 1
HEX_R = 12.0        # hex radius (center to corner), 1.2 ft
CHART_TOP = 370.0   # the chart shows the court up to 37 ft from the hoop; farther shots are heaves


def loc_zone(x, y, is3):
    """Vectorised location zone. Mid-range and above-the-break 3s split left/center/right at the paint edges (x = ±80)."""
    x, y, is3 = np.asarray(x, float), np.asarray(y, float), np.asarray(is3, bool)
    side = np.where(x < -80, 0, np.where(x > 80, 2, 1))  # left, center, right
    rim = np.hypot(x, y) <= 40
    paint = (np.abs(x) <= 80) & (y <= 137.5)
    two = np.where(rim, 0, np.where(paint, 1, 2 + side))
    corner = (np.abs(x) >= 220) & (y <= 92)
    three = np.where(corner, np.where(x < 0, 5, 6), 7 + side)
    z = np.where(is3, three, two)
    return np.where((np.hypot(x, y) >= 400) | (y > CHART_TOP), HEAVE, z)


def hex_axial(x, y, R=HEX_R):
    """Pointy-top hex grid: nearest hex (q, r) for each point, via cube rounding."""
    q = (math.sqrt(3) / 3 * np.asarray(x, float) - np.asarray(y, float) / 3) / R
    r = (2 / 3 * np.asarray(y, float)) / R
    s = -q - r
    rq, rr, rs = np.round(q), np.round(r), np.round(s)
    dq, dr, ds = np.abs(rq - q), np.abs(rr - r), np.abs(rs - s)
    fix_q = (dq > dr) & (dq > ds)
    fix_r = ~fix_q & (dr > ds)
    rq = np.where(fix_q, -rr - rs, rq)
    rr = np.where(fix_r, -rq - rs, rr)
    return rq.astype(int), rr.astype(int)


def pair_counts(codes, made, n):
    """Flat [att0, made0, att1, made1, ...] for integer codes in [0, n)."""
    a = np.bincount(codes, minlength=n)
    m = np.bincount(codes, weights=made, minlength=n).astype(int)
    return np.stack([a, m], 1).ravel().tolist()


def player_names(ids, shots):
    try:
        from nba_api.stats.static import players
        full = {p["id"]: p["full_name"] for p in players.get_players()}
    except Exception:  # nba_api missing: fall back to the play-by-play surname
        full = {}
    last = shots.drop_duplicates("shooterId").set_index("shooterId")["shooter"].to_dict()
    return {int(p): full.get(int(p)) or last.get(p) or str(p) for p in ids}


def teams_in_order(df):
    """Teams a player shot for that season, in the order he played for them ("BKN/PHI" after a trade)."""
    first = df.sort_values("gameId").drop_duplicates("offTeam")
    return "/".join(first["offTeam"])


def current_rosters(path):
    """Latest season in bbref_rosters.parquet -> (roster rows, season, fetch date); empty if not fetched."""
    if not path or not Path(path).exists():
        return pd.DataFrame(), None, None
    r = pd.read_parquet(path)
    season = sorted(r["season"].unique())[-1]
    r = r[r["season"] == season].drop_duplicates("bbref_id", keep="last")
    return r, season, r["fetched"].max()


def bio(row):
    return {"team": row["team"], "pos": row["pos"], "ht": row["height"], "wt": row["weight"], "exp": row["exp"],
            "college": row["college"], "born": row["birth_date"], "no": row["number"], "bref": row["bbref_id"]}


def build_shotcharts_payload(shots_path, rosters_path="data/context/bbref_rosters.parquet"):
    shots = pd.read_parquet(shots_path, columns=["gameId", "season", "shooterId", "shooter", "offTeam", "zone", "style",
                                                 "made", "assisted", "is3", "dist", "x", "y"])
    shots = shots[shots["shooterId"] > 0].reset_index(drop=True)
    seasons = sorted(shots["season"].unique().tolist())
    shots["s"] = shots["season"].map({s: i for i, s in enumerate(seasons)})
    shots["lz"] = loc_zone(shots["x"], shots["y"], shots["is3"] == 1)
    shots["ty"] = shots["zone"].map({z: i for i, z in enumerate(ZONES)})
    shots["st"] = shots["style"].map({s: i for i, s in enumerate(STYLES)})

    # hex grid: only hexes that someone shot from, in the charted part of the court
    on_chart = shots["lz"] != HEAVE
    q, r = hex_axial(shots["x"], shots["y"])
    keys = pd.Series(list(zip(q, r)))
    hexes = sorted(set(keys[on_chart.to_numpy()]))
    hex_idx = {h: i for i, h in enumerate(hexes)}
    shots["hx"] = np.where(on_chart, keys.map(lambda k: hex_idx.get(k, -1)), -1)
    H = len(hexes)

    def record(df):
        made = df["made"].to_numpy()
        h = df[df["hx"] >= 0]
        ha = np.bincount(h["hx"], minlength=H)
        hm = np.bincount(h["hx"], weights=h["made"], minlength=H).astype(int)
        nz = np.flatnonzero(ha)
        t3 = df["is3"] == 1
        return {
            "g": int(df["gameId"].nunique()),
            "lz": pair_counts(df["lz"].to_numpy(), made, len(LOC_ZONES)),
            "ty": pair_counts(df["ty"].to_numpy(), made, len(ZONES)),
            "st": pair_counts(df["st"].to_numpy(), made, len(STYLES)),
            "t3": [int(t3.sum()), int(df.loc[t3, "made"].sum())],
            "ast": int(df.loc[df["made"] == 1, "assisted"].sum()),
            "d": int(round(df["dist"].sum())),
            "h": np.stack([nz, ha[nz], hm[nz]], 1).ravel().tolist(),
        }

    league = []
    for s, g in shots.groupby("s", sort=True):
        rec = record(g)
        rec.pop("h")
        ha = np.bincount(g.loc[g["hx"] >= 0, "hx"], minlength=H)
        hm = np.bincount(g.loc[g["hx"] >= 0, "hx"], weights=g.loc[g["hx"] >= 0, "made"], minlength=H).astype(int)
        rec.update({"s": int(s), "games": int(g["gameId"].nunique()), "players": int(g["shooterId"].nunique()),
                    "ha": ha.tolist(), "hm": hm.tolist()})
        league.append(rec)

    order = shots["shooterId"].value_counts().index
    names = player_names(order, shots)
    players = []
    for pid, g in shots.groupby("shooterId"):
        recs = []
        for s, gs in g.groupby("s", sort=True):
            rec = record(gs)
            rec.update({"s": int(s), "t": teams_in_order(gs)})
            recs.append(rec)
        players.append({"id": int(pid), "n": names[int(pid)], "fga": int(len(g)), "p": recs})
    players.sort(key=lambda p: -p["fga"])
    for p in players:
        del p["fga"]

    # where everyone is now
    roster, roster_season, fetched = current_rosters(rosters_path)
    by_id = {int(r["personId"]): r for _, r in roster.iterrows() if pd.notna(r["personId"])}
    for p in players:
        r = by_id.pop(p["id"], None)
        if r is not None:
            p["now"] = bio(r)
    matched = {int(r["personId"]) for _, r in roster.iterrows() if pd.notna(r["personId"])} - set(by_id)
    for _, r in roster.iterrows():  # rostered players with no shots yet (rookies, or not in these seasons)
        if pd.isna(r["personId"]) or int(r["personId"]) not in matched:
            players.append({"id": int(r["personId"]) if pd.notna(r["personId"]) else 0, "n": r["player"], "p": [], "now": bio(r)})
    team_names = {t["abbreviation"]: t["full_name"] for t in nba_teams.get_teams()}

    return {
        "seasons": seasons, "loc_zones": LOC_ZONES, "types": ZONES, "styles": STYLES,
        "hex": {"r": HEX_R, "top": CHART_TOP, "qr": [list(map(int, h)) for h in hexes]},
        "league": league, "players": players, "teams": team_names,
        "roster": {"season": roster_season, "fetched": fetched, "source": "Basketball-Reference", "n": int(len(roster))},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="data/processed/shots.parquet")
    ap.add_argument("--rosters", default="data/context/bbref_rosters.parquet")
    args = ap.parse_args()
    payload = build_shotcharts_payload(args.shots, args.rosters)
    size = len(json.dumps(payload, separators=(",", ":")))
    n_recs = sum(len(p["p"]) for p in payload["players"])
    now = sum(1 for p in payload["players"] if p.get("now"))
    print(f"{payload['roster']['season']} rosters: {now} players on a team now, "
          f"{sum(1 for p in payload['players'] if not p['p'])} of them with no shots in the data yet")
    print(f"{len(payload['players'])} players, {n_recs} player-seasons, {len(payload['hex']['qr'])} hexes, "
          f"seasons {payload['seasons']} -> {size / 1e6:.2f} MB of JSON")


if __name__ == "__main__":
    main()
