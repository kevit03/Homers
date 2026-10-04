"""Build one row per field-goal attempt with shot type, on-floor lineups, likely defender and coaches.

Shot zone and style come from the play-by-play shot description, distance and court location.
Lineups are rebuilt from substitutions. The primary defender is the on-floor opponent who
guarded the shooter most in that game (from the NBA matchup feed, weighted by shots attempted).

    python -m models.build_shots --raw data/raw --context data/context --out data/processed/shots.parquet

All 30 seasons: seasons already in --reuse are copied from it, older ones are built (in parallel) and their
missing court coordinates backfilled from NBA.com shot charts (fetch/fetch_shotchart.py), joined on
GAME_ID + GAME_EVENT_ID == actionNumber + shooter. Shots with no coordinates anywhere get a zone from the
shot description and coord_imputed = 1.

    python -m models.build_shots --out data/processed/shots_30.parquet --reuse data/processed/shots.parquet \
        --workers 8
"""
import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from models.tokenize_pbp import parse_clock, seconds_remaining

ZONES = ["LAYUP", "DUNK", "FLOATER", "HOOK", "SHORT_MID", "LONG_MID", "CORNER_3", "ABOVE_BREAK_3"]
STYLES = ["STANDARD", "PULLUP", "STEPBACK", "FADEAWAY", "DRIVING", "CUTTING", "PUTBACK", "ALLEY_OOP", "RUNNING"]
TEAM_ID_MIN = 1610612737  # team-level rows use the team id as personId


def shot_zone(sub, desc, dist, x, y, is3):
    if is3:
        return "CORNER_3" if abs(x) >= 220 and y <= 92 else "ABOVE_BREAK_3"
    if "dunk" in sub:
        return "DUNK"
    if "float" in sub:
        return "FLOATER"
    if "hook" in sub:
        return "HOOK"
    if "layup" in sub or "tip" in sub or "finger roll" in sub or dist <= 3:
        return "LAYUP"
    return "SHORT_MID" if dist <= 14 else "LONG_MID"


def impute_zone(sub, is3):
    """Zone for a shot with no coordinates at all: the description's shot type, else the commonest zone.

    LONG_MID is a placeholder for a jumper of unknown length; main() re-assigns it from the shooter's own
    split of short and long jumpers that season.
    """
    if is3:
        return "ABOVE_BREAK_3"  # ~80% of old threes; corner vs above the break is unknowable here
    if any(k in sub for k in ("dunk", "float", "hook", "layup", "tip", "finger roll")):
        return shot_zone(sub, "", 0, 0, 0, False)
    return "LONG_MID"


def fill_jumpers(shots):
    """Imputed 2PT jumpers: SHORT_MID or LONG_MID by the shooter's (else the league's) majority that season."""
    known = shots[(shots["coord_imputed"] == 0) & shots["zone"].isin(["SHORT_MID", "LONG_MID"])]
    share = known.assign(short=known["zone"].eq("SHORT_MID")).groupby(["season", "shooterId"])["short"].agg(["mean", "size"])
    league = known.assign(short=known["zone"].eq("SHORT_MID")).groupby("season")["short"].mean()
    todo = (shots["coord_imputed"] == 1) & (shots["is3"] == 0) & (shots["zone"] == "LONG_MID")
    if not todo.any():
        return shots
    key = pd.MultiIndex.from_arrays([shots.loc[todo, "season"], shots.loc[todo, "shooterId"]])
    own = share.reindex(key)
    p = np.where(own["size"].fillna(0).to_numpy() >= 10, own["mean"].to_numpy(),
                 league.reindex(shots.loc[todo, "season"]).to_numpy())
    shots.loc[todo, "zone"] = np.where(np.nan_to_num(p, nan=0.0) > 0.5, "SHORT_MID", "LONG_MID")
    return shots


def shot_style(sub):
    for key, style in (("putback", "PUTBACK"), ("tip", "PUTBACK"), ("alley oop", "ALLEY_OOP"), ("cutting", "CUTTING"),
                       ("step back", "STEPBACK"), ("pull", "PULLUP"), ("fadeaway", "FADEAWAY"),
                       ("turnaround", "FADEAWAY"), ("driving", "DRIVING"), ("running", "RUNNING")):
        if key in sub:
            return style
    return "STANDARD"


def load_matchups(ctx, season, gid):
    p = ctx / "matchups" / season / f"{gid}.parquet"
    if not p.exists():
        return None
    m = pd.read_parquet(p)
    for c in ("personIdOff", "personIdDef"):
        m[c] = pd.to_numeric(m[c], errors="coerce").astype("Int64")
    for c in ("partialPossessions", "matchupFieldGoalsAttempted"):
        m[c] = pd.to_numeric(m[c], errors="coerce").fillna(0.0)
    return m


def norm(name):
    return unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().strip().lower()


def name_map(df, m):
    """team tricode -> {display name: personId} using every name form seen in the game."""
    names = defaultdict(dict)
    rows = df[(df["personId"] > 0) & (df["personId"] < TEAM_ID_MIN) & (df["teamTricode"] != "")]
    for r in rows[["teamTricode", "personId", "playerName", "playerNameI"]].drop_duplicates().itertuples():
        names[r.teamTricode][norm(r.playerName)] = r.personId
        names[r.teamTricode][norm(r.playerNameI)] = r.personId
    if m is not None:
        for r in m[["teamTricode", "personIdOff", "familyNameOff", "nameIOff"]].drop_duplicates().itertuples():
            if pd.notna(r.personIdOff):
                names[r.teamTricode].setdefault(norm(r.familyNameOff), int(r.personIdOff))
                names[r.teamTricode].setdefault(norm(r.nameIOff), int(r.personIdOff))
    return names


def lineups(df, names):
    """Return {row index: {tricode: frozenset of 5 personIds}} for the on-floor players at each event."""
    teams_ = [t for t in df["teamTricode"].unique() if t]
    on_floor, prev_end = {}, {t: set() for t in teams_}
    for period, pdf in df.groupby("period", sort=True):
        subs_in = {t: [] for t in teams_}  # (row position, player) in order
        first_seen = {}
        for pos, r in enumerate(pdf.itertuples()):
            t = r.teamTricode
            if not t:
                continue
            if r.actionType == "Substitution":
                m = re.match(r"SUB:\s*(.+?)\s+FOR\s+(.+)", str(r.description))
                inc = names[t].get(norm(m.group(1))) if m else None
                if r.personId and r.personId < TEAM_ID_MIN:
                    first_seen.setdefault((t, int(r.personId)), "out")
                if inc:
                    first_seen.setdefault((t, int(inc)), "in")
            elif 0 < r.personId < TEAM_ID_MIN:
                first_seen.setdefault((t, int(r.personId)), "play")
        cur = {}
        for t in teams_:
            starters = [p for (tt, p), how in first_seen.items() if tt == t and how != "in"]
            if len(starters) < 5:  # someone played the whole period without a recorded action
                starters += [p for p in prev_end[t] if p not in starters and first_seen.get((t, p)) != "in"]
            cur[t] = set(starters[:5])
        for r in pdf.itertuples():
            t = r.teamTricode
            if t and r.actionType == "Substitution":
                m = re.match(r"SUB:\s*(.+?)\s+FOR\s+(.+)", str(r.description))
                inc = names[t].get(norm(m.group(1))) if m else None
                if r.personId in cur[t]:
                    cur[t].discard(int(r.personId))
                if inc:
                    cur[t].add(int(inc))
            elif t and 0 < r.personId < TEAM_ID_MIN and r.personId not in cur[t] and len(cur[t]) < 5:
                cur[t].add(int(r.personId))  # a player we missed shows up by making a play
            on_floor[r.Index] = {tt: frozenset(s) for tt, s in cur.items()}
        prev_end = {t: set(s) for t, s in cur.items()}
    return on_floor


def load_shotchart(ctx, season):
    """{gameId: DataFrame indexed by GAME_EVENT_ID} from data/context/shotchart/<season>.parquet, or {}."""
    p = ctx / "shotchart" / f"{season}.parquet"
    if not p.exists():
        return {}
    sc = pd.read_parquet(p, columns=["GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "PERIOD", "LOC_X", "LOC_Y",
                                     "SHOT_DISTANCE", "SHOT_ZONE_BASIC"])
    sc["GAME_ID"] = sc["GAME_ID"].astype(str).str.zfill(10)
    sc = sc.drop_duplicates(["GAME_ID", "GAME_EVENT_ID"])
    return {gid: g.set_index("GAME_EVENT_ID") for gid, g in sc.groupby("GAME_ID")}


def primary_defender(m, shooter, def5):
    if m is None or not def5:
        return None, 0.0
    sub = m[(m["personIdOff"] == shooter) & (m["personIdDef"].isin(list(def5)))]
    if not len(sub):
        return None, 0.0
    w = (sub["matchupFieldGoalsAttempted"] + 0.1 * sub["partialPossessions"]).to_numpy(float)
    if w.sum() <= 0:
        return None, 0.0
    k = int(np.argmax(w))
    return int(sub["personIdDef"].iloc[k]), float(w[k] / w.sum())


def process_game(df, ctx, coaches, sc=None):
    """sc: this game's shot-chart rows indexed by GAME_EVENT_ID (load_shotchart), used to fill missing coordinates."""
    df = df.copy()
    df["actionNumber"] = pd.to_numeric(df["actionNumber"], errors="coerce")
    df["period"] = pd.to_numeric(df["period"], errors="coerce").fillna(1).astype(int)
    # Late scorekeeper entries get high actionNumbers, so order by game clock first.
    df["_clock"] = df["clock"].map(parse_clock)
    df = df.sort_values(["period", "_clock", "actionNumber"], ascending=[True, False, True]).reset_index(drop=True)
    df["personId"] = pd.to_numeric(df["personId"], errors="coerce").fillna(0).astype(int)
    df["teamId"] = pd.to_numeric(df["teamId"], errors="coerce").fillna(0).astype(int)
    df["period"] = pd.to_numeric(df["period"], errors="coerce").fillna(1).astype(int)
    for col in ("scoreHome", "scoreAway"):
        df[col] = pd.to_numeric(df[col], errors="coerce").ffill().fillna(0)
    df["teamTricode"] = df["teamTricode"].fillna("").astype(str).replace("None", "")
    for col in ("actionType", "subType", "location"):  # 11 games of 2000-01 pad these with spaces
        if col in df:
            df[col] = df[col].fillna("").astype(str).str.strip()
    gid, season = str(df["gameId"].iloc[0]), str(df["season"].iloc[0])

    m = load_matchups(ctx, season, gid)
    names = name_map(df, m)
    floor = lineups(df, names)
    side = df[df["location"].isin(["h", "v"]) & (df["teamId"] > 0)].drop_duplicates("teamId")
    team_of = dict(zip(side["location"], zip(side["teamId"], side["teamTricode"])))
    if set(team_of) != {"h", "v"}:
        return []

    out, prev_h, prev_a = [], 0.0, 0.0
    for r in df.itertuples():
        if r.actionType in ("Made Shot", "Missed Shot") and r.location in ("h", "v"):
            off_id, off_t = team_of[r.location]
            def_id, def_t = team_of["v" if r.location == "h" else "h"]
            sub = str(r.subType).lower()
            desc = str(r.description)
            dist = float(pd.to_numeric(r.shotDistance, errors="coerce") or 0)
            x = float(pd.to_numeric(r.xLegacy, errors="coerce") or 0)
            y = float(pd.to_numeric(r.yLegacy, errors="coerce") or 0)
            is3 = str(r.shotValue).startswith("3") or "3PT" in desc
            # Pre-2010 feeds leave a quarter of shots at x = y = distance = 0: take NBA.com's shot chart
            # (same coordinate system) when it has the same event by the same shooter.
            hit = None
            if sc is not None and pd.notna(r.actionNumber) and int(r.actionNumber) in sc.index:
                h = sc.loc[int(r.actionNumber)]
                if int(h["PLAYER_ID"]) == int(r.personId):
                    hit = h
            no_xy = x == 0 and y == 0 and dist == 0
            if no_xy and hit is not None:
                x, y, dist = float(hit["LOC_X"]), float(hit["LOC_Y"]), float(hit["SHOT_DISTANCE"])
            elif dist == 0 and (x or y):
                dist = float(round(np.hypot(x, y) / 10))  # old 3PT rows: coordinates but distance 0
            imputed = int(no_xy and hit is None)
            lu = floor.get(r.Index, {})
            off5 = sorted(lu.get(off_t, frozenset()))
            def5 = sorted(lu.get(def_t, frozenset()))
            dfd, conf = primary_defender(m, r.personId, def5)
            margin = (prev_h - prev_a) if r.location == "h" else (prev_a - prev_h)
            p = int(r.period)
            out.append({
                "gameId": gid, "season": season, "period": p,
                "sec": seconds_remaining(p, parse_clock(r.clock)), "margin": margin, "is_home": int(r.location == "h"),
                "shooterId": int(r.personId), "shooter": str(r.playerName),
                "offTeamId": int(off_id), "offTeam": off_t, "defTeamId": int(def_id), "defTeam": def_t,
                "offCoachId": coaches.get((season, off_id), 0), "defCoachId": coaches.get((season, def_id), 0),
                "zone": impute_zone(sub, is3) if imputed else shot_zone(sub, desc, dist, x, y, is3),
                "style": shot_style(sub), "subType": str(r.subType),
                "made": int(r.actionType == "Made Shot"), "assisted": int("AST)" in desc),
                "is3": int(is3), "dist": dist, "x": x, "y": y,
                "off5": off5, "def5": def5, "defenderId": dfd or 0, "defenderConf": conf,
                "coord_imputed": imputed, "sc_match": int(hit is not None),
            })
        prev_h, prev_a = r.scoreHome, r.scoreAway
    return out


def build_season(job):
    """Worker: (season, files, context dir, coaches) -> (rows, skipped game messages)."""
    season, files, ctx, coaches = job
    charts = load_shotchart(ctx, season)
    rows, skipped = [], []
    for f in files:
        try:
            rows += process_game(pd.read_parquet(f), ctx, coaches, charts.get(f.stem))
        except Exception as e:
            skipped.append(f"{f.name}: {e}")
    print(f"  {season}: {len(files)} games, {len(rows):,} shots, shot chart {'yes' if charts else 'no'}", flush=True)
    return rows, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--context", default="data/context")
    ap.add_argument("--out", default="data/processed/shots.parquet")
    ap.add_argument("--limit", type=int, default=0, help="only process this many games (for testing)")
    ap.add_argument("--seasons", nargs="+", help="only these season folders, e.g. 2021-22 2022-23 (default: all)")
    ap.add_argument("--reuse", help="copy the seasons this shots parquet already has instead of rebuilding them")
    ap.add_argument("--workers", type=int, default=1, help="seasons built in parallel")
    args = ap.parse_args()
    ctx = Path(args.context)

    coaches = {}
    cpath = ctx / "coaches.parquet"
    if cpath.exists():
        c = pd.read_parquet(cpath)
        coaches = {(s, int(t)): int(cid) for s, t, cid in zip(c["season"], c["teamId"], c["coachId"])}

    files = sorted(Path(args.raw).rglob("*.parquet"))
    if args.seasons:
        files = [f for f in files if f.parent.name in args.seasons]
    reused = None
    if args.reuse:
        reused = pd.read_parquet(args.reuse)
        have = set(reused["season"].unique())
        if args.seasons:
            reused = reused[reused["season"].isin(args.seasons)]
        files = [f for f in files if f.parent.name not in have]
        reused = reused.assign(coord_imputed=0, sc_match=0)
        print(f"reusing {len(reused):,} shots ({len(have)} seasons) from {args.reuse}")
    if args.limit:
        files = files[:args.limit]
    by_season = defaultdict(list)
    for f in files:
        by_season[f.parent.name].append(f)
    jobs = [(s, fs, ctx, coaches) for s, fs in sorted(by_season.items())]
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.workers) as ex:
            results = list(ex.map(build_season, jobs))
    else:
        results = [build_season(j) for j in jobs]
    rows, skipped = [], 0
    for r, sk in results:
        rows += r
        skipped += len(sk)
        for m in sk:
            print(f"skip {m}")
    shots = pd.DataFrame(rows)
    if reused is not None:
        shots = pd.concat([shots, reused], ignore_index=True) if len(shots) else reused
        shots = shots.sort_values(["season", "gameId"], kind="mergesort").reset_index(drop=True)
    for c in ("coord_imputed", "sc_match"):
        shots[c] = shots[c].astype("int8")
    shots = fill_jumpers(shots)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shots.to_parquet(args.out)

    full = (shots["off5"].str.len() == 5) & (shots["def5"].str.len() == 5)
    print(f"{len(shots):,} shots from {shots['gameId'].nunique()} games ({skipped} skipped)")
    print(f"full lineups on {full.mean():.1%} of shots | shooter in own lineup "
          f"{np.mean([s in o for s, o in zip(shots['shooterId'], shots['off5'])]):.1%}")
    print(f"defender estimated on {(shots['defenderId'] > 0).mean():.1%} of shots | coach known on "
          f"{(shots['offCoachId'] > 0).mean():.1%}")
    print(shots["zone"].value_counts(normalize=True).round(3).to_dict())
    print(shots["style"].value_counts(normalize=True).round(3).to_dict())
    built = shots[shots["season"].isin(by_season)]
    per = None
    if len(built):
        per = built.groupby("season").agg(shots=("gameId", "size"), shotchart_match=("sc_match", "mean"),
                                          coord_imputed=("coord_imputed", "mean")).round(4)
        print(per.to_string())
    meta = {"zones": ZONES, "styles": STYLES}
    if per is not None:
        meta["coverage"] = per.reset_index().to_dict("records")
    json.dump(meta, open(Path(args.out).with_suffix(".json"), "w"))


if __name__ == "__main__":
    main()
