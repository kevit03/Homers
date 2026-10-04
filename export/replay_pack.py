"""Compact formats for Game replay games. No torch here, so build_site.py can import it.

Three formats, all read by unpackGame / rpSeason in dashboard_template.html:

v1  (old) one dict per game of fixed-width base-62 strings (b62 in export_dashboard): ~28 chars an event.
v2  one dict per game, {"v": 2, ...}: what pack_game writes into the page and into games/<season>.js.
      tok  one char per token (ALPHA)
      sec  0.1 s left, as deltas from the previous event (start 2880.0 s), zigzag varints
      pm   Tempo's home win chance in 1/1000, deltas from the previous event (start 0.5), zigzag varints
      ti   the top-5 next-token guesses, five chars an event
      tp   their probabilities in 1/100, five varints an event
      per / diff / tot are derived on the page from the tokens (PERIOD_START, scoring plays). They are only written,
      as zigzag varint deltas, for a game where the derivation is wrong. "nt": 1 means the game has no running total.
      pb (the logistic baseline) is not stored: the page computes it from DATA.baseline, sec and diff.
v3  one dict per season, {"v": 3, ...}: what build_site.py ships to the hosted site (pack_season). The v2 columns of
      every game end to end, and the top-5 guesses range-coded with adaptive context models (top: base64), which
      the page decodes for the whole season the first time a game of it is opened.
"""
import base64
import json

import numpy as np

ALPHA = "".join(chr(c) for c in range(32, 127) if chr(c) not in '"\\<')  # 92 chars, JSON- and </script>-safe
AIDX = {c: i for i, c in enumerate(ALPHA)}
HALF = 46  # varint digits: 0..45 end a number, 46..91 carry 46 more values each (little-endian)
PERIOD_START = 29
POINTS = {3: 2, 5: 3, 7: 1, 16: -2, 18: -3, 20: -1}  # H_2PT/3PT/FT_MAKE, A_2PT/3PT/FT_MAKE
COLS = ("tok", "sec", "pm", "ti", "tp", "per", "diff", "tot", "pb", "top_i", "top_p", "nt")


def vint(values, signed=False):
    out = []
    for v in values:
        v = int(v)
        if signed:
            v = 2 * v if v >= 0 else -2 * v - 1
        while v >= HALF:
            out.append(ALPHA[HALF + v % HALF])
            v //= HALF
        out.append(ALPHA[v])
    return "".join(out)


def vdec(s, n=None, signed=False, pos=0):
    """Read n varints (all of them if n is None) from s at pos; returns (values, next pos)."""
    out = []
    while (n is None and pos < len(s)) or (n is not None and len(out) < n):
        v, mul = 0, 1
        while True:
            c = AIDX[s[pos]]
            pos += 1
            if c < HALF:
                v += c * mul
                break
            v += (c - HALF) * mul
            mul *= HALF
        out.append((v >> 1) ^ -(v & 1) if signed else v)
    return out, pos


def derive(tok):
    """per, diff and tot as the page derives them from the tokens."""
    tok = np.asarray(tok, dtype=np.int64)
    per = np.maximum(np.cumsum(tok == PERIOD_START), 1)
    pts = np.zeros(len(tok), dtype=np.int64)
    for t, p in POINTS.items():
        pts[tok == t] = p
    return per, np.cumsum(pts), np.cumsum(np.abs(pts))


def pack_game_v2(g):
    tok = np.asarray(g["tok"], dtype=np.int64).ravel()
    assert tok.min(initial=0) >= 0 and tok.max(initial=0) < HALF, "vocab too big for one char a token"
    sec10 = np.rint(np.asarray(g["sec"], dtype=np.float64) * 10).astype(np.int64)
    pm = np.rint(np.asarray(g["pm"], dtype=np.float64) * 1000).astype(np.int64)
    ti = np.asarray(g["top_i"], dtype=np.int64).reshape(len(tok), -1)
    q = np.rint(np.asarray(g["top_p"], dtype=np.float64).reshape(len(tok), -1) * 100).astype(np.int64)
    out = {"v": 2, "id": g["id"], "season": g["season"], "home_win": g["home_win"],
           "tok": "".join(ALPHA[t] for t in tok),
           "sec": vint(-np.diff(np.r_[28800, sec10]), True),
           "pm": vint(np.diff(np.r_[500, pm]), True),
           "ti": "".join(ALPHA[t] for t in ti.ravel()), "tp": vint(q.ravel())}
    per, diff, tot = derive(tok)
    for k, want, got in (("per", g["per"], per), ("diff", g["diff"], diff), ("tot", g.get("tot"), tot)):
        if want is None:
            continue
        want = np.rint(np.asarray(want, dtype=np.float64)).astype(np.int64)
        if not np.array_equal(want, got):
            out[k] = vint(np.diff(np.r_[0, want]), True)
    if g.get("tot") is None:
        out["nt"] = 1
    return out


def _b62(s, w, scale=1, off=0):
    B = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    d = np.array([B.index(c) for c in s], dtype=np.int64).reshape(-1, w)
    v = np.zeros(len(d), dtype=np.int64)
    for j in range(w):
        v = v * 62 + d[:, j]
    return (v - off) / scale


def unpack_game(pg):
    """Any per-game format (v1 or v2) back to arrays: tok, per, sec, diff, tot (or None), pm, top_i (n, 5), top_p (n, 5)."""
    if pg.get("v") == 2:
        tok = np.array([AIDX[c] for c in pg["tok"]], dtype=np.int64)
        n = len(tok)
        sec = 2880 - np.cumsum(vdec(pg["sec"], n, True)[0]) / 10
        pm = (500 + np.cumsum(vdec(pg["pm"], n, True)[0])) / 1000
        ti = np.array([AIDX[c] for c in pg["ti"]], dtype=np.int64).reshape(n, -1)
        tp = np.array(vdec(pg["tp"])[0], dtype=np.int64).reshape(n, -1) / 100
        per, diff, tot = derive(tok)
        per, diff, tot = (np.cumsum(vdec(pg[k], n, True)[0]) if pg.get(k) else d for k, d in (("per", per), ("diff", diff), ("tot", tot)))
        return {"tok": tok, "per": per, "sec": sec, "diff": diff, "tot": None if pg.get("nt") else tot, "pm": pm, "top_i": ti, "top_p": tp}
    tok = _b62(pg["tok"], 1).astype(np.int64)
    return {"tok": tok, "per": _b62(pg["per"], 1).astype(np.int64), "sec": _b62(pg["sec"], 3, 10),
            "diff": _b62(pg["diff"], 2, 1, 100).astype(np.int64), "tot": _b62(pg["tot"], 2).astype(np.int64) if pg.get("tot") else None,
            "pm": _b62(pg["pm"], 2, 1000), "top_i": _b62(pg["top_i"], 1).astype(np.int64).reshape(len(tok), -1),
            "top_p": _b62(pg["top_p"], 2, 1000).reshape(len(tok), -1)}


# ---- v3: range-coded top-5 guesses -------------------------------------------------------------------------------
# A carryless range coder (Subbotin) with adaptive frequency tables, one per context. dashboard_template.html's
# rpTop mirrors it step for step: change one and you must change the other.
RC_TOP, RC_BOT, RC_INC, RC_LIM = 1 << 24, 1 << 16, 24, 1 << 15


class _Enc:
    def __init__(self):
        self.low, self.range, self.out, self.tabs = 0, 0xFFFFFFFF, bytearray(), {}

    def put(self, key, n, sym):
        f = self.tabs.get(key)
        if f is None:
            f = self.tabs[key] = [1] * n + [n]  # the last entry is the total
        assert 0 <= sym < n, (key, n, sym)
        r = self.range // f[n]
        self.low += sum(f[:sym]) * r
        self.range = r * f[sym]
        while True:
            if (self.low >> 24) == ((self.low + self.range) >> 24):
                pass
            elif self.range < RC_BOT:
                self.range = (-self.low) % RC_BOT
            else:
                break
            self.out.append(self.low >> 24)
            self.low = (self.low % RC_TOP) << 8
            self.range = (self.range % RC_TOP) << 8
        f[sym] += RC_INC
        f[n] += RC_INC
        if f[n] > RC_LIM:
            t = 0
            for i in range(n):
                f[i] = (f[i] + 1) >> 1
                t += f[i]
            f[n] = t

    def finish(self):
        for _ in range(4):
            self.out.append(self.low >> 24)
            self.low = (self.low % RC_TOP) << 8
        return bytes(self.out)


def _top_code(enc, ni, tok, ti, q):
    """The context model, event by event: each guess given the token and the guesses before it, then each
    probability given the one above it (they're sorted, so it can't be bigger) and the running sum."""
    for e in range(len(tok)):
        t, r = int(tok[e]), [int(x) for x in ti[e]]
        a, b, c, d = r[0], r[1], r[2], r[3]
        enc.put(8 * t, ni, a)
        enc.put(1 + 8 * (t + ni * a), ni, b)
        enc.put(2 + 8 * (t + ni * (a + ni * b)), ni, c)
        enc.put(3 + 8 * (t + ni * (a + ni * (b + ni * c))), ni, d)
        enc.put(4 + 8 * (b + ni * (c + ni * d)), ni, r[4])
        qe = [int(x) for x in q[e]]
        enc.put(5 + 8 * (t + ni * (a + ni * b)), 101, qe[0])
        s = qe[0]
        for k in range(1, 5):
            enc.put(6 + 8 * (k + 8 * (qe[k - 1] + 101 * (min(s >> 2, 25) if k > 1 else 0))), qe[k - 1] + 1, qe[k])
            s += qe[k]


def pack_season(games):
    """A season of per-game dicts (v1 or v2, plus the extra keys write_replay_seasons adds) as one v3 dict."""
    meta, tok, sec, pm, lens, tops = [], [], [], [], [], []
    for pg in games:
        u = unpack_game(pg)
        n = len(u["tok"])
        lens.append(n)
        q = np.rint(u["top_p"] * 100).astype(np.int64)
        assert u["top_i"].shape == (n, 5) and q.min(initial=0) >= 0 and q.max(initial=0) <= 100 \
            and (np.diff(q, axis=1) <= 0).all(), pg["id"]
        tops.append((u["tok"], u["top_i"], q))
        v2 = pg if pg.get("v") == 2 else pack_game_v2({**pg, **u, "top_p": q / 100})
        tok.append(v2["tok"]); sec.append(v2["sec"]); pm.append(v2["pm"])
        meta.append({k: v for k, v in pg.items() if k not in COLS and k != "v"}
                    | {k: v2[k] for k in ("per", "diff", "tot", "nt") if k in v2})
    ni = max(int(max(t.max(initial=0), i.max(initial=0))) for t, i, _ in tops) + 1 if tops else 1
    enc = _Enc()
    for t, i, q in tops:
        _top_code(enc, ni, t, i, q)
    return {"v": 3, "ni": ni, "len": lens, "games": meta, "tok": "".join(tok), "sec": "".join(sec), "pm": "".join(pm),
            "top": base64.b64encode(enc.finish()).decode()}


def read_games_js(path):
    """(season, games) from a games/<season>.js written by write_replay_seasons: HOMERS_GAMES("2016-17",[...]);"""
    text = open(path).read()
    head = text.index("(") + 1
    comma = text.index(",", head)
    return json.loads(text[head:comma]), json.loads(text[comma + 1:text.rindex(")")])


def season_file_v3(path):
    """games/<season>.js (any format) -> (season, the same file's text in v3, gzipped at level 9). For build_site.py."""
    import gzip
    season, games = read_games_js(path)
    if isinstance(games, list):
        games = pack_season(games)
    text = f"HOMERS_GAMES({json.dumps(season)},{json.dumps(games, separators=(',', ':'))});\n"
    return season, gzip.compress(text.encode(), 9, mtime=0)
