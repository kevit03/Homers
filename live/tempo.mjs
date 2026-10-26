/* Tempo in JavaScript, for games in progress: the Live tab runs it in the browser and api/record.js on Vercel.

   loadTempo(meta, bytes)   weights from export/export_live.py (tempo.json + tempo.bin, float16)
   espnGame(summary)        ESPN's play-by-play (site.api.espn.com .../summary?event=<id>) as Tempo's tokens, the way
                            models/tokenize_pbp.process_game reads NBA.com's PlayByPlayV3 for training
   tempoCall(wp, ev)        Tempo's call: the first play after tip-off that puts either team at CALL_AT or better

   The forward pass matches models/model.py (pre-LayerNorm GPT, exact GELU, 4 game-state features) one position at a
   time with a key/value cache, so a poll only runs the plays that are new since the last one. */

export const CALL_AT = 0.8;  // 2025-26 backtest, first crossing: right 80% of the time, with a median 28 minutes left (90%: 89%, 17 min)

function f16(bytes) {  // float16 -> Float32Array
  const u = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 1), out = new Float32Array(u.length);
  for (let i = 0; i < u.length; i++) {
    const h = u[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    out[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return out;
}

function erf(x) {  // Abramowitz & Stegun 7.1.26 would be 1.5e-7 off; this series/continued fraction is float32-exact
  const ax = Math.abs(x);
  if (ax > 6) return Math.sign(x);
  if (ax < 2.5) {  // Maclaurin series
    let sum = x, term = x, x2 = x * x;
    for (let n = 1; n < 60; n++) {
      term *= -x2 / n;
      const t = term / (2 * n + 1);
      sum += t;
      if (Math.abs(t) < 1e-17) break;
    }
    return sum * 2 / Math.sqrt(Math.PI);
  }
  let f = 0;  // erfc continued fraction for the tails
  for (let n = 60; n >= 1; n--) f = n / 2 / (ax + f);
  const erfc = Math.exp(-ax * ax) / Math.sqrt(Math.PI) / (ax + f);
  return Math.sign(x) * (1 - erfc);
}
const gelu = x => 0.5 * x * (1 + erf(x / Math.SQRT2));

function linear(W, b, x, nOut, nIn, out) {  // torch layout: W is [out, in]
  for (let o = 0; o < nOut; o++) {
    let s = b ? b[o] : 0;
    const r = o * nIn;
    for (let i = 0; i < nIn; i++) s += W[r + i] * x[i];
    out[o] = s;
  }
  return out;
}
function layerNorm(x, g, b, out) {
  const n = x.length;
  let mu = 0, v = 0;
  for (let i = 0; i < n; i++) mu += x[i];
  mu /= n;
  for (let i = 0; i < n; i++) v += (x[i] - mu) ** 2;
  const inv = 1 / Math.sqrt(v / n + 1e-5);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mu) * inv * g[i] + b[i];
  return out;
}

/* models/model.make_features with no Elo: time left, margin, period, margin for the time left */
export function features(sec, diff, per) {
  const minutes = Math.max(sec, 0) / 60;
  return [sec / 2880, diff / 20, Math.min(per, 6) / 4, diff / Math.sqrt(minutes + 1) / 5];
}

export function loadTempo(meta, bytes) {
  const all = f16(bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes)), w = {};
  for (const t of meta.tensors) w[t.name] = all.subarray(t.offset, t.offset + t.shape.reduce((a, b) => a * b, 1));
  const { n_layer: L, n_head: NH, n_embd: C, block_size: B } = meta.config, HD = C / NH, scale = 1 / Math.sqrt(HD);
  const blk = [...Array(L)].map((_, l) => {
    const p = k => w[`blocks.${l}.${k}`];
    return { ln1g: p("ln1.weight"), ln1b: p("ln1.bias"), ln2g: p("ln2.weight"), ln2b: p("ln2.bias"),
             qkvW: p("attn.qkv.weight"), qkvB: p("attn.qkv.bias"), projW: p("attn.proj.weight"), projB: p("attn.proj.bias"),
             fcW: p("mlp.0.weight"), fcB: p("mlp.0.bias"), outW: p("mlp.2.weight"), outB: p("mlp.2.bias") };
  });
  const x = new Float32Array(C), h = new Float32Array(C), qkv = new Float32Array(3 * C), y = new Float32Array(C),
        o = new Float32Array(C), m = new Float32Array(4 * C), att = new Float32Array(B);

  /* one game: push() events in order; wp[i] is the home team's win chance after event i */
  function session() {
    const K = blk.map(() => new Float32Array(B * C)), V = blk.map(() => new Float32Array(B * C));
    const s = { n: 0, wp: [], keys: [] };
    s.step = (tok, f) => {
      const t = s.n;
      if (t >= B) return s.wp[s.wp.length - 1];  // past the context the model trained on: hold the last reading
      const te = w["tok_emb.weight"].subarray(tok * C, tok * C + C), pe = w["pos_emb.weight"].subarray(t * C, t * C + C);
      linear(w["feat_proj.weight"], w["feat_proj.bias"], f, C, f.length, x);
      for (let i = 0; i < C; i++) x[i] += te[i] + pe[i];
      for (let l = 0; l < L; l++) {
        const b = blk[l];
        linear(b.qkvW, b.qkvB, layerNorm(x, b.ln1g, b.ln1b, h), 3 * C, C, qkv);
        K[l].set(qkv.subarray(C, 2 * C), t * C);
        V[l].set(qkv.subarray(2 * C, 3 * C), t * C);
        for (let hd = 0; hd < NH; hd++) {
          const q0 = hd * HD;
          let mx = -Infinity;
          for (let j = 0; j <= t; j++) {
            let d = 0;
            const k0 = j * C + q0;
            for (let i = 0; i < HD; i++) d += qkv[q0 + i] * K[l][k0 + i];
            att[j] = d * scale;
            if (att[j] > mx) mx = att[j];
          }
          let z = 0;
          for (let j = 0; j <= t; j++) z += (att[j] = Math.exp(att[j] - mx));
          for (let i = 0; i < HD; i++) {
            let acc = 0;
            for (let j = 0; j <= t; j++) acc += att[j] * V[l][j * C + q0 + i];
            y[q0 + i] = acc / z;
          }
        }
        linear(b.projW, b.projB, y, C, C, o);
        for (let i = 0; i < C; i++) x[i] += o[i];
        linear(b.fcW, b.fcB, layerNorm(x, b.ln2g, b.ln2b, h), 4 * C, C, m);
        for (let i = 0; i < 4 * C; i++) m[i] = gelu(m[i]);
        linear(b.outW, b.outB, m, C, 4 * C, o);
        for (let i = 0; i < C; i++) x[i] += o[i];
      }
      layerNorm(x, w["ln_f.weight"], w["ln_f.bias"], h);
      let z = w["wp_head.bias"][0];
      for (let i = 0; i < C; i++) z += w["wp_head.weight"][i] * h[i];
      s.n++;
      return 1 / (1 + Math.exp(-z));
    };
    /* events: [{tok, sec, diff, per}]; appended after what's already in */
    s.push = events => {
      for (const e of events) { s.wp.push(s.step(e.tok, features(e.sec, e.diff, e.per))); s.keys.push(evKey(e)); }
      return s;
    };
    /* the feed can fix a play after the fact: keep the longest matching prefix and run the rest again */
    s.sync = events => {
      let k = 0;
      while (k < s.keys.length && k < events.length && s.keys[k] === evKey(events[k])) k++;
      s.n = Math.min(s.n, k); s.wp.length = k; s.keys.length = k;
      return s.push(events.slice(k));
    };
    return s;
  }
  return { session, vocab: meta.vocab, config: meta.config };
}
const evKey = e => `${e.tok}|${e.sec}|${e.diff}|${e.per}`;

/* ---------- ESPN play-by-play -> Tempo tokens ---------- */
const VOCAB = ["<pad>", "<bos>", "<eos>", ...["H", "A"].flatMap(s => ["2PT_MAKE", "2PT_MISS", "3PT_MAKE", "3PT_MISS", "FT_MAKE", "FT_MISS",
  "OREB", "DREB", "TOV", "FOUL", "VIOLATION", "TIMEOUT", "JUMPBALL"].map(e => `${s}_${e}`)), "PERIOD_START", "PERIOD_END"];
export const STOI = Object.fromEntries(VOCAB.map((t, i) => [t, i]));

// ESPN's own abbreviations where they differ from NBA.com's tricodes (which the rest of the site uses)
const TRI = { GS: "GSW", NY: "NYK", SA: "SAS", NO: "NOP", UTAH: "UTA", WSH: "WAS", PHO: "PHX" };
export const nbaTri = a => TRI[a] || a;

function parseClock(v) {  // "11:40", "45.3", "0.0"
  const s = String(v ?? "0"), m = s.match(/^(\d+):(\d+(?:\.\d+)?)$/);
  return m ? +m[1] * 60 + +m[2] : parseFloat(s) || 0;
}
const secondsLeft = (per, clock) => (per <= 4 ? (4 - per) * 720 + clock : clock);

/* what tokenize_pbp.classify would call this play, given which side last missed */
function classify(p, side, lastMiss) {
  const t = String(p.type?.text || "").replace(/\s+/g, " ").trim(), tl = t.toLowerCase();
  if (tl === "end period") return "PERIOD_END";
  if (!side) return null;
  if (p.shootingPlay) {
    if (tl.startsWith("free throw")) return `${side}_FT_${p.scoringPlay ? "MAKE" : "MISS"}`;
    if (!p.scoringPlay && tl.includes("heave")) return null;  // NBA.com files missed heaves apart from missed shots
    const three = p.pointsAttempted === 3 || /three point/i.test(p.text || "");
    return `${side}_${three ? "3PT" : "2PT"}_${p.scoringPlay ? "MAKE" : "MISS"}`;
  }
  if (tl.includes("rebound")) return lastMiss ? `${side}_${side === lastMiss ? "OREB" : "DREB"}` : null;
  if (tl.includes("turnover") || /^(traveling|double dribble|discontinued dribble|palming|backcourt)/.test(tl)) return `${side}_TOV`;
  if (tl.includes("foul") || tl.includes("technical") || tl === "offensive charge") return `${side}_FOUL`;  // NBA.com: technicals are fouls
  if (tl.includes("violation") || tl.includes("goaltending") || tl.startsWith("delay") || tl === "kicked ball" || /^(double )?lane$/.test(tl)) return `${side}_VIOLATION`;
  if (tl.includes("timeout") || tl === "challenge") return `${side}_TIMEOUT`;  // NBA.com: a coach's challenge is a timeout
  if (tl === "jumpball" || tl === "jump ball") return "H_JUMPBALL";  // NBA.com puts every jump ball on the home side
  return null;  // substitutions, reviews, ejections, end of game...
}

/* summary: ESPN's summary JSON. Returns the teams, the state of play and Tempo's input, one event per token:
   {tok, sec, diff, per, hs, as, text, clock} after <bos>, ending in <eos> once the game is final. */
export function espnGame(summary) {
  const comp = summary.header.competitions[0], st = comp.status.type;
  const team = ha => {
    const c = comp.competitors.find(x => x.homeAway === ha);
    return { id: String(c.team.id), abbr: c.team.abbreviation, tri: nbaTri(c.team.abbreviation), name: c.team.displayName,
             nick: c.team.name, color: c.team.color, alt: c.team.alternateColor, score: +(c.score || 0), winner: !!c.winner };
  };
  const home = team("home"), away = team("away");
  // game clock order, the feed's own order breaking ties (tokenize_pbp.game_order)
  const plays = (summary.plays || []).map((p, i) => ({ p, i, per: +(p.period?.number || 1), clk: parseClock(p.clock?.displayValue) }))
    .sort((a, b) => a.per - b.per || b.clk - a.clk || a.i - b.i);
  const ev = [{ tok: STOI["<bos>"], sec: 2880, diff: 0, per: 1, hs: 0, as: 0, text: "Tip-off", clock: "12:00" }];
  let hs = 0, as = 0, lastMiss = null, curPer = 0;
  const add = (tok, per, clk, text, clock) => ev.push({ tok: STOI[tok], sec: secondsLeft(per, clk), diff: hs - as, per, hs, as, text, clock });
  for (const { p, per, clk } of plays) {
    hs = Math.max(hs, +p.homeScore || 0); as = Math.max(as, +p.awayScore || 0);  // a team's score never goes down
    if (per !== curPer) {  // ESPN has no start-of-period play; NBA.com's feed (and so training) does
      curPer = per;
      add("PERIOD_START", per, per <= 4 ? 720 : 300, `Start of ${per <= 4 ? `Q${per}` : `OT${per - 4 > 1 ? per - 4 : ""}`}`, per <= 4 ? "12:00" : "5:00");
    }
    const tid = p.team?.id ? String(p.team.id) : null, side = tid === home.id ? "H" : tid === away.id ? "A" : null;
    const tok = classify(p, side, lastMiss);
    if (!tok) continue;
    if (/_(2PT|3PT|FT)_MISS$/.test(tok)) lastMiss = tok[0];
    add(tok, per, clk, p.text || p.type?.text || "", p.clock?.displayValue || "");
  }
  const final = st.completed && st.state === "post";
  if (final) {
    hs = Math.max(hs, home.score); as = Math.max(as, away.score);
    ev.push({ tok: STOI["<eos>"], sec: 0, diff: hs - as, per: ev[ev.length - 1].per, hs, as, text: "Final", clock: "0.0" });
  }
  return { id: String(summary.header.id), date: comp.date, home, away, events: ev, final, state: st.state, detail: st.shortDetail,
           period: comp.status.period, clock: comp.status.displayClock, homeWin: final ? hs > as : null };
}

/* Tempo's call: the first event after tip-off where either team reaches CALL_AT. Locked from then on. */
export function tempoCall(wp, ev, at = CALL_AT) {
  for (let i = 1; i < wp.length; i++) {
    if (wp[i] >= at || wp[i] <= 1 - at) {
      const e = ev[i];
      return { i, home: wp[i] >= 0.5, p: Math.max(wp[i], 1 - wp[i]), sec: e.sec, per: e.per, clock: e.clock, hs: e.hs, as: e.as };
    }
  }
  return null;
}

/* Minutes of the game still to play when the call came (regulation only; an OT call reads 0) */
export const minutesLeft = c => (c.per <= 4 ? c.sec / 60 : 0);
