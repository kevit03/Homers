/* Tempo's record this season: every finished game, the call Tempo made in it (live/tempo.mjs tempoCall) and whether
   it was right. A call is a pure function of the play-by-play, so replaying a finished game gives the same call the Live tab
   showed while it was on.

   build_record.mjs writes a snapshot (live/record.json) at deploy time; api/record.js adds the games finished since then. */
import { espnGame, tempoCall, CALL_AT } from "./tempo.mjs";

const ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba";
export const SEASON_START = "20261020";  // 2026-27 opening night (ESPN's dates are US Eastern)
const COUNTED = new Set([2, 3, 5]);      // ESPN season types: regular season, playoffs, play-in (not preseason)

async function getJSON(url, tries = 3) {
  for (let k = 0; ; k++) {
    try {
      const r = await fetch(url, { signal: AbortSignal.timeout(15000) });
      if (!r.ok) throw new Error(`${r.status} from ${url}`);
      return await r.json();
    } catch (e) {
      if (k >= tries - 1) throw e;
      await new Promise(res => setTimeout(res, 500 * (k + 1)));
    }
  }
}

export const etDay = (d = new Date()) => {  // "20261026", today in New York
  const p = Object.fromEntries(new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit" })
    .formatToParts(d).map(x => [x.type, x.value]));
  return `${p.year}${p.month}${p.day}`;
};
function nextDay(day) {
  const d = new Date(Date.UTC(+day.slice(0, 4), +day.slice(4, 6) - 1, +day.slice(6, 8) + 1));
  return d.toISOString().slice(0, 10).replace(/-/g, "");
}

/* One finished game, replayed through Tempo. */
export async function gameResult(T, id) {
  const g = espnGame(await getJSON(`${ESPN}/summary?event=${id}`));
  if (!g.final) return null;
  const s = T.session().push(g.events), c = tempoCall(s.wp, g.events);
  return { id, date: g.date, h: g.home.tri, a: g.away.tri, hs: g.home.score, as: g.away.score, homeWin: g.homeWin,
           call: c && { home: c.home, p: +c.p.toFixed(3), sec: Math.round(c.sec), per: c.per, clock: c.clock, hs: c.hs, as: c.as },
           right: c ? c.home === g.homeWin : null };
}

async function pool(items, n, fn) {
  const out = [];
  let i = 0;
  await Promise.all([...Array(Math.min(n, items.length))].map(async () => { while (i < items.length) { const k = i++; out[k] = await fn(items[k]); } }));
  return out;
}

/* prev: an earlier record (or null). Scans the days after prev.through up to today and adds the games that finished.
   `budget` caps how many new games one call replays (each is a ~400 KB download plus ~0.3 s of model);
   the rest wait for the next call, and `behind` says how many. */
export async function updateRecord(T, prev, { today = etDay(), budget = Infinity, log = () => {} } = {}) {
  const games = new Map((prev?.games || []).map(g => [g.id, g]));
  let through = prev?.through || null, day = through ? nextDay(through) : SEASON_START, todo = [], openDay = null;
  for (; day <= today; day = nextDay(day)) {
    const sb = await getJSON(`${ESPN}/scoreboard?dates=${day}`);
    const evs = (sb.events || []).filter(e => COUNTED.has(e.season?.type));
    const done = evs.filter(e => e.status?.type?.completed);
    todo.push(...done.filter(e => !games.has(e.id)).map(e => e.id));
    if (done.length < evs.length || day === today) openDay ??= day;  // games still to finish: look at this day again next time
    if (openDay == null) through = day;
  }
  const run = todo.slice(0, budget);
  log(`${run.length} new finished games to replay${todo.length > run.length ? ` (${todo.length - run.length} left for later)` : ""}`);
  let missed = todo.length - run.length;
  for (const r of await pool(run, 6, id => gameResult(T, id).catch(e => (log(`  ${id}: ${e.message}`), null))))
    if (r) games.set(r.id, r); else missed++;
  if (missed) through = prev?.through || null;  // scan these days again next time; games already replayed aren't redone
  return summarize([...games.values()], through, missed);
}

export function summarize(list, through, behind = 0) {
  const games = list.sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : a.id < b.id ? -1 : 1));
  const called = games.filter(g => g.call);
  return { season: "2026-27", callAt: CALL_AT, through, behind, updated: new Date().toISOString(),
           right: called.filter(g => g.right).length, wrong: called.filter(g => !g.right).length, uncalled: games.length - called.length,
           // 2025-26, held out of training, the same rule (export: scratch backtest, runs/base): what to expect
           backtest: { season: "2025-26", right: 1054, wrong: 261, medianMinutesLeft: 27.9 },
           games };
}
