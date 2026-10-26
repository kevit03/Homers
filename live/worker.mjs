/* The Live tab's model thread: fetches a game's play-by-play from ESPN, turns it into Tempo's tokens and runs Tempo
   on the plays it hasn't seen yet, so polling nine games never stalls the page.

   in:  { id }                       ESPN event id to (re)load
   out: { id, game, wp, call } | { id, error } */
import { loadTempo, espnGame, tempoCall } from "./tempo.mjs";

const ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba";
const ready = (async () => {
  const [meta, bin] = await Promise.all([fetch(new URL("tempo.json", import.meta.url)).then(r => r.json()),
                                         fetch(new URL("tempo.bin", import.meta.url)).then(r => r.arrayBuffer())]);
  return loadTempo(meta, new Uint8Array(bin));
})();
const sessions = new Map();
let queue = Promise.resolve();  // one game at a time: the model's scratch buffers are shared

self.onmessage = ({ data: { id } }) => {
  queue = queue.then(async () => {
    try {
      const T = await ready;
      const r = await fetch(`${ESPN}/summary?event=${id}`, { cache: "no-store" });
      if (!r.ok) throw new Error(`ESPN answered ${r.status}`);
      const g = espnGame(await r.json());
      const s = (sessions.get(id) || sessions.set(id, T.session()).get(id)).sync(g.events);
      const ev = g.events.map(({ tok, sec, per, hs, as, text, clock }) => ({ tok, sec, per, hs, as, text, clock }));
      self.postMessage({ id, game: { ...g, events: ev }, wp: s.wp.slice(), call: tempoCall(s.wp, g.events) });
    } catch (e) {
      self.postMessage({ id, error: String(e.message || e) });
    }
  });
};
