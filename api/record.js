/* GET /api/record: Tempo's record this season (live/record.mjs). Starts from the snapshot shipped with the deploy
   (live/record.json) and replays the games finished since; Vercel's CDN keeps the answer for a few minutes. */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { loadTempo } from "../live/tempo.mjs";
import { updateRecord } from "../live/record.mjs";

const dir = join(process.cwd(), "live");
let T = null, memo = null;  // kept between requests while the function stays warm

export default async function handler(req, res) {
  try {
    T ??= loadTempo(JSON.parse(readFileSync(join(dir, "tempo.json"))), readFileSync(join(dir, "tempo.bin")));
    memo ??= JSON.parse(readFileSync(join(dir, "record.json")));
    memo = await updateRecord(T, memo, { budget: 80 });
    res.setHeader("Cache-Control", "public, max-age=60, s-maxage=300, stale-while-revalidate=3600");
    res.status(200).json(memo);
  } catch (e) {
    res.setHeader("Cache-Control", "no-store");
    res.status(502).json({ error: String(e.message || e) });
  }
}
