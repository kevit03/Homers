/* Refresh live/record.json, the record snapshot the site ships with (export/build_site.py runs this before a deploy).
     node live/build_record.mjs          # adds the games finished since the last snapshot
     node live/build_record.mjs --fresh  # replays the whole season again */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { loadTempo } from "./tempo.mjs";
import { updateRecord } from "./record.mjs";

const here = new URL(".", import.meta.url).pathname, file = here + "record.json";
const T = loadTempo(JSON.parse(readFileSync(here + "tempo.json")), readFileSync(here + "tempo.bin"));
const prev = !process.argv.includes("--fresh") && existsSync(file) ? JSON.parse(readFileSync(file)) : null;
const rec = await updateRecord(T, prev, { log: m => console.log(m) });
writeFileSync(file, JSON.stringify(rec) + "\n");
console.log(`${file}: ${rec.games.length} games through ${rec.through}, Tempo ${rec.right}-${rec.wrong}${rec.uncalled ? `, ${rec.uncalled} never called` : ""}${rec.behind ? `, ${rec.behind} not replayed yet` : ""}`);
