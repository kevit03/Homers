/* Player search test: every "find a player" box finds players the way people type them.

   For each search box with a dropdown (shot charts, matchup scorer and defender, player tendencies) it types into the
   real input and reads the dropdown. It checks known queries (no accents, initials, last name first, hyphens,
   apostrophes, extra spaces), that every player in the box's list can be found by his name typed without accents,
   and that the keyboard works. It also checks that the matchup boxes never offer the player already on the other
   side, and that the table filters (shot chart table, defenders, fantasy board) are accent-blind. Team names work in
   every box and filter: "knicks", "NYK", "new york", "New York Knicks" and "ny" list only Knicks, and "knicks brunson"
   puts Brunson first.

   Run it on a built dashboard.html served over http, in the browser console:

     const s = document.createElement("script"); s.src = "tests/player_search.js"; document.head.append(s);
     // then
     await testPlayerSearch()
*/
window.testPlayerSearch = async function testPlayerSearch() {
  const fails = [], passes = { cases: 0, everyone: 0, keys: 0, filters: 0, teams: 0 };
  const plain = s => s.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase();
  const type = (input, text) => { input.focus(); input.value = text; input.dispatchEvent(new Event("input", { bubbles: true })); };
  const pop = input => input.closest(".ps").querySelector(".ps-pop");
  const names = input => [...pop(input).querySelectorAll(".ps-opt .ps-name")].map(n => n.textContent);
  const key = (input, k) => input.dispatchEvent(new KeyboardEvent("keydown", { key: k, bubbles: true, cancelable: true }));
  const tick = () => new Promise(r => setTimeout(r));

  // each box and the names it should be able to find, read at the moment of testing
  const BOXES = [
    ["scFind", "shots", () => SC.players.map(p => p.n)],
    ["muOff", "matchups", () => MU_OFFS.filter(p => p.id !== mu.def.id).map(p => p.n)],
    ["muDef", "matchups", () => MU_DEFS.filter(p => p.id !== mu.off.id).map(p => p.n)],
    ["plFind", "players", () => SH.players.map(p => p.name)],
  ].filter(([id, tab]) => $(id) && tabHasContent(tab));
  // query -> the player who must come first (skipped for a box whose list doesn't have him)
  const CASES = [["jokic", "Nikola Jokić"], ["JOKIC", "Nikola Jokić"], ["doncic", "Luka Dončić"], ["porzingis", "Kristaps Porziņģis"],
    ["valanciunas", "Jonas Valančiūnas"], ["sga", "Shai Gilgeous-Alexander"], ["gilgeous alexander", "Shai Gilgeous-Alexander"],
    ["shai gil", "Shai Gilgeous-Alexander"], ["lebron j", "LeBron James"], ["james lebron", "LeBron James"], ["deaaron", "De'Aaron Fox"],
    ["de'aaron fox", "De'Aaron Fox"], ["pj washington", "P.J. Washington"], ["  giannis  ", "Giannis Antetokounmpo"]];

  for (const [id, tab, list] of BOXES) {
    showTab(tab);
    const input = $(id), p = pop(input);
    type(input, "");
    if (p.hidden || !p.querySelector(".ps-opt")) fails.push({ box: id, test: "focusing shows suggestions" });
    for (const [q, want] of CASES) {
      if (!list().includes(want)) continue;
      type(input, q);
      const got = names(input);
      if (got[0] === want) passes.cases++; else fails.push({ box: id, test: `"${q}" puts ${want} first`, got: got.slice(0, 3) });
    }
    // team names: every result is on that team, and a name after the team narrows to him
    const teamsOf = () => [...p.querySelectorAll(".ps-opt")].map(o => (o.querySelector(".ps-team") || {}).textContent || "");
    for (const q of ["knicks", "NYK", "new york", "New York Knicks", "ny"]) {
      type(input, q);
      const ts = teamsOf();
      if (ts.length && ts.every(t => t === "NYK")) passes.teams++; else fails.push({ box: id, test: `"${q}" lists only Knicks`, got: ts.slice(0, 5) });
    }
    const knick = list().find(n => n === "Jalen Brunson");
    if (knick) {
      type(input, "knicks brunson");
      if (names(input)[0] === knick && teamsOf()[0] === "NYK") passes.teams++; else fails.push({ box: id, test: '"knicks brunson" puts Brunson first', got: names(input).slice(0, 3) });
    }
    type(input, "xyzzy qq");
    if (!p.querySelector(".ps-empty")) fails.push({ box: id, test: "says when nothing matches" });
    // keyboard
    type(input, "curry");
    const before = names(input);
    key(input, "ArrowDown");
    const sel = p.querySelector('.ps-opt[aria-selected="true"] .ps-name');
    if (before.length > 1 && (!sel || sel.textContent !== before[1])) fails.push({ box: id, test: "ArrowDown moves to the next result", got: sel && sel.textContent });
    else passes.keys++;
    key(input, "Escape");
    if (!p.hidden) fails.push({ box: id, test: "Esc closes the list" }); else passes.keys++;
    type(input, before[0] ? plain(before[0]) : "curry");
    key(input, "Enter");
    const picked = id === "scFind" ? SC.players[sc.player].n : id === "muOff" ? mu.off.n : id === "muDef" ? mu.def.n : PL && PL.name;
    if (!p.hidden || input.value || (before[0] && picked !== before[0])) fails.push({ box: id, test: "Enter picks the top result and clears the box", picked });
    else passes.keys++;
    input.blur();
    await tick();
  }

  // everyone can be found by his own name typed without accents
  for (const [id, tab, list] of BOXES) {
    showTab(tab);
    const input = $(id), all = list();
    for (const [k, name] of all.entries()) {
      type(input, plain(name));
      if (names(input).includes(name)) passes.everyone++;
      else fails.push({ box: id, test: "find by own name", name, got: names(input).slice(0, 3) });
      if (k % 200 === 0) await tick();
    }
    type(input, ""); input.blur();
  }

  // matchups: never the player already on the other side
  if (BOXES.some(([id]) => id === "muOff")) {
    showTab("matchups");
    type($("muOff"), plain(mu.def.n));
    if (names($("muOff")).includes(mu.def.n)) fails.push({ box: "muOff", test: "doesn't offer the current defender", name: mu.def.n });
    type($("muDef"), plain(mu.off.n));
    if (names($("muDef")).includes(mu.off.n)) fails.push({ box: "muDef", test: "doesn't offer the current scorer", name: mu.off.n });
    for (const id of ["muOff", "muDef"]) { type($(id), ""); $(id).blur(); }
  }

  // table filters keep accented players when typed without accents
  const FILTERS = [["lbFind", "shots", "#lbBody tr"], ["dFilter", "players", "#dTable tr"], ["pfSearch", "players", "#pfRows tr"]]
    .filter(([id, tab]) => $(id) && tabHasContent(tab));
  for (const [id, tab, rowSel] of FILTERS) {
    showTab(tab);
    for (const [q, want] of [["jokic", "Jokić"], ["doncic", "Dončić"]]) {
      type($(id), q);
      const rows = [...document.querySelectorAll(rowSel)].map(r => r.textContent);
      if (rows.some(t => t.includes(want))) passes.filters++; else fails.push({ box: id, test: `"${q}" keeps ${want}`, rows: rows.length });
    }
    type($(id), "new york knicks");
    const rows = [...document.querySelectorAll(rowSel)];
    if (rows.length && rows.every(r => r.textContent.includes("NYK"))) passes.teams++; else fails.push({ box: id, test: '"new york knicks" keeps only Knicks', rows: rows.length });
    type($(id), ""); $(id).blur();
  }
  showTab("shots");
  return { passes, failures: fails.length, first: fails.slice(0, 12) };
};
