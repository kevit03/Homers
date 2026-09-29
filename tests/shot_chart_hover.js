/* Shot chart hover test: every spot on the court must show its popup when the pointer is over it.

   It uses the browser's own hit-testing: for points inside each spot it asks what element is really under the
   pointer (document.elementFromPoint), sends that element a mousemove at that point, and checks that the popup
   is showing, is about that spot (same makes and attempts, or the same zone), has no NaN/undefined in it,
   and sits inside the chart panel.

     Hex view   every spot of every player (season "Everything"): the centre, near the coloured hex's edge,
                and in the rest of its cell. For a sample of players, every season option and 13 points per spot.
     Zones view points on a grid over each zone, plus points lying on the court lines drawn over the zones.

   Run it on a built dashboard.html served over http (e.g. python3 -m http.server), in the browser console:

     const s = document.createElement("script"); s.src = "tests/shot_chart_hover.js"; document.head.append(s);
     // then
     await testShotChartHover()          // about 2-4 minutes; progress in window.scHoverTest
*/
window.testShotChartHover = async function testShotChartHover({ samplePlayers = 40, zonePlayers = 40, gridStep = 7 } = {}) {
  const out = window.scHoverTest = { done: false, renders: 0, spots: 0, points: 0, skipped: 0, fails: [], nFails: 0, byKind: {} };
  const svg = $("scCourt"), tip = $("scTip"), panel = $("scCourtPanel"), nav = document.querySelector("nav");
  const saved = { player: sc.player, season: sc.season, view: sc.view, nav: nav && nav.style.position, y: scrollY };
  showTab("shots");
  if (nav) nav.style.position = "static";  // a sticky menu would cover the top of the court
  const fail = (kind, info) => {
    out.nFails++; out.byKind[kind] = (out.byKind[kind] || 0) + 1;
    if (out.fails.length < 60) out.fails.push({ kind, player: SC.players[sc.player].n, season: sc.season, view: sc.view, ...info });
  };
  const toClient = (x, y) => { const m = svg.getScreenCTM(); return [m.a * x + m.c * y + m.e, m.b * x + m.d * y + m.f]; };
  const inView = (cx, cy) => { const r = svg.getBoundingClientRect(); return cx > r.left + 1 && cx < r.right - 1 && cy > r.top + 1 && cy < r.bottom - 1 && cy < innerHeight - 1 && cy > 1; };
  const bad = /NaN|undefined|null|Infinity/;

  // hover the point (user units) and return the popup text, or a failure reason
  function hoverAt(x, y) {
    let [cx, cy] = toClient(x, y);
    if (!inView(cx, cy)) {
      const r = svg.getBoundingClientRect();
      if (cx <= r.left + 1 || cx >= r.right - 1) return { skip: true };  // outside the drawn court
      scrollBy(0, cy - innerHeight / 2); [cx, cy] = toClient(x, y);
      if (!inView(cx, cy)) return { skip: true };
    }
    const el = document.elementFromPoint(cx, cy);
    tip.hidden = true; tip.innerHTML = "";
    if (!el || !svg.contains(el)) return { reason: "pointer not over the chart", el: el && (el.id || el.className) };
    el.dispatchEvent(new MouseEvent("mousemove", { bubbles: true, clientX: cx, clientY: cy }));
    if (tip.hidden || !tip.textContent.trim()) return { reason: "no popup", el: el.getAttribute("class") || el.tagName };
    const pr = panel.getBoundingClientRect(), tr = tip.getBoundingClientRect();
    if (tr.left < pr.left - 1 || tr.top < pr.top - 1 || tr.right > pr.right + 1 || tr.bottom > pr.bottom + 1) return { reason: "popup outside the panel", text: tip.textContent };
    if (bad.test(tip.textContent)) return { reason: "bad value in popup", text: tip.textContent };
    return { text: tip.textContent, el };
  }
  const check = (kind, x, y, expect, info) => {
    const r = hoverAt(x, y);
    if (r.skip) { out.skipped++; return; }
    out.points++;
    if (r.reason) fail(r.reason, { ...info, x: +x.toFixed(1), y: +y.toFixed(1), el: r.el, text: r.text });
    else if (!r.text.includes(expect)) fail("popup for the wrong spot", { ...info, expect, text: r.text.slice(0, 80) });
  };
  const pts = s => s.getAttribute("points").trim().split(/\s+/).map(p => p.split(",").map(Number));
  const centre = v => [v.reduce((a, p) => a + p[0], 0) / v.length, v.reduce((a, p) => a + p[1], 0) / v.length];
  const lerp = ([ax, ay], [bx, by], t) => [ax + (bx - ax) * t, ay + (by - ay) * t];

  function render(player, season, view) {
    sc.player = player; sc.season = String(season); sc.view = view;
    scDraw(); out.renders++;
    window.scrollTo(0, svg.getBoundingClientRect().top + scrollY - 8);
  }
  function hexSpots(dense) {
    const p = SC.players[sc.player];
    if (!scRecs(p).length) return;
    const o = scSum(scRecs(p));
    for (const hit of svg.querySelectorAll(".hit")) {
      const i = +hit.dataset.h, [a, m] = o.h.get(i), vis = hit.nextElementSibling, cell = pts(hit), dot = pts(vis), c = centre(cell);
      const expect = `${m} of ${a} made`, info = { spot: i };
      out.spots++;
      const where = [c, lerp(c, dot[0], .8)];                                  // centre, and inside the coloured hex near its edge
      if (Math.hypot(...cell[0].map((v, k) => v - c[k])) - Math.hypot(...dot[0].map((v, k) => v - c[k])) > 2)
        where.push(lerp(lerp(c, dot[1], 1), cell[1], .5));                     // in the cell outside the coloured hex
      if (dense) for (let k = 0; k < 6; k++) where.push(lerp(c, dot[k], .9), lerp(c, lerp(cell[k], cell[(k + 1) % 6], .5), .92));
      for (const [x, y] of where) check("hex", x, y, expect, info);
    }
  }
  function zoneSpots() {
    const zones = [...svg.querySelectorAll("path[data-z]")];
    const zoneAt = (x, y) => zones.find(z => z.isPointInFill(new DOMPoint(x, y)));
    const vb = svg.viewBox.baseVal, probes = [];
    for (let x = vb.x + gridStep / 2; x < vb.x + vb.width; x += gridStep)
      for (let y = vb.y + gridStep / 2; y < vb.y + vb.height; y += gridStep) probes.push([x, y]);
    for (const line of svg.querySelectorAll(".line")) {                        // points right on the court lines
      if (!line.getTotalLength) continue;
      const L = line.getTotalLength();
      for (let d = 0; d < L; d += 9) { const q = line.getPointAtLength(d); probes.push([q.x, q.y]); }
    }
    for (const [x, y] of probes) {
      const z = zoneAt(x, y);
      if (!z) continue;
      // a point exactly on the border of two zones belongs to either; only test points clearly inside one
      if (zones.filter(o => o !== z && (o.isPointInFill(new DOMPoint(x + 1.5, y)) || o.isPointInFill(new DOMPoint(x - 1.5, y)) ||
          o.isPointInFill(new DOMPoint(x, y + 1.5)) || o.isPointInFill(new DOMPoint(x, y - 1.5)))).length) continue;
      out.spots++;
      check("zone", x, y, SC.loc_zones[+z.dataset.z], { zone: SC.loc_zones[+z.dataset.z] });
    }
  }
  const tick = () => new Promise(r => setTimeout(r));

  try {
    const shooters = SC.players.map((p, i) => [i, p]).filter(([, p]) => p.p.length);
    const all = scHasPO() ? "all" : SC.seasons.length > 1 ? "all" : "0";
    // 1. every player, every spot, season "Everything"
    for (const [k, [i]] of shooters.entries()) {
      render(i, all, "hex"); hexSpots(false);
      if (k % 25 === 0) { out.progress = `hex, every player: ${k}/${shooters.length}`; await tick(); }
    }
    // 2. a sample of players (busiest and quietest), every season option, 13 points per spot
    const opts = [...$("scSeason").options].map(o => o.value);
    const sample = [...shooters.slice(0, samplePlayers / 2), ...shooters.slice(-samplePlayers / 2)];
    for (const [k, [i]] of sample.entries()) {
      for (const s of opts) { render(i, s, "hex"); hexSpots(true); }
      out.progress = `hex, every season: ${k + 1}/${sample.length}`; await tick();
    }
    // 3. zones view
    const zsample = shooters.filter((_, k) => k % Math.max(1, Math.floor(shooters.length / zonePlayers)) === 0).slice(0, zonePlayers);
    for (const [k, [i]] of zsample.entries()) {
      for (const s of [all, opts[0]]) { render(i, s, "zones"); zoneSpots(); }
      out.progress = `zones: ${k + 1}/${zsample.length}`; await tick();
    }
  } finally {
    Object.assign(sc, { player: saved.player, season: saved.season, view: saved.view });
    if (nav) nav.style.position = saved.nav || "";
    tip.hidden = true; scRender(); window.scrollTo(0, saved.y);
    out.done = true; out.progress = "done";
  }
  return { renders: out.renders, spots: out.spots, points: out.points, skipped: out.skipped, failures: out.nFails, byKind: out.byKind, first: out.fails.slice(0, 8) };
};
