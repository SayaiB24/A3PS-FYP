// Hand-rolled SVG charts. Thin marks, recessive hairline grid, one y-axis per
// chart, hover layer by default, and a table-view twin for every chart.

import { el, svg, cssVar } from "./util.js";

// ---------------------------------------------------------------------------
// tooltip + table registry
// ---------------------------------------------------------------------------

const tipEl = () => document.getElementById("tooltip");

export function showTip(evt, title, rows) {
  const t = tipEl();
  t.replaceChildren();
  if (title) t.appendChild(el("div", { class: "tt-title", text: title }));
  for (const r of rows) {
    const row = el("div", { class: "tt-row" });
    if (r.color) row.appendChild(el("span", { class: "swatch line", style: { background: r.color } }));
    row.appendChild(el("b", { text: r.value }));
    if (r.name) row.appendChild(el("span", { class: "n", text: r.name }));
    t.appendChild(row);
  }
  t.classList.remove("hidden");
  const pad = 14, w = t.offsetWidth, h = t.offsetHeight;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
  if (y + h > window.innerHeight - 8) y = evt.clientY - h - pad;
  t.style.left = `${Math.max(8, x)}px`;
  t.style.top = `${Math.max(8, y)}px`;
}
export const hideTip = () => tipEl().classList.add("hidden");

const tables = new Map();   // chart host id -> {columns, rows}
const tableOpen = new Set();

export function registerTable(hostId, columns, rows) {
  tables.set(hostId, { columns, rows });
  if (tableOpen.has(hostId)) renderTable(hostId);
}

function renderTable(hostId) {
  const host = document.getElementById(hostId);
  if (!host) return;
  let wrap = host.nextElementSibling;
  if (!wrap || !wrap.classList.contains("tablewrap")) {
    wrap = el("div", { class: "tablewrap" });
    host.after(wrap);
  }
  const t = tables.get(hostId);
  wrap.replaceChildren();
  if (!t || !t.rows.length) { wrap.appendChild(el("div", { class: "muted", text: "no data" })); return; }
  const tbl = el("table", { class: "data" });
  tbl.appendChild(el("tr", {}, t.columns.map((c) => el("th", { class: c.num ? "num" : "", text: c.label }))));
  for (const r of t.rows) {
    tbl.appendChild(el("tr", {}, t.columns.map((c) =>
      el("td", { class: c.num ? "num" : "", text: r[c.key] == null ? "—" : String(r[c.key]) }))));
  }
  wrap.appendChild(tbl);
}

export function wireTableButtons(root = document) {
  root.querySelectorAll("[data-table]").forEach((b) => {
    b.addEventListener("click", () => {
      const id = b.dataset.table;
      const host = document.getElementById(id);
      if (tableOpen.has(id)) {
        tableOpen.delete(id);
        const w = host && host.nextElementSibling;
        if (w && w.classList.contains("tablewrap")) w.remove();
        b.textContent = "table";
      } else {
        tableOpen.add(id);
        renderTable(id);
        b.textContent = "hide table";
      }
    });
  });
}

// Legend: always present for >= 2 series; line key for lines, dot for points.
export function legendRow(items) {
  return el("div", { class: "legend", style: { margin: "0 0 6px" } }, items.map((it) =>
    el("span", { class: "lg-item" }, [
      el("span", { class: `swatch${it.kind === "line" ? " line" : ""}`, style: { background: it.color,
        borderRadius: it.kind === "dot" ? "50%" : null } }), it.name])));
}

function addLegend(host, o, items) {
  const list = o.legend || items;
  if (o.legend === false || !list || list.length < 2) return;
  host.prepend(legendRow(list));
}

// ---------------------------------------------------------------------------
// scales / axes
// ---------------------------------------------------------------------------

export function linear([d0, d1], [r0, r1]) {
  const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  const f = (v) => r0 + (v - d0) * k;
  f.invert = (p) => (k === 0 ? d0 : d0 + (p - r0) / k);
  return f;
}

export function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => raw <= s) || raw;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

function frame(host, height, m) {
  const W = Math.max(280, host.clientWidth || 480);
  const s = svg("svg", { class: "chart", viewBox: `0 0 ${W} ${height}`, role: "img" });
  host.replaceChildren(s);
  return { s, W, H: height, x0: m.l, x1: W - m.r, y0: height - m.b, y1: m.t };
}

function yAxis(s, f, y, ticks, fmtY, title) {
  for (const v of ticks) {
    const py = y(v);
    s.appendChild(svg("line", { class: "gridline", x1: f.x0, x2: f.x1, y1: py, y2: py }));
    s.appendChild(svg("text", { x: f.x0 - 6, y: py + 3.5, "text-anchor": "end" }, fmtY(v)));
  }
  if (title) s.appendChild(svg("text", { class: "title", x: f.x0, y: f.y1 - 8 }, title));
}

function xAxis(s, f, x, ticks, fmtX, title) {
  s.appendChild(svg("line", { class: "axisline", x1: f.x0, x2: f.x1, y1: f.y0, y2: f.y0 }));
  for (const v of ticks) {
    s.appendChild(svg("text", { x: x(v), y: f.y0 + 15, "text-anchor": "middle" }, fmtX(v)));
  }
  if (title) s.appendChild(svg("text", { class: "title", x: f.x1, y: f.y0 + 30, "text-anchor": "end" }, title));
}

function refLine(s, f, x, y, r) {
  const color = r.color || null;
  if (r.axis === "y") {
    const py = y(r.v);
    s.appendChild(svg("line", { class: "ref", x1: f.x0, x2: f.x1, y1: py, y2: py, stroke: color }));
    if (r.label) s.appendChild(svg("text", { x: f.x1 - 2, y: py - 4, "text-anchor": "end" }, r.label));
  } else {
    const px = x(r.v);
    if (px < f.x0 - 1 || px > f.x1 + 1) return;
    s.appendChild(svg("line", {
      class: r.solid ? "" : "ref", x1: px, x2: px, y1: f.y1, y2: f.y0,
      stroke: color || cssVar("--muted"), "stroke-width": r.solid ? 1.5 : 1,
      "stroke-dasharray": r.dash || null,
    }));
    if (r.label) s.appendChild(svg("text", { x: px + 3, y: f.y1 + 10 }, r.label));
  }
}

// ---------------------------------------------------------------------------
// line chart (multi-series, crosshair snaps to nearest x, one tooltip lists all)
// ---------------------------------------------------------------------------

export function lineChart(host, o) {
  const m = { l: 44, r: 16, t: 22, b: o.xTitle ? 38 : 26 };
  const f = frame(host, o.height || 260, m);
  const { s } = f;
  const series = o.series.filter((se) => se.points.length);
  if (!series.length) { host.replaceChildren(el("div", { class: "empty-state", text: o.empty || "no data" })); return; }
  const xs = series.flatMap((se) => se.points.map((p) => p.x));
  const ys = series.flatMap((se) => se.points.map((p) => p.y)).filter((v) => v != null);
  const xd = o.xDomain || [Math.min(...xs), Math.max(...xs)];
  let yd = o.yDomain || [Math.min(0, ...ys), Math.max(...ys)];
  if (yd[0] === yd[1]) yd = [yd[0] - 1, yd[1] + 1];
  const x = linear(xd, [f.x0 + 4, f.x1 - 4]), y = linear(yd, [f.y0, f.y1]);
  const fmtY = o.yFormat || ((v) => String(v));
  const fmtX = o.xFormat || ((v) => String(v));
  yAxis(s, f, y, niceTicks(yd[0], yd[1], 4), fmtY, o.yTitle);
  xAxis(s, f, x, o.xTicks || niceTicks(xd[0], xd[1], 6), fmtX, o.xTitle);
  (o.refs || []).forEach((r) => refLine(s, f, x, y, r));

  for (const se of series) {
    const pts = se.points.filter((p) => p.y != null);
    if (se.connect !== false && pts.length > 1) {
      s.appendChild(svg("polyline", {
        points: pts.map((p) => `${x(p.x)},${y(p.y)}`).join(" "), fill: "none",
        stroke: se.color, "stroke-width": 2, "stroke-linejoin": "round",
        "stroke-dasharray": se.dash || null, opacity: se.faded ? 0.35 : 1,
      }));
    }
    if (se.showPoints || pts.length === 1 || se.connect === false) {
      for (const p of pts) {
        s.appendChild(svg("circle", { cx: x(p.x), cy: y(p.y), r: 4, fill: se.color,
          stroke: cssVar("--surface"), "stroke-width": 2 }));
      }
    }
    for (const mk of se.markers || []) {
      s.appendChild(svg("circle", { cx: x(mk.x), cy: y(mk.y), r: 6.5, fill: "none",
        stroke: se.color, "stroke-width": 2 }));
    }
  }
  (o.overlays || []).forEach((fn) => fn(s, f, x, y));

  // Hover layer: vertical crosshair snapping to the nearest x in any series.
  const allX = [...new Set(series.flatMap((se) => se.points.map((p) => p.x)))].sort((a, b) => a - b);
  const cross = svg("line", { class: "crosshair hidden", y1: f.y1, y2: f.y0 });
  s.appendChild(cross);
  const hit = svg("rect", { class: "hit", x: f.x0, y: f.y1, width: f.x1 - f.x0, height: f.y0 - f.y1 });
  s.appendChild(hit);
  const toLocal = (evt) => {
    const r = s.getBoundingClientRect();
    return (evt.clientX - r.left) * (f.W / r.width);
  };
  const nearestX = (px) => {
    const v = x.invert(px);
    let best = allX[0];
    for (const c of allX) if (Math.abs(c - v) < Math.abs(best - v)) best = c;
    return best;
  };
  hit.addEventListener("pointermove", (evt) => {
    const xv = nearestX(toLocal(evt));
    cross.setAttribute("x1", x(xv)); cross.setAttribute("x2", x(xv));
    cross.classList.remove("hidden");
    const rows = [];
    for (const se of series) {
      if (se.faded) continue;   // de-emphasised context lines stay out of the readout
      const p = se.points.find((q) => q.x === xv);
      if (p && p.y != null) rows.push({ color: se.color, value: fmtY(p.y), name: p.label || se.name });
    }
    showTip(evt, o.tipTitle ? o.tipTitle(xv) : fmtX(xv), rows);
  });
  hit.addEventListener("pointerleave", () => { cross.classList.add("hidden"); hideTip(); });
  if (o.onClick) {
    hit.style.cursor = "pointer";
    hit.addEventListener("click", (evt) => o.onClick(x.invert(toLocal(evt)), nearestX(toLocal(evt))));
  }
  addLegend(host, o, series.filter((se) => se.name && !se.faded)
    .map((se) => ({ name: se.name, color: se.color, kind: "line" })));
  return { x, y, f, svg: s };
}

// ---------------------------------------------------------------------------
// scatter (nearest-point hover, 24px minimum reach)
// ---------------------------------------------------------------------------

export function scatter(host, o) {
  const m = { l: 44, r: 16, t: 22, b: 38 };
  const f = frame(host, o.height || 280, m);
  const { s } = f;
  const pts = o.series.flatMap((se) => se.points.map((p) => ({ ...p, se })));
  if (!pts.length) { host.replaceChildren(el("div", { class: "empty-state", text: o.empty || "no data" })); return; }
  const x = linear(o.xDomain, [f.x0 + 4, f.x1 - 4]), y = linear(o.yDomain, [f.y0, f.y1]);
  yAxis(s, f, y, niceTicks(o.yDomain[0], o.yDomain[1], 4), o.yFormat, o.yTitle);
  xAxis(s, f, x, niceTicks(o.xDomain[0], o.xDomain[1], 5), o.xFormat, o.xTitle);
  (o.refs || []).forEach((r) => refLine(s, f, x, y, r));
  for (const p of pts) {
    const cx = x(p.x), cy = y(p.y);
    if (p.ring) s.appendChild(svg("circle", { cx, cy, r: 8, fill: "none", stroke: p.se.color, "stroke-width": 2 }));
    s.appendChild(svg("circle", { cx, cy, r: p.r || 4.5, fill: p.se.color,
      stroke: cssVar("--surface"), "stroke-width": 2, opacity: p.faded ? 0.35 : 0.9 }));
  }
  const ring = svg("circle", { r: 9, fill: "none", stroke: cssVar("--ink"), "stroke-width": 1.5, class: "hidden" });
  s.appendChild(ring);
  const hit = svg("rect", { class: "hit", x: f.x0, y: f.y1, width: f.x1 - f.x0, height: f.y0 - f.y1 });
  s.appendChild(hit);
  const nearest = (evt) => {
    const r = s.getBoundingClientRect(), k = f.W / r.width;
    const px = (evt.clientX - r.left) * k, py = (evt.clientY - r.top) * k;
    let best = null, bd = Infinity;
    for (const p of pts) {
      const d = Math.hypot(x(p.x) - px, y(p.y) - py);
      if (d < bd) { bd = d; best = p; }
    }
    return bd <= 24 ? best : null;
  };
  hit.addEventListener("pointermove", (evt) => {
    const p = nearest(evt);
    if (!p) { ring.classList.add("hidden"); hideTip(); hit.style.cursor = ""; return; }
    ring.setAttribute("cx", x(p.x)); ring.setAttribute("cy", y(p.y)); ring.classList.remove("hidden");
    hit.style.cursor = o.onClick ? "pointer" : "";
    showTip(evt, p.title || p.se.name, p.rows || [{ color: p.se.color, value: `${o.xFormat(p.x)}, ${o.yFormat(p.y)}` }]);
  });
  hit.addEventListener("pointerleave", () => { ring.classList.add("hidden"); hideTip(); });
  if (o.onClick) hit.addEventListener("click", (evt) => { const p = nearest(evt); if (p) o.onClick(p); });
  addLegend(host, o, o.series.filter((se) => se.points.length).map((se) => ({ name: se.name, color: se.color, kind: "dot" })));
}

// ---------------------------------------------------------------------------
// dot plot: one row per metric, one dot per run on a shared axis
// ---------------------------------------------------------------------------

export function dotPlot(host, o) {
  const rowH = 30, labelW = Math.min(230, Math.max(150, (host.clientWidth || 480) * 0.36));
  const m = { l: labelW, r: 18, t: 8, b: 26 };
  const f = frame(host, m.t + m.b + rowH * o.rows.length, m);
  const { s } = f;
  if (!o.rows.length) { host.replaceChildren(el("div", { class: "empty-state", text: o.empty || "no data" })); return; }
  const x = linear(o.domain, [f.x0 + 6, f.x1 - 6]);
  for (const v of niceTicks(o.domain[0], o.domain[1], 5)) {
    s.appendChild(svg("line", { class: "gridline", x1: x(v), x2: x(v), y1: f.y1, y2: f.y0 }));
    s.appendChild(svg("text", { x: x(v), y: f.y0 + 15, "text-anchor": "middle" }, o.format(v)));
  }
  const dots = [];
  o.rows.forEach((row, i) => {
    const cy = f.y1 + rowH * i + rowH / 2;
    s.appendChild(svg("line", { class: "gridline", x1: f.x0, x2: f.x1, y1: cy, y2: cy }));
    s.appendChild(svg("text", { class: "lab", x: f.x0 - 10, y: cy + 4, "text-anchor": "end" }, row.label));
    // Nudge exact collisions vertically so no run hides behind another.
    const seen = new Map();
    for (const d of row.values) {
      if (d.v == null) continue;
      const key = Math.round(x(d.v) / 3);
      const k = seen.get(key) || 0; seen.set(key, k + 1);
      const cy2 = cy + (k ? (k % 2 ? 1 : -1) * Math.ceil(k / 2) * 5 : 0);
      s.appendChild(svg("circle", { cx: x(d.v), cy: cy2, r: 5, fill: d.color,
        stroke: cssVar("--surface"), "stroke-width": 2 }));
      dots.push({ cx: x(d.v), cy: cy2, row, d });
    }
  });
  const hit = svg("rect", { class: "hit", x: 0, y: f.y1, width: f.W, height: f.y0 - f.y1 });
  s.appendChild(hit);
  hit.addEventListener("pointermove", (evt) => {
    const r = s.getBoundingClientRect(), k = f.W / r.width;
    const py = (evt.clientY - r.top) * k;
    const idx = Math.floor((py - f.y1) / rowH);
    const row = o.rows[idx];
    if (!row) { hideTip(); return; }
    // One tooltip per row: every run's value for that metric.
    showTip(evt, row.label, row.values.map((d) => ({
      color: d.color, value: d.v == null ? "—" : o.format(d.v), name: d.name + (d.note ? ` · ${d.note}` : ""),
    })));
  });
  hit.addEventListener("pointerleave", hideTip);
  const seen = new Map();
  o.rows.forEach((r) => r.values.forEach((d) => seen.set(d.name, d.color)));
  addLegend(host, o, [...seen].map(([name, color]) => ({ name, color, kind: "dot" })));
}

// ---------------------------------------------------------------------------
// confusion-matrix small multiples (HTML: cells shaded on one sequential hue)
// ---------------------------------------------------------------------------

export function confusionGrid(host, items) {
  host.replaceChildren();
  if (!items.length) {
    host.appendChild(el("div", { class: "empty-state", text: "No selected run has per-clip rows." }));
    return;
  }
  const wrap = el("div", { style: { display: "flex", flexWrap: "wrap", gap: "18px" } });
  const ramp = ["--heat-0", "--heat-1", "--heat-2", "--heat-3", "--heat-4"].map(cssVar);
  for (const it of items) {
    const cm = it.cm, n = cm.TP + cm.FP + cm.FN + cm.TN;
    const cell = (k, label) => {
      const frac = n ? cm[k] / n : 0;
      const step = Math.min(4, Math.floor(frac * 5 / 0.6));
      const dark = step >= 2;
      return el("td", {
        style: { background: ramp[step], color: dark ? "#fff" : "#0b0b0b", textAlign: "center",
                 padding: "8px 10px", minWidth: "64px", borderRadius: "6px" },
        title: `${label}: ${cm[k]} of ${n}`,
      }, [el("div", { style: { fontSize: "18px", fontWeight: "650" }, text: String(cm[k]) }),
          el("div", { style: { fontSize: "10px" }, text: label })]);
    };
    const tbl = el("table", { style: { borderCollapse: "separate", borderSpacing: "3px", fontSize: "11px" } }, [
      el("tr", {}, [el("td"), el("td", { class: "muted", text: "fired", style: { textAlign: "center" } }),
        el("td", { class: "muted", text: "silent", style: { textAlign: "center" } })]),
      el("tr", {}, [el("td", { class: "muted", text: "positive" }), cell("TP", "TP"), cell("FN", "FN")]),
      el("tr", {}, [el("td", { class: "muted", text: "negative" }), cell("FP", "FP"), cell("TN", "TN")]),
    ]);
    wrap.appendChild(el("div", {}, [
      el("div", { style: { display: "flex", alignItems: "center", gap: "6px", marginBottom: "4px", fontSize: "12px" } },
        [el("span", { class: "swatch", style: { background: it.color } }), el("b", { text: it.name })]),
      el("div", { class: "muted", style: { fontSize: "11px", marginBottom: "4px" }, text: it.sub }),
      tbl,
    ]));
  }
  host.appendChild(wrap);
}
