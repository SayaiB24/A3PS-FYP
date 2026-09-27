// A3PS results dashboard v2: filters, run comparison, history, drill-down.
// Read-only: every byte comes from GET /api/* (scripts/serve_dashboard_v2.py).

import {
  $, el, api, fmt, debounce, slotColor, cssVar, METRICS, RATE_METRICS, LEAD_METRICS,
  metricOf, OUTCOME, shortDate,
} from "./util.js";
import {
  lineChart, scatter, dotPlot, confusionGrid, registerTable, wireTableButtons,
} from "./charts.js";
import { initLive, onLiveShow, onLiveFilters, openInLive } from "./live.js";

const FAMILIES = ["learned", "ablation", "ensemble", "baseline"];
const SPLIT_ORDER = ["eval_v1", "train_val", "train_val_v2", "eval_v2"];
const KIND_LABEL = {
  selection_val: "checkpoint val", sweep: "sweep", final_read: "final read",
  ablation_rerun: "ablation re-run", ensemble_grid: "ensemble grid", per_clip: "per-clip eval",
};

const S = {
  runs: [], byId: new Map(), splits: {},
  f: { preset: "all", from: null, to: null, families: new Set(FAMILIES), split: "", clip: "",
       thr: 0.7, classes: new Set(), classOff: new Set() },
  selected: [], slot: new Map(), hidden: new Set(), evalChoice: new Map(), details: new Map(),
  drill: { evalId: null, rows: [], outcomes: new Set(["TP", "FN", "FP", "TN"]), aboveOnly: false,
           sort: { key: "clip_id", dir: 1 }, clip: null },
  view: "live",
};

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------

async function boot() {
  setupTheme();
  setupTabs();
  wireTableButtons();
  $("status").textContent = "loading runs…";
  const data = await api("/api/runs");
  S.runs = data.runs;
  S.splits = data.splits;
  S.runs.forEach((r) => S.byId.set(r.id, r));
  $("status").textContent = `${S.runs.length} runs` + (data.warnings.length ? ` · ${data.warnings.length} warning(s)` : "");
  if (data.warnings.length) $("status").title = data.warnings.join("\n");
  setupFilters();
  // Default comparison: the committed run vs the threshold-system baseline.
  for (const id of ["k1p0_v2_selfix_s1234", "baseline_threshold_system"]) if (S.byId.has(id)) toggleSelect(id, true);
  readHash();
  renderAll();
  window.addEventListener("resize", debounce(renderAll, 150));
}

function setupTheme() {
  const saved = (() => { try { return localStorage.getItem("a3ps-theme"); } catch { return null; } })();
  if (saved) document.documentElement.dataset.theme = saved;
  $("theme-btn").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("a3ps-theme", next); } catch { /* private mode */ }
    renderAll();
  });
}

function setupTabs() {
  document.querySelectorAll(".tabs button").forEach((b) =>
    b.addEventListener("click", () => showView(b.dataset.view)));
}

function showView(v) {
  S.view = v;
  document.querySelectorAll(".tabs button").forEach((b) =>
    b.setAttribute("aria-selected", String(b.dataset.view === v)));
  document.querySelectorAll(".view").forEach((s) => s.classList.toggle("active", s.id === `view-${v}`));
  try { history.replaceState(null, "", `#${v}`); } catch { /* file:// */ }
  renderAll();
}

function readHash() {
  const v = (location.hash || "").slice(1);
  if (["live", "compare", "history", "drill"].includes(v)) showView(v);
}

// ---------------------------------------------------------------------------
// filters
// ---------------------------------------------------------------------------

function setupFilters() {
  const fam = $("f-family");
  for (const f of FAMILIES) {
    const n = S.runs.filter((r) => r.family === f).length;
    const b = el("button", { class: "chip", "aria-pressed": "true", text: `${f} (${n})` });
    b.addEventListener("click", () => {
      S.f.families.has(f) ? S.f.families.delete(f) : S.f.families.add(f);
      b.setAttribute("aria-pressed", String(S.f.families.has(f)));
      renderAll();
    });
    fam.appendChild(b);
  }
  const sp = $("f-split");
  for (const k of SPLIT_ORDER) sp.appendChild(el("option", { value: k, text: S.splits[k]?.label || k }));
  sp.addEventListener("change", () => { S.f.split = sp.value; renderAll(); });

  const preset = $("f-date-preset");
  preset.addEventListener("change", () => {
    S.f.preset = preset.value;
    const custom = preset.value === "custom";
    $("f-date-from").classList.toggle("hidden", !custom);
    $("f-date-to").classList.toggle("hidden", !custom);
    renderAll();
  });
  for (const id of ["f-date-from", "f-date-to"]) {
    $(id).addEventListener("change", () => {
      S.f.from = $("f-date-from").value || null;
      S.f.to = $("f-date-to").value || null;
      renderAll();
    });
  }
  $("f-clip").addEventListener("input", debounce(() => { S.f.clip = $("f-clip").value.trim(); renderAll(); }, 150));
  const thr = $("f-thr");
  thr.addEventListener("input", () => {
    S.f.thr = +thr.value;
    $("f-thr-val").textContent = S.f.thr.toFixed(2);
    renderAllLight();
  });
  $("cmp-clear").addEventListener("click", () => {
    [...S.selected].forEach((id) => toggleSelect(id, false));
    renderAll();
  });
  renderClassFilter();
}

function registerClasses(classes) {
  let added = false;
  for (const c of classes) if (!S.f.classes.has(c)) { S.f.classes.add(c); added = true; }
  if (added) renderClassFilter();
}

function renderClassFilter() {
  const pop = $("f-classes-pop");
  pop.replaceChildren();
  if (!S.f.classes.size) {
    pop.appendChild(el("div", { class: "muted", style: { fontSize: "12px" },
      text: "Classes appear once a clip with tracks is loaded." }));
  }
  for (const c of [...S.f.classes].sort()) {
    const cb = el("input", { type: "checkbox", checked: !S.f.classOff.has(c) });
    cb.addEventListener("change", () => {
      cb.checked ? S.f.classOff.delete(c) : S.f.classOff.add(c);
      renderClassFilter();
      renderAllLight();
    });
    pop.appendChild(el("label", {}, [cb, ` ${c}`]));
  }
  const on = [...S.f.classes].filter((c) => !S.f.classOff.has(c));
  $("f-classes-sum").textContent = !S.f.classOff.size ? "all" : `${on.length} of ${S.f.classes.size}`;
}

function dateBounds() {
  if (S.f.preset === "all") return [null, null];
  if (S.f.preset === "custom") return [S.f.from, S.f.to];
  const newest = S.runs.map((r) => r.date).filter(Boolean).sort().pop();
  if (!newest) return [null, null];
  const d = new Date(newest);
  d.setDate(d.getDate() - Number(S.f.preset));
  return [d.toISOString().slice(0, 10), null];
}

function runPasses(r) {
  if (!S.f.families.has(r.family)) return false;
  const [from, to] = dateBounds();
  const day = (r.date || "").slice(0, 10);
  if (from && (!day || day < from)) return false;
  if (to && (!day || day > to)) return false;
  if (S.f.split && !r.evals.some((e) => e.split === S.f.split)) return false;
  return true;
}

const filteredRuns = () => S.runs.filter(runPasses);

// ---------------------------------------------------------------------------
// selection / colour (colour follows the run, never its rank)
// ---------------------------------------------------------------------------

function toggleSelect(id, on) {
  const has = S.selected.includes(id);
  if (on === undefined) on = !has;
  if (on && !has) {
    const used = new Set(S.slot.values());
    let slot = 0;
    while (used.has(slot) && slot < 8) slot++;
    if (slot >= 8) { $("status").textContent = "8 runs max in one comparison (palette limit)"; return; }
    S.selected.push(id);
    S.slot.set(id, slot);
    S.hidden.delete(id);
  } else if (!on && has) {
    S.selected = S.selected.filter((x) => x !== id);
    S.slot.delete(id);
  }
}

const colorOf = (id) => slotColor(S.slot.get(id) ?? 0);

function chosenEval(run) {
  const id = S.evalChoice.get(run.id) || run.headline_eval;
  let ev = run.evals.find((e) => e.id === id) || run.evals[0];
  if (S.f.split && ev && ev.split !== S.f.split) {
    ev = run.evals.find((e) => e.split === S.f.split) || ev;
  }
  return ev;
}

const evalLabel = (e) =>
  `${KIND_LABEL[e.kind] || e.kind} · ${e.split_label}` +
  (e.threshold != null ? ` · ${e.threshold}${e.confirm != null ? `/${e.confirm}f` : ""}` : "");

// ---------------------------------------------------------------------------
// render dispatch
// ---------------------------------------------------------------------------

const liveCtx = () => ({ getFilters: () => S.f, registerClasses: registerClasses });

function renderAll() {
  if (S.view === "live") { initLive(liveCtx()); onLiveShow(); }
  renderRunList();
  if (S.view === "compare") renderCompare();
  if (S.view === "history") renderHistory();
  if (S.view === "drill") renderDrill();
}

// Cheap re-render for slider/class changes.
function renderAllLight() {
  if (S.view === "live") onLiveFilters();
  if (S.view === "compare") renderCompare();
  if (S.view === "drill") renderDrill();
}

// ---------------------------------------------------------------------------
// compare
// ---------------------------------------------------------------------------

function renderRunList() {
  const host = $("runlist");
  host.replaceChildren();
  const runs = filteredRuns();
  // Selected runs that the filters now exclude stay listed, so they can be removed.
  const extra = S.selected.map((id) => S.byId.get(id)).filter((r) => !runs.includes(r));
  for (const fam of FAMILIES) {
    const group = runs.concat(extra).filter((r) => r.family === fam);
    if (!group.length) continue;
    host.appendChild(el("div", { class: "fam-head", text: fam }));
    group.sort((a, b) => (b.date || "").localeCompare(a.date || "") || a.id.localeCompare(b.id));
    for (const r of group) {
      const sel = S.selected.includes(r.id);
      const ev = chosenEval(r);
      const v2 = metricOf(ev, "useful_warning_rate_v2"), fa = metricOf(ev, "false_alarm_rate");
      const cb = el("input", { type: "checkbox", checked: sel, "aria-label": `select ${r.label}` });
      const row = el("div", { class: "runrow", title: r.notes.join("\n") || null }, [
        cb,
        el("div", {}, [
          el("div", { class: "rname" }, [
            sel ? el("span", { class: "swatch", style: { background: colorOf(r.id), marginRight: "6px" } }) : null,
            r.label]),
          el("div", { class: "rmeta", text: `${shortDate(r.date)} · ${ev ? ev.split_label : "no eval"}` +
            (runs.includes(r) ? "" : " · filtered out") }),
        ]),
        el("div", { class: "rval num", text: v2 != null ? `v2 ${fmt(v2)}` : (fa != null ? `FA ${fmt(fa)}` : "") }),
      ]);
      row.addEventListener("click", (e) => {
        if (e.target !== cb) cb.checked = !cb.checked;
        toggleSelect(r.id, cb.checked);
        renderAll();
      });
      host.appendChild(row);
    }
  }
  if (!host.children.length) host.appendChild(el("div", { class: "empty-state", text: "No runs match the filters." }));
}

function visibleRuns() {
  return S.selected.filter((id) => !S.hidden.has(id)).map((id) => S.byId.get(id));
}

async function ensureDetails(ids) {
  const missing = ids.filter((id) => !S.details.has(id));
  await Promise.all(missing.map(async (id) => S.details.set(id, await api(`/api/run/${encodeURIComponent(id)}`))));
}

function renderCompare() {
  const legend = $("cmp-legend");
  legend.replaceChildren();
  const empty = S.selected.length === 0;
  $("cmp-empty").classList.toggle("hidden", !empty);
  $("cmp-charts").classList.toggle("hidden", empty);
  for (const id of S.selected) {
    const r = S.byId.get(id);
    const vis = el("input", { type: "checkbox", checked: !S.hidden.has(id), "aria-label": `show ${r.label}` });
    vis.addEventListener("change", () => { vis.checked ? S.hidden.delete(id) : S.hidden.add(id); renderCompare(); });
    const sel = el("select", { "aria-label": `evaluation for ${r.label}` });
    for (const e of r.evals) sel.appendChild(el("option", { value: e.id, text: evalLabel(e) }));
    sel.value = chosenEval(r)?.id || "";
    sel.addEventListener("change", () => { S.evalChoice.set(id, sel.value); renderCompare(); renderRunList(); });
    legend.appendChild(el("span", { class: `lg-item${S.hidden.has(id) ? " off" : ""}` }, [
      el("label", {}, [vis, el("span", { class: "swatch", style: { background: colorOf(id) } }), r.label]),
      r.evals.length ? sel : el("span", { class: "muted", text: "no evaluations" }),
    ]));
  }
  if (empty) return;

  const runs = visibleRuns();
  const evs = runs.map((r) => ({ r, ev: chosenEval(r) }));
  const splits = [...new Set(evs.map((x) => x.ev?.split).filter(Boolean))];
  const warn = $("cmp-split-warn");
  warn.classList.toggle("hidden", splits.length <= 1);
  warn.textContent = splits.length > 1
    ? `The visible runs are measured on different splits (${splits.map((s) => S.splits[s]?.label || s).join(", ")}). ` +
      "Numbers across splits are not directly comparable; use the Split filter or the per-run evaluation dropdown to align them."
    : "";

  // 1. headline rates
  const rateRows = RATE_METRICS.map((k) => ({
    label: METRICS[k].label + (METRICS[k].derived ? " · derived" : ""),
    key: k,
    values: evs.map(({ r, ev }) => ({ name: r.label, color: colorOf(r.id), v: metricOf(ev, k),
      note: ev ? ev.split_label : "" })),
  })).filter((row) => row.values.some((d) => d.v != null));
  dotPlot($("ch-metrics"), { rows: rateRows, domain: [0, 1], format: (v) => v.toFixed(2),
    empty: "No rate metrics for the visible runs." });
  registerTable("ch-metrics",
    [{ key: "run", label: "run" }, { key: "eval", label: "evaluation" },
     ...RATE_METRICS.map((k) => ({ key: k, label: METRICS[k].label, num: true }))],
    evs.map(({ r, ev }) => Object.fromEntries([["run", r.label], ["eval", ev ? evalLabel(ev) : "—"],
      ...RATE_METRICS.map((k) => [k, fmt(metricOf(ev, k))])])));

  // 2. lead times (own axis: seconds)
  const leadRows = LEAD_METRICS.map((k) => ({
    label: METRICS[k].label, key: k,
    values: evs.map(({ r, ev }) => ({ name: r.label, color: colorOf(r.id), v: metricOf(ev, k) })),
  })).filter((row) => row.values.some((d) => d.v != null));
  const leadMax = Math.max(1, ...leadRows.flatMap((r) => r.values.map((d) => d.v || 0)));
  dotPlot($("ch-lead"), { rows: leadRows, domain: [0, Math.ceil(leadMax * 1.1)], format: (v) => `${v.toFixed(1)} s`,
    empty: "No lead-time metrics for the visible runs." });
  registerTable("ch-lead", [{ key: "run", label: "run" }, ...LEAD_METRICS.map((k) => ({ key: k, label: METRICS[k].label, num: true }))],
    evs.map(({ r, ev }) => Object.fromEntries([["run", r.label], ...LEAD_METRICS.map((k) => [k, fmt(metricOf(ev, k), "s")])])));

  // 3. training curves (needs per-run history)
  const withHist = runs.filter((r) => r.n_epochs > 0);
  ensureDetails(withHist.map((r) => r.id)).then(() => renderTrainChart(withHist));

  // 4. operating-point trade-off
  const opSeries = runs.map((r) => {
    const chosen = chosenEval(r);
    return {
      name: r.label, color: colorOf(r.id),
      points: r.evals.filter((e) => metricOf(e, "false_alarm_rate") != null && metricOf(e, "useful_warning_rate") != null)
        .map((e) => ({
          x: metricOf(e, "false_alarm_rate"), y: metricOf(e, "useful_warning_rate"),
          ring: chosen && e.id === chosen.id,
          faded: chosen && e.split !== chosen.split,
          r: e.threshold != null && Math.abs(e.threshold - S.f.thr) < 1e-6 ? 5.5 : 4,
          title: `${r.label}`,
          rows: [
            { color: colorOf(r.id), value: fmt(metricOf(e, "useful_warning_rate")), name: "useful (legacy)" },
            { value: fmt(metricOf(e, "false_alarm_rate")), name: "false-alarm rate" },
            { value: evalLabel(e), name: "" },
          ],
        })),
    };
  });
  scatter($("ch-op"), {
    series: opSeries, xDomain: [0, 1], yDomain: [0, 1],
    xFormat: (v) => v.toFixed(1), yFormat: (v) => v.toFixed(2),
    xTitle: "false-alarm rate →", yTitle: "useful-warning rate (legacy)",
    refs: [{ axis: "x", v: 0.2, label: "FA 0.20" }],
    empty: "No stored operating points for the visible runs.",
  });
  registerTable("ch-op", [{ key: "run", label: "run" }, { key: "eval", label: "evaluation" },
    { key: "u", label: "useful", num: true }, { key: "fa", label: "FA", num: true }],
    runs.flatMap((r) => r.evals.filter((e) => metricOf(e, "false_alarm_rate") != null)
      .map((e) => ({ run: r.label, eval: evalLabel(e), u: fmt(metricOf(e, "useful_warning_rate")), fa: fmt(metricOf(e, "false_alarm_rate")) }))));

  // 5. PR + 6. confusion (only where per-clip rows exist)
  const withRows = evs.map(({ r, ev }) => {
    const e = ev && ev.derived ? ev : r.evals.find((x) => x.derived);
    return e ? { r, ev: e, swapped: e !== ev } : null;
  }).filter(Boolean);
  const prSeries = withRows.filter((x) => x.ev.derived.pr_curve?.length).map(({ r, ev }) => ({
    name: `${r.label} (AP ${fmt(ev.derived.clip_AP_peak)})`, color: colorOf(r.id),
    points: [{ x: 0, y: ev.derived.pr_curve[0].precision },
      ...ev.derived.pr_curve.map((p) => ({ x: p.recall, y: p.precision, label: `${r.label} · thr ${p.threshold.toFixed(3)}` }))],
  }));
  const noRows = runs.filter((r) => !withRows.some((x) => x.r === r)).map((r) => r.label);
  lineChart($("ch-pr"), {
    series: prSeries, xDomain: [0, 1], yDomain: [0, 1], height: 250,
    xFormat: (v) => v.toFixed(1), yFormat: (v) => v.toFixed(2), xTitle: "recall →", yTitle: "precision",
    tipTitle: (x) => `recall ${x.toFixed(3)}`, showPoints: false,
    empty: "None of the visible runs has per-clip rows.",
  });
  if (noRows.length) {
    $("ch-pr").appendChild(el("div", { class: "note", text: `No per-clip rows (totals only): ${noRows.join(", ")}.` }));
  }
  registerTable("ch-pr", [{ key: "run", label: "run" }, { key: "t", label: "peak ≥", num: true },
    { key: "r", label: "recall", num: true }, { key: "p", label: "precision", num: true }],
    withRows.flatMap(({ r, ev }) => (ev.derived.pr_curve || []).map((p) =>
      ({ run: r.label, t: p.threshold.toFixed(4), r: fmt(p.recall), p: fmt(p.precision) }))));
  confusionGrid($("ch-cm"), withRows.map(({ r, ev, swapped }) => ({
    name: r.label, color: colorOf(r.id), cm: ev.derived.confusion,
    sub: `${ev.split_label}${ev.threshold != null ? ` · thr ${ev.threshold}/${ev.confirm}f` : ""} · ` +
         `P ${fmt(ev.derived.precision)} R ${fmt(ev.derived.recall)} F1 ${fmt(ev.derived.f1)}` +
         (swapped ? " · (from the run's per-clip evaluation)" : ""),
  })));
}

function renderTrainChart(runs) {
  const key = $("ch-train-metric").value;
  const series = runs.map((r) => {
    const d = S.details.get(r.id);
    const pts = (d?.history || []).map((h) => ({ x: h.epoch, y: h[key] ?? null }));
    const best = pts.find((p) => p.x === r.best_epoch);
    return { name: r.label, color: colorOf(r.id), points: pts, markers: best ? [best] : [] };
  });
  lineChart($("ch-train"), {
    series, height: 250, xTitle: "epoch →", yFormat: (v) => (key === "train_loss" ? v.toFixed(2) : v.toFixed(2)),
    xFormat: (v) => String(Math.round(v)), tipTitle: (x) => `epoch ${x}`,
    empty: "No per-epoch history for the visible runs (ensembles and baselines have none).",
  });
  registerTable("ch-train", [{ key: "run", label: "run" }, { key: "epoch", label: "epoch", num: true },
    { key: "v", label: key, num: true }, { key: "best", label: "selected" }],
    series.flatMap((s) => s.points.map((p) => ({ run: s.name, epoch: p.x, v: fmt(p.y, key === "train_loss" ? "loss" : "rate"),
      best: s.markers[0] && s.markers[0].x === p.x ? "✓" : "" }))));
}
$("ch-train-metric").addEventListener("change", () => renderTrainChart(visibleRuns().filter((r) => r.n_epochs > 0)));

// ---------------------------------------------------------------------------
// history
// ---------------------------------------------------------------------------

function setupHistoryMetric() {
  const sel = $("hist-metric");
  if (sel.options.length) return;
  for (const k of ["useful_warning_rate_v2", "useful_warning_rate", "false_alarm_rate", "mean_AP",
    "mean_lead_s", "f1"]) {
    sel.appendChild(el("option", { value: k, text: METRICS[k].label + (METRICS[k].derived ? " · derived" : "") }));
  }
  sel.addEventListener("change", renderHistory);
  $("hist-source").addEventListener("change", renderHistory);
}

function renderHistory() {
  setupHistoryMetric();
  const key = $("hist-metric").value, source = $("hist-source").value;
  const runs = filteredRuns().slice().sort((a, b) =>
    (a.date || "").localeCompare(b.date || "") || a.id.localeCompare(b.id));
  const pick = (r) => (source === "headline"
    ? r.evals.find((e) => e.id === r.headline_eval)
    : r.evals.find((e) => e.kind === "selection_val") || r.evals.find((e) => e.id === r.headline_eval));
  const kind = METRICS[key].kind;
  const bySplit = new Map();
  runs.forEach((r, i) => {
    const ev = pick(r);
    const v = metricOf(ev, key);
    if (v == null || !ev) return;
    if (S.f.split && ev.split !== S.f.split) return;
    if (!bySplit.has(ev.split)) bySplit.set(ev.split, []);
    bySplit.get(ev.split).push({ x: i, y: v, label: `${r.label} · ${S.splits[ev.split]?.label || ev.split}` });
  });
  const series = SPLIT_ORDER.filter((s) => bySplit.has(s)).map((s) => ({
    name: S.splits[s]?.label || s, color: slotColor(SPLIT_ORDER.indexOf(s)), points: bySplit.get(s), showPoints: true,
  }));
  // Tick at the first run of each distinct date.
  const ticks = [];
  runs.forEach((r, i) => { if (i === 0 || shortDate(r.date) !== shortDate(runs[i - 1].date)) ticks.push(i); });
  const yMax = kind === "rate" ? 1 : Math.max(1, ...series.flatMap((s) => s.points.map((p) => p.y))) * 1.1;
  lineChart($("ch-hist"), {
    series, height: 300, xDomain: [0, Math.max(1, runs.length - 1)], yDomain: [0, yMax],
    xTicks: ticks, xFormat: (i) => shortDate(runs[Math.round(i)]?.date).slice(5),
    yFormat: (v) => (kind === "rate" ? v.toFixed(2) : `${v.toFixed(1)} s`),
    xTitle: "runs, in commit order →", yTitle: METRICS[key].label,
    tipTitle: (i) => `${runs[i]?.label} · ${shortDate(runs[i]?.date)}`,
    refs: key === "false_alarm_rate" ? [{ axis: "y", v: 0.2, label: "target ≤ 0.20" }] : [],
    onClick: (_, i) => { const r = runs[i]; if (r) { toggleSelect(r.id, true); showView("compare"); } },
    empty: "No run in the filtered set carries this metric.",
  });
  // Direct label on the latest point of each split series (selective labelling).
  registerTable("ch-hist", [{ key: "run", label: "run" }, { key: "date", label: "date" }, { key: "split", label: "split" },
    { key: "v", label: METRICS[key].label, num: true }],
    series.flatMap((s) => s.points.map((p) => ({ run: runs[p.x].label, date: shortDate(runs[p.x].date), split: s.name,
      v: fmt(p.y, kind) }))));

  const tbl = $("hist-table");
  tbl.replaceChildren(el("tr", {}, ["date", "run", "family", "headline evaluation", "best epoch",
    "useful v2", "useful", "FA", "mean AP", "lead", "F1 (derived)", "notes"].map((h, i) =>
    el("th", { class: i >= 4 && i <= 10 ? "num" : "", text: h }))));
  for (const r of runs.slice().reverse()) {
    const ev = r.evals.find((e) => e.id === r.headline_eval);
    const tr = el("tr", { class: "clickable" + (S.selected.includes(r.id) ? " sel" : ""), title: "add to comparison" }, [
      el("td", { class: "num", text: shortDate(r.date) }), el("td", { text: r.label }), el("td", { text: r.family }),
      el("td", { text: ev ? evalLabel(ev) : "—" }), el("td", { class: "num", text: r.best_epoch ?? "—" }),
      ...["useful_warning_rate_v2", "useful_warning_rate", "false_alarm_rate", "mean_AP"].map((k) =>
        el("td", { class: "num", text: fmt(metricOf(ev, k)) })),
      el("td", { class: "num", text: fmt(metricOf(ev, "mean_lead_s") ?? metricOf(ev, "mTTA_s"), "s") }),
      el("td", { class: "num", text: fmt(metricOf(ev, "f1")) }),
      el("td", { class: "muted", text: r.notes.join(" ") }),
    ]);
    tr.addEventListener("click", () => { toggleSelect(r.id, true); showView("compare"); });
    tbl.appendChild(tr);
  }
}

// ---------------------------------------------------------------------------
// drill-down
// ---------------------------------------------------------------------------

function drillEvals() {
  return S.runs.flatMap((r) => r.evals.filter((e) => e.has_rows).map((e) => ({ r, e })));
}

async function renderDrill() {
  const sel = $("drill-eval");
  const opts = drillEvals();
  if (!sel.options.length) {
    for (const { r, e } of opts) sel.appendChild(el("option", { value: e.id, text: `${r.label} · ${evalLabel(e)}` }));
    sel.addEventListener("change", () => { S.drill.evalId = sel.value; S.drill.rows = []; renderDrill(); });
    const oc = $("drill-oc");
    for (const k of ["TP", "FN", "FP", "TN"]) {
      const b = el("button", { class: "chip", "aria-pressed": "true" }, [
        el("span", { class: `oc ${k}` }, [el("i"), OUTCOME[k].label])]);
      b.addEventListener("click", () => {
        S.drill.outcomes.has(k) ? S.drill.outcomes.delete(k) : S.drill.outcomes.add(k);
        b.setAttribute("aria-pressed", String(S.drill.outcomes.has(k)));
        renderDrill();
      });
      oc.appendChild(b);
    }
    const ab = el("button", { class: "chip", "aria-pressed": "false", text: "only peak ≥ Risk filter" });
    ab.addEventListener("click", () => {
      S.drill.aboveOnly = !S.drill.aboveOnly;
      ab.setAttribute("aria-pressed", String(S.drill.aboveOnly));
      renderDrill();
    });
    oc.appendChild(ab);
    // Default: the held-out read if present.
    const held = opts.find((x) => x.e.held_out) || opts[0];
    if (held) S.drill.evalId = held.e.id;
    sel.value = S.drill.evalId || "";
  }
  const cur = opts.find((x) => x.e.id === S.drill.evalId);
  if (!cur) { $("drill-sub").textContent = "No evaluation with per-clip rows was found."; return; }
  if (!S.drill.rows.length) {
    $("drill-sub").textContent = "loading rows…";
    S.drill.rows = (await api(`/api/rows/${encodeURIComponent(cur.e.id)}`)).rows;
  }
  const { r, e } = cur;
  $("drill-sub").textContent = `${r.label} · ${e.split_label}` +
    (e.threshold != null ? ` · decision rule thr ${e.threshold} held ${e.confirm} frames` : "") +
    ` · source ${e.source}. ${S.splits[e.split]?.note || ""}`;

  const d = e.derived || {}, cm = d.confusion || {};
  const tiles = $("drill-tiles");
  tiles.replaceChildren(
    ...["TP", "FN", "FP", "TN"].map((k) => el("div", { class: "tile" }, [
      el("div", { class: "t-label" }, [el("span", { class: `oc ${k}` }, [el("i"), k])]),
      el("div", { class: "t-val", text: String(cm[k] ?? "—") }),
      el("div", { class: "t-sub", text: OUTCOME[k].label })])),
    ...[["precision", d.precision], ["recall", d.recall], ["F1", d.f1]].map(([l, v]) =>
      el("div", { class: "tile derived" }, [el("div", { class: "t-label", text: l }),
        el("div", { class: "t-val", text: fmt(v) })])),
    ...["useful_warning_rate_v2", "useful_warning_rate", "false_alarm_rate"].filter((k) => metricOf(e, k) != null)
      .map((k) => el("div", { class: "tile" }, [el("div", { class: "t-label", text: METRICS[k].label }),
        el("div", { class: "t-val", text: fmt(metricOf(e, k)) })])),
  );

  const q = S.f.clip.toLowerCase();
  let rows = S.drill.rows.filter((x) => S.drill.outcomes.has(x.outcome) &&
    (!q || x.clip_id.toLowerCase().includes(q)) &&
    (!S.drill.aboveOnly || (x.peak_prob != null && x.peak_prob >= S.f.thr)));
  const { key, dir } = S.drill.sort;
  rows = rows.slice().sort((a, b) => {
    const va = a[key], vb = b[key];
    if (va == null) return 1; if (vb == null) return -1;
    return (typeof va === "number" ? va - vb : String(va).localeCompare(String(vb), undefined, { numeric: true })) * dir;
  });
  $("drill-count").textContent = `${rows.length} of ${S.drill.rows.length} clips`;

  renderStrip(rows);
  renderDrillTable(rows);
  if (S.drill.clip) renderClipPanel(S.drill.clip, false);
}

function hashJitter(s) {
  let h = 0;
  for (const c of s) h = (h * 31 + c.charCodeAt(0)) | 0;
  return ((h >>> 0) % 1000) / 1000 - 0.5;
}

function renderStrip(rows) {
  const series = ["TP", "FN", "FP", "TN"].map((k) => ({
    name: OUTCOME[k].label, color: OUTCOME[k].color(),
    points: rows.filter((x) => x.outcome === k && x.peak_prob != null).map((x) => ({
      x: x.peak_prob, y: (x.label === 1 ? 1 : 0) + hashJitter(x.clip_id) * 0.5, clip: x.clip_id,
      title: `clip ${x.clip_id}`,
      rows: [{ color: OUTCOME[k].color(), value: x.peak_prob.toFixed(3), name: "peak_prob" },
             { value: OUTCOME[k].label, name: x.label ? "positive" : "negative" },
             { value: x.verdict_v2 || x.verdict || "—", name: "verdict" }],
    })),
  }));
  scatter($("ch-strip"), {
    series, xDomain: [0, 1], yDomain: [-0.5, 1.5], height: 220,
    xFormat: (v) => v.toFixed(1), yFormat: (v) => (v === 1 ? "pos" : v === 0 ? "neg" : ""),
    xTitle: "whole-clip peak_prob →",
    refs: [{ axis: "x", v: S.f.thr, label: `Risk ≥ ${S.f.thr.toFixed(2)}` }],
    onClick: (p) => selectClip(p.clip),
    empty: "No clips with a peak probability (the reactive baseline has no score).",
  });
  registerTable("ch-strip", [{ key: "clip", label: "clip" }, { key: "label", label: "label" },
    { key: "oc", label: "outcome" }, { key: "p", label: "peak_prob", num: true }],
    rows.map((x) => ({ clip: x.clip_id, label: x.label ? "pos" : "neg", oc: x.outcome, p: fmt(x.peak_prob) })));
}

const DRILL_COLS = [
  ["clip_id", "clip"], ["label", "label"], ["outcome", "outcome"], ["verdict", "verdict"],
  ["verdict_v2", "verdict v2"], ["first_fire_t", "first fire (s)", 1], ["alert_t", "alert (s)", 1],
  ["event_t", "event (s)", 1], ["lead_s", "lead (s)", 1], ["peak_prob", "peak_prob", 1], ["data", "data"],
];

function renderDrillTable(rows) {
  const tbl = $("drill-table");
  const head = el("tr", {}, DRILL_COLS.map(([k, l, num]) => {
    const arrow = S.drill.sort.key === k ? (S.drill.sort.dir > 0 ? " ▲" : " ▼") : "";
    const th = el("th", { class: `sortable${num ? " num" : ""}`, text: l + arrow });
    if (k !== "data") th.addEventListener("click", () => {
      S.drill.sort = { key: k, dir: S.drill.sort.key === k ? -S.drill.sort.dir : 1 };
      renderDrill();
    });
    return th;
  }));
  tbl.replaceChildren(head);
  for (const x of rows.slice(0, 500)) {
    const tr = el("tr", { class: "clickable" + (S.drill.clip === x.clip_id ? " sel" : "") }, [
      el("td", { class: "num", text: x.clip_id }),
      el("td", { text: x.label ? "pos" : "neg" }),
      el("td", {}, [el("span", { class: `oc ${x.outcome}` }, [el("i"), x.outcome])]),
      el("td", { text: x.verdict || "—" }), el("td", { text: x.verdict_v2 || "—" }),
      ...["first_fire_t", "alert_t", "event_t", "lead_s"].map((k) => el("td", { class: "num", text: x[k] == null ? "—" : x[k].toFixed(2) })),
      el("td", { class: "num", text: fmt(x.peak_prob) }),
      el("td", { class: "avail", text: [x.has_tracks ? "tracks" : null, x.has_risk_curve ? "curve" : null].filter(Boolean).join(" · ") || "video only" }),
    ]);
    tr.addEventListener("click", () => selectClip(x.clip_id));
    tbl.appendChild(tr);
  }
}

function selectClip(cid) {
  S.drill.clip = cid;
  renderDrill();
  renderClipPanel(cid, true);
}

async function renderClipPanel(cid, fetchNow) {
  const row = S.drill.rows.find((x) => x.clip_id === cid);
  $("clip-title").textContent = `Clip ${cid}`;
  const info = $("clip-info");
  const btn = $("clip-open-replay");
  btn.classList.remove("hidden");
  btn.onclick = () => { showView("live"); openInLive(cid, `${currentDrillRunLabel()} · ${row ? row.outcome : ""}`); };
  info.className = "";
  info.replaceChildren(el("dl", { class: "kv" }, row ? [
    el("dt", { text: "outcome" }), el("dd", {}, [el("span", { class: `oc ${row.outcome}` }, [el("i"), OUTCOME[row.outcome].label])]),
    el("dt", { text: "verdict" }), el("dd", { text: `${row.verdict || "—"}${row.verdict_v2 ? ` · v2 ${row.verdict_v2}` : ""}` }),
    el("dt", { text: "ground truth" }), el("dd", { text: row.label ? `alert ${fmt(row.alert_t, "s")} · event ${fmt(row.event_t, "s")}` : "negative clip" }),
    el("dt", { text: "decision rule" }), el("dd", { text: row.first_fire_t != null ? `fired at ${fmt(row.first_fire_t, "s")}` : "did not fire" }),
    el("dt", { text: "peak" }), el("dd", { text: `${fmt(row.peak_prob)}${row.peak_t != null ? ` at ${fmt(row.peak_t, "s")}` : ""}` }),
  ] : []));
  const host = $("ch-tracks");
  if (!row || !row.has_tracks) {
    host.replaceChildren(el("div", { class: "note", text: "No tracked-object data is stored for this clip, so there are no per-object risk trajectories. " +
      "Live replay still plays the video with ground-truth markers" + (row?.has_risk_curve ? " and the learned risk curve." : ".") }));
    return;
  }
  if (!fetchNow && host.dataset.clip === cid) { drawTrackRisk(host, host._clip, row); return; }
  host.classList.add("loading");
  try {
    const clip = await api(`/api/clip/${encodeURIComponent(cid)}`);
    host._clip = clip; host.dataset.clip = cid;
    registerClasses(new Set(clip.frames.flatMap((f) => f.tracks.map((t) => t.cls))));
    drawTrackRisk(host, clip, row);
  } catch (e) {
    host.replaceChildren(el("div", { class: "note warn", text: String(e.message || e) }));
  } finally { host.classList.remove("loading"); }
}

const currentDrillRunLabel = () => {
  const x = drillEvals().find((o) => o.e.id === S.drill.evalId);
  return x ? `${x.r.label} · ${x.e.split_label}` : "";
};

function drawTrackRisk(host, clip, row) {
  // One line per tracked object: its per-actor collision probability over time.
  const tracks = new Map();
  for (const f of clip.frames) for (const t of f.tracks) {
    if (t.p == null || S.f.classOff.has(t.cls)) continue;
    if (!tracks.has(t.id)) tracks.set(t.id, { id: t.id, cls: t.cls, pts: [], peak: 0 });
    const tr = tracks.get(t.id);
    tr.pts.push({ x: f.t, y: t.p });
    tr.peak = Math.max(tr.peak, t.p);
  }
  const ranked = [...tracks.values()].sort((a, b) => b.peak - a.peak);
  const top = ranked.slice(0, 5), rest = ranked.slice(5);
  const gray = cssVar("--neutral-mark");
  const series = [
    ...rest.map((t) => ({ name: `${t.cls} ID-${t.id}`, color: gray, points: t.pts, faded: true })),
    ...top.map((t, i) => ({ name: `${t.cls} ID-${t.id} (peak ${t.peak.toFixed(2)})`, color: slotColor(i), points: t.pts })),
  ];
  const refs = [{ axis: "y", v: S.f.thr, label: `Risk ≥ ${S.f.thr.toFixed(2)}` }];
  const gt = clip.gt || {};
  if (gt.alert_t != null) refs.push({ axis: "x", v: gt.alert_t, label: "alert", color: cssVar("--s1"), dash: "5 3" });
  if (gt.event_t != null) refs.push({ axis: "x", v: gt.event_t, label: "impact", color: cssVar("--ink"), solid: true });
  if (row && row.first_fire_t != null) refs.push({ axis: "x", v: row.first_fire_t, label: "fired", color: cssVar("--serious"), dash: "2 3" });
  host.replaceChildren();
  const legend = el("div", { class: "legend" }, top.map((t, i) =>
    el("span", { class: "lg-item" }, [el("span", { class: "swatch line", style: { background: slotColor(i) } }), `${t.cls} ID-${t.id}`])));
  if (rest.length) legend.appendChild(el("span", { class: "lg-item" }, [el("span", { class: "swatch line", style: { background: gray } }), `${rest.length} other actors`]));
  const chartHost = el("div");
  host.append(el("div", { class: "sub", text: `Per-object risk trajectories · ${clip.overlay_source === "full" ? "per-actor geometric collision probability (threshold system)" : "slim overlay"} · top 5 by peak coloured` }), legend, chartHost);
  lineChart(chartHost, {
    series, height: 230, yDomain: [0, 1], xTitle: "time (s) →", yFormat: (v) => v.toFixed(1), legend: false,
    xFormat: (v) => v.toFixed(0), tipTitle: (x) => `t = ${x.toFixed(2)} s`, refs,
    empty: "No actor in the visible classes has a collision probability.",
  });
}

boot().catch((e) => {
  $("status").textContent = `failed: ${e.message || e}`;
  console.error(e);
});
