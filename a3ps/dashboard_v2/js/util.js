// Shared helpers: DOM, formatting, metric catalogue, colour roles, API access.

export const $ = (id) => document.getElementById(id);

export function el(tag, attrs = {}, children = []) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "text") n.textContent = v;
    else if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (k === "style" && typeof v === "object") Object.assign(n.style, v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of [].concat(children)) {
    if (c == null) continue;
    n.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return n;
}

const SVGNS = "http://www.w3.org/2000/svg";
export function svg(tag, attrs = {}, text) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) n.setAttribute(k, v);
  if (text != null) n.textContent = text;   // labels are data: never innerHTML
  return n;
}

export const cssVar = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// Categorical slots in fixed order (validated palette, both modes).
export const SLOTS = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--s8"];
export const slotColor = (i) => cssVar(SLOTS[i % SLOTS.length]);
export const STATUS = { good: "--good", warning: "--warning", serious: "--serious", critical: "--critical" };
export const statusColor = (s) => cssVar(STATUS[s]);
export const OUTCOME = {
  TP: { label: "true positive", color: () => statusColor("good") },
  FN: { label: "missed (FN)", color: () => statusColor("critical") },
  FP: { label: "false alarm (FP)", color: () => statusColor("serious") },
  TN: { label: "true negative", color: () => cssVar("--neutral-mark") },
};

export function fmt(v, kind = "rate") {
  if (v == null || Number.isNaN(v)) return "—";
  if (kind === "rate") return v.toFixed(3);
  if (kind === "s") return `${v.toFixed(2)} s`;
  if (kind === "int") return String(Math.round(v));
  if (kind === "loss") return v.toFixed(4);
  return String(v);
}

// Metric catalogue. `derived` metrics are computed by the dashboard from per-clip
// rows; everything else is the pipeline's own number, shown under its own name.
export const METRICS = {
  useful_warning_rate_v2: { label: "useful-warning rate · v2 (episode)", kind: "rate", better: "up" },
  useful_warning_rate: { label: "useful-warning rate · legacy", kind: "rate", better: "up" },
  false_alarm_rate: { label: "false-alarm rate", kind: "rate", better: "down" },
  mean_AP: { label: "mean AP (pre-event cutoffs)", kind: "rate", better: "up" },
  AP: { label: "AP (whole-clip peak)", kind: "rate", better: "up" },
  detection_recall: { label: "detection recall", kind: "rate", better: "up" },
  frac_premature: { label: "fraction premature", kind: "rate", better: "down" },
  precision: { label: "precision", kind: "rate", better: "up", derived: true },
  recall: { label: "recall", kind: "rate", better: "up", derived: true },
  f1: { label: "F1", kind: "rate", better: "up", derived: true },
  mean_lead_s: { label: "mean lead (s)", kind: "s", better: "up" },
  mean_lead_vs_event_v2: { label: "mean lead vs event · v2 (s)", kind: "s", better: "up" },
  mTTA_s: { label: "mTTA (s)", kind: "s", better: "up" },
};
export const RATE_METRICS = ["useful_warning_rate_v2", "useful_warning_rate", "false_alarm_rate",
  "mean_AP", "AP", "precision", "recall", "f1"];
export const LEAD_METRICS = ["mean_lead_s", "mean_lead_vs_event_v2", "mTTA_s"];

export function metricOf(ev, key) {
  if (!ev) return null;
  const m = ev.metrics || {};
  if (m[key] != null) return m[key];
  const d = ev.derived || {};
  return d[key] != null ? d[key] : null;
}

const cache = new Map();
export async function api(path) {
  if (cache.has(path)) return cache.get(path);
  const p = fetch(path, { cache: "no-store" }).then(async (r) => {
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.error || `${r.status} ${path}`);
    return body;
  });
  cache.set(path, p);
  p.catch(() => cache.delete(path));
  return p;
}

export function debounce(fn, ms = 120) {
  let t = null;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

export const shortDate = (iso) => (iso ? iso.slice(0, 10) : "—");
