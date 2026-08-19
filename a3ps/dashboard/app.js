"use strict";

// A3PS dashboard. Reads a clip's events.json (the ClipResult schema) and drives
// the video-overlay canvas, agent console, flash banner, and risk timeline.
// Everything is schema-driven, so real Phase-4 output plugs in unchanged.

const $ = (id) => document.getElementById(id);
const els = {
  select: $("clip-select"),
  stage: document.querySelector(".stage"),
  video: $("video"),
  canvas: $("overlay"),
  debug: $("debug"),
  banner: $("banner"),
  bannerText: $("banner-text"),
  vignette: $("vignette"),
  threats: $("threats"),
  threshold: $("threshold"),
  eventLog: $("event-log"),
  timeline: $("risk-timeline"),
  gapLabel: $("gap-label"),
  compareWrap: $("compare-wrap"),
  compare: $("model-compare"),
  comparePanel: $("compare-panel"),
  compareRows: $("compare-rows"),
  compareExplain: $("compare-explain"),
  systemBadge: $("system-badge"),
  threatsNote: $("threats-note"),
  scrubber: $("scrubber"),
  timeReadout: $("time-readout"),
  btnBack: $("btn-back"),
  btnPlay: $("btn-play"),
  btnFwd: $("btn-fwd"),
  layers: {
    masks: $("layer-masks"), trails: $("layer-trails"),
    predictions: $("layer-predictions"), corridor: $("layer-corridor"),
  },
};

const RISK = {
  safe: [46, 204, 113], caution: [245, 166, 35], danger: [231, 76, 60],
};
const riskRGB = (lvl) => RISK[lvl] || RISK.safe;
const rgba = ([r, g, b], a) => `rgba(${r},${g},${b},${a})`;

const state = {
  clipId: null, meta: {}, frames: [],
  currentIndex: -1, currentFrame: null,
  events: [], perFrameRisk: [], thresholdSeries: [], threshold: 0.65,
  reactiveMarkers: [],     // {ev, t, gap} per VIRTUAL_BRAKE with a reactive trigger
  duration: 0,
  log: [],                 // rendered event-log entries (occurred events)
  userScrolledUp: false,   // suppress auto-scroll when the user scrolled up
  risk: null,              // clips/<id>/risk.json — learned-vs-threshold comparison
};

const nowMs = () => performance.now();
const LLM_DELAY_MS = 800;  // reveal AI narrative this long after the template

// ---------------------------------------------------------------------------
// geometry helpers (reactive-ADAS marker: distance of actor to ego corridor)
// ---------------------------------------------------------------------------

function pointInPoly(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) &&
        x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-9) + xi) inside = !inside;
  }
  return inside;
}
function distToSeg(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const l2 = dx * dx + dy * dy || 1e-9;
  let t = ((px - ax) * dx + (py - ay) * dy) / l2;
  t = Math.max(0, Math.min(1, t));
  const cx = ax + t * dx, cy = ay + t * dy;
  return Math.hypot(px - cx, py - cy);
}
function distToPoly(x, y, poly) {
  if (pointInPoly(x, y, poly)) return 0;
  let d = Infinity;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    d = Math.min(d, distToSeg(x, y, poly[j][0], poly[j][1], poly[i][0], poly[i][1]));
  }
  return d;
}

// Reactive-ADAS comparison. For each VIRTUAL_BRAKE, find the first frame where
// that event's actor is physically close to the ego corridor — what a naive
// proximity-based ADAS would trigger on. Prefer BEV metres (centroid_bev to a
// BEV corridor polygon < 2.0 m); fall back to image pixels (< 8% of frame
// height) when BEV isn't available. A3PS (the VIRTUAL_BRAKE) fires earlier.
function computeReactiveMarkers() {
  const pxThresh = 0.08 * (state.meta.height || 720);
  const markers = [];
  for (const ev of state.events) {
    if (ev.type !== "VIRTUAL_BRAKE") continue;
    let rt = null;
    for (const f of state.frames) {
      const tr = (f.tracks || []).find((t) => t.id === ev.actor_id);
      if (!tr) continue;
      const bevPoly = f.ego && f.ego.corridor_poly_bev;
      let near;
      if (tr.centroid_bev && bevPoly) {
        near = distToPoly(tr.centroid_bev[0], tr.centroid_bev[1], bevPoly) < 2.0;
      } else {
        const imgPoly = f.ego && f.ego.corridor_poly_img;
        near = imgPoly &&
          distToPoly(tr.centroid_img[0], tr.centroid_img[1], imgPoly) < pxThresh;
      }
      if (near) { rt = f.t; break; }
    }
    if (rt != null) markers.push({ ev, t: rt, gap: rt - ev.t });
  }
  return markers;
}

// ---------------------------------------------------------------------------
// data loading
// ---------------------------------------------------------------------------

function showStageMessage(html) {
  let el = document.getElementById("stage-msg");
  if (!el) {
    el = document.createElement("div");
    el.id = "stage-msg";
    el.className = "stage-msg";
    els.stage.appendChild(el);
  }
  el.innerHTML = html;
  el.classList.remove("hidden");
}

function hideStageMessage() {
  const el = document.getElementById("stage-msg");
  if (el) el.classList.add("hidden");
}

async function loadManifest() {
  let raw = [];
  let fetchFailed = false;
  try {
    raw = await (await fetch("clips/manifest.json", { cache: "no-store" })).json();
  } catch (e) {
    console.error("manifest load failed", e);
    fetchFailed = true;
  }

  // Two accepted shapes: the grouped/labelled form written by
  // scripts/update_manifest.py ({clips:[{id,group,label}]}), and a plain array
  // of ids from older manifests. Normalise to the former.
  const clips = Array.isArray(raw)
    ? raw.map((id) => ({ id, group: "", label: id }))
    : (raw && Array.isArray(raw.clips) ? raw.clips : []);

  els.select.innerHTML = "";
  let group = null, target = els.select;
  clips.forEach((c) => {
    if (c.group && c.group !== group) {
      group = c.group;
      target = document.createElement("optgroup");
      target.label = group;
      els.select.appendChild(target);
    } else if (!c.group) {
      target = els.select;
    }
    const o = document.createElement("option");
    o.value = c.id;
    o.textContent = c.label || c.id;
    target.appendChild(o);
  });

  if (clips.length) {
    hideStageMessage();
    loadClip(clips[0].id);
    return;
  }

  // Never fail silently: a blank stage with an empty dropdown is impossible to
  // diagnose from the UI, so say what went wrong and what to do about it.
  showStageMessage(
    fetchFailed
      ? `<strong>Could not load clips/manifest.json</strong>
         <p>Is the server running from the repo root?</p>
         <pre>python scripts/serve_dashboard.py --port 8000</pre>`
      : `<strong>No clips listed in the manifest</strong>
         <p>If the manifest file looks fine, your browser is probably running a
         cached copy of <code>app.js</code> — hard-refresh with
         <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>R</kbd>.</p>
         <p>Otherwise rebuild it:</p>
         <pre>python scripts/update_manifest.py --require-video</pre>`
  );
}

async function loadClip(id) {
  if (!id) return;
  state.clipId = id;
  Object.assign(state, {
    frames: [], events: [], perFrameRisk: [], thresholdSeries: [],
    currentIndex: -1, currentFrame: null, reactiveMarkers: [],
    log: [], userScrolledUp: false, risk: null,
  });

  // events.json carries the full per-frame track state and drives the video
  // overlay. It is ~20-50 MB per clip, so the learned-head demo clips ship
  // WITHOUT it (they only need risk.json). A missing file is therefore normal,
  // not an error: the overlay simply stays empty for those clips.
  try {
    const res = await fetch(`clips/${id}/events.json`, { cache: "no-store" });
    if (res.ok) {
      const clip = await res.json();
      state.meta = clip.meta || {};
      state.frames = (clip.frames || []).slice().sort((a, b) => a.t - b.t);
      state.events = (clip.events || []).slice().sort((a, b) => a.t - b.t);
    }
  } catch (e) { console.warn(`no overlay data for ${id}`, e); }

  // risk.json is the Phase IV comparison payload (learned curve + the old
  // threshold system's curve + ground-truth markers). Also optional: the
  // original dev clips predate it.
  try {
    const res = await fetch(`clips/${id}/risk.json`, { cache: "no-store" });
    if (res.ok) state.risk = await res.json();
  } catch (e) { console.warn(`no risk.json for ${id}`, e); }

  // Derived series.
  state.perFrameRisk = state.frames.map((f) =>
    (f.tracks || []).reduce((m, t) =>
      Math.max(m, (t.prediction && t.prediction.collision_prob) || 0), 0));
  const last = state.frames[state.frames.length - 1];
  const base = (state.meta.config && state.meta.config.base_threshold) || 0.75;
  // Threshold as a step function over time (per-frame active_threshold).
  state.thresholdSeries = state.frames.map((f) =>
    (f.context && f.context.active_threshold != null) ? f.context.active_threshold : base);
  state.threshold = state.thresholdSeries[0] || 0.65;
  state.reactiveMarkers = computeReactiveMarkers();
  // Duration: prefer the overlay's last frame; fall back to the risk payload so
  // the comparison view scales correctly on clips that ship without events.json.
  state.duration = last ? last.t
    : (state.risk && state.risk.duration_s) ? state.risk.duration_s : 0;

  renderCompare();
  initEventLog();

  // Demo raw.mp4 files are gitignored (~23 MB each), so after a fresh clone the
  // curves and verdicts are present but the video is not. Say so instead of
  // showing a black rectangle.
  hideStageMessage();
  try {
    const head = await fetch(`clips/${id}/raw.mp4`, { method: "HEAD" });
    if (!head.ok) throw new Error(String(head.status));
    els.video.src = `clips/${id}/raw.mp4`;
    els.video.load();
  } catch (e) {
    els.video.removeAttribute("src");
    els.video.load();
    showStageMessage(
      `<strong>No video for clip ${id}</strong>
       <p>Risk curves and verdicts below are still valid — only the video file is
       missing. Demo videos are gitignored, so recreate them with:</p>
       <pre>python scripts/build_dashboard_demo.py --clips auto --n 6</pre>`
    );
  }
}

// ---------------------------------------------------------------------------
// frame sync
// ---------------------------------------------------------------------------

function binarySearchFrame(t) {
  const f = state.frames;
  let lo = 0, hi = f.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (f[mid].t <= t) { ans = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return ans;
}
function updateCurrentFrame() {
  if (!state.frames.length) { state.currentFrame = null; state.currentIndex = -1; return; }
  const idx = binarySearchFrame(els.video.currentTime);
  state.currentIndex = idx;
  const margin = 1.5 / (state.meta.fps || 30);
  const last = state.frames.length - 1;
  state.currentFrame =
    idx < 0 || (idx === last && els.video.currentTime > state.frames[last].t + margin)
      ? null : state.frames[idx];
}

// ---------------------------------------------------------------------------
// overlay canvas
// ---------------------------------------------------------------------------

function sizeCanvasToVideo() {
  const vw = els.video.videoWidth, vh = els.video.videoHeight;
  if (vw && vh && (els.canvas.width !== vw || els.canvas.height !== vh)) {
    els.canvas.width = vw; els.canvas.height = vh;
  }
}
function scaleFactors() {
  const mw = state.meta.width || els.canvas.width || 1;
  const mh = state.meta.height || els.canvas.height || 1;
  return [els.canvas.width / mw, els.canvas.height / mh];
}
function polyPath(ctx, poly, sx, sy) {
  ctx.beginPath();
  poly.forEach(([x, y], i) => (i ? ctx.lineTo(x * sx, y * sy) : ctx.moveTo(x * sx, y * sy)));
  ctx.closePath();
}
function drawOverlay() {
  const ctx = els.canvas.getContext("2d");
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  const cf = state.currentFrame;
  els.debug.textContent = cf ? `frame ${cf.frame_idx}` : "frame –";
  if (!cf) return;
  const [sx, sy] = scaleFactors();
  const L = els.layers;

  if (L.corridor.checked && cf.ego && cf.ego.corridor_poly_img) {
    polyPath(ctx, cf.ego.corridor_poly_img, sx, sy);
    ctx.fillStyle = "rgba(77,163,255,0.12)"; ctx.fill();
    ctx.strokeStyle = "rgba(77,163,255,0.5)"; ctx.lineWidth = 1.5; ctx.stroke();
  }
  const tracks = cf.tracks || [];
  for (const tr of tracks) {
    const c = riskRGB(tr.risk_level);
    if (L.masks.checked && tr.mask_poly && tr.mask_poly.length > 2) {
      polyPath(ctx, tr.mask_poly, sx, sy);
      ctx.fillStyle = rgba(c, 0.35); ctx.fill();
      ctx.strokeStyle = rgba(c, 1); ctx.lineWidth = 2; ctx.stroke();
    }
    if (L.trails.checked && tr.history_img && tr.history_img.length > 1) {
      const h = tr.history_img;
      ctx.lineWidth = 2;
      for (let i = 0; i < h.length - 1; i++) {
        ctx.strokeStyle = rgba(c, (i + 1) / (h.length - 1));
        ctx.beginPath();
        ctx.moveTo(h[i][0] * sx, h[i][1] * sy);
        ctx.lineTo(h[i + 1][0] * sx, h[i + 1][1] * sy);
        ctx.stroke();
      }
    }
    if (L.predictions.checked && tr.prediction && tr.prediction.mean_img) {
      const m = tr.prediction.mean_img, n = m.length;
      m.forEach(([x, y], i) => {
        const f = i / Math.max(1, n - 1);
        ctx.fillStyle = rgba(c, 1 - 0.85 * f);
        ctx.beginPath();
        ctx.arc(x * sx, y * sy, Math.max(1, 4 * (1 - f)), 0, Math.PI * 2);
        ctx.fill();
      });
    }
  }
  ctx.font = "16px ui-monospace, Menlo, Consolas, monospace";
  ctx.textBaseline = "bottom";
  for (const tr of tracks) {
    const c = riskRGB(tr.risk_level);
    const px = tr.bbox[0] * sx, py = tr.bbox[1] * sy;
    const label = `${tr.cls} ID-${tr.id}`;
    const w = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(0,0,0,0.6)"; ctx.fillRect(px, py - 20, w + 10, 20);
    ctx.fillStyle = rgba(c, 1); ctx.fillText(label, px + 5, py - 3);
  }
}

// ---------------------------------------------------------------------------
// agent console
// ---------------------------------------------------------------------------

function updateThreats() {
  const cf = state.currentFrame;
  const ranked = !cf ? [] : (cf.tracks || [])
    .filter((t) => t.prediction && t.prediction.collision_prob != null)
    .sort((a, b) => b.prediction.collision_prob - a.prediction.collision_prob)
    .slice(0, 3);
  if (!ranked.length) { els.threats.innerHTML = '<li class="empty">—</li>'; return; }
  els.threats.innerHTML = "";
  for (const tr of ranked) {
    const p = tr.prediction.collision_prob;
    const c = rgba(riskRGB(tr.risk_level), 1);
    const li = document.createElement("li");
    li.innerHTML =
      `<span class="who">${tr.cls} ID-${tr.id}</span>` +
      `<span class="bar"><span style="width:${Math.round(p * 100)}%;background:${c}"></span></span>` +
      `<span class="val">${p.toFixed(2)}</span>`;
    els.threats.appendChild(li);
  }
}

function learnedRiskAt(t) {
  const curve = state.risk && state.risk.learned && state.risk.learned.curve;
  if (!curve || !curve.length) return null;
  let best = null, bestDt = Infinity;
  for (const [ct, cp] of curve) {
    const dt = Math.abs(ct - t);
    if (dt < bestDt) { bestDt = dt; best = cp; }
  }
  return bestDt <= 0.5 ? best : null;
}

function updateThreshold() {
  // Phase IV clips are driven by the learned head, whose decision rule is a flat
  // probability threshold held for N consecutive frames -- NOT the old system's
  // context-lowered active_threshold. Showing the latter here would name a
  // parameter that has no effect on what you are watching.
  if (state.risk) {
    const op = state.risk.operating_point || {};
    const thr = op.threshold != null ? op.threshold : 0.7;
    const p = learnedRiskAt(els.video.currentTime || 0);
    const band = p == null ? "muted"
      : (p >= thr ? "danger" : (p >= 0.6 * thr ? "caution" : "safe"));
    els.threshold.innerHTML =
      `Learned risk: <b class="risk-${band}">${p == null ? "–" : p.toFixed(2)}</b>` +
      ` <span style="color:var(--muted)">fires at ≥ ${thr.toFixed(2)} held ` +
      `${op.confirm != null ? op.confirm : 5} frames</span>`;
    return;
  }

  const cf = state.currentFrame;
  const ctx = cf && cf.context;
  const flags = (ctx && ctx.flags) || [];
  const base = (state.meta.config && state.meta.config.base_threshold) || 0.75;
  const thr = (ctx && ctx.active_threshold) != null ? ctx.active_threshold : state.threshold;
  const reason = flags.length ? flags.join(", ").replace(/_/g, " ") : "context";
  els.threshold.innerHTML = thr < base - 1e-9
    ? `Threshold: <b>${thr.toFixed(2)}</b> ↓ <span style="color:var(--muted)">(lowered: ${reason})</span>`
    : `Threshold: <b>${Number(thr).toFixed(2)}</b>`;
}

// ---- event log: render events as they "occur" (t <= currentTime) ----------
// Live-appends during playback with the LLM narrative revealed 800 ms after the
// template (typewriter); reconstructs instantly on seek.

function _placeholder(on) {
  const empty = els.eventLog.querySelector("li.empty");
  if (on && !state.log.length && !empty) {
    els.eventLog.innerHTML = '<li class="empty">no events</li>';
  } else if (!on && empty) {
    empty.remove();
  }
}

function _makeEntry(ev, appearedAt) {
  const li = document.createElement("li");
  li.className = ev.type === "VIRTUAL_BRAKE" ? "brake" : "alert";
  const icon = ev.type === "VIRTUAL_BRAKE" ? "⛔" : "⚠";
  const tpl = ev.explanation_template || `${ev.type.replace(/_/g, " ")} ${ev.actor_cls} ID-${ev.actor_id}`;
  const row = document.createElement("div");
  row.className = "ev-row";
  row.innerHTML = `<span class="t">${ev.t.toFixed(2)}</span><span class="ic">${icon}</span><span class="tpl"></span>`;
  row.querySelector(".tpl").textContent = tpl;
  li.appendChild(row);

  let llmEl = null, llmText = null;
  if (ev.explanation_llm) {
    llmEl = document.createElement("div");
    llmEl.className = "llm hidden";
    llmEl.innerHTML = `<span class="badge">✦ AI narrative</span><span class="llm-text"></span>`;
    llmText = llmEl.querySelector(".llm-text");
    li.appendChild(llmEl);
  }
  li.addEventListener("click", () => { els.video.currentTime = ev.t; });
  els.eventLog.appendChild(li);
  return { ev, appearedAt, chars: 0, li, llmEl, llmText };
}

function initEventLog() {
  els.eventLog.innerHTML = '<li class="empty">no events</li>';
  state.log = [];
}

function rebuildEventLogInstant() {
  // Seek: reconstruct exactly the occurred events, with any LLM fully shown.
  const t = els.video.currentTime;
  els.eventLog.innerHTML = "";
  state.log = [];
  for (const ev of state.events) {
    if (ev.t > t) break;
    state.log.push(_makeEntry(ev, 0));   // appearedAt 0 => LLM shown immediately
  }
  if (!state.log.length) _placeholder(true);
  updateEventLogAnimations();
  autoScrollLog();
}

function syncEventLog() {
  const t = els.video.currentTime;
  let k = 0;
  while (k < state.events.length && state.events[k].t <= t) k++;

  if (k < state.log.length) {                 // stepped back a little: trim
    while (state.log.length > k) state.log.pop().li.remove();
    if (!state.log.length) _placeholder(true);
  }
  while (state.log.length < k) {              // new event(s) occurred: append
    _placeholder(false);
    state.log.push(_makeEntry(state.events[state.log.length], nowMs()));
    autoScrollLog();
  }
  updateEventLogAnimations();
}

function updateEventLogAnimations() {
  for (const e of state.log) {
    if (!e.ev.explanation_llm) continue;
    const full = e.ev.explanation_llm.length;
    if (e.chars >= full) {
      e.llmEl.classList.remove("hidden", "streaming");
      if (e.llmText.textContent.length !== full) e.llmText.textContent = e.ev.explanation_llm;
      continue;
    }
    const instant = e.appearedAt === 0;
    if (instant) {
      e.chars = full;
      e.llmEl.classList.remove("hidden");
      e.llmText.textContent = e.ev.explanation_llm;
    } else if (nowMs() - e.appearedAt >= LLM_DELAY_MS) {
      e.llmEl.classList.remove("hidden");
      e.chars = Math.min(full, e.chars + 2);        // typewriter
      e.llmText.textContent = e.ev.explanation_llm.slice(0, e.chars);
      e.llmEl.classList.toggle("streaming", e.chars < full);
      autoScrollLog();
    }
  }
}

function autoScrollLog() {
  if (!state.userScrolledUp) els.eventLog.scrollTop = els.eventLog.scrollHeight;
}

function updateBanner() {
  // Full-width banner + red vignette for 1.2 s after a VIRTUAL_BRAKE.
  const t = els.video.currentTime;
  const brake = state.events.filter((e) => e.type === "VIRTUAL_BRAKE" && e.t <= t).pop();
  const active = !!brake && t <= brake.t + 1.2;
  els.banner.classList.toggle("hidden", !active);
  els.vignette.classList.toggle("hidden", !active);
  if (active) {
    els.bannerText.textContent =
      `VIRTUAL BRAKING — TTC ${brake.ttc_s != null ? brake.ttc_s.toFixed(1) : "–"} s`;
  }
}

// ---------------------------------------------------------------------------
// risk timeline
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Phase IV comparison: learned risk head vs the old threshold system
// ---------------------------------------------------------------------------

const VERDICT_CLASS = {
  useful: "good", clean: "good",
  "too early": "bad", "false alarm": "bad", miss: "bad", "too late": "bad",
};

function renderCompare() {
  const r = state.risk;
  const show = !!r;
  els.compareWrap.classList.toggle("hidden", !show);
  els.comparePanel.classList.toggle("hidden", !show);

  // Say plainly which system is driving what you are looking at -- the two
  // colour the overlay by different quantities, so this is not cosmetic.
  els.systemBadge.classList.remove("hidden");
  els.systemBadge.classList.toggle("learned", show);
  els.systemBadge.textContent = show
    ? "learned head (GRU) · scene risk"
    : "threshold system · per-actor risk";
  els.systemBadge.title = show
    ? "Actor colour is the learned head's frame-level risk, applied scene-wide. "
      + "The head pools features across actors, so it does not attribute risk to "
      + "an individual actor."
    : "Actor colour is this actor's own collision probability from the "
      + "pre-Phase IV threshold risk engine.";
  els.threatsNote.classList.toggle("hidden", !show);

  if (!show) return;

  const L = r.learned || {}, T = r.threshold_system || {};
  const op = r.operating_point || {};
  const fmt = (t) => (t == null || Number.isNaN(t) ? "—" : `${t.toFixed(2)} s`);
  const lead = (t) => {
    const te = (r.markers || {}).time_of_event;
    if (t == null || te == null) return "";
    return ` (${(te - t).toFixed(2)} s before impact)`;
  };

  const row = (name, verdict, when, extra) => `
    <div class="cmp-row">
      <div class="cmp-name">${name}</div>
      <div class="cmp-verdict ${VERDICT_CLASS[verdict] || ""}">${verdict}</div>
      <div class="cmp-when">${when}${extra || ""}</div>
    </div>`;

  els.compareRows.innerHTML =
    row("Learned head (GRU)", L.verdict || "—",
        fmt(L.event ? L.event.t : null), lead(L.event ? L.event.t : null)) +
    row("Threshold system", T.verdict || "—",
        fmt(T.first_alert_t),
        `${lead(T.first_alert_t)}${T.n_events ? ` · ${T.n_events} events` : ""}`) +
    `<div class="cmp-foot">clip ${r.clip_id} ·
       ${r.label === 1 ? "positive (risky event)" : "negative (ordinary driving)"} ·
       operating point thr ${op.threshold} / confirm ${op.confirm}</div>`;

  els.compareExplain.innerHTML = L.event && L.event.explanation
    ? `<div class="cmp-expl-label">explanation on intervention</div>
       <div class="cmp-expl-text">${L.event.explanation}</div>`
    : `<div class="cmp-expl-label">explanation on intervention</div>
       <div class="cmp-expl-text muted">No alert fired${
         r.label === 0 ? " — correct, this is ordinary driving." : "."}</div>`;
}

function sizeCompare() {
  const c = els.compare;
  const w = c.clientWidth || 800;
  if (c.width !== w) c.width = w;
  if (c.height !== 96) c.height = 96;
}

function drawCurve(ctx, curve, W, H, pad, color, fill) {
  if (!curve || !curve.length) return;
  const plotH = H - pad * 2;
  const yOf = (p) => pad + (1 - p) * plotH;
  if (fill) {
    ctx.beginPath();
    ctx.moveTo(tX(curve[0][0]), H);
    curve.forEach(([t, p]) => ctx.lineTo(tX(t), yOf(p)));
    ctx.lineTo(tX(curve[curve.length - 1][0]), H);
    ctx.closePath();
    ctx.fillStyle = fill; ctx.fill();
  }
  ctx.beginPath();
  curve.forEach(([t, p], i) => {
    const x = tX(t), y = yOf(p);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.stroke();
}

function drawCompare() {
  const r = state.risk;
  if (!r) return;
  sizeCompare();
  const c = els.compare, ctx = c.getContext("2d");
  const W = c.width, H = c.height, pad = 8;
  const plotH = H - pad * 2;
  const yOf = (p) => pad + (1 - p) * plotH;
  ctx.clearRect(0, 0, W, H);

  // The old system's curve first, so the learned curve reads on top of it.
  drawCurve(ctx, (r.threshold_system || {}).curve, W, H, pad,
            "rgba(231,76,60,0.85)", "rgba(231,76,60,0.10)");
  drawCurve(ctx, (r.learned || {}).curve, W, H, pad,
            "#2ecc71", "rgba(46,204,113,0.14)");

  // Committed decision threshold for the learned head.
  const thr = (r.operating_point || {}).threshold;
  if (thr != null) {
    ctx.strokeStyle = "rgba(46,204,113,0.55)"; ctx.setLineDash([5, 4]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, yOf(thr)); ctx.lineTo(W, yOf(thr)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = "10px ui-monospace, monospace"; ctx.textBaseline = "bottom";
    ctx.fillStyle = "rgba(46,204,113,0.9)";
    ctx.fillText(`thr ${thr}`, 3, yOf(thr) - 1);
  }

  // Ground-truth markers: these are what make an alert judgeable.
  const m = r.markers || {};
  const vline = (t, color, label, dash) => {
    if (t == null) return;
    const x = tX(t);
    ctx.strokeStyle = color; ctx.lineWidth = 1.5;
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = "10px ui-monospace, monospace"; ctx.textBaseline = "top";
    ctx.fillStyle = color;
    const tw = ctx.measureText(label).width;
    ctx.fillText(label, Math.min(W - tw - 2, x + 3), 2);
  };
  vline(m.time_of_alert, "#4aa3f0", "alert", [4, 3]);
  vline(m.time_of_event, "#e6e8eb", "impact", null);

  // Where each system actually fired.
  const tri = (t, color, up) => {
    if (t == null) return;
    const x = tX(t), y = up ? H - pad : pad;
    ctx.fillStyle = color;
    ctx.beginPath();
    if (up) { ctx.moveTo(x - 5, y); ctx.lineTo(x + 5, y); ctx.lineTo(x, y - 8); }
    else { ctx.moveTo(x - 5, y); ctx.lineTo(x + 5, y); ctx.lineTo(x, y + 8); }
    ctx.closePath(); ctx.fill();
  };
  tri((r.learned || {}).event ? r.learned.event.t : null, "#2ecc71", true);
  tri((r.threshold_system || {}).first_alert_t, "#e74c3c", false);

  // playhead
  const px = tX(els.video.currentTime || 0);
  ctx.strokeStyle = "rgba(230,232,235,0.8)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke();
}

function sizeTimeline() {
  const c = els.timeline;
  const w = c.clientWidth || 800;
  if (c.width !== w) c.width = w;
  if (c.height !== 64) c.height = 64;
}
function tX(t) {
  const dur = state.duration || els.video.duration || 1;
  return (t / dur) * els.timeline.width;
}
function drawTimeline() {
  sizeTimeline();
  const c = els.timeline, ctx = c.getContext("2d");
  const W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  const pad = 6, plotH = H - pad * 2;
  const yOf = (p) => pad + (1 - p) * plotH;

  // risk sparkline (area)
  if (state.perFrameRisk.length) {
    ctx.beginPath();
    ctx.moveTo(0, H);
    state.frames.forEach((f, i) => ctx.lineTo(tX(f.t), yOf(state.perFrameRisk[i])));
    ctx.lineTo(W, H); ctx.closePath();
    ctx.fillStyle = "rgba(231,76,60,0.18)"; ctx.fill();
    ctx.beginPath();
    state.frames.forEach((f, i) => {
      const x = tX(f.t), y = yOf(state.perFrameRisk[i]);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.strokeStyle = "#e74c3c"; ctx.lineWidth = 1.5; ctx.stroke();
  }
  // active threshold as a STEP function over time (per-frame active_threshold)
  if (state.thresholdSeries.length) {
    ctx.strokeStyle = "rgba(245,166,35,0.85)"; ctx.setLineDash([4, 3]); ctx.lineWidth = 1;
    ctx.beginPath();
    let prevY = null;
    state.frames.forEach((f, i) => {
      const x = tX(f.t), y = yOf(state.thresholdSeries[i]);
      if (i === 0) ctx.moveTo(x, y);
      else { ctx.lineTo(x, prevY); ctx.lineTo(x, y); }
      prevY = y;
    });
    ctx.stroke(); ctx.setLineDash([]);
  }

  // event markers (top), clickable via the timeline click handler
  ctx.font = "12px ui-monospace, monospace"; ctx.textBaseline = "top";
  state.events.forEach((e) => {
    const x = tX(e.t);
    ctx.strokeStyle = e.type === "VIRTUAL_BRAKE" ? "#e74c3c" : "#f5a623";
    ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, H - pad); ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fillText(e.type === "VIRTUAL_BRAKE" ? "⛔" : "⚠", x + 2, pad);
  });

  // proactive (▼ A3PS) vs reactive-ADAS (▽) markers, one pair per VIRTUAL_BRAKE
  ctx.font = "10px ui-monospace, monospace"; ctx.textBaseline = "bottom";
  const triY = H - pad;
  for (const m of state.reactiveMarkers) {
    const xp = tX(m.ev.t), xr = tX(m.t);
    ctx.strokeStyle = "rgba(139,147,161,0.6)"; ctx.setLineDash([2, 2]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(xp, triY - 3); ctx.lineTo(xr, triY - 3); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#2ecc71"; ctx.beginPath();                 // ▼ filled (A3PS)
    ctx.moveTo(xp - 5, triY - 8); ctx.lineTo(xp + 5, triY - 8); ctx.lineTo(xp, triY);
    ctx.closePath(); ctx.fill();
    ctx.strokeStyle = "#8b93a1"; ctx.lineWidth = 1.5; ctx.beginPath();  // ▽ hollow
    ctx.moveTo(xr - 5, triY - 8); ctx.lineTo(xr + 5, triY - 8); ctx.lineTo(xr, triY);
    ctx.closePath(); ctx.stroke();
    if (m.gap > 0) {
      const label = `+${m.gap.toFixed(1)}s`;
      const tw = ctx.measureText(label).width;
      const lx = Math.max(2, Math.min(W - tw - 2, (xp + xr) / 2 - tw / 2));
      ctx.fillStyle = "#2ecc71"; ctx.fillText(label, lx, triY - 10);
    }
  }

  // playhead
  const px = tX(els.video.currentTime || 0);
  ctx.strokeStyle = "#e6e8eb"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke();
}

function updateGapLabel() {
  const m = state.reactiveMarkers[0];
  els.gapLabel.textContent = (m && m.gap > 0)
    ? `reactive trigger — A3PS fired +${m.gap.toFixed(1)} s earlier`
    : "";
}

// ---------------------------------------------------------------------------
// transport
// ---------------------------------------------------------------------------

function syncTransport() {
  const dur = state.duration || els.video.duration || 0;
  const t = els.video.currentTime || 0;
  els.timeReadout.textContent = `${t.toFixed(2)} / ${dur.toFixed(1)} s`;
  if (dur) els.scrubber.value = Math.round((t / dur) * 1000);
  els.btnPlay.textContent = els.video.paused ? "▶" : "⏸";
}

// ---------------------------------------------------------------------------
// main loop + wiring
// ---------------------------------------------------------------------------

function tick() {
  updateCurrentFrame();
  drawOverlay();
  updateThreats();
  updateThreshold();
  syncEventLog();
  updateBanner();
  drawTimeline();
  drawCompare();
  updateGapLabel();
  syncTransport();
  requestAnimationFrame(tick);
}

els.select.addEventListener("change", (e) => loadClip(e.target.value));
els.video.addEventListener("loadedmetadata", () => {
  sizeCanvasToVideo();
  if (els.video.videoWidth && els.video.videoHeight && els.stage) {
    els.stage.style.aspectRatio = `${els.video.videoWidth} / ${els.video.videoHeight}`;
  }
  if (!state.duration) state.duration = els.video.duration || 0;
});
els.video.addEventListener("timeupdate", updateCurrentFrame);
// A seek jumps time non-linearly: rebuild the log to match instantly.
els.video.addEventListener("seeked", rebuildEventLogInstant);
// Track whether the user scrolled the console up (suppress auto-scroll if so).
els.eventLog.addEventListener("scroll", () => {
  const el = els.eventLog;
  state.userScrolledUp = el.scrollHeight - el.scrollTop - el.clientHeight > 8;
});
window.addEventListener("resize", () => { sizeCanvasToVideo(); sizeTimeline(); });

els.btnPlay.addEventListener("click", () =>
  els.video.paused ? els.video.play() : els.video.pause());
els.btnBack.addEventListener("click", () =>
  els.video.currentTime = Math.max(0, els.video.currentTime - 1));
els.btnFwd.addEventListener("click", () =>
  els.video.currentTime = Math.min(state.duration || els.video.duration || 0,
    els.video.currentTime + 1));
els.scrubber.addEventListener("input", () => {
  const dur = state.duration || els.video.duration || 0;
  els.video.currentTime = (els.scrubber.value / 1000) * dur;
});
els.timeline.addEventListener("click", (e) => {
  const rect = els.timeline.getBoundingClientRect();
  const dur = state.duration || els.video.duration || 0;
  let t = ((e.clientX - rect.left) / rect.width) * dur;
  // Snap to a nearby event or reactive marker (~10px) so they're clickable.
  const tol = (10 / rect.width) * dur;
  const cands = state.events.map((ev) => ev.t)
    .concat(state.reactiveMarkers.map((m) => m.t));
  let best = null, bestD = Infinity;
  for (const c of cands) { const d = Math.abs(c - t); if (d < bestD) { bestD = d; best = c; } }
  if (best != null && bestD <= tol) t = best;
  els.video.currentTime = Math.max(0, Math.min(dur, t));
});

loadManifest();
requestAnimationFrame(tick);
