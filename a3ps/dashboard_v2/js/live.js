// Live view: the legacy dashboard's playback window (clip dropdown from the
// same manifest, labelled video with masks and alert banners, event log) with a
// synced driving-scene simulator beside it. Both panes are driven by the one
// video clock, so an alert fires in both at the same instant.

import { $, el, api, cssVar, slotColor, svg } from "./util.js";
import { lineChart, legendRow } from "./charts.js";
import { createSim } from "./sim.js";

const RISK = { safe: [46, 204, 113], caution: [245, 166, 35], danger: [231, 76, 60] };
const rgba = ([r, g, b], a) => `rgba(${r},${g},${b},${a})`;
const BANNER_S = 1.5;      // banner + vignette after an event (legacy used 1.2 s for brakes)
const HIGHLIGHT_S = 2.5;   // how long the simulator keeps the alerting actor highlighted

let ctx = null;
const L = {
  inited: false, ready: false, pending: null, id: null, clip: null, frames: [], events: [], dur: 0,
  sim: null, tl: null, playhead: null, logKey: null, loading: false,
};

const on = (name) => { const cb = document.querySelector(`[data-lv="${name}"]`); return !cb || cb.checked; };
const isDark = () => {
  const t = document.documentElement.dataset.theme;
  return t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
};

// ---------------------------------------------------------------------------
// init / clip loading
// ---------------------------------------------------------------------------

export async function initLive(c) {
  if (L.inited) return;
  L.inited = true;
  ctx = c;
  L.sim = createSim($("lv-sim"));
  const v = $("lv-video");
  $("lv-clip").addEventListener("change", (e) => loadLive(e.target.value));
  $("lv-play").addEventListener("click", togglePlay);
  $("lv-back").addEventListener("click", () => seek(v.currentTime - 1));
  $("lv-fwd").addEventListener("click", () => seek(v.currentTime + 1));
  $("lv-prev").addEventListener("click", () => jumpAlert(-1));
  $("lv-next").addEventListener("click", () => jumpAlert(+1));
  $("lv-scrub").addEventListener("input", () => seek(($("lv-scrub").value / 1000) * (L.dur || v.duration || 0)));
  $("lv-speed").addEventListener("change", () => { v.playbackRate = +$("lv-speed").value; });
  $("lv-simmode").addEventListener("change", draw);
  document.querySelectorAll("[data-lv]").forEach((cb) => cb.addEventListener("change", draw));
  v.addEventListener("loadedmetadata", () => {
    const cv = $("lv-canvas");
    cv.width = v.videoWidth; cv.height = v.videoHeight;
    if (!L.dur) L.dur = v.duration;
    buildTimeline();
  });
  v.addEventListener("seeked", draw);
  window.addEventListener("keydown", (e) => {
    if (!$("view-live").classList.contains("active") || /INPUT|SELECT/.test(e.target.tagName)) return;
    if (e.key === " ") { e.preventDefault(); togglePlay(); }
    if (e.key === "ArrowRight") seek(v.currentTime + (e.shiftKey ? 1 : 1 / 30));
    if (e.key === "ArrowLeft") seek(v.currentTime - (e.shiftKey ? 1 : 1 / 30));
    if (e.key === "n") jumpAlert(+1);
    if (e.key === "p") jumpAlert(-1);
  });

  const m = await api("/api/manifest");
  const sel = $("lv-clip");
  sel.replaceChildren();
  let group = null, target = sel;
  for (const c2 of m.clips) {           // same grouping the legacy dashboard renders
    if (c2.group && c2.group !== group) {
      group = c2.group;
      target = el("optgroup", { label: group });
      sel.appendChild(target);
    } else if (!c2.group) target = sel;
    target.appendChild(el("option", { value: c2.id, text: c2.label || c2.id }));
  }
  L.ready = true;
  if (L.pending) { openInLive(L.pending.cid, L.pending.label); L.pending = null; }
  else {
    const want = new URLSearchParams(location.search).get("clip");
    const first = m.clips.find((x) => x.id === want) || m.clips[0];
    if (first) { sel.value = first.id; loadLive(first.id); }
    else msg("No clips listed in dashboard/clips/manifest.json.");
  }
  requestAnimationFrame(loop);
}

// Open any clip (e.g. from the drill-down), even one the manifest does not list:
// it is added to the dropdown under its own group so the selection stays honest.
export function openInLive(cid, label) {
  if (!L.ready) { L.pending = { cid, label }; return; }
  const sel = $("lv-clip");
  if (![...sel.options].some((o) => o.value === cid)) {
    let grp = sel.querySelector('optgroup[data-extra="1"]');
    if (!grp) { grp = el("optgroup", { label: "Opened from drill-down", "data-extra": "1" }); sel.appendChild(grp); }
    grp.appendChild(el("option", { value: cid, text: `${cid}${label ? ` · ${label}` : ""}` }));
  }
  sel.value = cid;
  loadLive(cid);
}

export function onLiveShow() { if (L.clip) { buildTimeline(); draw(); } }
export function onLiveFilters() { if (L.clip) { buildTimeline(); draw(); } }

function msg(text) {
  const m = $("lv-msg");
  m.textContent = text || "";
  m.classList.toggle("hidden", !text);
}

export async function loadLive(id) {
  if (!id) return;
  L.id = id;
  L.loading = true;
  const v = $("lv-video");
  v.pause();
  msg(`Loading clip ${id}…`);
  let clip;
  try {
    clip = await api(`/api/clip/${encodeURIComponent(id)}?overlay=legacy&hz=15`);
  } catch (e) {
    msg(`Could not load clip ${id}: ${e.message || e}`);
    return;
  } finally { L.loading = false; }
  if (L.id !== id) return;          // user picked another clip meanwhile
  L.clip = clip;
  L.frames = clip.frames;
  L.events = clip.events.slice().sort((a, b) => a.t - b.t);
  L.dur = clip.meta.duration_s || (L.frames.length ? L.frames[L.frames.length - 1].t : 0);
  L.logKey = null;
  smoothGround();
  ctx.registerClasses(new Set(L.frames.flatMap((f) => f.tracks.map((t) => t.cls))));

  if (clip.video_url) {
    v.src = clip.video_url;
    v.load();
    msg("");
  } else {
    v.removeAttribute("src"); v.load();
    msg(`No video for clip ${id}. The simulator and timeline still play from the overlay data.`);
  }
  v.playbackRate = +$("lv-speed").value;

  const learned = !!clip.decision;
  const badge = $("lv-badge");
  badge.classList.remove("hidden");
  badge.classList.toggle("learned", learned);
  badge.textContent = learned ? "alerts: learned head (GRU)" : "alerts: threshold system";
  badge.title = learned
    ? "Alerts come from the learned risk head. Actor colour is its scene-level risk, applied to every actor."
    : "Alerts come from the pre-Phase IV threshold risk engine. Actor colour is each actor's own collision probability.";
  const bc = clip.bev_counts || {};
  $("lv-note").textContent =
    (bc.projected ? "Simulator positions are projected from the image with the pipeline's own GroundPlane " +
      "(default calibration), because this overlay did not store centroid_bev. "
      : "Simulator positions are the pipeline's centroid_bev. ") +
    "Forecast paths are the pipeline's Kalman-CV forecast (mean_img) projected onto the ground. " +
    "The ego is fixed at the origin: nothing in the pipeline estimates ego speed, so the road does not scroll.";
  buildLog();
  buildTimeline();
  draw();
}

// Ground positions from a monocular homography jitter frame to frame; a short
// centred moving average per track keeps the simulator readable without
// changing where anything is.
function smoothGround() {
  const byId = new Map();
  L.frames.forEach((f, i) => f.tracks.forEach((t) => {
    if (!t.c_bev) return;
    if (!byId.has(t.id)) byId.set(t.id, []);
    byId.get(t.id).push({ i, t });
  }));
  for (const seq of byId.values()) {
    seq.forEach((s, k) => {
      let sx = 0, sy = 0, n = 0;
      for (let j = Math.max(0, k - 2); j <= Math.min(seq.length - 1, k + 2); j++) {
        if (Math.abs(seq[j].i - s.i) > 4) continue;
        sx += seq[j].t.c_bev[0]; sy += seq[j].t.c_bev[1]; n++;
      }
      s.t.sm = [sx / n, sy / n];
    });
  }
  L.frames.forEach((f) => { f.byId = new Map(f.tracks.map((t) => [t.id, t])); });
}

// ---------------------------------------------------------------------------
// clock
// ---------------------------------------------------------------------------

const now = () => $("lv-video").currentTime || 0;
function seek(t) {
  const v = $("lv-video");
  v.currentTime = Math.max(0, Math.min(L.dur || v.duration || 0, t));
  draw();
}
function togglePlay() {
  const v = $("lv-video");
  if (!v.src) return;
  v.paused ? v.play() : v.pause();
}
function jumpAlert(dir) {
  const t = now();
  const evs = L.events;
  if (!evs.length) return;
  // Land 1.5 s before the alert so it can be watched happening.
  const ev = dir > 0 ? evs.find((e) => e.t - 1.5 > t + 0.05) : [...evs].reverse().find((e) => e.t - 1.5 < t - 0.3);
  if (ev) { seek(ev.t - 1.5); $("lv-video").play(); }
}

function frameIndex(t) {
  const f = L.frames;
  let lo = 0, hi = f.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (f[mid].t <= t) { ans = mid; lo = mid + 1; } else hi = mid - 1;
  }
  if (ans === f.length - 1 && t > f[ans].t + 0.5) return -1;
  return ans;
}

function activeEvent(t, windowS) {
  let hit = null;
  for (const e of L.events) { if (e.t <= t && t <= e.t + windowS) hit = e; if (e.t > t) break; }
  return hit;
}

function loop() {
  if (L.clip && $("view-live").classList.contains("active")) draw();
  requestAnimationFrame(loop);
}

// ---------------------------------------------------------------------------
// drawing
// ---------------------------------------------------------------------------

function draw() {
  if (!L.clip) return;
  const t = now();
  const i = frameIndex(t);
  const frame = i >= 0 ? L.frames[i] : null;
  const banner = activeEvent(t, BANNER_S);
  const highlight = activeEvent(t, HIGHLIGHT_S);
  drawVideo(frame, highlight);
  drawSim(t, i, highlight);
  drawBanners(banner);
  drawHud(frame, t, i);
  syncLog(t);
  const v = $("lv-video");
  $("lv-time").textContent = `${t.toFixed(2)} / ${(L.dur || 0).toFixed(1)} s`;
  if (L.dur) $("lv-scrub").value = Math.round((t / L.dur) * 1000);
  $("lv-play").textContent = v.paused ? "▶" : "⏸";
  if (L.playhead && L.tl) {
    const x = L.tl.x(Math.min(t, L.dur));
    L.playhead.setAttribute("x1", x); L.playhead.setAttribute("x2", x);
  }
}

function visible(tracks) {
  const off = ctx.getFilters().classOff;
  return tracks.filter((t) => !off.has(t.cls));
}

function drawVideo(frame, alertEv) {
  const cv = $("lv-canvas"), g = cv.getContext("2d");
  g.clearRect(0, 0, cv.width, cv.height);
  $("lv-frame").textContent = frame ? `frame ${frame.i}` : "frame –";
  if (!frame) return;
  const sx = cv.width / (L.clip.meta.width || cv.width), sy = cv.height / (L.clip.meta.height || cv.height);
  const lw = Math.max(1.5, cv.width / 700);
  const poly = (pts) => { g.beginPath(); pts.forEach(([x, y], k) => (k ? g.lineTo(x * sx, y * sy) : g.moveTo(x * sx, y * sy))); g.closePath(); };

  if (on("corridor") && frame.corridor_img) {
    poly(frame.corridor_img);
    g.fillStyle = alertEv ? "rgba(231,76,60,0.16)" : "rgba(77,163,255,0.12)"; g.fill();
    g.strokeStyle = alertEv ? "rgba(231,76,60,0.6)" : "rgba(77,163,255,0.5)"; g.lineWidth = lw; g.stroke();
  }
  const tracks = visible(frame.tracks);
  for (const tr of tracks) {
    const c = RISK[tr.lvl] || RISK.safe;
    if (on("masks") && tr.mask && tr.mask.length > 2) {
      poly(tr.mask);
      g.fillStyle = rgba(c, 0.35); g.fill();
      g.strokeStyle = rgba(c, 1); g.lineWidth = lw; g.stroke();
    }
    if (on("trails") && tr.trail && tr.trail.length > 1) {
      for (let k = 0; k < tr.trail.length - 1; k++) {
        g.strokeStyle = rgba(c, (k + 1) / (tr.trail.length - 1)); g.lineWidth = lw;
        g.beginPath(); g.moveTo(tr.trail[k][0] * sx, tr.trail[k][1] * sy);
        g.lineTo(tr.trail[k + 1][0] * sx, tr.trail[k + 1][1] * sy); g.stroke();
      }
    }
    if (on("predictions") && tr.pred_img && tr.pred_img.length) {
      const n = tr.pred_img.length;
      tr.pred_img.forEach(([x, y], k) => {
        const f = k / Math.max(1, n - 1);
        g.fillStyle = rgba(c, 1 - 0.85 * f);
        g.beginPath(); g.arc(x * sx, y * sy, Math.max(1.5, 4 * (1 - f)) * lw / 1.5, 0, Math.PI * 2); g.fill();
      });
    }
    if (on("boxes")) {
      const [x1, y1, x2, y2] = tr.bbox;
      g.strokeStyle = rgba(c, 0.9); g.lineWidth = lw;
      g.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
    }
  }
  // Labels last so masks never cover them.
  const fs = Math.round(21 * lw / 1.8);   // readable at half-width (side by side)
  g.font = `${fs}px ui-monospace, Menlo, Consolas, monospace`;
  g.textBaseline = "bottom";
  for (const tr of tracks) {
    const isAlert = alertEv && alertEv.actor_id === tr.id;
    const c = isAlert ? RISK.danger : (RISK[tr.lvl] || RISK.safe);
    const [x1, y1, x2, y2] = tr.bbox;
    if (isAlert) {
      g.strokeStyle = rgba(RISK.danger, 1); g.lineWidth = lw * 2.2;
      g.strokeRect(x1 * sx - 3, y1 * sy - 3, (x2 - x1) * sx + 6, (y2 - y1) * sy + 6);
    }
    if (!on("labels") && !isAlert) continue;
    const text = isAlert
      ? `${alertEv.type === "VIRTUAL_BRAKE" ? "⛔" : "⚠"} ${tr.cls} ID-${tr.id}`
      : `${tr.cls} ID-${tr.id}${tr.p != null ? ` ${tr.p.toFixed(2)}` : ""}`;
    const w = g.measureText(text).width, h = fs + 6;
    g.fillStyle = isAlert ? "rgba(200,40,40,0.9)" : "rgba(0,0,0,0.6)";
    g.fillRect(x1 * sx, y1 * sy - h, w + 10, h);
    g.fillStyle = isAlert ? "#fff" : rgba(c, 1);
    g.fillText(text, x1 * sx + 5, y1 * sy - 3);
  }
}

function drawSim(t, i, alertEv) {
  let actors = [];
  let corridor = null;
  if (i >= 0) {
    const f0 = L.frames[i], f1 = L.frames[i + 1];
    const a = f1 ? Math.max(0, Math.min(1, (t - f0.t) / Math.max(1e-6, f1.t - f0.t))) : 0;
    corridor = f0.corridor_bev;
    actors = visible(f0.tracks).filter((tr) => tr.sm).map((tr) => {
      const nx = f1 && f1.byId.get(tr.id);
      // Interpolate between overlay frames so motion is continuous at 60 fps.
      const x = nx && nx.sm ? tr.sm[0] + (nx.sm[0] - tr.sm[0]) * a : tr.sm[0];
      const y = nx && nx.sm ? tr.sm[1] + (nx.sm[1] - tr.sm[1]) * a : tr.sm[1];
      const dx = x - tr.c_bev[0], dy = y - tr.c_bev[1];
      const isAlert = alertEv && alertEv.actor_id === tr.id;
      return {
        id: tr.id, cls: tr.cls, x, y, lvl: tr.lvl, p: tr.p,
        pred: (tr.pred_bev || []).map(([px, py]) => [px + dx, py + dy]),
        alert: isAlert ? { type: alertEv.type, ttc: alertEv.ttc_s } : null,
      };
    });
  }
  L.sim.draw({
    mode: $("lv-simmode").value, actors, corridor, alert: alertEv, t, dark: isDark(),
    layers: { forecast: on("sim-forecast"), zone: on("sim-zone"), labels: on("sim-labels") },
  });
}

function drawBanners(ev) {
  for (const [b, vg] of [["lv-banner", "lv-vig"], ["lv-banner2", "lv-vig2"]]) {
    const banner = $(b), vig = $(vg);
    banner.classList.toggle("hidden", !ev);
    vig.classList.toggle("hidden", !ev || ev.type !== "VIRTUAL_BRAKE");
    if (!ev) continue;
    const brake = ev.type === "VIRTUAL_BRAKE";
    banner.classList.toggle("brake", brake);
    banner.textContent = `${brake ? "⛔ VIRTUAL BRAKING" : "⚠ COLLISION ALERT"} — ${ev.actor_cls} ID-${ev.actor_id}` +
      `${ev.ttc_s != null ? ` · TTC ${ev.ttc_s.toFixed(1)} s` : ""} · P ${ev.collision_prob.toFixed(2)}`;
  }
}

function drawHud(frame, t, i) {
  const top = frame ? visible(frame.tracks).filter((x) => x.p != null).sort((a, b) => b.p - a.p).slice(0, 3) : [];
  const learned = L.clip.curves.find((c) => c.id === "learned");
  let scene = null;
  if (learned) {
    let best = null, bd = Infinity;
    for (const [ct, cp] of learned.points) { const d = Math.abs(ct - t); if (d < bd) { bd = d; best = cp; } if (ct > t + 1) break; }
    if (bd <= 0.5) scene = best;
  }
  const hud = $("lv-hud");
  hud.replaceChildren(...[
    el("div", { text: `t ${t.toFixed(2)} s · ${frame ? `frame ${frame.i}` : "no overlay"} · ${frame ? visible(frame.tracks).length : 0} actors` }),
    scene != null ? el("div", { text: `scene risk (learned) ${scene.toFixed(2)}` }) : null,
    ...top.map((x) => el("div", { text: `${x.cls} ${x.id}  P ${x.p.toFixed(2)}${x.ttc != null ? `  TTC ${x.ttc.toFixed(1)}s` : ""}` })),
  ].filter(Boolean));
}

// ---------------------------------------------------------------------------
// event log (agent console)
// ---------------------------------------------------------------------------

function buildLog() {
  const ul = $("lv-log");
  ul.replaceChildren();
  if (!L.events.length) {
    ul.appendChild(el("li", { class: "empty", text: "No alerts on this clip." }));
    return;
  }
  for (const ev of L.events) {
    const brake = ev.type === "VIRTUAL_BRAKE";
    const li = el("li", { class: `ev ${brake ? "brake" : "alert"} future` }, [
      el("div", { class: "ev-row" }, [
        el("span", { class: "t num", text: ev.t.toFixed(2) }),
        el("span", { class: "ic", text: brake ? "⛔" : "⚠" }),
        el("b", { text: brake ? "VIRTUAL BRAKE" : "ALERT" }),
        el("span", { class: "muted", text: `${ev.actor_cls} ID-${ev.actor_id} · P ${ev.collision_prob.toFixed(2)}${ev.ttc_s != null ? ` · TTC ${ev.ttc_s.toFixed(1)} s` : ""}` }),
      ]),
      ev.explanation_template ? el("div", { class: "ev-text", text: ev.explanation_template }) : null,
      ev.explanation_llm ? el("div", { class: "ev-text llm", text: `✦ ${ev.explanation_llm}` }) : null,
    ]);
    li.title = "Jump to 1.5 s before this alert";
    li.addEventListener("click", () => { seek(ev.t - 1.5); $("lv-video").play(); });
    ev._li = li;
    ul.appendChild(li);
  }
}

function syncLog(t) {
  const active = activeEvent(t, BANNER_S);
  const key = `${L.events.filter((e) => e.t <= t).length}|${active ? active.event_id : ""}`;
  if (key === L.logKey) return;
  L.logKey = key;
  let lastPast = null;
  for (const ev of L.events) {
    if (!ev._li) continue;
    const past = ev.t <= t;
    ev._li.classList.toggle("future", !past);
    ev._li.classList.toggle("active", ev === active);
    if (past) lastPast = ev._li;
  }
  // Scroll only the log box; scrollIntoView would also move the whole page.
  if (lastPast) {
    const ul = $("lv-log");
    const top = lastPast.offsetTop;   // .evlog is position:relative
    if (top < ul.scrollTop || top + lastPast.offsetHeight > ul.scrollTop + ul.clientHeight) {
      ul.scrollTop = Math.max(0, top - ul.clientHeight + lastPast.offsetHeight + 8);
    }
  }
}

// ---------------------------------------------------------------------------
// risk timeline
// ---------------------------------------------------------------------------

function buildTimeline() {
  const host = $("lv-timeline");
  if (!L.clip || !host.clientWidth) return;
  const series = [];
  const learned = L.clip.curves.find((c) => c.id === "learned");
  if (learned) series.push({ name: "learned head (GRU) · scene risk", color: slotColor(0),
    points: learned.points.map(([t, p]) => ({ x: t, y: p })) });
  const dur = L.dur || 1;
  if (!series.length) series.push({ name: "", color: "transparent", points: [{ x: 0, y: null }, { x: dur, y: null }] });

  const dec = L.clip.decision || {};
  const thr = (dec.operating_point || {}).threshold;
  const refs = [];
  if (thr != null) refs.push({ axis: "y", v: thr, label: `alert threshold ${thr}` });
  const gt = L.clip.gt || {};
  if (gt.alert_t != null) refs.push({ axis: "x", v: gt.alert_t, label: "time_of_alert", color: cssVar("--s1"), dash: "5 3" });
  if (gt.event_t != null) refs.push({ axis: "x", v: gt.event_t, label: "impact", color: cssVar("--ink"), solid: true });

  // Only the learned head's own decisions; the threshold system's events are not drawn here.
  const marks = [dec.event, dec.brake_event].filter((ev) => ev && ev.t != null);
  const overlays = [(s, f, x) => {
    for (const ev of marks) {
      const brake = ev.type === "VIRTUAL_BRAKE";
      const px = x(ev.t), col = brake ? "#e74c3c" : "#f5a623";
      s.appendChild(svg("line", { x1: px, x2: px, y1: f.y1, y2: f.y0, stroke: col, "stroke-width": 1.5, opacity: 0.8 }));
      const tri = svg("path", { d: `M${px - 6},${f.y1 - 2} L${px + 6},${f.y1 - 2} L${px},${f.y1 + 8} Z`, fill: col });
      tri.appendChild(svg("title", {}, `${ev.type} ${ev.actor_cls} ID-${ev.actor_id} @ ${ev.t.toFixed(2)} s`));
      s.appendChild(tri);
    }
    L.playhead = svg("line", { x1: f.x0, x2: f.x0, y1: f.y1 - 4, y2: f.y0, stroke: cssVar("--ink"), "stroke-width": 1.5 });
    s.appendChild(L.playhead);
  }];
  const chart = lineChart(host, {
    series, height: 170, xDomain: [0, dur], yDomain: [0, 1], xTitle: "time (s) →",
    xFormat: (v) => v.toFixed(0), yFormat: (v) => v.toFixed(1), refs, overlays, legend: false,
    tipTitle: (xv) => `t = ${xv.toFixed(2)} s`, onClick: (xv) => seek(xv),
  });
  L.tl = chart ? { x: chart.x } : null;
  host.prepend(legendRow([
    ...series.filter((s) => s.name).map((s) => ({ name: s.name, color: s.color, kind: "line" })),
    { name: "▼ alert", color: "#f5a623" }, { name: "▼ virtual brake", color: "#e74c3c" },
  ]));
}
