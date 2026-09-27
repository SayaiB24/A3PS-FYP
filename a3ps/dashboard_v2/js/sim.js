// Lightweight driving-scene simulator: re-renders the tracked scene from the
// pipeline's ground-plane positions, either from a chase camera behind the ego
// vehicle (perspective, 2.5D cuboids) or from straight above. Plain canvas, no
// engine: it only visualises what the pipeline produced, frame by frame.
//
// World frame (same as centroid_bev): x = metres to the right, y = metres ahead
// of the ego's front bumper, z = up.

const RISK_RGB = { safe: [46, 204, 113], caution: [245, 166, 35], danger: [231, 76, 60] };
const rgba = ([r, g, b], a = 1) => `rgba(${r},${g},${b},${a})`;
const shade = ([r, g, b], k) => [r * k, g * k, b * k].map((v) => Math.max(0, Math.min(255, Math.round(v))));

// Footprint (width, length, height) in metres by class.
const DIMS = {
  car: [1.8, 4.4, 1.5], truck: [2.5, 8.0, 3.2], bus: [2.6, 11, 3.2],
  motorcycle: [0.8, 2.1, 1.4], bicycle: [0.7, 1.8, 1.6], person: [0.6, 0.6, 1.75],
};
const dimsOf = (cls) => DIMS[cls] || [1.6, 3.5, 1.5];

const ROAD_HALF = 7.2, LANE = 3.6;

// ---------------------------------------------------------------------------
// cameras
// ---------------------------------------------------------------------------

function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function norm(a) { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }

function chaseCamera(W, H) {
  // Framed on the 5-40 m band where the tracker's ground positions are usable.
  const pos = [0, -8.5, 6.8], target = [0, 22, 0];
  const f = norm(sub(target, pos));
  const r = norm(cross(f, [0, 0, 1]));
  const u = cross(r, f);
  const fl = (H / 2) / Math.tan((40 * Math.PI) / 360);
  const NEAR = 0.6;
  const toCam = (p) => { const d = sub(p, pos); return [dot(d, r), dot(d, u), dot(d, f)]; };
  const screen = ([xc, yc, zc]) => [W / 2 + (fl * xc) / zc, H / 2 - (fl * yc) / zc];
  return {
    kind: "chase", pos, NEAR, toCam, screen,
    project: (p) => { const c = toCam(p); return c[2] < NEAR ? null : screen(c); },
    horizonY: () => screen(toCam([0, 1e5, 0]))[1],
    scaleAt: (p) => { const c = toCam(p); return c[2] < NEAR ? 0 : fl / c[2]; },
  };
}

function topCamera(W, H) {
  const yMin = -6, span = 40;
  const ppm = Math.min(H / span, W / 24);
  const screen = ([x, y]) => [W / 2 + x * ppm, H - (y - yMin) * ppm];
  return {
    kind: "top", ppm, yMin, span,
    toCam: (p) => [p[0], p[1], 1], screen,
    project: (p) => screen(p), horizonY: () => 0, scaleAt: () => ppm, NEAR: -Infinity,
  };
}

// Clip a polygon (camera space) against the near plane, then project it.
function projectPoly(cam, pts) {
  if (cam.kind === "top") return pts.map((p) => cam.screen(p));
  const c = pts.map(cam.toCam);
  const out = [];
  for (let i = 0; i < c.length; i++) {
    const a = c[i], b = c[(i + 1) % c.length];
    const ina = a[2] >= cam.NEAR, inb = b[2] >= cam.NEAR;
    if (ina) out.push(a);
    if (ina !== inb) {
      const k = (cam.NEAR - a[2]) / (b[2] - a[2]);
      out.push([a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, cam.NEAR]);
    }
  }
  return out.length >= 3 ? out.map(cam.screen) : null;
}

function projectLine(cam, a, b) {
  if (cam.kind === "top") return [cam.screen(a), cam.screen(b)];
  let ca = cam.toCam(a), cb = cam.toCam(b);
  if (ca[2] < cam.NEAR && cb[2] < cam.NEAR) return null;
  const clip = (p, q) => {
    const k = (cam.NEAR - p[2]) / (q[2] - p[2]);
    return [p[0] + (q[0] - p[0]) * k, p[1] + (q[1] - p[1]) * k, cam.NEAR];
  };
  if (ca[2] < cam.NEAR) ca = clip(ca, cb);
  if (cb[2] < cam.NEAR) cb = clip(cb, ca);
  return [cam.screen(ca), cam.screen(cb)];
}

function fillPoly(g, pts, style) {
  if (!pts) return;
  g.beginPath();
  pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
  g.closePath();
  g.fillStyle = style;
  g.fill();
}

// ---------------------------------------------------------------------------
// scene pieces
// ---------------------------------------------------------------------------

function drawEnvironment(g, cam, W, H, dark) {
  if (cam.kind === "chase") {
    const hy = Math.max(0, Math.min(H, cam.horizonY()));
    const sky = g.createLinearGradient(0, 0, 0, hy);
    sky.addColorStop(0, dark ? "#0f1826" : "#8fb8e6");
    sky.addColorStop(1, dark ? "#2a3444" : "#dfe9f3");
    g.fillStyle = sky; g.fillRect(0, 0, W, hy);
    g.fillStyle = dark ? "#1d2a1f" : "#9db38c"; g.fillRect(0, hy, W, H - hy);
  } else {
    g.fillStyle = dark ? "#1d2a1f" : "#a9bd98"; g.fillRect(0, 0, W, H);
  }
  // road surface
  fillPoly(g, projectPoly(cam, [[-ROAD_HALF, -20, 0], [ROAD_HALF, -20, 0], [ROAD_HALF, 160, 0], [-ROAD_HALF, 160, 0]]),
    dark ? "#2b2d31" : "#5b5e63");
  // kerbs (solid) and lane lines (dashed)
  const line = (x, y0, y1, color, w) => {
    const seg = projectLine(cam, [x, y0, 0.01], [x, y1, 0.01]);
    if (!seg) return;
    const s0 = cam.scaleAt([x, Math.max(y0, -2), 0]);
    g.strokeStyle = color; g.lineWidth = Math.max(1, Math.min(6, w * (cam.kind === "top" ? cam.ppm : s0)));
    g.beginPath(); g.moveTo(...seg[0]); g.lineTo(...seg[1]); g.stroke();
  };
  for (const x of [-ROAD_HALF, ROAD_HALF]) line(x, -20, 160, "#e8e8e0", 0.15);
  for (const x of [-LANE * 1.5, -LANE / 2, LANE / 2, LANE * 1.5]) {
    for (let y = -18; y < 150; y += 9) line(x, y, y + 3, x === -LANE / 2 ? "#f2c94c" : "#e8e8e0", 0.12);
  }
  // distance marks
  g.font = "11px system-ui, sans-serif";
  g.fillStyle = dark ? "rgba(255,255,255,.55)" : "rgba(0,0,0,.55)";
  for (let y = 10; y <= 60; y += 10) {
    const p = cam.project([ROAD_HALF + 0.6, y, 0]);
    if (p && p[0] < W - 30) g.fillText(`${y} m`, p[0] + 2, p[1]);
  }
}

function drawZone(g, cam, corridor, alert) {
  if (!corridor || corridor.length < 3) return;
  const pts = projectPoly(cam, corridor.map(([x, y]) => [x, Math.max(y, -3), 0.02]));
  if (!pts) return;
  fillPoly(g, pts, alert ? "rgba(231,76,60,0.28)" : "rgba(77,163,255,0.22)");
  g.strokeStyle = alert ? "rgba(231,76,60,0.9)" : "rgba(77,163,255,0.8)"; g.lineWidth = 1.5;
  g.beginPath(); pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.closePath(); g.stroke();
}

function cuboid(bx, by, cls) {
  // (bx, by) is the ground point under the bottom-centre of the image box: the
  // edge of the object nearest the camera. The body extends away from the ego.
  const [w, l, h] = dimsOf(cls);
  const x0 = bx - w / 2, x1 = bx + w / 2, y0 = by, y1 = by + l;
  const v = [[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0], [x0, y0, h], [x1, y0, h], [x1, y1, h], [x0, y1, h]];
  // faces with outward normals
  return {
    h, center: [bx, by + l / 2, h / 2], top: [bx, by + l / 2, h],
    faces: [
      { idx: [4, 5, 6, 7], n: [0, 0, 1], k: 1.12 },   // top
      { idx: [0, 1, 5, 4], n: [0, -1, 0], k: 0.95 },  // rear (faces the ego)
      { idx: [3, 2, 6, 7], n: [0, 1, 0], k: 0.7 },    // front
      { idx: [0, 3, 7, 4], n: [-1, 0, 0], k: 0.8 },   // left
      { idx: [1, 2, 6, 5], n: [1, 0, 0], k: 0.8 },    // right
    ].map((f) => ({ ...f, pts: f.idx.map((i) => v[i]) })),
    foot: [v[0], v[1], v[2], v[3]],
  };
}

function drawActor(g, cam, a, dark) {
  const base = a.color || RISK_RGB[a.lvl] || RISK_RGB.safe;
  const alpha = a.dim ? 0.35 : 1;
  const cb = cuboid(a.x, a.y, a.cls);
  if (cam.kind === "chase") {
    // soft contact shadow
    fillPoly(g, projectPoly(cam, cb.foot.map(([x, y]) => [x + 0.15, y - 0.2, 0.01])), `rgba(0,0,0,${0.35 * alpha})`);
    const vis = cb.faces.filter((f) => {
      const c = f.pts.reduce((s, p) => [s[0] + p[0] / 4, s[1] + p[1] / 4, s[2] + p[2] / 4], [0, 0, 0]);
      return dot(f.n, sub(cam.pos, c)) > 0;
    });
    for (const f of vis) {
      const pts = projectPoly(cam, f.pts);
      fillPoly(g, pts, rgba(shade(base, f.k), 0.92 * alpha));
      if (pts) {
        g.strokeStyle = rgba(shade(base, 0.45), 0.8 * alpha); g.lineWidth = 1;
        g.beginPath(); pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.closePath(); g.stroke();
      }
    }
  } else {
    const pts = projectPoly(cam, cb.foot);
    fillPoly(g, pts, rgba(base, 0.9 * alpha));
    g.strokeStyle = rgba(shade(base, 0.45), alpha); g.lineWidth = 1.2;
    g.beginPath(); pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.closePath(); g.stroke();
  }
  return cb;
}

function drawEgo(g, cam) {
  const cb = cuboid(0, -4.6, "car");
  const blue = [58, 134, 229];
  if (cam.kind === "chase") {
    for (const f of cb.faces) {
      const c = f.pts.reduce((s, p) => [s[0] + p[0] / 4, s[1] + p[1] / 4, s[2] + p[2] / 4], [0, 0, 0]);
      if (dot(f.n, sub(cam.pos, c)) <= 0) continue;
      fillPoly(g, projectPoly(cam, f.pts), rgba(shade(blue, f.k), 0.55));
    }
  } else {
    fillPoly(g, projectPoly(cam, cb.foot), rgba(blue, 0.95));
  }
  const p = cam.project([0, -2.3, cam.kind === "chase" ? 1.9 : 0]);
  if (p) {
    g.font = "bold 11px system-ui, sans-serif"; g.fillStyle = "#fff";
    g.textAlign = "center"; g.fillText("EGO", p[0], p[1] + (cam.kind === "top" ? 4 : 0)); g.textAlign = "left";
  }
}

function insidePoly(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-9) + xi) inside = !inside;
  }
  return inside;
}

function drawForecast(g, cam, a, corridor) {
  if (!a.pred || a.pred.length < 1) return null;
  const color = a.alert ? RISK_RGB.danger : (RISK_RGB[a.lvl] || RISK_RGB.safe);
  const path = [[a.x, a.y], ...a.pred];
  g.setLineDash([6, 5]);
  g.strokeStyle = rgba(color, a.dim ? 0.3 : 0.95);
  g.lineWidth = a.alert ? 3 : 1.8;
  for (let i = 0; i < path.length - 1; i++) {
    const seg = projectLine(cam, [...path[i], 0.04], [...path[i + 1], 0.04]);
    if (!seg) continue;
    g.beginPath(); g.moveTo(...seg[0]); g.lineTo(...seg[1]); g.stroke();
  }
  g.setLineDash([]);
  // first predicted point inside the ego corridor = the predicted conflict point
  if (!corridor) return null;
  const hit = a.pred.find(([x, y]) => insidePoly(x, y, corridor));
  return hit || null;
}

function drawConflict(g, cam, pt, t) {
  const p = cam.project([pt[0], pt[1], 0.05]);
  if (!p) return;
  const s = Math.max(5, Math.min(18, 0.8 * cam.scaleAt([pt[0], pt[1], 0])));
  g.strokeStyle = "rgba(231,76,60,0.95)"; g.lineWidth = 3;
  g.beginPath(); g.moveTo(p[0] - s, p[1] - s * 0.5); g.lineTo(p[0] + s, p[1] + s * 0.5);
  g.moveTo(p[0] + s, p[1] - s * 0.5); g.lineTo(p[0] - s, p[1] + s * 0.5); g.stroke();
  g.font = "bold 11px system-ui, sans-serif"; g.fillStyle = "rgba(231,76,60,1)";
  g.fillText("predicted conflict", p[0] + s + 4, p[1] + 4);
}

function ring(g, cam, x, y, r, style, width) {
  const pts = [];
  for (let i = 0; i < 28; i++) {
    const a = (i / 28) * Math.PI * 2;
    pts.push([x + r * Math.cos(a), y + r * Math.sin(a), 0.05]);
  }
  const proj = projectPoly(cam, pts);
  if (!proj) return;
  g.strokeStyle = style; g.lineWidth = width;
  g.beginPath(); proj.forEach(([px, py], i) => (i ? g.lineTo(px, py) : g.moveTo(px, py))); g.closePath(); g.stroke();
}

function label(g, x, y, text, bg, fg = "#fff") {
  g.font = "12px system-ui, sans-serif";
  const w = g.measureText(text).width + 10;
  g.fillStyle = bg; g.fillRect(x - w / 2, y - 18, w, 17);
  g.fillStyle = fg; g.textAlign = "center"; g.fillText(text, x, y - 5); g.textAlign = "left";
}

// ---------------------------------------------------------------------------
// public
// ---------------------------------------------------------------------------

export function createSim(canvas) {
  const g = canvas.getContext("2d");
  let cssW = 0, cssH = 0;
  function fit() {
    const dpr = window.devicePixelRatio || 1;
    const r = canvas.getBoundingClientRect();
    if (!r.width) return false;
    if (r.width !== cssW || r.height !== cssH) {
      cssW = r.width; cssH = r.height;
      canvas.width = Math.round(cssW * dpr); canvas.height = Math.round(cssH * dpr);
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    return true;
  }
  return {
    /** scene: {mode, actors:[{id,cls,x,y,lvl,p,pred,alert,dim}], corridor, alert, t, layers, dark} */
    draw(scene) {
      if (!fit()) return;
      const W = cssW, H = cssH;
      g.clearRect(0, 0, W, H);
      const cam = scene.mode === "top" ? topCamera(W, H) : chaseCamera(W, H);
      drawEnvironment(g, cam, W, H, scene.dark);
      if (scene.layers.zone) drawZone(g, cam, scene.corridor, !!scene.alert);

      // far-to-near so nearer bodies overdraw farther ones
      const actors = scene.actors.filter((a) => a.y > -6 && a.y < 120 && Math.abs(a.x) < 45)
        .sort((a, b) => (b.y - a.y) || (Math.abs(b.x) - Math.abs(a.x)));
      const conflicts = [];
      if (scene.layers.forecast) {
        for (const a of actors) {
          const c = drawForecast(g, cam, a, scene.corridor);
          if (c && a.alert) conflicts.push(c);   // one marker: where the alerting actor's forecast meets the ego path
        }
      }
      const pulse = 0.5 + 0.5 * Math.sin(scene.t * 8);
      const ranked = actors.filter((a) => a.p != null).sort((a, b) => b.p - a.p).slice(0, 4);
      const labelled = new Set([...ranked, ...actors.filter((a) => a.y < 22)]);
      for (const a of actors) {
        if (a.alert) {
          ring(g, cam, a.x, a.y + dimsOf(a.cls)[1] / 2, 2.6 + pulse * 0.8, `rgba(231,76,60,${0.5 + 0.5 * pulse})`, 3);
          // tether from the threatening actor to the ego bumper
          const seg = projectLine(cam, [a.x, a.y, 0.05], [0, 0, 0.05]);
          if (seg) {
            g.strokeStyle = "rgba(231,76,60,0.8)"; g.lineWidth = 2; g.setLineDash([3, 4]);
            g.beginPath(); g.moveTo(...seg[0]); g.lineTo(...seg[1]); g.stroke(); g.setLineDash([]);
          }
        }
        const cb = drawActor(g, cam, a, scene.dark);
        if ((scene.layers.labels && labelled.has(a)) || a.alert) {
          const p = cam.project(cam.kind === "top" ? [a.x, a.y + dimsOf(a.cls)[1] + 0.5, 0] : [cb.top[0], cb.top[1], cb.h + 0.5]);
          if (p) {
            const txt = a.alert ? `⚠ ${a.cls} ${a.id}${a.alert.ttc != null ? ` · TTC ${a.alert.ttc.toFixed(1)} s` : ""}`
              : `${a.cls} ${a.id}${a.p != null ? ` · ${a.p.toFixed(2)}` : ""}`;
            label(g, p[0], p[1], txt, a.alert ? "rgba(200,40,40,0.92)" : "rgba(0,0,0,0.6)");
          }
        }
      }
      for (const c of conflicts) drawConflict(g, cam, c, scene.t);
      drawEgo(g, cam);
    },
  };
}
