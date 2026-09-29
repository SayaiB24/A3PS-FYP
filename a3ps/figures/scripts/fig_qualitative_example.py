#!/usr/bin/env python
"""Figure B: qualitative example on clip 879 (train_val_v2 positive).

Panel (a): keyframe at the GRU alert-onset frame (frame_idx 544, t=18.133 s),
drawn the way the v2 web dashboard draws it: a line-for-line port of
``drawVideo()`` in dashboard_v2/js/live.js (mask fill + outline, fading trail,
fading forecast dots, box, and a "cls ID-id P" label on a dark box for every
actor, all coloured by risk level). It is fed exactly the per-frame payload
the dashboard receives, built by the dashboard's own
``DashboardStore._slim_frames`` (scripts/dashboard_v2_data.py) from the cached
eval/anticipation/879/events.json, over the raw frame from
data/nexar/videos/00879.mp4 at native resolution. Nothing is re-run. Two
print-only departures, both display-only: the corridor outline is a dark dashed
line (the dashboard's is a faint 50%-alpha blue), and the Nexar watermark strip
is cropped out of view.

Panel (b): p_t vs time from a fresh forward pass of the LOCKED checkpoint
(risk_gru_k1p0_v2_selfix_s1234.pt) over the CACHED feature file
data/features/train_val_v2/879.npz -- the same numbers behind Figure A and
Table IV. Inference only; find_alert_episodes/first_alert_time from
a3ps.risk.anticipation_loss, reused unmodified.

    python figures/scripts/fig_qualitative_example.py

Writes figures/fig_qualitative_example.{png,pdf,svg}.
"""

import os
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import torch  # noqa: E402

from a3ps.common.schema import ClipResult  # noqa: E402
from a3ps.features.extract import load_features  # noqa: E402
from a3ps.risk.anticipation_loss import find_alert_episodes, first_alert_time  # noqa: E402
from a3ps.risk.temporal import RiskGRU  # noqa: E402
from dashboard_v2_data import DashboardStore  # noqa: E402

CLIP_ID = "879"
CHECKPOINT = os.path.join(ROOT, "notebooks", "models", "risk_gru_k1p0_v2_selfix_s1234.pt")
FEATURES_NPZ = os.path.join(ROOT, "data", "features", "train_val_v2", f"{CLIP_ID}.npz")
EVENTS_JSON = os.path.join(ROOT, "eval", "anticipation", CLIP_ID, "events.json")
VIDEO_PATH = os.path.join(ROOT, "data", "nexar", "videos", "00879.mp4")

THRESHOLD, CONFIRM = 0.70, 8
KEYFRAME_IDX = 544  # nearest events.json frame to the GRU alert onset (18.133 s)

# Display-only crop of the Nexar watermark band (native pixels from the top).
TOP_CROP_PX = 85
# Label size for print; the dashboard's own is ~21 px, i.e. ~4 pt at this width.
LABEL_PT = 5.0

OUT_BASE = os.path.join(ROOT, "figures", "fig_qualitative_example")

INK = "#222222"
BLUE = "#2F5D8A"
GREY = "#6E6E6E"
WINDOW_FILL = "#E1E8F1"

# dashboard_v2/js/live.js
RISK = {"safe": (46, 204, 113), "caution": (245, 166, 35), "danger": (231, 76, 60)}


def rgba(c, a):
    return (c[0] / 255, c[1] / 255, c[2] / 255, a)


# ---------------------------------------------------------------------------
# panel (b): cached-feature inference
# ---------------------------------------------------------------------------

def panel_b_data():
    model, extra = RiskGRU.load(CHECKPOINT)
    model.eval()
    d = load_features(FEATURES_NPZ)
    meta = d["meta"]
    X = torch.from_numpy(d["X"])
    t = torch.from_numpy(d["t"])
    with torch.no_grad():
        probs = torch.sigmoid(model(X))
    onset = first_alert_time(probs, t, THRESHOLD, CONFIRM)
    episodes = find_alert_episodes(probs, t, THRESHOLD, CONFIRM)
    return {
        "t": t.tolist(), "p": probs.tolist(),
        "alert_t": meta["alert_time_s"], "event_t": meta["event_time_s"],
        "onset": onset, "episodes": episodes,
    }


# ---------------------------------------------------------------------------
# panel (a): dashboard payload + raw frame
# ---------------------------------------------------------------------------

def extract_raw_frame(video_path, frame_idx):
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"could not read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def panel_a_data():
    """The dashboard's own slim payload for exactly this frame, plus the raw image."""
    clip = ClipResult.load_json(EVENTS_JSON)
    record = next(f for f in clip.frames if f.frame_idx == KEYFRAME_IDX)
    # A one-frame view so _slim_frames' 10 Hz subsampling can't pick a neighbour.
    one = types.SimpleNamespace(meta=clip.meta, frames=[record])
    slim = DashboardStore._slim_frames(one, 10.0, 16)[0]
    assert slim["i"] == KEYFRAME_IDX
    img = extract_raw_frame(VIDEO_PATH, KEYFRAME_IDX)
    return slim, img, clip.meta


def draw_dashboard_frame(ax, frame, meta, pt_per_px):
    """Port of drawVideo() in dashboard_v2/js/live.js, in native pixel coords.

    No alert event is active at this frame (the rule-based events in this clip
    are at 4.1 s, 20.3 s and 39.9 s; the dashboard highlights one for 2.5 s),
    so the non-alert branch is the one the dashboard takes here.
    """
    from matplotlib.patches import Circle, Polygon, Rectangle

    width = meta.get("width") or 1280
    lw = max(1.5, width / 700)         # dashboard line width, px (sx = sy = 1)
    lwp = lw * pt_per_px               # same, in points

    if frame.get("corridor_img"):
        ax.add_patch(Polygon(frame["corridor_img"], closed=True,
                             facecolor=rgba((77, 163, 255), 0.12), edgecolor="none", zorder=2))
        xs = [p[0] for p in frame["corridor_img"]] + [frame["corridor_img"][0][0]]
        ys = [p[1] for p in frame["corridor_img"]] + [frame["corridor_img"][0][1]]
        ax.plot(xs, ys, color="white", linewidth=2.2, zorder=3, solid_capstyle="round")
        ax.plot(xs, ys, color="#0B2545", linewidth=1.0, linestyle=(0, (4, 2)), zorder=3)

    for tr in frame["tracks"]:
        c = RISK.get(tr["lvl"], RISK["safe"])
        if tr.get("mask") and len(tr["mask"]) > 2:
            ax.add_patch(Polygon(tr["mask"], closed=True, facecolor=rgba(c, 0.35),
                                 edgecolor=rgba(c, 1), linewidth=lwp, zorder=4))
        trail = [p for p in (tr.get("trail") or []) if p]
        if len(trail) > 1:
            for k in range(len(trail) - 1):
                ax.plot([trail[k][0], trail[k + 1][0]], [trail[k][1], trail[k + 1][1]],
                        color=rgba(c, (k + 1) / (len(trail) - 1)), linewidth=lwp, zorder=5)
        pred = [p for p in (tr.get("pred_img") or []) if p]
        n = len(pred)
        for k, (x, y) in enumerate(pred):
            f = k / max(1, n - 1)
            ax.add_patch(Circle((x, y), max(1.5, 4 * (1 - f)) * lw / 1.5,
                                facecolor=rgba(c, 1 - 0.85 * f), edgecolor="none", zorder=6))
        x1, y1, x2, y2 = tr["bbox"]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                               edgecolor=rgba(c, 0.9), linewidth=lwp, zorder=7))

    # Labels last so masks never cover them (same as the dashboard). Text and
    # colour are the dashboard's; only the POSITION of labels that would sit on
    # top of each other (the queue of distant cars) is moved, with a leader
    # line back to the box corner the dashboard anchors them to.
    labels = []
    for tr in sorted(frame["tracks"], key=lambda tr: tr["bbox"][0]):
        c = RISK.get(tr["lvl"], RISK["safe"])
        x1, y1 = tr["bbox"][0], tr["bbox"][1]
        text = f"{tr['cls']} ID-{tr['id']}" + (f" {tr['p']:.2f}" if tr.get("p") is not None else "")
        txt = ax.text(x1, y1, text, color=rgba(c, 1), fontsize=LABEL_PT, family="monospace",
                      ha="left", va="bottom", zorder=9, clip_on=True,
                      bbox=dict(facecolor=(0, 0, 0, 0.6), edgecolor="none", pad=1.0))
        labels.append((txt, tr["bbox"], c))
    return labels


def rail_labels(fig, ax, labels, boxes, img_w, row_bottoms):
    """Lay the dashboard labels out in rows along the top of the frame, ordered
    left-to-right by their actor's box centre, each with a thin leader line to
    the top-centre of its box. Only label POSITIONS change; text and colour
    are the dashboard's."""
    renderer = fig.canvas.get_renderer()
    px_per_disp = img_w / ax.get_window_extent().width
    widths = [t.get_bbox_patch().get_window_extent(renderer).width * px_per_disp
              for t, _, _ in labels]
    gap = 12.0
    n_rows = len(row_bottoms)
    order = sorted(range(len(labels)), key=lambda i: (boxes[i][0] + boxes[i][2]) / 2)
    rows = [[] for _ in range(n_rows)]
    for k, i in enumerate(order):
        rows[k % n_rows].append(i)

    for r, idxs in enumerate(rows):
        # desired left edges centred on each box, then pushed apart left->right
        xs = []
        for i in idxs:
            cx = (boxes[i][0] + boxes[i][2]) / 2
            x = max(2.0, cx - widths[i] / 2)
            if xs:
                prev = idxs[len(xs) - 1]
                x = max(x, xs[-1] + widths[prev] + gap)
            xs.append(x)
        overflow = (xs[-1] + widths[idxs[-1]]) - (img_w - 2.0) if xs else 0
        if overflow > 0:   # slide the row back left, keeping order and gaps
            for j in range(len(xs) - 1, -1, -1):
                xs[j] -= overflow
                if j and xs[j] < xs[j - 1] + widths[idxs[j - 1]] + gap:
                    overflow = xs[j - 1] + widths[idxs[j - 1]] + gap - xs[j]
                    xs[j - 1] -= 0  # already moved by the loop on the next pass
            for j in range(1, len(xs)):
                xs[j] = max(xs[j], xs[j - 1] + widths[idxs[j - 1]] + gap)
        for i, x in zip(idxs, xs):
            txt, _, c = labels[i]
            y = row_bottoms[r]
            txt.set_position((x, y))
            bx = (boxes[i][0] + boxes[i][2]) / 2
            ax.plot([x + widths[i] / 2, bx], [y, boxes[i][1]], color=rgba(c, 0.9),
                    linewidth=0.45, zorder=8)


def main():
    pb = panel_b_data()
    frame, img, meta = panel_a_data()
    keyframe_t = frame["t"]

    print(f"panel (b) onset (cached-feature inference): {pb['onset']}")
    print(f"panel (a) keyframe: frame_idx={frame['i']} t={keyframe_t}")
    if pb["onset"] is not None and abs(pb["onset"] - keyframe_t) > 1e-3:
        print(f"NOTE: cached-feature onset ({pb['onset']}) and keyframe time "
              f"({keyframe_t}) differ by {abs(pb['onset'] - keyframe_t):.3f}s")
    else:
        print("CONFIRMED: panel (a)'s rendered frame is the same frame as "
              f"panel (b)'s keyframe marker (frame_idx {frame['i']}, t={keyframe_t:.3f}s).")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "svg.fonttype": "path", "font.size": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.16, 2.4))

    # ---- panel (a) ----
    h, w = img.shape[:2]
    axa.imshow(img)
    axa.set_xlim(0, w)
    axa.set_ylim(h, TOP_CROP_PX)
    axa.set_xticks([])
    axa.set_yticks([])
    for spine in axa.spines.values():
        spine.set_visible(False)

    # ---- panel (b) ----
    t = pb["t"]
    p = pb["p"]
    alert_t, event_t = pb["alert_t"], pb["event_t"]

    axb.axvspan(alert_t, event_t, color=WINDOW_FILL, zorder=0)
    axb.plot(t, p, color=BLUE, linewidth=1.0, zorder=3)
    axb.axhline(THRESHOLD, color=GREY, linestyle="--", linewidth=0.7, zorder=1)
    axb.axvline(alert_t, color=INK, linestyle=":", linewidth=0.7, zorder=1)
    axb.axvline(event_t, color=INK, linestyle=":", linewidth=0.7, zorder=1)

    bar_y = -0.06
    for ep in pb["episodes"]:
        axb.plot([ep["confirm_t"], ep["end_t"]], [bar_y, bar_y], color=BLUE,
                 linewidth=3.0, solid_capstyle="butt", clip_on=False, zorder=3)

    axb.scatter([keyframe_t], [p[min(range(len(t)), key=lambda i: abs(t[i] - keyframe_t))]],
                s=26, facecolor=BLUE, edgecolor=INK, linewidth=0.6, zorder=4)

    # t_a/t_e labels offset to either side so the dotted lines don't cross them.
    off = 0.018 * (t[-1] - t[0])
    axb.text(alert_t - off, 1.02, r"$t_a$", fontsize=6.5, color=INK, ha="right", va="bottom")
    axb.text(event_t + off, 1.02, r"$t_e$", fontsize=6.5, color=INK, ha="left", va="bottom")
    axb.text(t[0] + 0.2, THRESHOLD + 0.02, r"$\theta=0.70$", fontsize=6.5, color=GREY,
             ha="left", va="bottom")

    axb.set_xlabel("time (s)", fontsize=7.5)
    axb.set_ylabel(r"$p_t$", fontsize=7.5)
    axb.set_ylim(-0.12, 1.05)
    axb.set_xlim(t[0], t[-1])
    axb.tick_params(labelsize=7)

    fig.tight_layout(pad=0.4, w_pad=1.6)
    fig.subplots_adjust(bottom=0.25)
    fig.canvas.draw()

    # Dashboard overlays go on after layout is fixed, so their pixel->point
    # scale (line widths) is exact.
    bbox = axa.get_window_extent()
    pt_per_px = (bbox.width / fig.dpi * 72.0) / w
    labels = draw_dashboard_frame(axa, frame, meta, pt_per_px)
    fig.canvas.draw()
    rail_labels(fig, axa, labels, [b for _, b, _ in labels], w,
                row_bottoms=(TOP_CROP_PX + 40, TOP_CROP_PX + 80, TOP_CROP_PX + 120))

    # "(a)"/"(b)" from the final axes positions, on one row.
    pos_a, pos_b = axa.get_position(), axb.get_position()
    label_y = 0.012
    fig.text(pos_a.x0, label_y, "(a)", fontsize=7.5, color=INK, ha="left", va="bottom")
    fig.text(pos_b.x0, label_y, "(b)", fontsize=7.5, color=INK, ha="left", va="bottom")

    os.makedirs(os.path.dirname(OUT_BASE), exist_ok=True)
    for ext, kw in ((".png", {"dpi": 600}), (".pdf", {}), (".svg", {})):
        fig.savefig(OUT_BASE + ext, **kw)
    plt.close(fig)
    print(f"wrote {OUT_BASE}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
