#!/usr/bin/env python
"""Qualitative before/alert/brake/after figure for one processed dashboard clip.

Given a clip already processed WITH rendering (i.e. via
``scripts/run_pipeline.py``, so it has both ``raw.mp4`` and ``events.json``
under ``dashboard/clips/<id>/``), this finds one ALERT -> VIRTUAL_BRAKE
incident and extracts four frames:

    1. one frame just before the SAFE -> ALERT transition
    2. the SAFE -> ALERT transition frame
    3. the VIRTUAL_BRAKE frame
    4. one frame just after the VIRTUAL_BRAKE

Each frame is pulled from ``raw.mp4`` with OpenCV and drawn with the EXACT
same overlays the dashboard uses -- mask tinted by ``risk_level``, predicted
path, ego corridor -- by reusing ``a3ps.pipeline.Pipeline._draw_frame`` /
``_draw_prediction`` directly (via a bare, ``__init__``-free instance, since
those two methods touch no other instance state -- so the qualitative figure
is guaranteed to look exactly like the dashboard, not a re-implementation
that could drift out of sync with it).

The four panels are arranged in a 2x2 grid with matplotlib, each labeled with
its ``risk_level`` and timestamp; the VIRTUAL_BRAKE panel is captioned with
the event's real ``explanation_template`` string. Saved at 300 DPI (PNG or
PDF, by extension) for print use in the report.

    python scripts/make_qual_figure.py --clip dashboard/clips/dev14
    python scripts/make_qual_figure.py --clip dashboard/clips/dev14 --out eval/figures/dev14_qual.pdf
"""

import argparse
import os
import sys
import textwrap

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a3ps.common.schema import ClipResult  # noqa: E402
from a3ps.pipeline import Pipeline  # noqa: E402

# Mirrors a3ps.pipeline.Pipeline.BRAKE_FLASH_S exactly, so the red
# "VIRTUAL BRAKE" border/vignette overlay matches the dashboard's own timing.
BRAKE_FLASH_S = 0.5


# ---------------------------------------------------------------------------
# finding the incident + the four target frames
# ---------------------------------------------------------------------------

def find_incident(clip: ClipResult, actor_id=None):
    """Return (alert_event, brake_event) for one ALERT->VIRTUAL_BRAKE incident.

    Picks the earliest VIRTUAL_BRAKE in the clip (or the one for ``actor_id``
    if given), then the ALERT event for that same actor immediately preceding
    it -- per the decision-engine state machine (a3ps/risk/decision.py),
    VIRTUAL_BRAKE only ever fires after that actor's own ALERT.
    """
    brakes = [e for e in clip.events if e.type == "VIRTUAL_BRAKE"]
    if actor_id is not None:
        brakes = [e for e in brakes if e.actor_id == actor_id]
    if not brakes:
        raise ValueError(
            "No VIRTUAL_BRAKE event found in this clip"
            + (f" for actor_id={actor_id}" if actor_id is not None else "")
            + " -- pick a positive clip with a real braking incident.")
    brake = min(brakes, key=lambda e: e.t)

    alerts = [e for e in clip.events
              if e.type == "ALERT" and e.actor_id == brake.actor_id and e.t <= brake.t]
    if not alerts:
        raise ValueError(
            f"No ALERT event found for actor_id={brake.actor_id} before its "
            f"VIRTUAL_BRAKE at t={brake.t:.2f} -- unexpected given the decision "
            f"engine's state machine; check events.json wasn't hand-edited.")
    alert = max(alerts, key=lambda e: e.t)   # the one immediately preceding brake
    return alert, brake


def pick_four_frames(clip: ClipResult, alert_event, brake_event):
    """(panel label, FrameRecord) x4, in display order.

    Uses each frame's POSITION in ``clip.frames`` (not frame_idx arithmetic)
    to find neighbors, so it stays correct even if frame_idx ever has gaps.
    """
    pos_by_frame_idx = {fr.frame_idx: i for i, fr in enumerate(clip.frames)}
    if alert_event.frame_idx not in pos_by_frame_idx:
        raise ValueError(f"ALERT frame_idx={alert_event.frame_idx} not found in events.json frames")
    if brake_event.frame_idx not in pos_by_frame_idx:
        raise ValueError(f"VIRTUAL_BRAKE frame_idx={brake_event.frame_idx} not found in events.json frames")

    alert_pos = pos_by_frame_idx[alert_event.frame_idx]
    brake_pos = pos_by_frame_idx[brake_event.frame_idx]
    before_alert_pos = max(0, alert_pos - 1)
    after_brake_pos = min(len(clip.frames) - 1, brake_pos + 1)

    return [
        ("before ALERT", clip.frames[before_alert_pos]),
        ("SAFE -> ALERT", clip.frames[alert_pos]),
        ("VIRTUAL_BRAKE", clip.frames[brake_pos]),
        ("after BRAKE", clip.frames[after_brake_pos]),
    ]


# ---------------------------------------------------------------------------
# frame extraction + dashboard-identical drawing
# ---------------------------------------------------------------------------

def extract_frame(cap, frame_idx):
    import cv2

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise IOError(f"Could not read frame {frame_idx} from raw.mp4 "
                       f"(video may be shorter than expected, or frame_idx/fps mismatch).")
    return frame


def draw_panel(frame, record, ego_poly, brake_event):
    """Draw this frame with the exact dashboard overlays.

    ``Pipeline._draw_frame``/``_draw_prediction`` reference no instance state
    other than each other (both are pure functions of their arguments plus
    the module-level RISK_COLORS), so a bare instance created via
    ``Pipeline.__new__`` (skipping ``__init__`` -- no YOLO/tracker load) is
    sufficient and keeps this script's output identical to the dashboard's by
    construction, not by copy-pasted drawing code that could drift.
    """
    stub = Pipeline.__new__(Pipeline)
    brake_flash = 0.0 <= (record.t - brake_event.t) <= BRAKE_FLASH_S
    return stub._draw_frame(frame, record, ego_poly, brake_flash=brake_flash)


def track_risk_level(record, actor_id):
    for tr in record.tracks:
        if tr.id == actor_id:
            return tr.risk_level
    return "n/a (actor not tracked this frame)"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Render a 2x2 before/ALERT/VIRTUAL_BRAKE/after qualitative figure.")
    p.add_argument("--clip", required=True,
                   help="Processed clip dir, e.g. dashboard/clips/dev14 "
                        "(needs raw.mp4 + events.json -- i.e. run_pipeline.py, "
                        "not eval_anticipation.py's render=False path).")
    p.add_argument("--actor-id", type=int, default=None,
                   help="Pick the incident for this actor id "
                        "(default: earliest VIRTUAL_BRAKE in the clip).")
    p.add_argument("--out", default=None,
                   help="Output path, .png or .pdf (default: eval/figures/<clip_id>_qual.png).")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    events_path = os.path.join(args.clip, "events.json")
    raw_path = os.path.join(args.clip, "raw.mp4")
    if not os.path.isfile(events_path):
        p.error(f"missing {events_path}")
    if not os.path.isfile(raw_path):
        p.error(f"missing {raw_path} -- this clip was processed without rendering "
                f"(e.g. by eval_anticipation.py, which uses render=False); "
                f"re-run it with scripts/run_pipeline.py instead.")

    clip = ClipResult.load_json(events_path)
    try:
        alert_event, brake_event = find_incident(clip, args.actor_id)
        panels = pick_four_frames(clip, alert_event, brake_event)
    except ValueError as exc:
        p.error(str(exc))

    ego_poly = None
    for _label, record in panels:
        if record.ego and record.ego.get("corridor_poly_img"):
            ego_poly = record.ego["corridor_poly_img"]
            break
    if ego_poly is None:
        p.error("no corridor_poly_img found on any of the four target frames")

    import cv2

    cap = cv2.VideoCapture(raw_path)
    if not cap.isOpened():
        p.error(f"cannot open {raw_path}")

    rendered = []
    try:
        for label, record in panels:
            frame = extract_frame(cap, record.frame_idx)
            frame = draw_panel(frame, record, ego_poly, brake_event)
            rendered.append((label, record, frame))
    finally:
        cap.release()

    import matplotlib
    matplotlib.use("Agg")   # headless
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (label, record, frame) in zip(axes.flat, rendered):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_xticks([])
        ax.set_yticks([])
        risk = track_risk_level(record, brake_event.actor_id)
        ax.set_title(f"{label}\nrisk_level={risk}   t={record.t:.2f}s", fontsize=10)
        if label == "VIRTUAL_BRAKE":
            caption = brake_event.explanation_template or "(no explanation_template on this event)"
            ax.set_xlabel("\n".join(textwrap.wrap(caption, width=55)), fontsize=8)

    clip_id = os.path.basename(os.path.normpath(args.clip))
    fig.suptitle(f"{clip_id} -- actor {brake_event.actor_cls} ID {brake_event.actor_id}",
                 fontsize=12)
    fig.tight_layout()

    out = args.out or os.path.join("eval", "figures", f"{clip_id}_qual.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
