"""Tests for the anticipation eval metrics + processed-clip reading (no GPU)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from a3ps.common.schema import (  # noqa: E402
    ClipResult, Event, FrameRecord, Prediction, TrackState,
)

import eval_anticipation as ea  # noqa: E402


# ---------------------------------------------------------------------------
# metric math
# ---------------------------------------------------------------------------

def test_average_precision_perfect_and_ranking():
    # Positives all score above negatives -> AP = 1.0.
    assert abs(ea.average_precision([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) - 1.0) < 1e-9
    # No positives -> NaN.
    assert ea.average_precision([0.5, 0.4], [0, 0]) != ea.average_precision([0.5], [0])  # NaN != NaN
    # One negative ranked above one positive: AP = 0.5.
    ap = ea.average_precision([0.9, 0.8], [0, 1])
    assert abs(ap - 0.5) < 1e-9


def _row(clip_id, label, event_time_s, first_alert_t, peak_prob, processed=True):
    return {
        "clip_id": clip_id, "label": label, "event_time_s": event_time_s,
        "first_alert_t": first_alert_t, "peak_prob": peak_prob, "processed": processed,
    }


def test_compute_metrics_recall_tta_false_alarms():
    rows = [
        _row("p1", 1, event_time_s=5.0, first_alert_t=3.0, peak_prob=0.9),   # anticipated, TTA 2.0
        _row("p2", 1, event_time_s=6.0, first_alert_t=6.5, peak_prob=0.8),   # alert AFTER event -> miss
        _row("p3", 1, event_time_s=4.0, first_alert_t=None, peak_prob=0.3),  # no alert -> miss
        _row("n1", 0, event_time_s=None, first_alert_t=None, peak_prob=0.1),  # clean
        _row("n2", 0, event_time_s=None, first_alert_t=2.0, peak_prob=0.7),   # false alarm
        _row("skip", 1, event_time_s=5.0, first_alert_t=1.0, peak_prob=0.9, processed=False),
    ]
    summary, detail = ea.compute_metrics(rows)

    assert summary["n_processed"] == 5           # the unprocessed row is excluded
    assert summary["n_positive"] == 3
    assert summary["n_negative"] == 2
    assert abs(summary["detection_recall"] - (1 / 3)) < 1e-9
    assert abs(summary["mTTA_s"] - 2.0) < 1e-9   # only p1 contributes
    assert summary["n_tta"] == 1
    assert abs(summary["false_alarm_rate"] - 0.5) < 1e-9


def test_compute_metrics_all_clean_negatives():
    rows = [
        _row("n1", 0, None, None, 0.1),
        _row("n2", 0, None, None, 0.2),
    ]
    summary, _ = ea.compute_metrics(rows)
    assert summary["false_alarm_rate"] == 0.0
    assert summary["n_positive"] == 0


# ---------------------------------------------------------------------------
# reading processed clips end-to-end (writes real events.json fixtures)
# ---------------------------------------------------------------------------

def _write_clip(clips_dir, clip_id, peak_prob, alert_t=None, alert_type="VIRTUAL_BRAKE"):
    """Write a minimal but real ClipResult -> <clips_dir>/<clip_id>/events.json."""
    pred = Prediction(horizon_s=4.0, dt=0.2, mean_img=[[0.0, 0.0]],
                      collision_prob=peak_prob, ttc_s=1.0)
    tr = TrackState(id=1, cls="person", bbox=[0, 0, 10, 10],
                    centroid_img=[5.0, 5.0], prediction=pred)
    clip = ClipResult(
        meta={"clip_id": clip_id},
        frames=[FrameRecord(frame_idx=0, t=0.0, tracks=[tr])],
        events=[],
    )
    if alert_t is not None:
        clip.events.append(Event(
            event_id=1, frame_idx=int(alert_t / 0.2), t=alert_t, type=alert_type,
            actor_id=1, actor_cls="person", collision_prob=peak_prob, threshold=0.75,
            ttc_s=1.0))
    out = os.path.join(clips_dir, str(clip_id))
    os.makedirs(out, exist_ok=True)
    clip.save_json(os.path.join(out, "events.json"))


def test_summarize_and_read_from_disk(tmp_path):
    clips_dir = str(tmp_path / "anticipation")
    _write_clip(clips_dir, "posA", peak_prob=0.88, alert_t=2.5)     # anticipated (event 5)
    _write_clip(clips_dir, "negB", peak_prob=0.12, alert_t=None)    # clean negative

    # Read posA back and summarize.
    clip = ClipResult.load_json(ea.clip_events_path(clips_dir, "posA"))
    first_alert_t, peak = ea.summarize_clip(clip, ea.DEFAULT_ALERT_TYPES)
    assert first_alert_t == 2.5
    assert abs(peak - 0.88) < 1e-9

    # Assemble rows the way main() does and check the metrics.
    rows = []
    for cid, label, te in [("posA", 1, 5.0), ("negB", 0, None)]:
        ev = ea.clip_events_path(clips_dir, cid)
        clip = ClipResult.load_json(ev)
        fa, pk = ea.summarize_clip(clip, ea.DEFAULT_ALERT_TYPES)
        rows.append({"clip_id": cid, "label": label, "event_time_s": te,
                     "first_alert_t": fa, "peak_prob": pk, "processed": True})
    summary, _ = ea.compute_metrics(rows)
    assert summary["detection_recall"] == 1.0
    assert summary["false_alarm_rate"] == 0.0
    assert abs(summary["mTTA_s"] - 2.5) < 1e-9    # 5.0 - 2.5
    assert abs(summary["AP"] - 1.0) < 1e-9


def test_alert_type_filter_ignores_threshold_lowered(tmp_path):
    # A THRESHOLD_LOWERED event alone must NOT count as an anticipation alert.
    clips_dir = str(tmp_path / "anticipation")
    _write_clip(clips_dir, "c1", peak_prob=0.5, alert_t=1.0, alert_type="THRESHOLD_LOWERED")
    clip = ClipResult.load_json(ea.clip_events_path(clips_dir, "c1"))
    first_alert_t, _ = ea.summarize_clip(clip, ea.DEFAULT_ALERT_TYPES)
    assert first_alert_t is None
