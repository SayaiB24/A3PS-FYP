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
# matched-subset mTTA (fair A3PS-vs-reactive comparison)
# ---------------------------------------------------------------------------

def _master_row(clip_id, label, event_time_s, a3ps_fa, reactive_fa, processed=True):
    return {
        "clip_id": clip_id, "label": label, "event_time_s": event_time_s,
        "a3ps_first_alert_t": a3ps_fa, "reactive_first_alert_t": reactive_fa,
        "processed": processed,
    }


def test_matched_subset_only_counts_positives_both_anticipate():
    rows = [
        # both anticipate: A3PS tta=2.0 (5-3), reactive tta=1.0 (5-4)
        _master_row("p1", 1, 5.0, a3ps_fa=3.0, reactive_fa=4.0),
        # only A3PS anticipates -> excluded from the matched subset
        _master_row("p2", 1, 6.0, a3ps_fa=5.5, reactive_fa=None),
        # only reactive anticipates -> excluded
        _master_row("p3", 1, 4.0, a3ps_fa=None, reactive_fa=3.5),
        # negative, irrelevant to matched mTTA
        _master_row("n1", 0, None, a3ps_fa=None, reactive_fa=None),
    ]
    m = ea.matched_subset_metrics(rows)
    assert m is not None
    assert m["n_matched"] == 1
    assert abs(m["a3ps_mtta_s"] - 2.0) < 1e-9
    assert abs(m["reactive_mtta_s"] - 1.0) < 1e-9
    assert abs(m["gain_s"] - 1.0) < 1e-9   # A3PS is 1.0s earlier on the matched subset


def test_matched_subset_none_when_no_overlap():
    rows = [
        _master_row("p1", 1, 5.0, a3ps_fa=3.0, reactive_fa=None),
        _master_row("p2", 1, 6.0, a3ps_fa=None, reactive_fa=5.0),
    ]
    assert ea.matched_subset_metrics(rows) is None


def test_matched_subset_ignores_unprocessed_rows():
    rows = [
        _master_row("p1", 1, 5.0, a3ps_fa=3.0, reactive_fa=4.0, processed=False),
    ]
    assert ea.matched_subset_metrics(rows) is None


# ---------------------------------------------------------------------------
# threshold sweep (Step 7.1): cache + in-memory precision/recall
# ---------------------------------------------------------------------------

def _clip_with_frame_probs(clip_id, probs_by_t):
    """A minimal real ClipResult with one frame per (t, prob) pair."""
    frames = []
    for t, prob in probs_by_t:
        pred = Prediction(horizon_s=4.0, dt=0.2, mean_img=[[0.0, 0.0]], collision_prob=prob)
        tr = TrackState(id=1, cls="car", bbox=[0, 0, 10, 10], centroid_img=[5.0, 5.0], prediction=pred)
        frames.append(FrameRecord(frame_idx=int(t / 0.5), t=t, tracks=[tr]))
    return ClipResult(meta={"clip_id": clip_id}, frames=frames, events=[])


def test_cache_frame_probs_round_trips(tmp_path):
    clip = _clip_with_frame_probs("c1", [(0.0, 0.3), (0.5, 0.7), (1.0, 0.95)])
    cache_dir = str(tmp_path / "cache")
    path = ea.cache_frame_probs(clip, "c1", cache_dir)
    assert os.path.isfile(path)

    loaded = ea.load_frame_prob_cache(cache_dir, "c1")
    assert loaded == [(0.0, 0.3), (0.5, 0.7), (1.0, 0.95)]


def test_cache_frame_probs_skips_existing_unless_forced(tmp_path):
    cache_dir = str(tmp_path / "cache")
    clip_a = _clip_with_frame_probs("c1", [(0.0, 0.1)])
    clip_b = _clip_with_frame_probs("c1", [(0.0, 0.9)])   # different content, same clip_id

    ea.cache_frame_probs(clip_a, "c1", cache_dir)
    ea.cache_frame_probs(clip_b, "c1", cache_dir)         # no force -> should NOT overwrite
    assert ea.load_frame_prob_cache(cache_dir, "c1") == [(0.0, 0.1)]

    ea.cache_frame_probs(clip_b, "c1", cache_dir, force=True)
    assert ea.load_frame_prob_cache(cache_dir, "c1") == [(0.0, 0.9)]


def test_load_frame_prob_cache_missing_returns_none(tmp_path):
    assert ea.load_frame_prob_cache(str(tmp_path), "does_not_exist") is None


def test_sweep_thresholds_precision_recall(tmp_path):
    cache_dir = str(tmp_path / "cache")
    # Positive clip: event at t=2.0. Frames before/at the event reach 0.7 then
    # 0.9; a later 0.95 frame is AFTER the event and must not count as a hit.
    pos_clip = _clip_with_frame_probs(
        "p1", [(0.0, 0.3), (0.5, 0.5), (1.0, 0.7), (1.5, 0.9), (2.5, 0.95)])
    # Negative clip: no event_time_s, so any frame crossing threshold is a
    # false alarm regardless of when it occurs.
    neg_clip = _clip_with_frame_probs("n1", [(0.0, 0.1), (0.5, 0.2)])

    ea.cache_frame_probs(pos_clip, "p1", cache_dir)
    ea.cache_frame_probs(neg_clip, "n1", cache_dir)

    rows = [
        {"clip_id": "p1", "label": 1, "event_time_s": 2.0, "processed": True},
        {"clip_id": "n1", "label": 0, "event_time_s": None, "processed": True},
    ]

    sweep = ea.sweep_thresholds(rows, cache_dir, thresholds=[0.15, 0.6, 0.8, 0.99])
    by_thr = {row["threshold"]: row for row in sweep}

    # thr=0.15: neg's 0.2 frame >= 0.15 -> FP; pos's <=te frames reach 0.9 -> TP.
    assert by_thr[0.15]["tp"] == 1 and by_thr[0.15]["fp"] == 1 and by_thr[0.15]["fn"] == 0
    # thr=0.6: neg never reaches 0.6 -> no FP; pos's t=1.0 (0.7, <= te) -> TP.
    assert by_thr[0.6]["tp"] == 1 and by_thr[0.6]["fp"] == 0 and by_thr[0.6]["fn"] == 0
    # thr=0.8: pos's t=1.5 (0.9, <= te) still qualifies -> TP.
    assert by_thr[0.8]["tp"] == 1 and by_thr[0.8]["fp"] == 0
    # thr=0.99: only the 0.95 frame reaches it, but that frame is AFTER the
    # event (t=2.5 > te=2.0) -> excluded -> FN, not TP.
    assert by_thr[0.99]["tp"] == 0 and by_thr[0.99]["fn"] == 1

    assert abs(by_thr[0.15]["precision"] - 0.5) < 1e-9   # 1 tp / (1 tp + 1 fp)
    assert by_thr[0.6]["precision"] == 1.0
    assert by_thr[0.6]["recall"] == 1.0


def test_sweep_thresholds_skips_rows_without_a_cache(tmp_path):
    cache_dir = str(tmp_path / "cache")   # empty -- nothing cached
    rows = [{"clip_id": "p1", "label": 1, "event_time_s": 2.0, "processed": True}]
    sweep = ea.sweep_thresholds(rows, cache_dir, thresholds=[0.5])
    assert sweep[0]["tp"] == 0 and sweep[0]["fp"] == 0 and sweep[0]["fn"] == 0


def test_write_pr_csv(tmp_path):
    path = str(tmp_path / "pr.csv")
    sweep = [
        {"threshold": 0.40, "precision": 1.0, "recall": 0.5, "tp": 1, "fp": 0, "fn": 1},
        {"threshold": 0.45, "precision": float("nan"), "recall": float("nan"),
         "tp": 0, "fp": 0, "fn": 0},
    ]
    ea.write_pr_csv(path, sweep)
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    assert "threshold,precision,recall" in content
    assert "0.40,1.0000,0.5000" in content
    assert "0.45,," in content   # NaN serializes to empty, not "nan"


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
