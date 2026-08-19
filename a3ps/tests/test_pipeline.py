"""Unit tests for Pipeline's per-stage latency aggregation (Table IV).

build_per_stage_ms() is a pure function of the accumulated timings dict, so
it's tested directly here without needing a real video/YOLO model (that full
path is verified by manually re-running scripts/run_pipeline.py on real dev
clips and checking meta.json -- see docs/runbooks/GPU_HANDOFF.md /
docs/design/metrics.md).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a3ps.pipeline import PER_STAGE_MS_NOTE, build_per_stage_ms  # noqa: E402


def test_build_per_stage_ms_averages_per_frame():
    timings = {"track": 100.0, "forecast": 20.0, "risk": 10.0, "render": 5.0}
    out = build_per_stage_ms(timings, n_frames=10)
    assert out == {
        "perception": 0.0,
        "tracking": 10.0,
        "forecasting": 2.0,
        "risk_decision": 1.0,
    }


def test_build_per_stage_ms_perception_always_zero_by_convention():
    # Perception (YOLO detect+segment) is fused into the single model.track()
    # call -- there is no independent perception measurement to report, so it
    # must always be 0.0 rather than fabricated, regardless of input.
    out = build_per_stage_ms({"track": 999.0}, n_frames=1)
    assert out["perception"] == 0.0


def test_build_per_stage_ms_zero_frames_does_not_divide_by_zero():
    out = build_per_stage_ms({"track": 100.0, "forecast": 50.0, "risk": 5.0}, n_frames=0)
    assert out == {"perception": 0.0, "tracking": 0.0, "forecasting": 0.0, "risk_decision": 0.0}


def test_build_per_stage_ms_missing_keys_default_to_zero():
    out = build_per_stage_ms({}, n_frames=5)
    assert out == {"perception": 0.0, "tracking": 0.0, "forecasting": 0.0, "risk_decision": 0.0}


def test_per_stage_ms_note_explains_the_fusion():
    # The honesty note must actually mention why perception reads 0 -- a
    # future reader of meta.json (or the paper) shouldn't have to guess.
    assert "model.track()" in PER_STAGE_MS_NOTE
    assert "perception" in PER_STAGE_MS_NOTE.lower()
