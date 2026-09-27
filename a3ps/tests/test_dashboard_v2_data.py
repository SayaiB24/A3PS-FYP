"""Tests for the read-only v2 dashboard data layer (scripts/dashboard_v2_data.py)."""

import glob
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import dashboard_v2_data as dv  # noqa: E402


def test_parse_tag_decodes_the_naming_convention():
    c = dv.parse_tag("k1p0_v2_selfix_paw2p0_s1235")
    assert c["kappa"] == 1.0 and c["pre_alert_weight"] == 2.0 and c["seed"] == 1235
    assert c["partition"] == "v2" and c["selector"] == "fixed (selfix)"
    assert dv.parse_tag("k3p0_clean")["partition"] == "clean"
    assert dv.parse_tag("abl_ego")["ablate"] == "ego"
    assert dv.parse_tag("cap_h128")["hidden"] == 128
    assert dv.run_family("abl_ttc") == "ablation"


def test_parse_sweep_md(tmp_path):
    p = tmp_path / "operating_point_sweep_x.md"
    p.write_text(
        "- checkpoint: `notebooks/models/risk_gru_k1p0_v2_s1234.pt` (best epoch 4)\n"
        "- scored on: `data/features/train_val_v2` — 205 clips, 103 positive\n\n"
        "| threshold | confirm | useful | useful n | too early | **false alarm** | mean lead (s) | mean AP |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 0.70 | 8 | 0.243 | 25/103 | 39 | **0.196** ✅ | 1.95 | 0.630 |\n", encoding="utf-8")
    sw = dv.parse_sweep_md(str(p))
    assert sw["tag"] == "k1p0_v2_s1234" and sw["best_epoch"] == 4
    assert sw["split"] == "train_val_v2" and sw["n_clips"] == 205
    (row,) = sw["rows"]
    assert row == {"threshold": 0.7, "confirm": 8, "useful_warning_rate": 0.243,
                   "n_too_early": 39, "false_alarm_rate": 0.196,
                   "mean_lead_s": 1.95, "mean_AP": 0.63}


def _row(label, fired, peak):
    return {"label": label, "outcome": dv.outcome_of(label, fired), "peak_prob": peak}


def test_derived_metrics_confusion_f1_and_pr():
    rows = [_row(1, True, 0.9), _row(1, False, 0.4), _row(0, True, 0.8), _row(0, False, 0.1)]
    d = dv.derived_metrics(rows)
    assert d["confusion"] == {"TP": 1, "FP": 1, "FN": 1, "TN": 1}
    assert d["precision"] == 0.5 and d["recall"] == 0.5 and d["f1"] == 0.5
    # PR by descending peak: 0.9 (1/1), 0.8 (1/2), 0.4 (2/3), 0.1 (2/4)
    assert [(p["recall"], round(p["precision"], 3)) for p in d["pr_curve"]] == \
        [(0.5, 1.0), (0.5, 0.5), (1.0, 0.667), (1.0, 0.5)]


def test_derived_metrics_consumes_ties_together():
    rows = [_row(1, True, 1.0), _row(0, True, 1.0), _row(1, True, 0.5)]
    curve = dv.derived_metrics(rows)["pr_curve"]
    assert curve[0] == {"threshold": 1.0, "recall": 0.5, "precision": 0.5}


def test_clean_removes_nan():
    assert dv._clean({"a": float("nan"), "b": [1.0, float("inf")]}) == {"a": None, "b": [1.0, None]}


needs_artifacts = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "eval", "final_eval_read_v2_s1234.json")),
    reason="repo eval artifacts not present")


@needs_artifacts
def test_store_reproduces_committed_numbers_and_writes_nothing():
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        watched = [p for pat in ("eval/**/*", "dashboard/**/*", "notebooks/models/*")
                   for p in glob.glob(pat, recursive=True) if os.path.isfile(p)]
        before = {p: os.path.getmtime(p) for p in watched}
        store = dv.DashboardStore(load_checkpoints=False)
        after = {p: os.path.getmtime(p) for p in watched}
        new = [p for pat in ("eval/**/*", "dashboard/**/*", "notebooks/models/*")
               for p in glob.glob(pat, recursive=True) if os.path.isfile(p) and p not in before]
    finally:
        os.chdir(cwd)
    assert before == after and not new, "the dashboard data layer must be read-only"

    run = store.runs["k1p0_v2_selfix_s1234"]
    final = next(e for e in run["evals"] if e["kind"] == "final_read")
    assert final["split"] == "eval_v2" and final["held_out"]
    assert final["metrics"]["useful_warning_rate_v2"] == pytest.approx(0.65)
    assert final["metrics"]["false_alarm_rate"] == pytest.approx(1 / 6)
    assert final["derived"]["confusion"] == {"TP": 42, "FP": 10, "FN": 18, "TN": 50}
    assert run["headline_eval"] == final["id"]

    base = store.runs["baseline_threshold_system"]["evals"][0]
    # Same numbers eval/anticipation.md reports for the pre-Phase IV system.
    assert base["metrics"]["false_alarm_rate"] == pytest.approx(0.767, abs=1e-3)
    assert base["metrics"]["useful_warning_rate"] == pytest.approx(0.05)
    assert len(store.rows[base["id"]]) == 120
