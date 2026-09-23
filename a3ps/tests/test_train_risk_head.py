"""Tests for the risk-head training loop -- above all, that batching is exact.

The training loop was changed from one forward pass per clip to a single padded
forward pass per batch (~5x faster). That optimisation is only valid because
RiskGRU is causal and its input norm is per-timestep; if either changed, padded
positions would leak into valid ones and the objective would shift *silently*
rather than crash. These tests are the guard against that.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from a3ps.risk.anticipation_loss import batch_anticipation_loss  # noqa: E402
from a3ps.risk.temporal import RiskGRU  # noqa: E402

from train_risk_head import batched_logits  # noqa: E402

D = 112


def _clip(T, seed, label=1, alert=None, event=None):
    g = torch.Generator().manual_seed(seed)
    return {
        "X": torch.randn(T, D, generator=g),
        "t": torch.arange(T, dtype=torch.float32) * 0.1,
        "label": label,
        "alert_time_s": alert,
        "event_time_s": event,
    }


def _model(seed=0):
    torch.manual_seed(seed)
    m = RiskGRU(D, hidden=64, layers=1, dropout=0.1)
    m.eval()          # dropout off: batching must be EXACT, not merely close
    return m


# ---------------------------------------------------------------------------
# the core equivalence claim
# ---------------------------------------------------------------------------

def test_batched_logits_match_per_clip_forward_exactly():
    """A padded batch's valid prefixes equal running each clip on its own."""
    m = _model()
    batch = [_clip(130, 1), _clip(84, 2), _clip(131, 3), _clip(99, 4)]

    with torch.no_grad():
        batched = batched_logits(m, batch)
        solo = [m(c["X"]) for c in batch]

    assert len(batched) == len(solo)
    for b, s, c in zip(batched, solo, batch):
        assert b.shape == s.shape == (c["X"].shape[0],)
        # Same maths, different kernel shapes -- allow only float noise.
        assert torch.allclose(b, s, atol=1e-5, rtol=1e-4), \
            f"max abs diff {(b - s).abs().max().item():.3e}"


def test_batched_loss_matches_per_clip_loss():
    """The end-to-end loss value is unchanged by batching the forward pass."""
    m = _model()
    batch = [
        _clip(130, 11, label=1, alert=9.0, event=12.5),
        _clip(120, 12, label=0),
        _clip(90, 13, label=1, alert=5.0, event=8.4),
        _clip(131, 14, label=0),
    ]
    common = dict(kappa=3.0, pre_alert_weight=0.5, pos_weight=1.0)

    def loss_from(logits):
        loss, stats = batch_anticipation_loss(
            logits, [c["t"] for c in batch], [c["label"] for c in batch],
            [c["alert_time_s"] for c in batch], [c["event_time_s"] for c in batch],
            **common)
        return float(loss), stats

    with torch.no_grad():
        lb, sb = loss_from(batched_logits(m, batch))
        ls, ss = loss_from([m(c["X"]) for c in batch])

    assert lb == pytest.approx(ls, abs=1e-6)
    assert sb == ss


def test_gradients_match_per_clip_forward():
    """Not just the value -- the gradients must match, or training differs."""
    batch = [
        _clip(130, 21, label=1, alert=9.0, event=12.5),
        _clip(88, 22, label=0),
        _clip(131, 23, label=1, alert=4.0, event=7.2),
    ]
    args = dict(kappa=3.0, pre_alert_weight=0.5, pos_weight=1.0)

    def grads(use_batched):
        m = _model(seed=7)
        logits = (batched_logits(m, batch) if use_batched
                  else [m(c["X"]) for c in batch])
        loss, _ = batch_anticipation_loss(
            logits, [c["t"] for c in batch], [c["label"] for c in batch],
            [c["alert_time_s"] for c in batch], [c["event_time_s"] for c in batch],
            **args)
        m.zero_grad()
        loss.backward()
        return {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}

    gb, gs = grads(True), grads(False)
    assert set(gb) == set(gs) and gb
    for name in gb:
        assert torch.allclose(gb[name], gs[name], atol=1e-5, rtol=1e-3), \
            f"{name}: max abs diff {(gb[name] - gs[name]).abs().max().item():.3e}"


# ---------------------------------------------------------------------------
# the properties that make it safe
# ---------------------------------------------------------------------------

def test_padding_cannot_affect_valid_positions():
    """Appending arbitrary junk after a clip leaves its own outputs untouched.

    This is the causality property the batching relies on. Zeros are what the
    helper actually pads with; random junk is the stronger test.
    """
    m = _model()
    x = _clip(100, 31)["X"]
    junk = torch.randn(40, D, generator=torch.Generator().manual_seed(99)) * 50.0

    with torch.no_grad():
        plain = m(x)
        padded = m(torch.cat([x, junk], dim=0))[:x.shape[0]]

    assert torch.allclose(plain, padded, atol=1e-5, rtol=1e-4)


def test_single_clip_batch_and_uniform_lengths():
    """Degenerate shapes: one clip, and a batch needing no padding at all."""
    m = _model()
    for batch in ([_clip(130, 41)], [_clip(130, 42), _clip(130, 43)]):
        with torch.no_grad():
            for b, c in zip(batched_logits(m, batch), batch):
                assert torch.allclose(b, m(c["X"]), atol=1e-5, rtol=1e-4)


def test_ragged_batch_preserves_each_length():
    """Each returned tensor is its own clip's length, not the padded length."""
    m = _model()
    lens = [37, 130, 84, 131, 12]
    batch = [_clip(T, 50 + i) for i, T in enumerate(lens)]
    with torch.no_grad():
        out = batched_logits(m, batch)
    assert [int(o.shape[0]) for o in out] == lens


# ---------------------------------------------------------------------------
# ablation must reach the validation set as well as the training set
# ---------------------------------------------------------------------------

def test_ablation_zeroes_val_features_too(tmp_path, monkeypatch):
    """--ablate with a separate --val-features dir must ablate BOTH sides.

    It used to zero only the training clips: the model trained on ablated inputs
    and was scored on intact ones, so every ablation result was a train/eval
    shift rather than a measurement of the missing information.
    """
    import numpy as np
    import train_risk_head as trh
    from a3ps.features.extract import frame_feature_dim, frame_feature_names, save_features

    dim = frame_feature_dim()
    names = list(frame_feature_names())
    ego_cols = [i for i, n in enumerate(names) if n.startswith("ego_")]
    assert ego_cols, "test needs the ego_* columns to exist"

    for split in ("tr", "va"):
        for i, label in enumerate((0, 1)):
            feats = {"X": np.ones((20, dim), np.float32), "t": np.arange(20) * 0.1,
                     "n_actors": np.ones(20), "has_ego": True}
            meta = {"clip_id": f"{split}{i}", "label": label,
                    "event_time_s": 1.5 if label else None,
                    "alert_time_s": 0.5 if label else None}
            save_features(str(tmp_path / split / f"{split}{i}.npz"), feats, meta)

    seen = {}

    def fake_train(model, train_clips, val_clips, args, on_improve=None, on_epoch=None):
        seen["train"], seen["val"] = train_clips, val_clips
        return None, []

    monkeypatch.setattr(trh, "train", fake_train)
    monkeypatch.setattr(sys, "argv", [
        "train_risk_head.py", "--features", str(tmp_path / "tr"),
        "--val-features", str(tmp_path / "va"), "--ablate", "ego"])
    trh.main()

    for side in ("train", "val"):
        for c in seen[side]:
            assert (c["X"][:, ego_cols] == 0).all(), f"ego not zeroed on {side}"
            other = [j for j in range(dim) if j not in ego_cols]
            assert (c["X"][:, other] == 1).all(), f"non-ego columns altered on {side}"


# ---------------------------------------------------------------------------
# the held-out split must not be reachable by selection
# ---------------------------------------------------------------------------

def _freeze_with_eval(tmp_path, eval_ids):
    """Write a minimal freeze file naming ``eval_ids`` as the held-out split."""
    import json
    path = tmp_path / "split_freeze.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "counts": {"eval": {"pos": len(eval_ids), "neg": 0}},
        "clips": {str(c): {"split": "eval", "label": 1} for c in eval_ids},
    }), encoding="utf-8")
    return str(path)


def _feature_dir(tmp_path, name, clip_ids):
    import numpy as np
    from a3ps.features.extract import frame_feature_dim, save_features

    dim = frame_feature_dim()
    for i, cid in enumerate(clip_ids):
        feats = {"X": np.ones((20, dim), np.float32), "t": np.arange(20) * 0.1,
                 "n_actors": np.ones(20), "has_ego": True}
        meta = {"clip_id": str(cid), "label": i % 2,
                "event_time_s": 1.5 if i % 2 else None,
                "alert_time_s": 0.5 if i % 2 else None}
        save_features(str(tmp_path / name / f"{cid}.npz"), feats, meta)
    return str(tmp_path / name)


def test_val_features_on_the_frozen_eval_split_is_refused(tmp_path, monkeypatch):
    """--val-features pointing at eval clips must refuse, not just warn.

    The old --allow-leakage guard compared *directories*, so `--features
    train_all --val-features eval` passed it while early-stopping every
    checkpoint on the test set (docs/status/eval_leakage_audit.md). Different
    directories, same leak.
    """
    import train_risk_head as trh

    train_dir = _feature_dir(tmp_path, "tr", [901, 902])
    val_dir = _feature_dir(tmp_path, "held", [11, 12])
    freeze = _freeze_with_eval(tmp_path, [11, 12])

    monkeypatch.setattr(sys, "argv", [
        "train_risk_head.py", "--features", train_dir,
        "--val-features", val_dir, "--split-freeze", freeze])
    with pytest.raises(SystemExit) as e:
        trh.main()
    assert "eval" in str(e.value) and "train_val" in str(e.value)


def test_final_eval_report_flag_permits_the_single_final_read(tmp_path, monkeypatch):
    """The refusal is overridable, but only by saying so explicitly."""
    import train_risk_head as trh

    train_dir = _feature_dir(tmp_path, "tr2", [903, 904])
    val_dir = _feature_dir(tmp_path, "held2", [21, 22])
    freeze = _freeze_with_eval(tmp_path, [21, 22])

    reached = {}

    def fake_train(model, train_clips, val_clips, args, on_improve=None, on_epoch=None):
        reached["yes"] = True
        return None, []

    monkeypatch.setattr(trh, "train", fake_train)
    monkeypatch.setattr(sys, "argv", [
        "train_risk_head.py", "--features", train_dir,
        "--val-features", val_dir, "--split-freeze", freeze,
        "--final-eval-report"])
    trh.main()
    assert reached.get("yes")


def test_training_on_the_frozen_eval_split_is_refused(tmp_path, monkeypatch):
    """--features is guarded too: training on eval is worse than validating on it."""
    import train_risk_head as trh

    train_dir = _feature_dir(tmp_path, "held3", [31, 32])
    freeze = _freeze_with_eval(tmp_path, [31, 32])

    monkeypatch.setattr(sys, "argv", [
        "train_risk_head.py", "--features", train_dir, "--split-freeze", freeze])
    with pytest.raises(SystemExit) as e:
        trh.main()
    assert "--features" in str(e.value)


def test_sweep_operating_point_has_no_eval_default(tmp_path, monkeypatch):
    """--features is required: the sweep can no longer land on eval by default."""
    import sweep_operating_point as sop

    monkeypatch.setattr(sys, "argv", ["sweep_operating_point.py"])
    with pytest.raises(SystemExit):
        sop.main()


def test_sweep_operating_point_refuses_eval_clips(tmp_path, monkeypatch):
    """Choosing an operating point on the held-out split must refuse."""
    import sweep_operating_point as sop

    feats = _feature_dir(tmp_path, "held4", [41, 42])
    freeze = _freeze_with_eval(tmp_path, [41, 42])

    monkeypatch.setattr(sys, "argv", [
        "sweep_operating_point.py", "--features", feats,
        "--split-freeze", freeze])
    with pytest.raises(SystemExit) as e:
        sop.main()
    assert "train_val" in str(e.value)


# ---------------------------------------------------------------------------
# per-clip dump must reproduce evaluate()'s own aggregates
# ---------------------------------------------------------------------------

def test_dump_rows_reproduces_aggregates():
    """evaluate(dump_rows=True) must not be a second, divergent scoring path.

    Built for docs/status/early_firing_diagnosis.md: the per-clip dump is only
    trustworthy if it is a readout of the exact numbers evaluate() already
    returns, not a parallel computation that could silently disagree.
    """
    import train_risk_head as trh

    m = _model(seed=7)
    clips = [
        dict(_clip(60, 1, label=1, alert=1.5, event=4.0), clip_id="p_useful"),
        dict(_clip(60, 2, label=1, alert=2.5, event=4.5), clip_id="p_early"),
        dict(_clip(60, 3, label=1, alert=None, event=None), clip_id="p_untimed"),
        dict(_clip(60, 4, label=0), clip_id="n_a"),
        dict(_clip(60, 5, label=0), clip_id="n_b"),
    ]
    threshold, confirm = 0.5, 3

    summary = trh.evaluate(m, clips, threshold, confirm=confirm)
    dumped = trh.evaluate(m, clips, threshold, confirm=confirm, dump_rows=True)

    # Same aggregate numbers with the flag on or off -- dumping never perturbs
    # scoring.
    for k in ("n_clips", "n_timed", "useful_warning_rate", "n_useful",
              "n_too_early", "false_alarm_rate", "mean_lead_s", "mean_AP"):
        a, b = summary[k], dumped[k]
        if a == a:  # not NaN
            assert a == pytest.approx(b, nan_ok=True)
        else:
            assert b != b

    rows = dumped["rows"]
    assert {r["clip_id"] for r in rows} == {c["clip_id"] for c in clips}

    # n_timed / n_useful / n_too_early / false_alarm_rate recomputed from the
    # dumped verdicts must match evaluate()'s own counters exactly.
    n_timed = sum(1 for r in rows if r["label"] == 1 and r["verdict"] != "untimed")
    n_useful = sum(1 for r in rows if r["verdict"] == "useful")
    n_early = sum(1 for r in rows if r["verdict"] == "too_early")
    n_neg = sum(1 for r in rows if r["label"] == 0)
    n_fa = sum(1 for r in rows if r["verdict"] == "false_alarm")
    assert n_timed == dumped["n_timed"]
    assert n_useful == dumped["n_useful"] == summary["n_useful"]
    assert n_early == dumped["n_too_early"] == summary["n_too_early"]
    fa_rate = n_fa / n_neg if n_neg else float("nan")
    assert fa_rate == pytest.approx(dumped["false_alarm_rate"])

    # mean_lead_s must equal the mean of dumped lead_s over useful+too_early
    # verdicts, exactly as evaluate()'s own `leads` list is built.
    leads = [r["lead_s"] for r in rows if r["verdict"] in ("useful", "too_early")]
    if leads:
        assert sum(leads) / len(leads) == pytest.approx(dumped["mean_lead_s"])
    else:
        assert dumped["mean_lead_s"] != dumped["mean_lead_s"]

    # The untimed positive must be excluded from n_timed, exactly like
    # evaluate()'s own `continue` on `te is None or ta is None`.
    untimed = [r for r in rows if r["clip_id"] == "p_untimed"][0]
    assert untimed["verdict"] == "untimed"
    assert untimed["clip_id"] not in {
        r["clip_id"] for r in rows if r["verdict"] in ("useful", "too_early", "missed", "late")}
