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
        return None, [], False, False

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
        return None, [], False, False

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

    # v2/prematurity aggregates must also be exact readouts of the dump.
    n_useful_v2 = sum(1 for r in rows if r["verdict_v2"] == "useful")
    assert n_useful_v2 == dumped["n_useful_v2"]
    assert (n_useful_v2 / n_timed if n_timed else float("nan")) == pytest.approx(
        dumped["useful_warning_rate_v2"], nan_ok=True)
    n_premature = sum(1 for r in rows if r.get("premature") is True)
    assert (n_premature / n_timed if n_timed else float("nan")) == pytest.approx(
        dumped["frac_premature"], nan_ok=True)
    n_gross = sum(1 for r in rows if r.get("gross_premature") is True)
    assert n_gross == dumped["n_gross_premature"]
    # v2 can only ever credit a superset of legacy's useful clips.
    assert n_useful_v2 >= n_useful


# ---------------------------------------------------------------------------
# corrected (episode-based) useful-warning definition
# -- docs/design/useful_warning_definition.md
# ---------------------------------------------------------------------------

def test_legacy_numbers_reproduce_committed_checkpoint_bit_for_bit():
    """The legacy definition must be untouched by the v2/prematurity work.

    Locks in the exact numbers already reported in
    docs/status/repartition_results.md section 4 for seed 1234 on
    train_val_v2 at threshold 0.5 / confirm 3, scored from the checkpoint
    actually committed to the repo -- not a synthetic clip, so this catches a
    regression the synthetic tests below could miss.
    """
    import os
    import train_risk_head as trh
    from a3ps.risk.temporal import RiskGRU

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ckpt = os.path.join(root, "notebooks", "models", "risk_gru_k1p0_v2_s1234.pt")
    feats = os.path.join(root, "data", "features", "train_val_v2")
    if not (os.path.isfile(ckpt) and os.path.isdir(feats)):
        pytest.skip("checkpoint or train_val_v2 features not present")

    model, extra = RiskGRU.load(ckpt)
    clips = trh.load_clips(feats)
    v = trh.evaluate(model, clips, 0.5, confirm=3)

    assert v["n_timed"] == 103
    assert v["n_useful"] == 34
    assert v["n_too_early"] == 63
    assert v["useful_warning_rate"] == pytest.approx(0.3300970873786408)
    assert v["false_alarm_rate"] == pytest.approx(0.6666666666666666)
    assert v["mean_lead_s"] == pytest.approx(2.6963195892805905)
    assert v["mean_AP"] == pytest.approx(0.6295955829547311)


def test_v2_credits_an_alert_that_spans_the_whole_window():
    """An alert confirmed before alert_t and never dropped, spanning the whole
    window, must score useful under v2 and too_early under legacy -- this is
    exactly the clip-1013 case documented in early_firing_diagnosis.md."""
    import torch
    import train_risk_head as trh

    t = torch.arange(60, dtype=torch.float32) * 0.1   # 0.0 .. 5.9s
    alert_t, event_t = 3.0, 4.0
    # Hot (logit +10 -> prob ~1) for the WHOLE window and slightly before it,
    # confirmed well ahead of alert_t; quiet elsewhere.
    logits = torch.full((60,), -10.0)
    logits[20:45] = 10.0   # t=2.0 .. 4.4, covers [alert_t=3.0, event_t=4.0]

    class _M:
        def eval(self): return self
        def __call__(self, X): return logits

    clip = {"clip_id": "spanning", "X": torch.zeros(60, D), "t": t, "label": 1,
           "alert_time_s": alert_t, "event_time_s": event_t}
    v = trh.evaluate(_M(), [clip], threshold=0.5, confirm=5, dump_rows=True)
    row = v["rows"][0]

    assert row["verdict"] == "too_early"       # legacy: first crossing < alert_t
    assert row["verdict_v2"] == "useful"       # v2: the window IS covered
    assert row["premature"] is True            # earliest episode is before alert_t
    assert v["n_useful"] == 0
    assert v["n_useful_v2"] == 1


def test_v2_does_not_credit_an_alert_entirely_before_the_window():
    """A spike that decays back down before alert_t must stay not-useful under
    BOTH definitions (legacy calls this too_early rather than missed, since it
    DID fire -- just not inside the window), and must be counted premature."""
    import torch
    import train_risk_head as trh

    t = torch.arange(60, dtype=torch.float32) * 0.1
    alert_t, event_t = 3.0, 4.0
    logits = torch.full((60,), -10.0)
    logits[5:12] = 10.0    # t=0.5..1.1, well before alert_t=3.0, and it drops

    class _M:
        def eval(self): return self
        def __call__(self, X): return logits

    clip = {"clip_id": "early_spike", "X": torch.zeros(60, D), "t": t, "label": 1,
           "alert_time_s": alert_t, "event_time_s": event_t}
    v = trh.evaluate(_M(), [clip], threshold=0.5, confirm=5, dump_rows=True)
    row = v["rows"][0]

    assert row["verdict"] == "too_early"     # legacy: fired, but before alert_t
    assert row["verdict_v2"] == "missed"     # v2: the window itself was never covered
    assert row["premature"] is True
    assert row["gross_premature"] is True    # >5s early AND in first 20% of the clip
    assert v["n_useful"] == 0
    assert v["n_useful_v2"] == 0


def test_v2_does_not_credit_an_alert_entirely_after_the_event():
    """A confirmed alert firing only after event_t must not be useful under
    either definition."""
    import torch
    import train_risk_head as trh

    t = torch.arange(60, dtype=torch.float32) * 0.1
    alert_t, event_t = 1.0, 2.0
    logits = torch.full((60,), -10.0)
    logits[40:50] = 10.0   # t=4.0..4.9, well after event_t=2.0

    class _M:
        def eval(self): return self
        def __call__(self, X): return logits

    clip = {"clip_id": "late_fire", "X": torch.zeros(60, D), "t": t, "label": 1,
           "alert_time_s": alert_t, "event_time_s": event_t}
    v = trh.evaluate(_M(), [clip], threshold=0.5, confirm=5, dump_rows=True)
    row = v["rows"][0]

    assert row["verdict"] == "late"
    assert row["verdict_v2"] == "missed"
    assert row["premature"] is False
    assert v["n_useful"] == 0
    assert v["n_useful_v2"] == 0


# ---------------------------------------------------------------------------
# clips 544 and 1013 -- permanent regressions for the v2-is-a-superset-of-
# legacy guarantee. Both were real failures found on real checkpoints:
# clip 544 was the case that exposed the debounce-end-vs-run-start bug
# (v2 scored BELOW legacy before the fix); clip 1013 is the original
# clip-1013 case documented in early_firing_diagnosis.md that motivated the
# whole corrected definition. Neither must ever regress silently.
# ---------------------------------------------------------------------------

def _load_real_checkpoint_and_clip(seed_ckpt, clip_id):
    import os
    import train_risk_head as trh
    from a3ps.risk.temporal import RiskGRU

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ckpt = os.path.join(root, "notebooks", "models", seed_ckpt)
    feats = os.path.join(root, "data", "features", "train_val_v2")
    if not (os.path.isfile(ckpt) and os.path.isdir(feats)):
        pytest.skip("checkpoint or train_val_v2 features not present")
    model, extra = RiskGRU.load(ckpt)
    clips = trh.load_clips(feats)
    by_id = {c["clip_id"]: c for c in clips}
    if clip_id not in by_id:
        pytest.skip(f"clip {clip_id} not present in train_val_v2")
    return model, by_id[clip_id]


def test_clip_544_v2_is_not_below_legacy():
    """Regression for the debounce-end-vs-run-start bug.

    Before the fix, find_alert_episodes() credited an episode at the frame its
    confirm-length debounce COMPLETED rather than the run's own first frame.
    On this exact clip (seed 1236, thr 0.80/confirm 8) that pushed the
    confirmation timestamp from 20.803s (inside [alert_t=20.354, event_t=
    21.455], legacy verdict useful) to 21.505s (just past event_t), so v2
    scored the clip missed while legacy scored it useful -- v2 BELOW legacy,
    exactly the regression this test exists to catch if it ever reappears.
    """
    import train_risk_head as trh

    model, clip = _load_real_checkpoint_and_clip(
        "risk_gru_k1p0_v2_s1236.pt", "544")
    v = trh.evaluate(model, [clip], 0.80, confirm=8, dump_rows=True)
    row = v["rows"][0]

    assert row["verdict"] == "useful"
    assert row["verdict_v2"] == "useful"     # must NOT be "missed"
    assert v["n_useful_v2"] >= v["n_useful"]


def test_clip_1013_v2_credits_the_sustained_window_coverage():
    """Regression for the case that motivated the corrected definition.

    Clip 1013 (seed 1234, thr 0.60/confirm 5): one continuous elevated run
    starting well before alert_t=18.833 and lasting through event_t=19.300,
    so legacy's first-crossing rule scores it too_early -- identical to never
    warning at all -- while v2 must credit the sustained window coverage as
    useful. See docs/status/early_firing_diagnosis.md section 3 and
    docs/design/useful_warning_definition.md section 1.
    """
    import train_risk_head as trh

    model, clip = _load_real_checkpoint_and_clip(
        "risk_gru_k1p0_v2_s1234.pt", "1013")
    v = trh.evaluate(model, [clip], 0.60, confirm=5, dump_rows=True)
    row = v["rows"][0]

    assert row["verdict"] == "too_early"     # legacy: penalised for firing early
    assert row["verdict_v2"] == "useful"     # v2: window coverage credited
    assert row["premature"] is True
    assert v["n_useful"] == 0
    assert v["n_useful_v2"] == 1


def test_v2_useful_set_is_a_superset_of_legacy_across_the_full_grid():
    """The property itself, checked directly rather than inferred from two
    clips: for every (threshold, confirm) cell and every v2 checkpoint,
    n_useful_v2 must never fall below n_useful. This is what actually failed
    (6 of 45 cells) before the run-start fix; this test is the permanent
    guard against that regression reappearing under a future change to
    find_alert_episodes() or evaluate()."""
    import os
    import train_risk_head as trh
    from a3ps.risk.temporal import RiskGRU

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    feats = os.path.join(root, "data", "features", "train_val_v2")
    if not os.path.isdir(feats):
        pytest.skip("train_val_v2 features not present")
    clips = trh.load_clips(feats)

    checkpoints = [f"risk_gru_k1p0_v2_s{s}.pt" for s in (1234, 1235, 1236)]
    checkpoints = [c for c in checkpoints
                  if os.path.isfile(os.path.join(root, "notebooks", "models", c))]
    if not checkpoints:
        pytest.skip("no v2 checkpoints present")

    violations = []
    for ckpt_name in checkpoints:
        model, extra = RiskGRU.load(os.path.join(root, "notebooks", "models", ckpt_name))
        for confirm in (3, 5, 8):
            for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
                v = trh.evaluate(model, clips, threshold, confirm=confirm)
                if v["n_useful_v2"] < v["n_useful"]:
                    violations.append((ckpt_name, threshold, confirm,
                                       v["n_useful"], v["n_useful_v2"]))

    assert not violations, f"v2 scored below legacy on: {violations}"


# ---------------------------------------------------------------------------
# checkpoint-selection fallback: near-chance picks must not be kept silently
# -- docs/status/selection_fix_and_tradeoff.md
# ---------------------------------------------------------------------------

def test_train_selection_fallback_picks_best_ap_not_near_chance_epoch(monkeypatch):
    """When no epoch ever meets fa_target AND the useful-first pick is near
    chance, train() must fall back to the best-mean-AP epoch and report
    fallback_used=True -- this is the seed-1236 bug (epoch 1, mean AP 0.501,
    chance, beat epoch 6, mean AP 0.648, real separation, purely because
    epoch 1 had a marginally better useful/-FA pair) reproduced deterministically.
    """
    import train_risk_head as trh

    # Scripted per-epoch validation: epoch 1 has the best useful/-FA pair (the
    # OLD rule keeps it) but chance-level AP; epoch 3 has real separation but
    # loses the primary tiebreak. None meet fa_target=0.20.
    scripted = [
        (0.50, 0.40, 0.51),   # epoch 1: wins on useful/-FA, chance-level AP
        (0.30, 0.60, 0.55),   # epoch 2: loses both
        (0.40, 0.45, 0.75),   # epoch 3: real separation, loses on useful/-FA
    ]

    def fake_evaluate(model, clips, threshold, confirm=3, dump_rows=False):
        useful, fa, ap = scripted[min(fake_evaluate.calls, len(scripted) - 1)]
        fake_evaluate.calls += 1
        return {"n_clips": 20, "n_timed": 10, "useful_warning_rate": useful,
                "n_useful": int(round(useful * 10)), "n_too_early": 0,
                "false_alarm_rate": fa, "mean_lead_s": 1.0,
                "AP": {0.5: ap, 1.0: ap, 1.5: ap}, "mean_AP": ap}
    fake_evaluate.calls = 0

    monkeypatch.setattr(trh, "evaluate", fake_evaluate)

    m = _model(seed=1)
    train_clips = [_clip(40, 1, label=1, alert=1.0, event=3.0),
                  _clip(40, 2, label=0)]
    val_clips = [_clip(20, i, label=1 if i % 2 else 0) for i in range(20)]

    class Args:
        lr = 1e-3
        weight_decay = 1e-4
        grad_clip = 5.0
        seed = 1
        batch_size = 2
        kappa = 3.0
        pre_alert_weight = 0.5
        pos_weight = 1.0
        threshold = 0.5
        confirm = 3
        fa_target = 0.20
        epochs = 3
        patience = 0

    best, history, fallback_used, degenerate = trh.train(m, train_clips, val_clips, Args())

    assert fallback_used is True
    assert best["epoch"] == 3
    assert best["val"]["mean_AP"] == pytest.approx(0.75)
    assert best["rank"][0] == 0        # fa_target was never met this run
    assert degenerate is False         # the fallback rescued it -- 0.75 is not near chance


def test_train_no_fallback_when_useful_first_pick_is_not_near_chance(monkeypatch):
    """A run that never meets fa_target but whose useful-first pick has real
    separation (like the currently-committed seeds 1234/1235) must be
    returned UNCHANGED -- the fallback only fires for near-chance picks."""
    import train_risk_head as trh

    scripted = [
        (0.34, 0.67, 0.63),   # epoch 1: wins useful/-FA, AND has real AP
        (0.29, 0.63, 0.59),
    ]

    def fake_evaluate(model, clips, threshold, confirm=3, dump_rows=False):
        useful, fa, ap = scripted[min(fake_evaluate.calls, len(scripted) - 1)]
        fake_evaluate.calls += 1
        return {"n_clips": 20, "n_timed": 10, "useful_warning_rate": useful,
                "n_useful": int(round(useful * 10)), "n_too_early": 0,
                "false_alarm_rate": fa, "mean_lead_s": 1.0,
                "AP": {0.5: ap, 1.0: ap, 1.5: ap}, "mean_AP": ap}
    fake_evaluate.calls = 0

    monkeypatch.setattr(trh, "evaluate", fake_evaluate)

    m = _model(seed=1)
    train_clips = [_clip(40, 1, label=1, alert=1.0, event=3.0),
                  _clip(40, 2, label=0)]
    val_clips = [_clip(20, i, label=1 if i % 2 else 0) for i in range(20)]

    class Args:
        lr = 1e-3
        weight_decay = 1e-4
        grad_clip = 5.0
        seed = 1
        batch_size = 2
        kappa = 3.0
        pre_alert_weight = 0.5
        pos_weight = 1.0
        threshold = 0.5
        confirm = 3
        fa_target = 0.20
        epochs = 2
        patience = 0

    best, history, fallback_used, degenerate = trh.train(m, train_clips, val_clips, Args())

    assert fallback_used is False
    assert best["epoch"] == 1
    assert best["val"]["mean_AP"] == pytest.approx(0.63)
    assert degenerate is False


def test_train_flags_degenerate_even_when_fa_target_met(monkeypatch):
    """The seed-1238 gap: an epoch can meet fa_target on the PRIMARY rule and
    still be chance-level. The fallback above never fires for it (it only
    fires when NO epoch meets fa_target), so the unconditional post-selection
    check is the only thing that can catch this -- and it must."""
    import train_risk_head as trh

    # epoch 1 meets fa_target (FA 0.10 <= 0.20) via the PRIMARY rule, so
    # rank[0] == 1 and the near-chance fallback never activates for it. Its
    # mean AP is 0.50 -- chance on this ~50/50 val set.
    scripted = [
        (0.30, 0.10, 0.50),   # epoch 1: FA-compliant, chance-level AP
        (0.35, 0.55, 0.65),   # epoch 2: not FA-compliant, real separation
    ]

    def fake_evaluate(model, clips, threshold, confirm=3, dump_rows=False):
        useful, fa, ap = scripted[min(fake_evaluate.calls, len(scripted) - 1)]
        fake_evaluate.calls += 1
        return {"n_clips": 20, "n_timed": 10, "useful_warning_rate": useful,
                "n_useful": int(round(useful * 10)), "n_too_early": 0,
                "false_alarm_rate": fa, "mean_lead_s": 1.0,
                "AP": {0.5: ap, 1.0: ap, 1.5: ap}, "mean_AP": ap}
    fake_evaluate.calls = 0

    monkeypatch.setattr(trh, "evaluate", fake_evaluate)

    m = _model(seed=1)
    train_clips = [_clip(40, 1, label=1, alert=1.0, event=3.0),
                  _clip(40, 2, label=0)]
    val_clips = [_clip(20, i, label=1 if i % 2 else 0) for i in range(20)]

    class Args:
        lr = 1e-3
        weight_decay = 1e-4
        grad_clip = 5.0
        seed = 1
        batch_size = 2
        kappa = 3.0
        pre_alert_weight = 0.5
        pos_weight = 1.0
        threshold = 0.5
        confirm = 3
        fa_target = 0.20
        epochs = 2
        patience = 0

    best, history, fallback_used, degenerate = trh.train(m, train_clips, val_clips, Args())

    assert best["rank"][0] == 1     # fa_target WAS met -- the primary rule ran normally
    assert fallback_used is False   # so the near-chance fallback never activates
    assert best["epoch"] == 1       # epoch 1 wins the primary rule (FA-compliant beats not)
    assert best["val"]["mean_AP"] == pytest.approx(0.50)
    assert degenerate is True       # but it must still be flagged as chance-level
