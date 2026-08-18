"""Tests for the alert-keyed anticipation loss and the temporal risk head.

The tests that matter here are the two that encode the project's actual failure
modes:

* ``test_standing_alarm_...`` -- the loss must prefer a well-timed warning over
  one that has been firing since frame 0. The current threshold system fails
  exactly this way (median 14.4 s before ``time_of_alert``, useful rate 3/60), and
  with ``pre_alert_weight=0`` the loss cannot see the difference at all.
* ``test_head_is_causal`` -- a non-causal head can observe the collision and then
  "anticipate" it, which inflates every metric simultaneously.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a3ps.risk.anticipation_loss import (  # noqa: E402
    DEFAULT_KAPPA,
    DEFAULT_PRE_ALERT_WEIGHT,
    alert_weights,
    anticipation_loss,
    batch_anticipation_loss,
    expected_lead_time,
    first_alert_time,
)
from a3ps.risk.temporal import RiskGRU, assert_causal, count_parameters  # noqa: E402

TA, TE = 15.5, 19.0


def _t(step=0.5, end=22.0):
    return torch.arange(0.0, end, step)


def _logits(t, fire_from=None, value=4.0):
    out = torch.full((len(t),), -value)
    if fire_from is not None:
        out[t >= fire_from] = value
    return out


# ---------------------------------------------------------------------------
# the weight curve
# ---------------------------------------------------------------------------

def test_weights_are_zero_outside_the_actionable_window():
    t = _t()
    w = alert_weights(t, TA, TE)
    assert torch.all(w[t < TA] == 0.0), "no weight before the actionable moment"
    assert torch.all(w[t > TE] == 0.0), "no weight after impact"
    assert torch.all(w[(t >= TA) & (t <= TE)] > 0.0)


def test_weight_rises_monotonically_toward_the_event():
    t = _t(step=0.25)
    w = alert_weights(t, TA, TE)
    inside = w[(t >= TA) & (t <= TE)]
    assert torch.all(inside[1:] >= inside[:-1])


def test_weight_endpoints_match_the_documented_formula():
    import math
    t = torch.tensor([TA, TE])
    w = alert_weights(t, TA, TE, kappa=3.0)
    assert float(w[1]) == pytest.approx(1.0)
    assert float(w[0]) == pytest.approx(math.exp(-3.0), rel=1e-5)


def test_degenerate_window_does_not_divide_by_zero():
    t = torch.tensor([18.9, 19.0, 19.1])
    w = alert_weights(t, 19.0, 19.0)
    assert torch.all(torch.isfinite(w))
    assert float(w.sum()) > 0.0


def test_expected_lead_time_shrinks_as_kappa_grows():
    lo = expected_lead_time(TA, TE, kappa=0.5)
    hi = expected_lead_time(TA, TE, kappa=6.0)
    assert lo > hi, "a sharper kappa asks for a later warning"
    assert expected_lead_time(TA, TE, kappa=0.0) == pytest.approx((TE - TA) / 2.0)
    assert expected_lead_time(19.0, 19.0) == 0.0


# ---------------------------------------------------------------------------
# per-clip loss
# ---------------------------------------------------------------------------

def test_negative_clip_is_penalised_for_any_firing():
    t = _t()
    quiet = anticipation_loss(_logits(t), t, label=0)
    loud = anticipation_loss(_logits(t, fire_from=0.0), t, label=0)
    assert float(loud) > float(quiet)


def test_late_warning_costs_more_than_a_timely_one():
    t = _t()
    timely = anticipation_loss(_logits(t, fire_from=TA), t, 1, TA, TE)
    late = anticipation_loss(_logits(t, fire_from=TE + 0.5), t, 1, TA, TE)
    assert float(late) > float(timely)


def test_standing_alarm_is_penalised_only_when_pre_alert_weight_is_on():
    """The core design decision, pinned: at 0.0 the loss is blind to it."""
    t = _t()
    standing = _logits(t, fire_from=0.0)
    timed = _logits(t, fire_from=TA)

    off_s = float(anticipation_loss(standing, t, 1, TA, TE, pre_alert_weight=0.0))
    off_t = float(anticipation_loss(timed, t, 1, TA, TE, pre_alert_weight=0.0))
    assert off_s == pytest.approx(off_t), (
        "with pre_alert_weight=0 the loss cannot distinguish a standing alarm "
        "from a well-timed warning -- this is why the default is non-zero")

    on_s = float(anticipation_loss(standing, t, 1, TA, TE,
                                   pre_alert_weight=DEFAULT_PRE_ALERT_WEIGHT))
    on_t = float(anticipation_loss(timed, t, 1, TA, TE,
                                   pre_alert_weight=DEFAULT_PRE_ALERT_WEIGHT))
    assert on_s > on_t


def test_default_pre_alert_weight_is_non_zero():
    assert DEFAULT_PRE_ALERT_WEIGHT > 0.0


def test_positive_without_times_falls_back_to_uniform_bce():
    t = _t()
    loss = anticipation_loss(_logits(t, fire_from=0.0), t, 1, None, None)
    assert float(loss) < 0.05, "firing everywhere satisfies a uniform target"


def test_window_outside_the_available_frames_contributes_nothing():
    """A truncated extraction window must not produce a garbage gradient."""
    t = torch.arange(0.0, 5.0, 0.5)         # clip ends long before TA
    loss = anticipation_loss(_logits(t), t, 1, TA, TE)
    assert float(loss) == 0.0


def test_loss_is_length_normalised():
    """A long clip must not dominate a short one just by having more frames."""
    short = anticipation_loss(_logits(_t(step=0.5)), _t(step=0.5), 0)
    long = anticipation_loss(_logits(_t(step=0.05)), _t(step=0.05), 0)
    assert float(short) == pytest.approx(float(long), rel=1e-4)


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="1-D and equal length"):
        anticipation_loss(torch.zeros(5), torch.zeros(6), 1, TA, TE)


def test_gradients_flow_to_the_frames_inside_the_window():
    t = _t()
    logits = torch.zeros(len(t), requires_grad=True)
    anticipation_loss(logits, t, 1, TA, TE, pre_alert_weight=0.0).backward()
    g = logits.grad
    assert torch.all(g[(t >= TA) & (t <= TE)] != 0.0)
    assert torch.all(g[t > TE] == 0.0), "post-event frames must not train the head"


# ---------------------------------------------------------------------------
# batch loss
# ---------------------------------------------------------------------------

def test_batch_stats_count_clip_kinds():
    t = _t()
    loss, stats = batch_anticipation_loss(
        [_logits(t, TA), _logits(t), _logits(t, TA)],
        [t, t, t],
        [1, 0, 1],
        [TA, None, None],
        [TE, None, None])
    assert stats["n_pos"] == 2 and stats["n_neg"] == 1
    assert stats["n_untimed"] == 1
    assert torch.isfinite(loss)


def test_batch_counts_clips_whose_window_missed_the_frames():
    short = torch.arange(0.0, 5.0, 0.5)
    _, stats = batch_anticipation_loss(
        [_logits(short)], [short], [1], [TA], [TE])
    assert stats["n_skipped"] == 1


def test_pos_weight_scales_the_positive_contribution():
    t = _t()
    args = ([_logits(t), _logits(t)], [t, t], [1, 0], [TA, None], [TE, None])
    low, _ = batch_anticipation_loss(*args, pos_weight=1.0)
    high, _ = batch_anticipation_loss(*args, pos_weight=5.0)
    assert float(high) > float(low)


def test_empty_and_ragged_batches_are_rejected():
    with pytest.raises(ValueError, match="empty batch"):
        batch_anticipation_loss([], [], [], [], [])
    t = _t()
    with pytest.raises(ValueError, match="same length"):
        batch_anticipation_loss([_logits(t)], [t], [1, 0], [TA], [TE])


# ---------------------------------------------------------------------------
# first_alert_time (the inference-side bridge to the eval script)
# ---------------------------------------------------------------------------

def test_first_alert_credits_the_start_of_a_confirmed_run():
    t = torch.arange(0.0, 2.0, 0.1)
    probs = torch.zeros(len(t))
    probs[t >= 1.0] = 0.9
    assert first_alert_time(probs, t, 0.5, frames_to_confirm=3) == pytest.approx(1.0)


def test_a_single_frame_spike_does_not_fire_with_debounce():
    t = torch.arange(0.0, 2.0, 0.1)
    probs = torch.zeros(len(t))
    probs[5] = 0.99
    assert first_alert_time(probs, t, 0.5, frames_to_confirm=3) is None
    assert first_alert_time(probs, t, 0.5, frames_to_confirm=1) == pytest.approx(0.5)


def test_never_crossing_returns_none():
    t = torch.arange(0.0, 2.0, 0.1)
    assert first_alert_time(torch.zeros(len(t)), t, 0.5) is None


# ---------------------------------------------------------------------------
# the head
# ---------------------------------------------------------------------------

def test_head_is_causal():
    m = RiskGRU(16)
    assert_causal(m, 16)


def test_bidirectional_head_is_rejected_by_the_causality_guard():
    import torch.nn as nn

    class Bidi(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.gru = nn.GRU(d, 8, 1, batch_first=True, bidirectional=True)
            self.head = nn.Linear(16, 1)

        def forward(self, x):
            return self.head(self.gru(x)[0]).squeeze(-1)

    with pytest.raises(AssertionError, match="not causal"):
        assert_causal(Bidi(16), 16)


def test_head_shapes_batched_and_unbatched():
    m = RiskGRU(12)
    assert tuple(m(torch.randn(4, 20, 12)).shape) == (4, 20)
    assert tuple(m(torch.randn(20, 12)).shape) == (20,)


def test_head_rejects_wrong_feature_width():
    m = RiskGRU(12)
    with pytest.raises(ValueError, match="feature width"):
        m(torch.randn(4, 20, 7))


def test_probs_are_in_range_and_leave_training_mode_untouched():
    m = RiskGRU(12)
    m.train()
    p = m.probs(torch.randn(20, 12))
    assert torch.all((p >= 0.0) & (p <= 1.0))
    assert m.training, "probs() must restore the previous mode"


def test_head_stays_small():
    """Capacity guard: 1,500 clips cannot support a large model."""
    assert count_parameters(RiskGRU(112)) < 100_000


def test_checkpoint_round_trip_preserves_outputs_and_provenance(tmp_path):
    m = RiskGRU(12, hidden=16)
    x = torch.randn(9, 12)
    before = m.probs(x)
    path = str(tmp_path / "head.pt")
    m.save(path, extra={"has_ego": False, "feature_dim": 12})
    loaded, extra = RiskGRU.load(path)
    assert torch.allclose(before, loaded.probs(x), atol=1e-6)
    assert extra["has_ego"] is False
    assert extra["feature_dim"] == 12


def test_head_can_learn_a_trivial_timed_pattern():
    """End-to-end: loss + head actually optimise toward a late-firing target."""
    torch.manual_seed(0)
    D, T = 4, 40
    t = torch.arange(T, dtype=torch.float32) * 0.25      # 0..10 s
    ta, te = 6.0, 9.0
    x = torch.zeros(T, D)
    x[t >= ta, 0] = 1.0                                  # the only informative cue
    model = RiskGRU(D, hidden=16, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(150):
        loss = anticipation_loss(model(x), t, 1, ta, te,
                                 pre_alert_weight=DEFAULT_PRE_ALERT_WEIGHT)
        opt.zero_grad()
        loss.backward()
        opt.step()
    probs = model.probs(x)
    assert float(probs[t >= te - 0.3].mean()) > 0.5, "should fire near the event"
    assert float(probs[t < ta].mean()) < 0.5, "should stay quiet before the alert"
