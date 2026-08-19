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
