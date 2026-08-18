"""Anticipation loss keyed to Nexar's ``time_of_alert``.

The shape of the problem
------------------------
A per-frame risk head emits ``p_t`` for every frame of a clip, but the label is
one event time per clip. A plain per-frame BCE against "1 everywhere in a
positive clip" is wrong twice over: it demands a high score 20 s before anything
is visible, and it gives no more weight to the frame just before impact than to
the first frame of the clip.

The accident-anticipation literature fixes the second half with an exponentially
weighted loss (Chan et al.'s DSA and successors), weighting a positive frame by
``exp(-(t_event - t))`` so a late miss is punished harder than an early one. What
those formulations do NOT have is a ground-truth "when did this become visible"
annotation, so they ramp all the way back to frame 0.

Nexar gives us one. ``time_of_alert`` is the annotated earliest actionable
moment, and it sits a measured 2.97-4.47 s before ``time_of_event`` (mean 3.49,
sd 0.40 over the 65 local positives). So the ramp can start where the dataset
says the hazard became actionable, instead of where our window-picking guessed.

The weighting
-------------
For a positive clip with alert time ``ta`` and event time ``te``, define
normalised progress through the actionable window::

    u(t) = (t - ta) / (te - ta)        clamped to [0, 1]

and weight the positive-frame BCE by::

    w(t) = exp(kappa * (u(t) - 1))     for ta <= t <= te

so ``w(te) = 1`` and ``w(ta) = exp(-kappa)`` (0.05 at the default kappa=3.0).
Frames after ``te`` are masked out entirely -- what the model says after the
collision is not anticipation. Frames before ``ta`` are handled by
``pre_alert_weight`` (see below).

Why exp-in-u rather than exp-in-seconds: the lead time varies by 1.5 s across
clips, so an absolute-time decay would implicitly weight short-lead clips
differently from long-lead ones. Normalising by each clip's own window makes the
curve mean the same thing everywhere.

Negatives are penalised on every frame for any positive output, normalised per
clip so a 60 s negative does not outweigh a 20 s one.

``pre_alert_weight`` -- why it is ON by default
-----------------------------------------------
Frames in a positive clip *before* ``ta`` are, by the dataset's own annotation,
ordinary driving, so firing there is a false alarm. It is tempting to leave that
term off and let the negative clips carry the "don't fire" signal. Measured, that
does not work::

    pre_alert_weight   standing-alarm loss   well-timed loss   separation
    0.00               0.0181                0.0181           +0.0000
    0.25               0.5204                0.0204           +0.5000
    0.50               1.0227                0.0227           +1.0000
    1.00               2.0272                0.0272           +2.0000

(batch of one positive + one negative; the "standing alarm" fires from frame 0 of
the positive and stays silent on the negative, i.e. it is already a perfect
clip-level classifier.)

At 0.0 the separation is *exactly* zero: the objective cannot distinguish a
warning delivered at the actionable moment from one that has been screaming since
the clip began. Negative clips do not rescue it, because a model can satisfy them
and still fire from frame 0 on every risky-looking clip -- which is precisely the
failure the current threshold system exhibits (it fires a median 14.4 s before
``time_of_alert``, and scores a useful-warning rate of 3/60).

The default is therefore **0.5**: enough to make standing alarms strictly worse
than well-timed ones, while staying below the positive term so that genuine early
evidence is discouraged rather than crushed. ``time_of_alert`` is a human
annotation with unknown jitter and hazards often become visible slightly before
they become actionable, so 0.5 rather than 1.0 -- and it remains the natural knob
to ablate.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

DEFAULT_KAPPA = 3.0

# See the module docstring: at 0.0 the loss cannot separate a standing alarm
# from a well-timed warning, so this is deliberately non-zero.
DEFAULT_PRE_ALERT_WEIGHT = 0.5


def alert_weights(t: torch.Tensor,
                  alert_t: float,
                  event_t: float,
                  kappa: float = DEFAULT_KAPPA) -> torch.Tensor:
    """Per-frame positive weight ``w(t)``; 0 outside ``[alert_t, event_t]``.

    ``t`` is a 1-D tensor of frame timestamps in seconds. Returns a tensor of the
    same shape. A degenerate window (``event_t <= alert_t``, which would mean the
    annotation is inconsistent) falls back to weighting only the frames at or
    just before ``event_t``, rather than dividing by zero.
    """
    span = float(event_t) - float(alert_t)
    inside = (t >= float(alert_t)) & (t <= float(event_t))
    if span <= 1e-6:
        # No usable window: weight exactly the frames at the event, uniformly.
        return inside.to(t.dtype)
    u = ((t - float(alert_t)) / span).clamp(0.0, 1.0)
    return torch.exp(float(kappa) * (u - 1.0)) * inside.to(t.dtype)


def anticipation_loss(logits: torch.Tensor,
                      t: torch.Tensor,
                      label: int,
                      alert_t: Optional[float] = None,
                      event_t: Optional[float] = None,
                      kappa: float = DEFAULT_KAPPA,
                      pre_alert_weight: float = DEFAULT_PRE_ALERT_WEIGHT,
                      ) -> torch.Tensor:
    """Loss for ONE clip. ``logits`` and ``t`` are 1-D and the same length.

    * label 1 with both times: alert-keyed weighted BCE toward 1 over
      ``[alert_t, event_t]``, plus ``pre_alert_weight`` * BCE toward 0 before
      ``alert_t``.
    * label 1 missing either time: falls back to a uniform BCE toward 1 over the
      whole clip. Unusable as anticipation supervision, so
      :func:`batch_anticipation_loss` reports how many clips took this path.
    * label 0: mean BCE toward 0 over every frame.

    Returns a scalar. Each branch is normalised by its own weight mass, so clip
    length and lead time do not change a clip's influence on the batch.
    """
    if logits.ndim != 1 or t.ndim != 1 or logits.shape != t.shape:
        raise ValueError(
            f"logits and t must be 1-D and equal length, got {tuple(logits.shape)} "
            f"and {tuple(t.shape)}")
    if logits.numel() == 0:
        return logits.sum() * 0.0

    if int(label) == 0:
        return F.binary_cross_entropy_with_logits(
            logits, torch.zeros_like(logits))

    if alert_t is None or event_t is None:
        return F.binary_cross_entropy_with_logits(
            logits, torch.ones_like(logits))

    w = alert_weights(t, alert_t, event_t, kappa)
    total = w.sum()
    if float(total) <= 0.0:
        # The clip's actionable window falls entirely outside the frames we have
        # (a truncated extraction window). Contribute nothing rather than a
        # gradient computed from no supervision.
        return logits.sum() * 0.0

    pos = (F.binary_cross_entropy_with_logits(
        logits, torch.ones_like(logits), reduction="none") * w).sum() / total

    if pre_alert_weight <= 0.0:
        return pos

    pre = (t < float(alert_t)).to(logits.dtype)
    pre_total = pre.sum()
    if float(pre_total) <= 0.0:
        return pos
    neg = (F.binary_cross_entropy_with_logits(
        logits, torch.zeros_like(logits), reduction="none") * pre).sum() / pre_total
    return pos + float(pre_alert_weight) * neg


def batch_anticipation_loss(logits_list: Sequence[torch.Tensor],
                            t_list: Sequence[torch.Tensor],
                            labels: Sequence[int],
                            alert_ts: Sequence[Optional[float]],
                            event_ts: Sequence[Optional[float]],
                            kappa: float = DEFAULT_KAPPA,
                            pre_alert_weight: float = DEFAULT_PRE_ALERT_WEIGHT,
                            pos_weight: float = 1.0):
    """Mean per-clip loss over a batch of variable-length clips.

    Returns ``(loss, stats)``. ``stats`` carries ``n_pos``, ``n_neg``,
    ``n_untimed`` (positives that fell back to uniform BCE) and ``n_skipped``
    (clips whose actionable window missed the available frames) -- all worth
    logging, because a training run where ``n_untimed`` is large is not learning
    anticipation no matter how the loss curve looks.

    ``pos_weight`` scales the positive clips' contribution. The Nexar training
    pool is 750/750, so the default 1.0 is right for the full set; it exists for
    the interim, when positives are scarce.
    """
    n = len(logits_list)
    if not (n == len(t_list) == len(labels) == len(alert_ts) == len(event_ts)):
        raise ValueError("all batch inputs must have the same length")
    if n == 0:
        raise ValueError("empty batch")

    stats = {"n_pos": 0, "n_neg": 0, "n_untimed": 0, "n_skipped": 0}
    terms = []
    weights = []
    for i in range(n):
        lbl = int(labels[i])
        if lbl == 1:
            stats["n_pos"] += 1
            if alert_ts[i] is None or event_ts[i] is None:
                stats["n_untimed"] += 1
            else:
                w = alert_weights(t_list[i], alert_ts[i], event_ts[i], kappa)
                if float(w.sum()) <= 0.0:
                    stats["n_skipped"] += 1
        else:
            stats["n_neg"] += 1

        terms.append(anticipation_loss(
            logits_list[i], t_list[i], lbl, alert_ts[i], event_ts[i],
            kappa=kappa, pre_alert_weight=pre_alert_weight))
        weights.append(float(pos_weight) if lbl == 1 else 1.0)

    wt = torch.tensor(weights, dtype=terms[0].dtype, device=terms[0].device)
    loss = (torch.stack(terms) * wt).sum() / wt.sum()
    return loss, stats


def first_alert_time(probs: torch.Tensor,
                     t: torch.Tensor,
                     threshold: float,
                     frames_to_confirm: int = 1) -> Optional[float]:
    """Timestamp of the first sustained threshold crossing, else None.

    The inference-side counterpart to the loss: converts a per-frame probability
    sequence into the single ``first_alert_t`` that
    ``scripts/eval_anticipation.py`` scores. ``frames_to_confirm`` mirrors the
    existing decision engine's debounce so a one-frame spike cannot fire.
    """
    if probs.ndim != 1 or probs.shape != t.shape:
        raise ValueError("probs and t must be 1-D and equal length")
    over = (probs >= float(threshold)).tolist()
    need = max(1, int(frames_to_confirm))
    run = 0
    for i, hot in enumerate(over):
        if hot:
            run += 1
            if run >= need:
                # Credit the START of the confirmed run, not its end: the model
                # actually knew at i-need+1, and charging it the debounce delay
                # would understate its lead time.
                return float(t[i - need + 1])
        else:
            run = 0
    return None


def expected_lead_time(alert_t: float, event_t: float,
                       kappa: float = DEFAULT_KAPPA) -> float:
    """Weighted-mean warning time under :func:`alert_weights`, in s before event.

    Diagnostic: what lead time is this loss actually asking for? At kappa=3 over
    a 3.5 s window it is ~1.0 s, so a large kappa quietly turns an anticipation
    objective into a detection one. Worth printing next to the chosen kappa.
    """
    span = float(event_t) - float(alert_t)
    if span <= 0:
        return 0.0
    k = float(kappa)
    if abs(k) < 1e-9:
        return span / 2.0
    # E[1-u] with density proportional to exp(k(u-1)) on u in [0, 1].
    mean_u = 1.0 / (1.0 - math.exp(-k)) - 1.0 / k
    return span * (1.0 - mean_u)
