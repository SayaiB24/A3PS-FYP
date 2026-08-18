"""Small temporal risk head: per-frame features -> per-frame risk probability.

This is what Phase IV became. The old Phase IV compared a physics-derived
collision probability against a hand-tuned threshold; this consumes that same
probability as one of ~112 input features and *learns* the discrimination
instead.

Deliberately small. The Nexar training pool is 750 positives + 750 negatives --
1,500 sequences. A GRU with hidden 64 over a 112-dim input is ~34 k parameters,
which is the right order for that much data; anything transformer-sized would
memorise the training clips and tell us nothing. The forecaster in Phase III is
an LSTM of similar scale (``a3ps/forecasting/seq2seq.py``, hidden 64, 2 layers),
so this also keeps the two learned components comparable in capacity.

Causality is the constraint that matters. The head must be strictly causal --
frame ``t``'s output may only depend on frames ``<= t`` -- because the whole
claim is anticipation. A bidirectional RNN or any non-causal attention would let
the model see the collision and then "predict" it, which would produce excellent
metrics and a worthless system. :class:`RiskGRU` is unidirectional for exactly
that reason, and :func:`assert_causal` is a runtime check that it stayed that way.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

DEFAULT_HIDDEN = 64
DEFAULT_LAYERS = 1


class RiskGRU(nn.Module):
    """Causal GRU over per-frame features -> one risk logit per frame.

    ``forward(x)`` takes ``(B, T, D)`` or ``(T, D)`` and returns logits with the
    batch/time dims preserved and the feature dim dropped. Logits, not
    probabilities: the loss in :mod:`a3ps.risk.anticipation_loss` is
    ``binary_cross_entropy_with_logits``, which is numerically stabler than
    sigmoid-then-BCE. Call :meth:`probs` when you want probabilities.

    ``input_norm`` applies a LayerNorm to the inputs. The feature vector mixes
    log-areas (order 1-10), normalised positions (order 0.1) and saturating
    probabilities (order 1), so without it the large-magnitude columns dominate
    the first layer purely by scale.
    """

    def __init__(self, input_dim: int, hidden: int = DEFAULT_HIDDEN,
                 layers: int = DEFAULT_LAYERS, dropout: float = 0.1,
                 input_norm: bool = True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.use_input_norm = bool(input_norm)

        self.norm = nn.LayerNorm(self.input_dim) if input_norm else nn.Identity()
        self.gru = nn.GRU(
            self.input_dim, self.hidden, self.layers,
            batch_first=True, bidirectional=False,
            dropout=(float(dropout) if self.layers > 1 else 0.0),
        )
        self.drop = nn.Dropout(float(dropout))
        self.head = nn.Linear(self.hidden, 1)

    def forward(self, x: torch.Tensor,
                state: Optional[torch.Tensor] = None,
                return_state: bool = False):
        squeeze = (x.ndim == 2)
        if squeeze:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(f"expected (B, T, D) or (T, D), got {tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"feature width {x.shape[-1]} != input_dim {self.input_dim}")

        h, new_state = self.gru(self.norm(x), state)
        logits = self.head(self.drop(h)).squeeze(-1)
        if squeeze:
            logits = logits.squeeze(0)
            new_state = new_state
        return (logits, new_state) if return_state else logits

    @torch.no_grad()
    def probs(self, x: torch.Tensor) -> torch.Tensor:
        """Per-frame risk probability in [0, 1]. Sets eval mode for the call."""
        was_training = self.training
        self.eval()
        try:
            return torch.sigmoid(self.forward(x))
        finally:
            if was_training:
                self.train()

    def config(self) -> Dict[str, Any]:
        return {"input_dim": self.input_dim, "hidden": self.hidden,
                "layers": self.layers, "input_norm": self.use_input_norm}

    def save(self, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Checkpoint weights + the config needed to rebuild, + provenance.

        ``extra`` should carry the feature schema version and whether ego features
        were present, so a checkpoint can never be silently applied to features it
        was not trained on.
        """
        torch.save({"state_dict": self.state_dict(),
                    "config": self.config(),
                    "extra": dict(extra or {})}, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> Tuple["RiskGRU", Dict[str, Any]]:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, ckpt.get("extra", {})


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def assert_causal(model: nn.Module, input_dim: int, T: int = 24,
                  tol: float = 1e-5) -> None:
    """Raise unless output ``t`` is unaffected by inputs after ``t``.

    Perturbs the second half of a sequence and checks the first half's logits do
    not move. Cheap, and it catches the single most damaging mistake available
    here: making the head bidirectional (or adding non-causal attention) turns
    anticipation into hindsight and inflates every metric at once.
    """
    model.eval()
    x = torch.randn(1, T, input_dim)
    base = model(x)
    half = T // 2
    x2 = x.clone()
    x2[:, half:, :] += 10.0
    after = model(x2)
    delta = (base[:, :half] - after[:, :half]).abs().max().item()
    if delta > tol:
        raise AssertionError(
            f"model is not causal: perturbing frames >= {half} moved earlier "
            f"logits by {delta:.3e} (tol {tol:.0e}). A non-causal risk head can "
            "see the collision before 'predicting' it.")
