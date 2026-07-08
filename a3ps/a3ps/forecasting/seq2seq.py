"""Tiny LSTM seq2seq forecaster (STRETCH, Week 3).

Encoder-decoder LSTM over trajectory deltas. Torch is imported lazily so this
module can be imported without torch present. Not yet implemented.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from a3ps.forecasting.base import Forecaster, Means, Point, Stds


class Seq2SeqForecaster(Forecaster):
    def __init__(
        self,
        weights_path: Optional[str] = None,
        hidden_size: int = 64,
        device: str = "cpu",
    ):
        self.weights_path = weights_path
        self.hidden_size = hidden_size
        self.device = device
        self._model = None

    def predict(self, history: List[Point], dt: float, horizon_s: float) -> Tuple[Means, Stds]:
        raise NotImplementedError("Seq2Seq LSTM not yet implemented (Week 3).")
