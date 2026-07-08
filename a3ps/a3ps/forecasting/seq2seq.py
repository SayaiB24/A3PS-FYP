"""Tiny LSTM seq2seq forecaster (STRETCH, Week 3).

Encoder LSTM (hidden 64, 2 layers) over the 10-point history; decoder LSTM
emits 20 future position-deltas plus a per-step log-variance head, so we get
native per-step stds (trained with Gaussian NLL). ``predict`` applies the same
normalization used during mining (translate last-history point to the origin,
rotate mean history heading to +y) on the way in and inverts it on the way out,
so it is a drop-in replacement for ``KalmanCVForecaster``.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from a3ps.forecasting.base import Forecaster, Means, Point, Stds

HIST = 10          # history points (2 s @ 5 Hz)
FUT = 20           # future points (4 s @ 5 Hz)
HIDDEN = 64
LAYERS = 2


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------

class Seq2SeqNet(nn.Module):
    """Encoder-decoder LSTM emitting per-step future deltas + log-variance.

    forward(history[B,HIST,2], target_deltas=None, steps=FUT)
        -> mean_pos[B,steps,2], logvar[B,steps,2]

    ``mean_pos`` is the cumulative sum of predicted deltas (positions relative
    to the last-history origin). ``logvar`` is the per-step *position*
    log-variance. Pass ``target_deltas`` for teacher forcing during training.
    """

    def __init__(self, hidden: int = HIDDEN, layers: int = LAYERS):
        super().__init__()
        self.hidden, self.layers = hidden, layers
        self.encoder = nn.LSTM(2, hidden, layers, batch_first=True)
        self.decoder = nn.LSTM(2, hidden, layers, batch_first=True)
        self.delta_head = nn.Linear(hidden, 2)
        self.logvar_head = nn.Linear(hidden, 2)

    def forward(self, history, target_deltas=None, steps: int = FUT):
        b = history.size(0)
        _, hc = self.encoder(history)                    # (h, c)
        inp = history.new_zeros(b, 1, 2)                 # start token = zero delta
        pos = history.new_zeros(b, 2)                    # last-history origin
        means, logvars = [], []
        for t in range(steps):
            out, hc = self.decoder(inp, hc)
            o = out[:, 0]
            delta = self.delta_head(o)
            logvar = self.logvar_head(o)
            pos = pos + delta
            means.append(pos)
            logvars.append(logvar)
            if target_deltas is not None and t < target_deltas.size(1):
                inp = target_deltas[:, t:t + 1, :]       # teacher forcing
            else:
                inp = delta.unsqueeze(1)
        return torch.stack(means, 1), torch.stack(logvars, 1)


def future_to_deltas(future):
    """Convert future positions (relative to origin) to step deltas.

    ``future`` is (B, FUT, 2); delta[0] = future[0] - 0, delta[t] = future[t] -
    future[t-1]. Used for teacher forcing and Gaussian-NLL targets.
    """
    prev = torch.cat([future.new_zeros(future.size(0), 1, 2), future[:, :-1]], 1)
    return future - prev


def gaussian_nll(mean_pos, logvar, target_pos):
    """Per-step diagonal Gaussian negative log-likelihood on positions."""
    inv = torch.exp(-logvar)
    return 0.5 * (logvar + (target_pos - mean_pos) ** 2 * inv).sum(-1).mean()


# ---------------------------------------------------------------------------
# forecaster wrapper
# ---------------------------------------------------------------------------

class Seq2SeqForecaster(Forecaster):
    def __init__(self, weights_path: Optional[str] = None,
                 hidden: int = HIDDEN, layers: int = LAYERS, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = Seq2SeqNet(hidden, layers).to(self.device)
        self.model.eval()
        self.weights_path = weights_path
        if weights_path:
            self.load(weights_path)

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(),
                    "hidden": self.model.hidden, "layers": self.model.layers}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.eval()

    @staticmethod
    def _normalize(history: np.ndarray):
        """Match mining: origin = last history pt, rotate mean heading to +y."""
        origin = history[-1].copy()
        shifted = history - origin
        mean_d = np.diff(history, axis=0).mean(axis=0)
        if math.hypot(mean_d[0], mean_d[1]) < 1e-6:
            theta = 0.0
        else:
            theta = (math.pi / 2.0) - math.atan2(mean_d[1], mean_d[0])
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s], [s, c]])
        hist_norm = shifted @ R.T
        return hist_norm, origin, theta, R

    @torch.no_grad()
    def predict(self, history: List[Point], dt: float, horizon_s: float) -> Tuple[Means, Stds]:
        hist = list(history)
        if len(hist) < 2:
            return [], []
        if len(hist) > HIST:
            hist = hist[-HIST:]
        elif len(hist) < HIST:
            hist = [hist[0]] * (HIST - len(hist)) + hist     # left-pad

        pts = np.asarray(hist, dtype=float)
        hist_norm, origin, theta, R = self._normalize(pts)

        x = torch.tensor(hist_norm[None], dtype=torch.float32, device=self.device)
        mean_pos, logvar = self.model(x, steps=FUT)
        mp = mean_pos[0].cpu().numpy()                       # (FUT, 2), normalized
        std_norm = np.exp(0.5 * logvar[0].cpu().numpy())     # (FUT, 2)

        # De-normalize: pos_orig = pos_norm @ R + origin.
        means_abs = mp @ R + origin
        # Rotate per-step diagonal covariance back by -theta.
        c, s = math.cos(theta), math.sin(theta)
        Rinv = np.array([[c, s], [-s, c]])                   # R(-theta)
        stds: Stds = []
        for k in range(FUT):
            cov = Rinv @ np.diag(std_norm[k] ** 2) @ Rinv.T
            stds.append([float(np.sqrt(max(cov[0, 0], 0.0))),
                         float(np.sqrt(max(cov[1, 1], 0.0)))])
        means: Means = [[float(px), float(py)] for px, py in means_abs]
        return means, stds
