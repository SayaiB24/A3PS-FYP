#!/usr/bin/env python
"""Evaluate forecasting quality: ADE / FDE for Kalman vs LSTM.

    python scripts/eval_forecast.py --trajectories data/trajectories.jsonl
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def ade(pred, gt) -> float:
    """Average Displacement Error over the horizon."""
    n = min(len(pred), len(gt))
    if n == 0:
        return float("nan")
    total = 0.0
    for (px, py), (gx, gy) in zip(pred[:n], gt[:n]):
        total += ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5
    return total / n


def fde(pred, gt) -> float:
    """Final Displacement Error (last predicted step)."""
    if not pred or not gt:
        return float("nan")
    (px, py), (gx, gy) = pred[-1], gt[-1]
    return ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5


def main() -> None:
    p = argparse.ArgumentParser(description="ADE/FDE: Kalman vs LSTM.")
    p.add_argument("--trajectories", default="data/trajectories.jsonl")
    p.add_argument("--history-s", type=float, default=2.0)
    p.add_argument("--horizon-s", type=float, default=4.0)
    args = p.parse_args()

    # TODO: load held-out trajectories, split history/future, run each
    # forecaster, and aggregate ADE/FDE per model.
    raise NotImplementedError("eval_forecast is a stub; helpers ade/fde are ready.")


if __name__ == "__main__":
    main()
