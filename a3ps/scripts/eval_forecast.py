#!/usr/bin/env python
"""ADE / FDE at 1/2/4 s: KalmanCVForecaster vs Seq2SeqForecaster.

Evaluates on the SAME val split the training notebook uses (all shards
concatenated in sorted order, 90/10 split with seed 42, val = first 10%).
Both forecasters run in the shard's normalized frame (last-history point at the
origin, heading +y), so metrics are directly comparable. Prints a table and
writes eval/forecast_table.md.

    python scripts/eval_forecast.py
    python scripts/eval_forecast.py --shards data/trajectories --weights models/seq2seq_v1.pt
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from a3ps.forecasting.kalman_cv import KalmanCVForecaster  # noqa: E402

DT = 0.2
HORIZON = 4.0
SEED = 42
VAL_FRAC = 0.10
HORIZONS_S = [1.0, 2.0, 4.0]          # -> steps 5, 10, 20


def load_val(shard_dir):
    shards = sorted(glob.glob(os.path.join(shard_dir, "*.npz")))
    if not shards:
        return None, None, None
    H = np.concatenate([np.load(s)["history"] for s in shards]).astype("float32")
    F = np.concatenate([np.load(s)["future"] for s in shards]).astype("float32")
    idx = np.random.default_rng(SEED).permutation(len(H))
    n_val = max(1, int(len(H) * VAL_FRAC))
    v = idx[:n_val]
    space = "px"
    try:
        import json
        space = json.load(open(os.path.join(shard_dir, "stats.json")))["space"]
        space = "m" if space == "bev" else "px"
    except Exception:
        pass
    return H[v], F[v], space


def ade_fde_at(pred, gt, step):
    """pred/gt: (N, FUT, 2). Returns (ADE over first `step`, FDE at step-1)."""
    d = np.linalg.norm(pred[:, :step] - gt[:, :step], axis=-1)   # (N, step)
    ade = float(d.mean())
    fde = float(np.linalg.norm(pred[:, step - 1] - gt[:, step - 1], axis=-1).mean())
    return ade, fde


def run_forecaster(fc, hist_batch):
    out = []
    for h in hist_batch:
        means, _ = fc.predict([list(map(float, p)) for p in h], DT, HORIZON)
        out.append(means)
    return np.asarray(out, dtype="float32")            # (N, FUT, 2)


def main():
    p = argparse.ArgumentParser(description="ADE/FDE: Kalman vs Seq2Seq.")
    p.add_argument("--shards", default="data/trajectories")
    p.add_argument("--weights", default="models/seq2seq_v1.pt")
    p.add_argument("--out", default="eval/forecast_table.md")
    args = p.parse_args()

    Hva, Fva, space = load_val(args.shards)
    if Hva is None:
        print(f"No shards in {args.shards}. Run scripts/mine_trajectories.py first.")
        return
    print(f"val windows: {len(Hva)}  (units: {space})")

    results = {}
    kal_pred = run_forecaster(KalmanCVForecaster(), Hva)
    results["Kalman-CV"] = kal_pred

    if os.path.isfile(args.weights):
        from a3ps.forecasting.seq2seq import Seq2SeqForecaster
        results["Seq2Seq-LSTM"] = run_forecaster(
            Seq2SeqForecaster(weights_path=args.weights), Hva)
    else:
        print(f"  (no weights at {args.weights} -> skipping Seq2Seq; train in Colab first)")

    steps = [int(round(h / DT)) for h in HORIZONS_S]
    # Build table.
    header = "| model | " + " | ".join(
        f"ADE@{int(h)}s | FDE@{int(h)}s" for h in HORIZONS_S) + " |"
    sep = "|" + "---|" * (1 + 2 * len(HORIZONS_S))
    lines = [f"# Forecast eval (val = {len(Hva)} windows, units: {space})", "",
             header, sep]
    print("\n" + header)
    for name, pred in results.items():
        cells = []
        for st in steps:
            ade, fde = ade_fde_at(pred, Fva, st)
            cells.append(f"{ade:.2f} | {fde:.2f}")
        row = f"| {name} | " + " | ".join(cells) + " |"
        lines.append(row)
        print(row)
    lines.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
