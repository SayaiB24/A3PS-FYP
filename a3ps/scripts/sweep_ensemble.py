#!/usr/bin/env python
"""Step 10.6: ensemble + temperature-scaling sweep on train_val_v2.

New, self-contained script. It IMPORTS ``evaluate`` and ``load_clips`` from
``train_risk_head`` and never modifies ``sweep_operating_point.py`` or any
checkpoint. Every variant is scored by the SAME ``evaluate()`` (v2 definition,
same threshold x confirm grid) as the single-checkpoint sweeps.

Variants
--------
* ``single``         each checkpoint alone (seed 1234 is the baseline)
* ``single_cal``     each checkpoint with its own fitted temperature
* ``ens``            unweighted mean of the members' per-frame probabilities
* ``ens_cal``        temperature fitted on the ensemble's own logit  (PRIMARY
                     calibrated candidate, declared before looking at results)
* ``ens_of_cal``     mean of the members' individually calibrated probabilities
                     (secondary; reported, not selectable as "the" candidate)

Calibration: temperature scaling, ``z -> z / T``, one scalar per model, fitted
by minimising the SAME anticipation loss the head was trained with
(``batch_anticipation_loss``, kappa/pre_alert_weight taken from the checkpoint)
over train_val_v2. Chosen over Platt because it is monotone with a single free
parameter, so it CANNOT change ranking (mean AP is invariant) -- any change in
useful-warning is attributable to the probability scale alone, which is exactly
the hypothesis being tested -- and one parameter on 205 clips overfits least.

The fit set and the scoring set are both train_val_v2 (as specified), so the
calibrated numbers are in-sample for T. ``--cv-folds`` adds a clip-level
K-fold check that fits T on the other folds, to size that optimism.
"""

import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from a3ps.common.splits import DEFAULT_FREEZE_PATH  # noqa: E402
from a3ps.risk.anticipation_loss import batch_anticipation_loss  # noqa: E402
from a3ps.risk.temporal import RiskGRU  # noqa: E402

from train_risk_head import (evaluate, guard_held_out, load_clips, _fmt)  # noqa: E402

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
CONFIRMS = [3, 5, 8]
FA_TARGET, LEAD_MIN = 0.20, 1.0


class LogitModel:
    """Duck-types the bit of RiskGRU that ``evaluate`` uses (eval + call -> logits)."""

    def __init__(self, fn):
        self.fn = fn

    def eval(self):
        return self

    def __call__(self, x):
        return self.fn(x)


def _logit(p):
    p = p.clamp(1e-6, 1 - 1e-6)
    return torch.log(p) - torch.log1p(-p)


def member_logits(model, clips):
    """{clip_id: logits} for one checkpoint, computed once."""
    model.eval()
    with torch.no_grad():
        return {c["clip_id"]: model(c["X"]) for c in clips}


def calib_loss(logit_by_clip, clips, T, kappa, paw):
    ls = [logit_by_clip[c["clip_id"]] / T for c in clips]
    with torch.no_grad():
        loss, _ = batch_anticipation_loss(
            ls, [c["t"] for c in clips], [c["label"] for c in clips],
            [c["alert_time_s"] for c in clips], [c["event_time_s"] for c in clips],
            kappa=kappa, pre_alert_weight=paw, pos_weight=1.0)
    return float(loss)


def fit_temperature(logit_by_clip, clips, kappa, paw):
    """Minimise the anticipation loss over log T in [ln 0.25, ln 8] (golden section)."""
    lo, hi = math.log(0.25), math.log(8.0)
    g = (math.sqrt(5) - 1) / 2
    f = lambda lt: calib_loss(logit_by_clip, clips, math.exp(lt), kappa, paw)
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(40):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return math.exp((a + b) / 2)


def make_model(logit_fns):
    return LogitModel(logit_fns)


def sweep(model, clips, keep_rows=False):
    rows = []
    for conf in CONFIRMS:
        for thr in THRESHOLDS:
            v = evaluate(model, clips, thr, confirm=conf, dump_rows=keep_rows)
            rows.append({"threshold": thr, "confirm": conf, **v})
    return rows


def best_cell(rows):
    """Best useful_v2 among cells passing FA, lead and gross-premature gates
    (the 'how close it got' reduction used in operating_point_selection.md)."""
    ok = [r for r in rows
          if r["false_alarm_rate"] <= FA_TARGET
          and r["mean_lead_vs_event_v2"] >= LEAD_MIN
          and r["n_gross_premature"] == 0]
    if not ok:
        return None
    return max(ok, key=lambda r: (r["useful_warning_rate_v2"],
                                  -r["false_alarm_rate"]))


def strip(r):
    return {k: v for k, v in r.items() if k != "rows"}


def grid_md(title, rows, ref_rows=None):
    L = [f"### {title}", "",
         "| thr | confirm | useful_v2 | n | FA | lead_vs_event_v2 | frac_premature "
         "| n_gross_premature | mean AP |" + (" Δuseful vs 1234 |" if ref_rows else ""),
         "|---|---|---|---|---|---|---|---|---|" + ("---|" if ref_rows else "")]
    for i, r in enumerate(rows):
        d = ""
        if ref_rows:
            d = f" {r['useful_warning_rate_v2'] - ref_rows[i]['useful_warning_rate_v2']:+.3f} |"
        L.append(f"| {r['threshold']:.2f} | {r['confirm']} "
                 f"| {_fmt(r['useful_warning_rate_v2'])} | {r['n_useful_v2']} "
                 f"| {_fmt(r['false_alarm_rate'])} "
                 f"| {_fmt(r['mean_lead_vs_event_v2'], 2)} "
                 f"| {_fmt(r['frac_premature'])} | {r['n_gross_premature']} "
                 f"| {_fmt(r['mean_AP'])} |" + d)
    L.append("")
    return L


def paired_bootstrap(rows_a, rows_b, n=5000, seed=0):
    """Paired bootstrap over timed positives of useful_v2(b) - useful_v2(a)."""
    da = {r["clip_id"]: r["verdict_v2"] == "useful" for r in rows_a if r["label"] == 1
          and r["verdict_v2"] != "untimed"}
    db = {r["clip_id"]: r["verdict_v2"] == "useful" for r in rows_b if r["label"] == 1
          and r["verdict_v2"] != "untimed"}
    ids = sorted(da)
    diffs = [int(db[i]) - int(da[i]) for i in ids]
    rng = random.Random(seed)
    m = len(ids)
    stats = sorted(sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return sum(diffs) / m, stats[int(0.025 * n)], stats[int(0.975 * n)]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", required=True,
                   help="Comma-separated seed=path pairs; the FIRST is the baseline.")
    p.add_argument("--features", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--split-freeze", default=DEFAULT_FREEZE_PATH)
    args = p.parse_args()
    guard_held_out(args.features, "--features", args.split_freeze, False,
                   "--final-eval-report")  # never passed: eval_v2 is refused

    pairs = [x.split("=", 1) for x in args.checkpoints.split(",")]
    names = [n for n, _ in pairs]
    base = names[0]
    clips = load_clips(args.features)
    n_pos = sum(c["label"] for c in clips)
    print(f"{len(clips)} clips ({n_pos} pos) from {args.features}")

    models, logits, extras, temps = {}, {}, {}, {}
    for n, path in pairs:
        m, ex = RiskGRU.load(path)
        models[n], extras[n] = m, ex
        logits[n] = member_logits(m, clips)
        if ex.get("degenerate_checkpoint"):
            raise SystemExit(f"{path} is flagged degenerate")
    kappa = extras[base].get("kappa", 1.0)
    paw = extras[base].get("pre_alert_weight", 0.5)

    def fn_single(n, T=1.0):
        return lambda x: models[n](x) / T

    # identity lookup: evaluate() calls model(c["X"]); map tensor -> clip_id
    by_x = {id(c["X"]): c["clip_id"] for c in clips}

    def lookup(n, x, T):
        return logits[n][by_x[id(x)]] / T

    for n in names:
        temps[n] = fit_temperature(logits[n], clips, kappa, paw)
        print(f"T[{n}] = {temps[n]:.3f}", flush=True)

    def ens_logit_by_clip(Ts):
        out = {}
        for c in clips:
            cid = c["clip_id"]
            p = torch.stack([torch.sigmoid(logits[n][cid] / Ts[n]) for n in names]).mean(0)
            out[cid] = _logit(p)
        return out

    ones = {n: 1.0 for n in names}
    ens_lg = ens_logit_by_clip(ones)
    T_ens = fit_temperature(ens_lg, clips, kappa, paw)
    print(f"T[ensemble] = {T_ens:.3f}", flush=True)
    ens_of_cal_lg = ens_logit_by_clip(temps)

    def wrap(lg, T=1.0):
        return LogitModel(lambda x: lg[by_x[id(x)]] / T)

    variants = {}
    for n in names:
        variants[f"single:{n}"] = wrap(logits[n])
        variants[f"single_cal:{n}"] = wrap(logits[n], temps[n])
    variants["ens"] = wrap(ens_lg)
    variants["ens_cal"] = wrap(ens_lg, T_ens)
    variants["ens_of_cal"] = wrap(ens_of_cal_lg)

    results, rowsdump = {}, {}
    for k, m in variants.items():
        rs = sweep(m, clips, keep_rows=True)
        results[k] = rs
        print(f"{k}: mean AP {_fmt(rs[0]['mean_AP'])}, best gated cell "
              f"{(lambda b: 'none' if b is None else (b['threshold'], b['confirm'], _fmt(b['useful_warning_rate_v2']), _fmt(b['false_alarm_rate'])))(best_cell(rs))}",
              flush=True)

    # ---- cross-validated calibration for the primary calibrated candidate ----
    K = args.cv_folds
    rng = random.Random(0)
    pos = [c["clip_id"] for c in clips if c["label"] == 1]
    neg = [c["clip_id"] for c in clips if c["label"] == 0]
    rng.shuffle(pos), rng.shuffle(neg)
    fold_of = {cid: i % K for i, cid in enumerate(pos)}
    fold_of.update({cid: i % K for i, cid in enumerate(neg)})
    Tfold = {}
    for f in range(K):
        tr = [c for c in clips if fold_of[c["clip_id"]] != f]
        Tfold[f] = fit_temperature(ens_lg, tr, kappa, paw)
    cv_lg = {c["clip_id"]: ens_lg[c["clip_id"]] / Tfold[fold_of[c["clip_id"]]]
             for c in clips}
    results["ens_cal_cv"] = sweep(wrap(cv_lg), clips, keep_rows=True)
    print(f"CV temperatures for ensemble: { {k: round(v, 3) for k, v in Tfold.items()} }")

    # ---- persist (new files only) ----
    os.makedirs(args.out_dir, exist_ok=True)
    payload = {"features": args.features, "n_clips": len(clips), "n_pos": n_pos,
               "members": dict(pairs), "temperatures": temps, "T_ensemble": T_ens,
               "T_cv_folds": Tfold, "kappa": kappa, "pre_alert_weight": paw,
               "grids": {k: [strip(r) for r in v] for k, v in results.items()}}
    with open(os.path.join(args.out_dir, "ensemble_calibration_grids.json"), "w") as fh:
        json.dump(payload, fh, indent=1, default=str)

    # ---- markdown ----
    base_rows = results[f"single:{base}"]
    L = ["# Ensemble + calibration grids (train_val_v2, v2 definition)", "",
         f"- members: {', '.join(f'{n}={p_}' for n, p_ in pairs)}",
         f"- scored on `{args.features}`: {len(clips)} clips, {n_pos} positive",
         f"- temperatures (fit on same set): "
         + ", ".join(f"{n}: {temps[n]:.3f}" for n in names)
         + f", ensemble: {T_ens:.3f}", ""]
    for k in ["ens", "ens_cal", "ens_of_cal", "ens_cal_cv"]:
        L += grid_md(k, results[k], base_rows)
    L += grid_md(f"single:{base} (baseline)", base_rows)
    for n in names[1:]:
        L += grid_md(f"single:{n}", results[f"single:{n}"], base_rows)
    with open(os.path.join(args.out_dir, "ensemble_calibration_grids.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(L))

    # ---- calibration-mismatch check at the shared cell ----
    def at(rs, thr, conf):
        return next(r for r in rs if r["threshold"] == thr and r["confirm"] == conf)

    print("\nshared cell thr0.70/c8 (useful_v2, FA):")
    U, C = [], []
    for n in names:
        a, b = at(results[f"single:{n}"], 0.7, 8), at(results[f"single_cal:{n}"], 0.7, 8)
        U.append(a["useful_warning_rate_v2"]), C.append(b["useful_warning_rate_v2"])
        print(f"  {n}: uncal {a['useful_warning_rate_v2']:.3f}/{a['false_alarm_rate']:.3f}"
              f"  cal {b['useful_warning_rate_v2']:.3f}/{b['false_alarm_rate']:.3f}")

    def sd(x):
        m = sum(x) / len(x)
        return (sum((v - m) ** 2 for v in x) / len(x)) ** 0.5
    print(f"  sd uncal {sd(U):.3f}  sd cal {sd(C):.3f}")

    # ---- comparison table ----
    print("\nBEST GATED CELL PER CANDIDATE")
    table = {}
    for k in [f"single:{base}", "ens", "ens_cal", "ens_of_cal", "ens_cal_cv"]:
        b = best_cell(results[k])
        table[k] = b
        if b is None:
            print(f"  {k}: no cell passes the gates")
            continue
        print(f"  {k}: thr {b['threshold']} c{b['confirm']}  useful_v2 "
              f"{b['useful_warning_rate_v2']:.3f}  FA {b['false_alarm_rate']:.3f}  "
              f"lead {b['mean_lead_vs_event_v2']:.2f}  mAP {b['mean_AP']:.3f}  "
              f"gross {b['n_gross_premature']}")
    bb = table[f"single:{base}"]
    for k in ["ens", "ens_cal", "ens_of_cal", "ens_cal_cv"]:
        b = table[k]
        if b is None:
            continue
        ra = at(results[f"single:{base}"], bb["threshold"], bb["confirm"])["rows"]
        rb = at(results[k], b["threshold"], b["confirm"])["rows"]
        d, lo, hi = paired_bootstrap(ra, rb)
        print(f"  paired boot (own best cell vs baseline best cell) {k}: "
              f"diff useful {d:+.3f} 95% CI [{lo:+.3f}, {hi:+.3f}]")
        table[k]["boot"] = [d, lo, hi]
    with open(os.path.join(args.out_dir, "comparison.json"), "w") as fh:
        json.dump({k: (None if v is None else strip(v)) for k, v in table.items()},
                  fh, indent=1, default=str)
    print(f"\nwrote {args.out_dir}")


if __name__ == "__main__":
    main()
