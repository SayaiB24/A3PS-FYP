#!/usr/bin/env python
"""Train the temporal risk head on extracted per-frame features.

    python scripts/train_risk_head.py --features data/features/train \\
        --val-features data/features/val --epochs 30 --out notebooks/models/risk_gru_v1.pt

Reads the .npz files written by ``scripts/extract_features.py``, trains
:class:`a3ps.risk.temporal.RiskGRU` with the alert-keyed anticipation loss, and
reports the three numbers the project cares about on the validation set:
useful-warning rate (keyed to ``alert_time_s``), official-style Nexar AP at the
500/1000/1500 ms pre-event cutoffs, and mean lead time.

Read this before trusting a number it prints
--------------------------------------------
The script refuses to run with ``--features`` and ``--val-features`` pointing at
the same directory unless ``--allow-leakage`` is given, and it stamps
``leakage_warning`` into the checkpoint when it is. It also prints, prominently:

* whether the features carry ego motion (``has_ego``). Features built from the
  cached ``eval/anticipation/`` pass do not, and a model trained on them was
  blind to ego motion.
* how many positives lacked usable event/alert times, since those clips give the
  loss no anticipation signal at all.

A clip-level (never frame-level) split is used for the internal ``--val-frac``
option, because frames from one clip are massively autocorrelated -- a
frame-level split would leak the answer and produce a validation curve that
means nothing.
"""

import argparse
import glob
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from a3ps.features.extract import frame_feature_dim, load_features  # noqa: E402
from a3ps.risk.anticipation_loss import (  # noqa: E402
    DEFAULT_KAPPA,
    DEFAULT_PRE_ALERT_WEIGHT,
    batch_anticipation_loss,
    expected_lead_time,
    first_alert_time,
)
from a3ps.risk.temporal import RiskGRU, assert_causal, count_parameters  # noqa: E402

from eval_anticipation import average_precision  # noqa: E402


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_clips(feature_dir):
    """Load every .npz in a directory into memory. These are ~20 KB each."""
    out = []
    for path in sorted(glob.glob(os.path.join(feature_dir, "*.npz"))):
        d = load_features(path)
        meta = d["meta"]
        out.append({
            "clip_id": meta.get("clip_id") or os.path.basename(path)[:-4],
            "X": torch.from_numpy(d["X"]),
            "t": torch.from_numpy(d["t"]),
            "label": int(meta.get("label", 0)),
            "event_time_s": meta.get("event_time_s"),
            "alert_time_s": meta.get("alert_time_s"),
            "has_ego": bool(meta.get("has_ego", False)),
        })
    return out


def split_by_clip(clips, val_frac, seed):
    """Stratified clip-level split. Never split frames within a clip."""
    pos = [c for c in clips if c["label"] == 1]
    neg = [c for c in clips if c["label"] == 0]
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_vp = max(1, int(round(len(pos) * val_frac))) if pos else 0
    n_vn = max(1, int(round(len(neg) * val_frac))) if neg else 0
    val = pos[:n_vp] + neg[:n_vn]
    train = pos[n_vp:] + neg[n_vn:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def evaluate(model, clips, threshold, cutoffs=(0.5, 1.0, 1.5), confirm=3):
    """Useful-warning rate, cutoff APs and lead time on a clip list."""
    model.eval()
    timelines, rows = [], []
    with torch.no_grad():
        for c in clips:
            probs = torch.sigmoid(model(c["X"]))
            fa = first_alert_time(probs, c["t"], threshold, confirm)
            rows.append({"clip": c, "first_alert_t": fa,
                         "peak": float(probs.max())})
            timelines.append(list(zip(c["t"].tolist(), probs.tolist())))

    useful = early = n_timed = 0
    leads, fa_neg = [], 0
    for r in rows:
        c = r["clip"]
        te, ta, fa = c["event_time_s"], c["alert_time_s"], r["first_alert_t"]
        if c["label"] == 0:
            fa_neg += int(fa is not None)
            continue
        if te is None or ta is None:
            continue
        n_timed += 1
        if fa is None or fa > te:
            continue
        leads.append(te - fa)
        if fa < ta:
            early += 1
        else:
            useful += 1

    def ap_at(tau):
        scores, labels = [], []
        for r, tl in zip(rows, timelines):
            c = r["clip"]
            if c["label"] == 1:
                if c["event_time_s"] is None:
                    continue
                cut = c["event_time_s"] - tau
                vals = [p for (t, p) in tl if t <= cut]
            else:
                vals = [p for (_, p) in tl]
            scores.append(max(vals) if vals else 0.0)
            labels.append(c["label"])
        return average_precision(scores, labels)

    aps = {tau: ap_at(tau) for tau in cutoffs}
    finite = [v for v in aps.values() if v == v]
    n_neg = sum(1 for r in rows if r["clip"]["label"] == 0)
    return {
        "n_clips": len(rows),
        "n_timed": n_timed,
        "useful_warning_rate": (useful / n_timed) if n_timed else float("nan"),
        "n_useful": useful,
        "n_too_early": early,
        "false_alarm_rate": (fa_neg / n_neg) if n_neg else float("nan"),
        "mean_lead_s": (sum(leads) / len(leads)) if leads else float("nan"),
        "AP": aps,
        "mean_AP": (sum(finite) / len(finite)) if finite else float("nan"),
    }


def _fmt(x, nd=3):
    return "n/a" if x is None or x != x else f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def batched_logits(model, batch):
    """Per-clip logits from ONE padded forward pass instead of one call per clip.

    Why this is safe to do by end-padding, with no masking in the loss:

    * :class:`RiskGRU` is a **unidirectional** GRU, so the output at step ``t``
      depends only on steps ``<= t``. Zeros appended after a clip's real frames
      cannot influence any valid position.
    * Its input normalisation is ``nn.LayerNorm(input_dim)``, which normalises
      across the FEATURE dimension within each timestep independently. Padding
      adds timesteps, not features, so no valid timestep's statistics change.
      (A BatchNorm, or any norm over the time axis, would break this.)

    So the valid prefix of each padded row is numerically identical to running
    that clip alone -- ``tests/test_train_risk_head.py`` asserts exactly that.
    The loss is already per-clip, so it needs no changes at all.

    The previous ``[model(c["X"]) for c in batch]`` gave an effective batch size
    of 1 and ~1.85x parallelism on 8 cores, because a batch-1 GRU is bound by
    sequential per-timestep kernel launches rather than arithmetic.
    """
    lens = [int(c["X"].shape[0]) for c in batch]
    width = int(batch[0]["X"].shape[1])
    x = torch.zeros(len(batch), max(lens), width,
                    dtype=batch[0]["X"].dtype)
    for i, c in enumerate(batch):
        x[i, :lens[i]] = c["X"]
    out = model(x)                                  # (B, T_max)
    return [out[i, :lens[i]] for i in range(len(batch))]


def train(model, train_clips, val_clips, args, on_improve=None, on_epoch=None):
    """Train the head, returning ``(best, history)``.

    ``on_improve(best)`` is called every time an epoch sets a new best validation
    score, and ``on_epoch(history)`` after every epoch. Both exist so the caller
    can persist progress *as it happens*: this loop keeps the best weights in
    memory, so without them a run that is interrupted -- or simply run for more
    epochs than it needed -- loses the best model entirely. That is not
    hypothetical; it cost us an 85-minute run whose best epoch was #5.

    Early stopping: if ``args.patience`` is > 0 and no epoch improves on the best
    score for that many consecutive epochs, training stops. On a 1,365-clip set
    the head peaks within ~5-10 epochs and then overfits, so the default budget
    is mostly spent making the result worse.
    """
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    best = None
    history = []
    stale = 0                      # epochs since the last improvement

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.Random(args.seed + epoch).shuffle(train_clips)
        total, nb = 0.0, 0
        agg = {"n_pos": 0, "n_neg": 0, "n_untimed": 0, "n_skipped": 0}

        for i in range(0, len(train_clips), args.batch_size):
            batch = train_clips[i:i + args.batch_size]
            logits = batched_logits(model, batch)
            loss, stats = batch_anticipation_loss(
                logits,
                [c["t"] for c in batch],
                [c["label"] for c in batch],
                [c["alert_time_s"] for c in batch],
                [c["event_time_s"] for c in batch],
                kappa=args.kappa,
                pre_alert_weight=args.pre_alert_weight,
                pos_weight=args.pos_weight,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            total += float(loss.detach())
            nb += 1
            for k in agg:
                agg[k] += stats[k]

        train_loss = total / max(nb, 1)
        val = evaluate(model, val_clips, args.threshold, confirm=args.confirm)
        history.append({"epoch": epoch, "train_loss": train_loss, **{
            "val_useful": val["useful_warning_rate"],
            "val_mean_AP": val["mean_AP"],
            "val_false_alarm": val["false_alarm_rate"]}})

        # Selection must respect the constraint that makes a result reportable.
        # Ranking on useful-warning alone previously kept epoch 5 (useful 0.833,
        # FA 0.583) over epoch 8 (useful 0.833, FA 0.550) -- equal on the score,
        # strictly better on false alarms -- because ties never updated and FA
        # was not part of the criterion at all.
        #
        # Rank: (meets the FA target, useful-warning, -FA). So an FA-compliant
        # epoch always beats a non-compliant one, ties on useful-warning break
        # toward lower FA, and if nothing is compliant the old behaviour applies
        # to whatever is available.
        score = val["useful_warning_rate"]
        fa = val["false_alarm_rate"]
        ok_fa = fa == fa and fa <= args.fa_target
        rank = (1 if ok_fa else 0, score if score == score else -1.0,
                -(fa if fa == fa else 1.0))
        improved = score == score and (best is None or rank > best["rank"])
        if improved:
            best = {"score": score, "rank": rank, "epoch": epoch, "val": val,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1

        # flush=True: without it a redirected/piped run shows nothing until the
        # process exits, which makes a long run impossible to monitor.
        print(f"epoch {epoch:3d}  loss {train_loss:.4f}  "
              f"val useful {_fmt(val['useful_warning_rate'])} "
              f"({val['n_useful']}/{val['n_timed']}, {val['n_too_early']} early)  "
              f"mAP {_fmt(val['mean_AP'])}  "
              f"FA {_fmt(val['false_alarm_rate'])}  "
              f"lead {_fmt(val['mean_lead_s'], 2)}s"
              f"{' ✓fa' if ok_fa else ''}"
              f"{'  <- best' if improved else ''}", flush=True)

        # Persist immediately, so an interrupted run still leaves the best model
        # and the history so far on disk.
        if improved and on_improve is not None:
            on_improve(best)
        if on_epoch is not None:
            on_epoch(history)

        if args.patience > 0 and stale >= args.patience:
            print(f"\nearly stop: no improvement for {stale} epoch(s) "
                  f"(best was epoch {best['epoch']} at "
                  f"{_fmt(best['score'])}). Stopping at epoch {epoch}/"
                  f"{args.epochs}.", flush=True)
            break

    if agg["n_untimed"]:
        print(f"\nnote: {agg['n_untimed']} positive-clip batches lacked "
              "event/alert times -- those gave the loss no anticipation signal.")
    return best, history


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True, help="Training .npz directory.")
    p.add_argument("--val-features", default=None,
                   help="Validation .npz directory. Omit to carve --val-frac "
                        "out of --features by clip.")
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--epochs", type=int, default=30,
                   help="Maximum epochs. Early stopping usually ends the run "
                        "sooner -- see --patience.")
    p.add_argument("--patience", type=int, default=4,
                   help="Stop after this many consecutive epochs with no "
                        "improvement in the validation useful-warning rate "
                        "(0 disables). On ~1.4k clips the head peaks within "
                        "5-10 epochs and overfits after, so the default keeps "
                        "runs short. The best checkpoint is written the moment "
                        "it appears, so stopping early never loses it.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--kappa", type=float, default=DEFAULT_KAPPA,
                   help="Anticipation-loss sharpness. Default %(default)s.")
    p.add_argument("--pre-alert-weight", type=float, default=DEFAULT_PRE_ALERT_WEIGHT,
                   help="Penalty on positive-clip frames before alert_time_s. "
                        "0 makes the loss blind to standing alarms -- see "
                        "a3ps/risk/anticipation_loss.py. Default %(default)s.")
    p.add_argument("--pos-weight", type=float, default=1.0)
    p.add_argument("--fa-target", type=float, default=0.20,
                   help="False-alarm rate the checkpoint selector treats as the "
                        "hard constraint (default %(default)s, per "
                        "docs/design/metrics.md). An epoch meeting it always "
                        "beats one that does not; ties on useful-warning rate "
                        "break toward lower FA.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Alert threshold used when scoring.")
    p.add_argument("--confirm", type=int, default=3,
                   help="Consecutive frames over threshold before alerting.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default=None, help="Checkpoint path (.pt).")
    p.add_argument("--history-json", default=None)
    p.add_argument("--allow-leakage", action="store_true",
                   help="Permit train and val to be the same clips. Produces a "
                        "plumbing check, NOT a result.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    clips = load_clips(args.features)
    if not clips:
        raise SystemExit(f"no .npz files in {args.features}")

    same_dir = (args.val_features and
                os.path.abspath(args.val_features) == os.path.abspath(args.features))
    if args.val_features and not same_dir:
        val_clips = load_clips(args.val_features)
        train_clips = clips
    elif same_dir:
        if not args.allow_leakage:
            raise SystemExit(
                "--features and --val-features are the same directory. That "
                "trains and validates on identical clips. Pass --allow-leakage "
                "if you want a plumbing check rather than a result.")
        train_clips, val_clips = clips, clips
    else:
        train_clips, val_clips = split_by_clip(clips, args.val_frac, args.seed)

    D = clips[0]["X"].shape[1]
    if D != frame_feature_dim():
        raise SystemExit(f"feature width {D} != current schema {frame_feature_dim()}")

    n_ego = sum(1 for c in clips if c["has_ego"])
    print(f"train {len(train_clips)} clips "
          f"({sum(1 for c in train_clips if c['label'] == 1)} pos) | "
          f"val {len(val_clips)} clips "
          f"({sum(1 for c in val_clips if c['label'] == 1)} pos)")
    print(f"feature width {D}")
    if n_ego == 0:
        print("!! NO clip carries ego-motion features (has_ego=False everywhere).\n"
              "   The ego slots are zeros, which is indistinguishable from a\n"
              "   stationary vehicle. Any result here is from a model blind to\n"
              "   ego motion and must be reported that way.")
    elif n_ego < len(clips):
        print(f"!! only {n_ego}/{len(clips)} clips carry ego features -- mixed "
              "inputs, the model cannot tell 'stationary' from 'not measured'.")
    if args.allow_leakage and same_dir:
        print("!! --allow-leakage: train == val. This is a PLUMBING CHECK, not a "
              "result. Do not quote any number below.")

    model = RiskGRU(D, hidden=args.hidden, layers=args.layers, dropout=args.dropout)
    assert_causal(model, D)
    print(f"RiskGRU hidden={args.hidden} layers={args.layers} "
          f"params={count_parameters(model):,} (causality check passed)")
    print(f"loss: kappa={args.kappa} pre_alert_weight={args.pre_alert_weight} "
          f"-> asks for a warning ~{expected_lead_time(0.0, 3.49, args.kappa):.2f}s "
          f"before impact on a mean-lead clip\n")

    # Save-on-improvement. Defined here (not inside train()) so train() stays a
    # pure loop and the persistence policy lives with the CLI that owns the paths.
    def checkpoint_extra(b):
        v = b["val"]
        return {
            "feature_dim": D,
            "has_ego": n_ego == len(clips),
            "n_clips_with_ego": n_ego,
            "kappa": args.kappa,
            "pre_alert_weight": args.pre_alert_weight,
            "threshold": args.threshold,
            "best_epoch": b["epoch"],
            "val": {k: (v[k] if not isinstance(v[k], dict) else
                        {str(a): c for a, c in v[k].items()}) for k in v},
            "leakage_warning": bool(args.allow_leakage and same_dir),
        }

    def save_best(b):
        if not args.out:
            return
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        state = model.state_dict()
        model.load_state_dict(b["state"])          # write the BEST weights...
        try:
            model.save(args.out, extra=checkpoint_extra(b))
        finally:
            model.load_state_dict(state)           # ...then restore current ones
        print(f"           saved best (epoch {b['epoch']}) -> {args.out}",
              flush=True)

    def save_history(h):
        if not args.history_json:
            return
        os.makedirs(os.path.dirname(args.history_json) or ".", exist_ok=True)
        with open(args.history_json, "w", encoding="utf-8") as fh:
            json.dump(h, fh, indent=2)

    t0 = time.perf_counter()
    best, history = train(model, train_clips, val_clips, args,
                          on_improve=save_best, on_epoch=save_history)
    print(f"\ntrained in {time.perf_counter() - t0:.0f}s")

    if best is None:
        print("no epoch produced a scorable validation result.")
        return

    model.load_state_dict(best["state"])
    v = best["val"]
    print(f"\nbest epoch {best['epoch']}:")
    print(f"  useful-warning rate : {_fmt(v['useful_warning_rate'])} "
          f"({v['n_useful']}/{v['n_timed']}; {v['n_too_early']} fired too early)")
    print(f"  false-alarm rate    : {_fmt(v['false_alarm_rate'])}")
    print(f"  mean lead time      : {_fmt(v['mean_lead_s'], 2)} s")
    for tau in sorted(v["AP"]):
        print(f"  AP @ -{tau * 1000:.0f} ms      : {_fmt(v['AP'][tau])}")
    print(f"  mean AP             : {_fmt(v['mean_AP'])}")

    # Both were already written during training (save-on-improvement above); these
    # final writes just guarantee the on-disk copy matches the returned best.
    if args.out:
        save_best(best)
    if args.history_json:
        save_history(history)
        print(f"wrote {args.history_json}")


if __name__ == "__main__":
    main()
