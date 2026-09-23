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
import statistics
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from a3ps.common.splits import (  # noqa: E402
    DEFAULT_FREEZE_PATH,
    HELD_OUT_SPLIT,
    held_out_clips_in,
    load_freeze,
)
from a3ps.features.extract import frame_feature_dim, load_features  # noqa: E402
from a3ps.risk.anticipation_loss import (  # noqa: E402
    DEFAULT_KAPPA,
    DEFAULT_PRE_ALERT_WEIGHT,
    batch_anticipation_loss,
    expected_lead_time,
    find_alert_episodes,
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


def guard_held_out(feature_dir, role, freeze_path, allowed, override_flag):
    """Refuse to touch the frozen held-out split for anything but final reporting.

    The pre-existing ``--allow-leakage`` guard only fired when ``--features``
    and ``--val-features`` were the *same directory*. ``train_all`` vs ``eval``
    are different directories, so it stayed silent while every checkpoint in
    this project was early-stopped and best-epoch-selected on the test set, and
    while the operating point, kappa and capacity calls were made on
    eval-scored numbers (``docs/status/eval_leakage_audit.md``).

    A number is only held-out if nothing upstream of it consulted these clips.
    So the default is refusal, and reading eval has to be an explicit, one-line
    statement of intent rather than the path of least resistance.
    """
    if not feature_dir:
        return
    hits = held_out_clips_in(feature_dir, load_freeze(freeze_path))
    if not hits:
        return
    if allowed:
        print(f"!! {override_flag}: {role} is {feature_dir}, which contains "
              f"{len(hits)} clip(s) frozen as '{HELD_OUT_SPLIT}'. This must be a "
              "FINAL REPORT on an already-fixed checkpoint and operating point. "
              "Anything selected after reading this is no longer held out.",
              flush=True)
        return
    raise SystemExit(
        f"{role} resolves to {feature_dir}, which contains {len(hits)} clip(s) "
        f"frozen as '{HELD_OUT_SPLIT}' in {freeze_path} (e.g. {hits[:5]}).\n"
        "Selecting anything against the held-out split -- best epoch, early "
        "stopping, threshold, kappa -- makes the final number a selection "
        "result, not a held-out one. Point this at data/features/train_val_v2 "
        f"instead.\nIf the checkpoint and operating point are already fixed and "
        f"this is the single final read, pass {override_flag}.")


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

GROSS_PREMATURE_SECONDS = 5.0
GROSS_PREMATURE_CLIP_FRACTION = 0.20


def evaluate(model, clips, threshold, cutoffs=(0.5, 1.0, 1.5), confirm=3,
            dump_rows=False):
    """Useful-warning rate, cutoff APs and lead time on a clip list.

    Every value this function returned before the corrected-definition work
    (``n_clips``, ``n_timed``, ``useful_warning_rate``, ``n_useful``,
    ``n_too_early``, ``false_alarm_rate``, ``mean_lead_s``, ``AP``,
    ``mean_AP``) is computed exactly as before and is the LEGACY definition:
    a clip is ``useful`` only if the very first sustained threshold crossing
    lands inside ``[alert_t, event_t]``. That definition, and every key name
    above, is preserved permanently -- see
    ``docs/design/useful_warning_definition.md`` -- because
    ``train()``'s epoch selection and every existing caller read these exact
    keys, and because every previously reported number in this project was
    measured under it.

    New keys, always present (not behind a flag, per
    ``docs/design/useful_warning_definition.md`` -- the two definitions are
    reported side by side, always, not opt-in): the corrected,
    episode-based definition, under which a clip is ``useful`` if ANY
    confirmed alert episode's active interval intersects
    ``[alert_t, event_t]``, not only the first one. ``n_useful_v2``,
    ``useful_warning_rate_v2``, ``mean_lead_vs_event_v2``,
    ``mean_lead_vs_alert_v2`` (means over v2-useful clips), and the
    prematurity axis: ``frac_premature`` (of timed positives, earliest
    confirmed episode anywhere in the clip begins before ``alert_t``),
    ``median_premature_s`` (median seconds early, over the premature ones),
    and ``n_gross_premature`` (earliest confirmed episode begins more than
    ``GROSS_PREMATURE_SECONDS`` before ``alert_t``, OR within the first
    ``GROSS_PREMATURE_CLIP_FRACTION`` of the clip's own duration -- the axis
    that keeps the old threshold engine's frame-0 firing failure mode
    visible regardless of which ``useful_warning_rate`` is being read).
    ``false_alarm_rate`` needs no v2 variant: a negative is a false alarm if
    it produces any confirmed episode at all, which is exactly what the
    existing ``fa is not None`` test already checks.

    ``dump_rows=True`` adds a ``"rows"`` key: one dict per clip carrying both
    definitions' verdicts and every quantity the aggregates above are
    computed from. It surfaces values this function already computes -- no
    scoring logic changes, and non-``dump_rows`` callers get the exact same
    dict shape they always did plus the always-present v2/prematurity keys.
    See ``tests/test_train_risk_head.py`` for tests that (a) the legacy keys
    reproduce pre-existing committed numbers bit for bit and (b) the dumped
    rows reproduce every aggregate exactly.
    """
    model.eval()
    timelines, rows = [], []
    with torch.no_grad():
        for c in clips:
            probs = torch.sigmoid(model(c["X"]))
            fa = first_alert_time(probs, c["t"], threshold, confirm)
            episodes = find_alert_episodes(probs, c["t"], threshold, confirm)
            peak_i = int(torch.argmax(probs))
            rows.append({"clip": c, "first_alert_t": fa, "episodes": episodes,
                         "peak": float(probs.max()),
                         "peak_t": float(c["t"][peak_i])})
            timelines.append(list(zip(c["t"].tolist(), probs.tolist())))

    useful = early = n_timed = 0
    leads, fa_neg = [], 0
    n_useful_v2 = 0
    leads_event_v2, leads_alert_v2 = [], []
    n_premature, premature_seconds, n_gross_premature = 0, [], 0
    detail = []
    for r in rows:
        c = r["clip"]
        te, ta, fa = c["event_time_s"], c["alert_time_s"], r["first_alert_t"]
        episodes = r["episodes"]
        window = (te - ta) if (te is not None and ta is not None) else None

        if c["label"] == 0:
            fa_neg += int(fa is not None)
            detail.append({
                "clip_id": c["clip_id"], "label": 0, "alert_t": ta, "event_t": te,
                "window_s": window, "first_fire_t": fa,
                "verdict": "false_alarm" if fa is not None else "clean",
                "verdict_v2": "false_alarm" if fa is not None else "clean",
                "lead_s": None, "offset_from_alert_s": None,
                "lead_vs_event_v2": None, "lead_vs_alert_v2": None,
                "premature": None, "gross_premature": None,
                "peak_prob": r["peak"], "peak_t": r["peak_t"]})
            continue
        if te is None or ta is None:
            detail.append({
                "clip_id": c["clip_id"], "label": 1, "alert_t": ta, "event_t": te,
                "window_s": window, "first_fire_t": fa, "verdict": "untimed",
                "verdict_v2": "untimed",
                "lead_s": None, "offset_from_alert_s": None,
                "lead_vs_event_v2": None, "lead_vs_alert_v2": None,
                "premature": None, "gross_premature": None,
                "peak_prob": r["peak"], "peak_t": r["peak_t"]})
            continue
        n_timed += 1
        if fa is None:
            verdict, lead = "missed", None
        elif fa > te:
            verdict, lead = "late", None
        elif fa < ta:
            verdict, lead = "too_early", (te - fa)
            early += 1
        else:
            verdict, lead = "useful", (te - fa)
            useful += 1
        if verdict in ("useful", "too_early"):
            leads.append(te - fa)

        # --- corrected definition: ANY episode's active interval [confirm_t,
        # end_t] intersecting [alert_t, event_t] makes the clip useful, not
        # only the first episode in the clip.
        intersecting = [e for e in episodes
                        if e["confirm_t"] <= te and e["end_t"] >= ta]
        if intersecting:
            earliest = min(intersecting, key=lambda e: e["confirm_t"])
            verdict_v2 = "useful"
            lead_event_v2 = te - earliest["confirm_t"]
            lead_alert_v2 = earliest["confirm_t"] - ta
            n_useful_v2 += 1
            leads_event_v2.append(lead_event_v2)
            leads_alert_v2.append(lead_alert_v2)
        else:
            verdict_v2 = "missed"
            lead_event_v2 = lead_alert_v2 = None

        # --- prematurity: earliest confirmed episode ANYWHERE in the clip
        # (not only ones intersecting the window), regardless of verdict.
        premature = gross_premature = False
        if episodes:
            earliest_overall = min(episodes, key=lambda e: e["confirm_t"])
            if earliest_overall["confirm_t"] < ta:
                premature = True
                early_by = ta - earliest_overall["confirm_t"]
                n_premature += 1
                premature_seconds.append(early_by)
                t0 = float(c["t"][0])
                t1 = float(c["t"][-1])
                dur = t1 - t0
                frac_in = ((earliest_overall["confirm_t"] - t0) / dur
                          if dur > 0 else 0.0)
                if (early_by > GROSS_PREMATURE_SECONDS
                        or frac_in < GROSS_PREMATURE_CLIP_FRACTION):
                    gross_premature = True
                    n_gross_premature += 1

        detail.append({
            "clip_id": c["clip_id"], "label": 1, "alert_t": ta, "event_t": te,
            "window_s": window, "first_fire_t": fa, "verdict": verdict,
            "verdict_v2": verdict_v2,
            "lead_s": lead,
            "offset_from_alert_s": (fa - ta) if fa is not None else None,
            "lead_vs_event_v2": lead_event_v2, "lead_vs_alert_v2": lead_alert_v2,
            "premature": premature, "gross_premature": gross_premature,
            "peak_prob": r["peak"], "peak_t": r["peak_t"]})

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
    out = {
        "n_clips": len(rows),
        "n_timed": n_timed,
        "useful_warning_rate": (useful / n_timed) if n_timed else float("nan"),
        "n_useful": useful,
        "n_too_early": early,
        "false_alarm_rate": (fa_neg / n_neg) if n_neg else float("nan"),
        "mean_lead_s": (sum(leads) / len(leads)) if leads else float("nan"),
        "AP": aps,
        "mean_AP": (sum(finite) / len(finite)) if finite else float("nan"),
        "n_useful_v2": n_useful_v2,
        "useful_warning_rate_v2": (n_useful_v2 / n_timed) if n_timed else float("nan"),
        "mean_lead_vs_event_v2": ((sum(leads_event_v2) / len(leads_event_v2))
                                  if leads_event_v2 else float("nan")),
        "mean_lead_vs_alert_v2": ((sum(leads_alert_v2) / len(leads_alert_v2))
                                  if leads_alert_v2 else float("nan")),
        "frac_premature": (n_premature / n_timed) if n_timed else float("nan"),
        "median_premature_s": (statistics.median(premature_seconds)
                               if premature_seconds else float("nan")),
        "n_gross_premature": n_gross_premature,
    }
    if dump_rows:
        out["rows"] = detail
    return out


def _fmt(x, nd=3):
    return "n/a" if x is None or x != x else f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

# Named feature groups for ablation, resolved against frame_feature_names() so a
# change to the feature layout can never silently ablate the wrong columns.
# Each entry is a predicate over a feature name.
ABLATION_GROUPS = {
    "ego": lambda n: n.startswith("ego_"),
    "collision_prob": lambda n: n.endswith("collision_prob") or n in ("mean_prob", "max_prob"),
    "ttc": lambda n: n.endswith("ttc_pred") or n.endswith("tau_area") or n.endswith("tau_h"),
    "corridor": lambda n: "corridor" in n,
    "kinematics": lambda n: any(n.endswith(s) for s in
                               ("_vx", "_vy", "_ax", "_ay", "_area_growth")),
    "appearance": lambda n: "_cls_" in n,
}


def resolve_ablation(spec):
    """Comma-separated group names -> (column indices, resolved names).

    Zeroing columns rather than removing them keeps the input width -- and
    therefore the architecture and parameter count -- identical across ablation
    arms, so a difference in the result is attributable to the information and
    not to a different-sized model.
    """
    from a3ps.features.extract import frame_feature_names

    if not spec:
        return [], []
    names = list(frame_feature_names())
    wanted = [g.strip() for g in spec.split(",") if g.strip()]
    unknown = [g for g in wanted if g not in ABLATION_GROUPS]
    if unknown:
        raise SystemExit(
            f"unknown ablation group(s) {unknown}; "
            f"available: {sorted(ABLATION_GROUPS)}")
    cols, hit = [], []
    for i, n in enumerate(names):
        if any(ABLATION_GROUPS[g](n) for g in wanted):
            cols.append(i)
            hit.append(n)
    if not cols:
        raise SystemExit(f"ablation {wanted} matched no feature columns")
    return cols, hit


def apply_ablation(clips, cols):
    """Zero the given feature columns in place, on every clip."""
    if not cols:
        return
    idx = torch.tensor(cols, dtype=torch.long)
    for c in clips:
        c["X"] = c["X"].clone()
        c["X"][:, idx] = 0.0


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


# How far above chance mean-AP has to be before a fa_target-noncompliant
# epoch is trusted as the run's selection, rather than treated as a likely
# collapse. Chance for AP on a clip set is its positive prevalence (a random
# ranking's expected AP equals the fraction of positives), not a fixed 0.5 --
# see ``train()``'s ``near_chance_ap`` computation. 0.05 was chosen because it
# cleanly separates every case seen so far: a genuinely undertrained epoch
# (mean AP 0.501 on a ~50/50 val set, effectively chance) from a real one
# (mean AP 0.59-0.68) by more than an order of magnitude of margin.
NEAR_CHANCE_AP_MARGIN = 0.05


def train(model, train_clips, val_clips, args, on_improve=None, on_epoch=None):
    """Train the head, returning ``(best, history, fallback_used)``.

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

    Checkpoint selection and its fallback
    --------------------------------------
    The primary rule ranks epochs by ``(meets fa_target, useful-warning, -FA)``
    (see the per-epoch loop below). When NO epoch in a run ever meets
    ``fa_target`` -- which turned out to be every run seen so far at the
    training loop's own scoring point -- that rule degrades to ranking purely
    by useful-warning with FA as a tiebreak, and useful-warning alone cannot
    tell a genuinely well-trained epoch from an undertrained one that happens
    to fire conservatively. On seed 1236 that picked epoch 1 (mean AP 0.501,
    chance) over epoch 6 (mean AP 0.648, real separation), because epoch 1's
    lower confidence gave it a marginally better useful/FA pair despite
    carrying almost no label information.

    The fix is a second, independent running best -- ``best_by_ap``, tracked by
    mean AP alone, never consulted for early stopping or the primary pick --
    used ONLY as a fallback, and only when both are true: (a) the primary pick
    never met ``fa_target`` during the run, and (b) the primary pick's own mean
    AP is within :data:`NEAR_CHANCE_AP_MARGIN` of chance. This deliberately
    does NOT touch runs like the ones that produced seeds 1234/1235 as
    currently committed: those also never met ``fa_target``, but their
    useful-first picks (mean AP 0.59-0.63) are nowhere near chance, so
    condition (b) never fires and they are returned unchanged. Swapping in
    ``best_by_ap`` sets ``fallback_used=True`` (also stamped in the checkpoint's
    ``extra`` dict as ``selection_fallback_used``) and re-fires ``on_improve``
    so the saved checkpoint on disk reflects the swap, not the near-chance pick
    that was "best" for most of the run.
    """
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    best = None
    best_by_ap = None              # independent running best, AP-ranked only;
                                    # consulted solely by the fallback below.
    history = []
    stale = 0                      # epochs since the last improvement

    n_pos_val = sum(1 for c in val_clips if c["label"] == 1)
    n_neg_val = sum(1 for c in val_clips if c["label"] == 0)
    n_val = n_pos_val + n_neg_val
    chance_ap = (n_pos_val / n_val) if n_val else 0.5
    near_chance_ap = chance_ap + NEAR_CHANCE_AP_MARGIN

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

        # Independent of the primary pick above: track the best-by-mean-AP
        # epoch too, purely as a fallback candidate (see the function
        # docstring). Never affects early stopping or on_improve on its own.
        ap = val["mean_AP"]
        if ap == ap and (best_by_ap is None or ap > best_by_ap["val"]["mean_AP"]):
            best_by_ap = {"epoch": epoch, "val": val,
                         "state": {k: v.clone() for k, v in model.state_dict().items()}}

        # flush=True: without it a redirected/piped run shows nothing until the
        # process exits, which makes a long run impossible to monitor.
        print(f"epoch {epoch:3d}  loss {train_loss:.4f}  "
              f"val useful {_fmt(val['useful_warning_rate'])} "
              f"({val['n_useful']}/{val['n_timed']}, {val['n_too_early']} early)  "
              f"mAP {_fmt(val['mean_AP'])}  "
              f"FA {_fmt(val['false_alarm_rate'])}  "
              f"lead {_fmt(val['mean_lead_s'], 2)}s"
              f"{'  [fa-ok]' if ok_fa else ''}"
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

    fallback_used = False
    if (best is not None and best["rank"][0] == 0
            and best["val"]["mean_AP"] == best["val"]["mean_AP"]
            and best["val"]["mean_AP"] < near_chance_ap
            and best_by_ap is not None and best_by_ap["epoch"] != best["epoch"]):
        fallback_used = True
        print(f"\n!! selection fallback: epoch {best['epoch']}'s mean AP "
              f"({_fmt(best['val']['mean_AP'])}) is within "
              f"{NEAR_CHANCE_AP_MARGIN} of chance ({_fmt(chance_ap)}) on this "
              f"val set, and no epoch ever met fa_target. Using epoch "
              f"{best_by_ap['epoch']} (mean AP {_fmt(best_by_ap['val']['mean_AP'])}) "
              "instead -- see NEAR_CHANCE_AP_MARGIN.", flush=True)
        best = {"score": best_by_ap["val"]["useful_warning_rate"],
               "rank": best["rank"], "epoch": best_by_ap["epoch"],
               "val": best_by_ap["val"], "state": best_by_ap["state"]}
        if on_improve is not None:
            on_improve(best)

    return best, history, fallback_used


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
    p.add_argument("--ablate", default="",
                   help="Comma-separated feature groups to ZERO OUT, for the "
                        "ablation table: ego, collision_prob, ttc, corridor, "
                        "kinematics, appearance. Columns are zeroed rather than "
                        "removed so the input width, architecture and parameter "
                        "count stay identical across arms.")
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
    p.add_argument("--split-freeze", default=DEFAULT_FREEZE_PATH,
                   help="Frozen split membership consulted by the held-out "
                        "guard (default %(default)s).")
    p.add_argument("--final-eval-report", action="store_true",
                   help="Permit --features/--val-features to contain clips "
                        "frozen as the held-out split. Only legitimate when the "
                        "checkpoint and operating point are ALREADY fixed and "
                        "this is the single final read.")
    args = p.parse_args()

    guard_held_out(args.features, "--features", args.split_freeze,
                   args.final_eval_report, "--final-eval-report")
    guard_held_out(args.val_features, "--val-features", args.split_freeze,
                   args.final_eval_report, "--final-eval-report")

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

    abl_cols, abl_names = resolve_ablation(args.ablate)
    if abl_cols:
        # Ablate validation too. With --val-features it is a separate list from
        # `clips`; zeroing only the training side trains on the ablated inputs
        # but scores on intact ones, which is a train/eval shift, not an
        # ablation. Zeroing is idempotent, so overlap between the lists is safe.
        apply_ablation(clips, abl_cols)
        apply_ablation(val_clips, abl_cols)
        print(f"!! ABLATION '{args.ablate}': zeroed {len(abl_cols)} of {D} feature "
              f"columns -> {abl_names[:6]}{' ...' if len(abl_names) > 6 else ''}")
        print("   Input width and parameter count are unchanged; any difference in "
              "the result is attributable to the missing information.")

    model = RiskGRU(D, hidden=args.hidden, layers=args.layers, dropout=args.dropout)
    assert_causal(model, D)
    print(f"RiskGRU hidden={args.hidden} layers={args.layers} "
          f"params={count_parameters(model):,} (causality check passed)")
    print(f"loss: kappa={args.kappa} pre_alert_weight={args.pre_alert_weight} "
          f"-> asks for a warning ~{expected_lead_time(0.0, 3.49, args.kappa):.2f}s "
          f"before impact on a mean-lead clip\n")

    # Placeholder until train() returns the real value below. checkpoint_extra
    # reads this by closure (Python resolves free variables at call time, not
    # def time), so the mid-training on_improve writes see this False
    # placeholder -- correct, since the fallback can only be decided once the
    # whole run is known -- and the final save_best() call after train()
    # returns sees the true value once `fallback_used` is reassigned.
    fallback_used = False

    # Save-on-improvement. Defined here (not inside train()) so train() stays a
    # pure loop and the persistence policy lives with the CLI that owns the paths.
    def checkpoint_extra(b):
        v = b["val"]
        return {
            "feature_dim": D,
            "has_ego": n_ego == len(clips),
            "n_clips_with_ego": n_ego,
            "kappa": args.kappa,
            "ablate": args.ablate or None,
            "ablated_columns": abl_cols or None,
            "pre_alert_weight": args.pre_alert_weight,
            "threshold": args.threshold,
            "best_epoch": b["epoch"],
            "val": {k: (v[k] if not isinstance(v[k], dict) else
                        {str(a): c for a, c in v[k].items()}) for k in v},
            "leakage_warning": bool(args.allow_leakage and same_dir),
            # Whether ANY epoch in this run met --fa-target. False means the
            # primary (fa_target, useful-warning, -FA) rank rule fell back to
            # comparing on useful-warning alone across every epoch -- see
            # train()'s docstring.
            "fa_target_met": bool(b["rank"][0] == 1),
            # True only if that fallback ALSO produced a near-chance pick,
            # triggering the mean-AP-based override. False (including on runs
            # where fa_target_met is also False) means the useful-first
            # fallback pick was kept because its mean AP was not near chance.
            "selection_fallback_used": bool(fallback_used),
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
    best, history, fallback_used = train(model, train_clips, val_clips, args,
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
    print(f"  fa_target ever met  : {bool(best['rank'][0] == 1)}")
    print(f"  selection fallback  : {fallback_used}"
          f"{' (near-chance override fired)' if fallback_used else ''}")

    # Both were already written during training (save-on-improvement above); these
    # final writes just guarantee the on-disk copy matches the returned best.
    if args.out:
        save_best(best)
    if args.history_json:
        save_history(history)
        print(f"wrote {args.history_json}")


if __name__ == "__main__":
    main()
