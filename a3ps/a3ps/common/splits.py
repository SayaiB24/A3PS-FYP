"""Frozen dev/eval split membership.

Why this module exists
----------------------
``scripts/prepare_nexar.py`` assigns splits with ``random.Random(SEED)`` +
``shuffle`` over the *remaining* clip pool. The shuffle order therefore depends
on how many clips are in the pool, so adding clips and re-running the script
**silently re-draws the held-out eval split**. That would invalidate every
number already measured against it (and the cached pipeline output under
``eval/anticipation/``) without changing a single line of code -- the kind of
error that is invisible in review and fatal in a writeup.

The fix is to record dev/eval membership once, in a file that lives OUTSIDE the
gitignored ``data/`` tree so it travels with the repo, and to make split
assignment consult it. Clips named in the freeze keep their recorded split
forever; clips not named in it can only ever land in ``train_traj`` (or stay
unused). Growing the dataset can then add training data but can never move a
clip into or out of the held-out set.

The freeze stores a label alongside each clip id purely as a tripwire: if a
clip's label ever disagrees with the freeze, the labels table changed under us
and the run should stop rather than quietly score against a different task.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Splits that are frozen once recorded. ``train_traj`` is deliberately absent:
# it is the only split allowed to grow.
FROZEN_SPLITS = ("dev", "eval")

DEFAULT_FREEZE_PATH = os.path.join("eval", "split_freeze.json")

SCHEMA_VERSION = 1


class SplitFreezeError(RuntimeError):
    """Raised when the live index contradicts the frozen split."""


def normalize_clip_id(clip_id: Any) -> str:
    """Canonical clip-id string: digits without leading zeros where numeric.

    The project writes clip ids three ways -- ``"00584"`` in labels.xlsx,
    ``"584"`` in index.csv, and ``584`` as an int in places -- so every lookup
    goes through here first. Non-numeric ids are passed through stripped.
    """
    s = str(clip_id).strip()
    if not s:
        return ""
    try:
        return str(int(s))
    except ValueError:
        return s


def _sort_key(cid: str):
    """Numeric ids sort numerically; anything else sorts after, lexically."""
    return (0, int(cid), "") if cid.isdigit() else (1, 0, cid)


# ---------------------------------------------------------------------------
# build / read
# ---------------------------------------------------------------------------

def build_freeze(records: Iterable[Dict[str, Any]], note: str = "") -> Dict[str, Any]:
    """Snapshot the dev/eval membership of ``records`` into a freeze dict.

    ``records`` are index rows (dicts with ``clip_id``, ``label``, ``split``).
    Only rows whose split is in :data:`FROZEN_SPLITS` are recorded.
    """
    clips: Dict[str, Dict[str, Any]] = {}
    for r in records:
        split = str(r.get("split") or "")
        if split not in FROZEN_SPLITS:
            continue
        cid = normalize_clip_id(r.get("clip_id"))
        if not cid:
            continue
        clips[cid] = {"split": split, "label": int(r.get("label") or 0)}

    counts: Dict[str, Dict[str, int]] = {}
    for meta in clips.values():
        c = counts.setdefault(meta["split"], {"pos": 0, "neg": 0})
        c["pos" if meta["label"] == 1 else "neg"] += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "note": note or (
            "Frozen dev/eval membership. Clips listed here keep this split "
            "permanently; clips absent from this file may only join train_traj. "
            "See a3ps/common/splits.py."
        ),
        "counts": counts,
        "clips": dict(sorted(clips.items(), key=lambda kv: _sort_key(kv[0]))),
    }


def save_freeze(freeze: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(freeze, fh, indent=2, sort_keys=False)
        fh.write("\n")


def load_freeze(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Read a freeze file, or return None when ``path`` is falsy/absent."""
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        freeze = json.load(fh)
    if int(freeze.get("schema_version", 0)) != SCHEMA_VERSION:
        raise SplitFreezeError(
            "{}: schema_version {} != {} -- refusing to guess at an older "
            "layout.".format(path, freeze.get("schema_version"), SCHEMA_VERSION)
        )
    if not isinstance(freeze.get("clips"), dict):
        raise SplitFreezeError("{}: missing or malformed 'clips' map.".format(path))
    return freeze


# ---------------------------------------------------------------------------
# apply / verify
# ---------------------------------------------------------------------------

def apply_freeze(records: List[Dict[str, Any]],
                 freeze: Dict[str, Any],
                 growable_split: str = "train_traj") -> Dict[str, Any]:
    """Overwrite each record's ``split`` from the freeze, in place.

    * A clip named in the freeze gets its frozen split.
    * A clip absent from the freeze gets ``growable_split`` if it is a negative,
      or ``""`` (unused) if it is a positive.

    Positives are deliberately NOT auto-added to any split: every positive is
    scarce and belongs to a deliberate decision, not to a default. New positives
    surface in the returned report under ``unpinned_positives`` so the caller
    can assign them explicitly.

    Raises :class:`SplitFreezeError` if a frozen clip's label changed, or if a
    frozen clip is missing from ``records`` entirely -- both mean the held-out
    set is no longer what was measured.
    """
    frozen = freeze.get("clips", {})
    by_id = {normalize_clip_id(r.get("clip_id")): r for r in records}

    missing = sorted(set(frozen) - set(by_id), key=_sort_key)
    if missing:
        raise SplitFreezeError(
            "{} frozen clip(s) absent from the index: {}{}. The held-out split "
            "cannot be reconstructed -- restore those clips (or their index "
            "rows) before rebuilding.".format(
                len(missing), missing[:10], " ..." if len(missing) > 10 else "")
        )

    relabelled = [cid for cid, meta in frozen.items()
                  if int(by_id[cid].get("label") or 0) != int(meta["label"])]
    if relabelled:
        relabelled.sort(key=_sort_key)
        raise SplitFreezeError(
            "{} frozen clip(s) changed label: {}{}. The labels table no longer "
            "matches what the frozen split was drawn from.".format(
                len(relabelled), relabelled[:10],
                " ..." if len(relabelled) > 10 else "")
        )

    unpinned_negatives: List[str] = []
    unpinned_positives: List[str] = []
    for rec in records:
        cid = normalize_clip_id(rec.get("clip_id"))
        meta = frozen.get(cid)
        if meta is not None:
            rec["split"] = meta["split"]
            continue
        if int(rec.get("label") or 0) == 1:
            rec["split"] = ""
            unpinned_positives.append(cid)
        else:
            rec["split"] = growable_split
            unpinned_negatives.append(cid)

    counts: Dict[str, int] = {}
    for rec in records:
        key = rec["split"] or "unused"
        counts[key] = counts.get(key, 0) + 1

    return {
        "n_frozen": len(frozen),
        # "unpinned" = not named in the freeze. That includes clips already
        # sitting in train_traj, not only newly-downloaded ones.
        "n_unpinned_negatives": len(unpinned_negatives),
        "n_unpinned_positives": len(unpinned_positives),
        "unpinned_positives": sorted(unpinned_positives, key=_sort_key),
        "counts": counts,
    }


def verify_freeze(records: Iterable[Dict[str, Any]],
                  freeze: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Check ``records`` already agree with the freeze. Returns (ok, problems).

    Read-only -- nothing is mutated. Use this in tests and as a pre-flight guard
    before scoring, so a drifted index is caught before it produces numbers that
    look fine.
    """
    frozen = freeze.get("clips", {})
    by_id = {normalize_clip_id(r.get("clip_id")): r for r in records}
    problems: List[str] = []

    for cid in sorted(frozen, key=_sort_key):
        meta = frozen[cid]
        rec = by_id.get(cid)
        if rec is None:
            problems.append(
                "clip {}: frozen as {} but absent from index".format(cid, meta["split"]))
            continue
        actual = str(rec.get("split") or "")
        if actual != meta["split"]:
            problems.append("clip {}: frozen as {} but index says {}".format(
                cid, meta["split"], actual or "(unused)"))
        if int(rec.get("label") or 0) != int(meta["label"]):
            problems.append("clip {}: frozen label {} but index says {}".format(
                cid, meta["label"], rec.get("label")))

    for cid in sorted(by_id, key=_sort_key):
        actual = str(by_id[cid].get("split") or "")
        if actual in FROZEN_SPLITS and cid not in frozen:
            problems.append(
                "clip {}: index says {} but it is not in the freeze (would "
                "silently enlarge the held-out set)".format(cid, actual))

    return (not problems), problems
