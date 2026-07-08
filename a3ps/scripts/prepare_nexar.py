#!/usr/bin/env python
"""Index Nexar clips, probe them with OpenCV, and build stratified splits.

Layout expected under ``--root`` (default data/nexar)::

    data/nexar/
      positive/   collision / near-collision clips  (label 1)
      negative/   normal-driving clips              (label 0)
      <annotation file>   optional CSV/JSON with per-clip event times

The exact annotation format that ships with Nexar varies, so this script
*inspects* the root for any ``*.csv`` / ``*.json`` and tries to auto-detect an
id column and an event-time column (common names below). If none is found,
``event_time_s`` is left blank and a note is printed — nothing crashes.

Outputs:
  * data/nexar/index.csv  with columns:
      clip_id, path, label, event_time_s, duration_s, fps, width, height, split
  * data/dev_clips/dev01.mp4 .. devNN.mp4  (copies of the dev split)
  * data/dev_clips/README.md  mapping devNN -> original clip

Splits (stratified, deterministic):
  * dev        : 10 negatives + 5 positives, the SHORTEST clips
  * eval       : 60 positives + 60 negatives, held out — NEVER used for tuning
  * train_traj : all remaining negatives (for trajectory mining)
  * (leftover positives get a blank split; they are unused)

    python scripts/prepare_nexar.py --root data/nexar
"""

import argparse
import csv
import json
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Candidate column names when auto-detecting an annotation table.
ID_KEYS = ["clip_id", "id", "video", "video_id", "filename", "file", "name", "clip"]
EVENT_KEYS = [
    "event_time_s", "event_time", "time_of_event", "event_s", "collision_time",
    "time_of_alert", "alert_time", "accident_time", "t_event",
]

# Split sizes (clamped down when the dataset is smaller).
DEV_NEG, DEV_POS = 10, 5
EVAL_NEG, EVAL_POS = 60, 60
SEED = 1234


# ---------------------------------------------------------------------------
# discovery + probing
# ---------------------------------------------------------------------------

def find_videos(folder):
    """Return sorted list of video paths under ``folder`` (recursive)."""
    out = []
    if not os.path.isdir(folder):
        return out
    for dirpath, _, names in os.walk(folder):
        for n in names:
            if os.path.splitext(n)[1].lower() in VIDEO_EXTS:
                out.append(os.path.join(dirpath, n))
    return sorted(out)


def probe(path):
    """Return dict(duration_s, fps, width, height) via OpenCV; safe on failure."""
    try:
        import cv2
    except ImportError:
        return {"duration_s": "", "fps": "", "width": "", "height": ""}

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"duration_s": "", "fps": "", "width": "", "height": ""}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = round(frames / fps, 3) if fps > 0 and frames > 0 else ""
    return {
        "duration_s": duration,
        "fps": round(fps, 3) if fps else "",
        "width": width or "",
        "height": height or "",
    }


# ---------------------------------------------------------------------------
# annotation auto-detection
# ---------------------------------------------------------------------------

def _norm_id(value):
    """Normalize an id/filename to a bare clip id (stem, lowercased)."""
    s = str(value).strip().replace("\\", "/").split("/")[-1]
    return os.path.splitext(s)[0].lower()


def _pick_key(fieldnames, candidates):
    lower = {f.lower(): f for f in fieldnames}
    for c in candidates:
        if c in lower:
            return lower[c]
    # substring fallback
    for f in fieldnames:
        fl = f.lower()
        if any(c in fl for c in candidates):
            return f
    return None


def load_event_times(root):
    """Search ``root`` for a CSV/JSON annotation and return {clip_id: event_time_s}.

    Returns ({}, note) if nothing usable is found.
    """
    ann_files = []
    for n in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        p = os.path.join(root, n)
        if os.path.isfile(p) and os.path.splitext(n)[1].lower() in (".csv", ".json"):
            ann_files.append(p)

    for p in ann_files:
        try:
            if p.lower().endswith(".csv"):
                mapping = _parse_csv_annotation(p)
            else:
                mapping = _parse_json_annotation(p)
        except Exception as exc:  # noqa: BLE001 - inspection helper, stay resilient
            print(f"  ! could not parse {os.path.basename(p)}: {exc}")
            continue
        if mapping:
            return mapping, f"using annotation file: {os.path.basename(p)}"

    return {}, "no usable annotation file found; event_time_s left blank"


def _to_seconds(val):
    try:
        return round(float(val), 3)
    except (TypeError, ValueError):
        return ""


def _parse_csv_annotation(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return {}
        id_key = _pick_key(reader.fieldnames, ID_KEYS)
        ev_key = _pick_key(reader.fieldnames, EVENT_KEYS)
        if id_key is None or ev_key is None:
            return {}
        out = {}
        for row in reader:
            cid = _norm_id(row.get(id_key, ""))
            if cid:
                out[cid] = _to_seconds(row.get(ev_key))
        return out


def _parse_json_annotation(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    rows = data if isinstance(data, list) else data.get("annotations", data.get("clips", []))
    if isinstance(rows, dict):
        # dict keyed by clip id -> {event_time: ...}
        out = {}
        for k, v in rows.items():
            ev = v.get("event_time_s", v.get("event_time")) if isinstance(v, dict) else v
            out[_norm_id(k)] = _to_seconds(ev)
        return out
    out = {}
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        id_key = _pick_key(list(r.keys()), ID_KEYS)
        ev_key = _pick_key(list(r.keys()), EVENT_KEYS)
        if id_key and ev_key:
            out[_norm_id(r[id_key])] = _to_seconds(r.get(ev_key))
    return out


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------

def _dur(rec):
    """Sort key: numeric duration, unknowns last."""
    d = rec["duration_s"]
    return d if isinstance(d, (int, float)) else float("inf")


def assign_splits(records):
    """Mutate each record's ``split`` field per the stratified policy."""
    for r in records:
        r["split"] = ""

    pos = [r for r in records if r["label"] == 1]
    neg = [r for r in records if r["label"] == 0]

    # dev: shortest clips.
    dev_neg = sorted(neg, key=_dur)[:min(DEV_NEG, len(neg))]
    dev_pos = sorted(pos, key=_dur)[:min(DEV_POS, len(pos))]
    dev_ids = {id(r) for r in dev_neg + dev_pos}
    for r in dev_neg + dev_pos:
        r["split"] = "dev"

    # eval: deterministic held-out sample from what's left.
    rng = random.Random(SEED)
    rem_neg = [r for r in neg if id(r) not in dev_ids]
    rem_pos = [r for r in pos if id(r) not in dev_ids]
    rng.shuffle(rem_neg)
    rng.shuffle(rem_pos)
    eval_neg = rem_neg[:min(EVAL_NEG, len(rem_neg))]
    eval_pos = rem_pos[:min(EVAL_POS, len(rem_pos))]
    eval_ids = {id(r) for r in eval_neg + eval_pos}
    for r in eval_neg + eval_pos:
        r["split"] = "eval"

    # train_traj: every remaining negative.
    for r in neg:
        if id(r) not in dev_ids and id(r) not in eval_ids:
            r["split"] = "train_traj"

    # leftover positives keep split="" (unused).
    return {
        "dev": len(dev_neg) + len(dev_pos),
        "eval": len(eval_neg) + len(eval_pos),
        "train_traj": sum(1 for r in records if r["split"] == "train_traj"),
        "unused": sum(1 for r in records if r["split"] == ""),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

FIELDS = ["clip_id", "path", "label", "event_time_s",
          "duration_s", "fps", "width", "height", "split"]


def build_records(root, event_times):
    records = []
    for label, sub in ((1, "positive"), (0, "negative")):
        for path in find_videos(os.path.join(root, sub)):
            clip_id = _norm_id(path)
            meta = probe(path)
            rel = os.path.relpath(path, start=os.path.dirname(root) or ".")
            ev = event_times.get(clip_id, "") if label == 1 else ""
            records.append({
                "clip_id": clip_id,
                "path": rel.replace("\\", "/"),
                "label": label,
                "event_time_s": ev,
                **meta,
            })
    return records


def copy_dev_clips(records, dev_dir):
    os.makedirs(dev_dir, exist_ok=True)
    dev = [r for r in records if r["split"] == "dev"]
    # negatives first, then positives; shortest first within each.
    dev.sort(key=lambda r: (r["label"], _dur(r)))
    mapping = []
    for i, r in enumerate(dev, start=1):
        name = f"dev{i:02d}.mp4"
        # r["path"] is relative to the parent of root (e.g. "nexar/negative/x.mp4"
        # under data/). Resolve it against data/ (the parent of dev_dir).
        src = os.path.normpath(os.path.join(dev_dir, "..", r["path"]))
        dst = os.path.join(dev_dir, name)
        if os.path.isfile(src):
            shutil.copyfile(src, dst)
        mapping.append((name, r["clip_id"], "positive" if r["label"] == 1 else "negative",
                        r["duration_s"]))
    return mapping


def write_dev_readme(dev_dir, mapping):
    lines = [
        "# Dev clips",
        "",
        "The 15 shortest clips (10 negative + 5 positive), copied and renamed for",
        "daily development. Generated by `scripts/prepare_nexar.py`.",
        "",
        "| dev file | original clip_id | label | duration_s |",
        "|----------|------------------|-------|------------|",
    ]
    for name, clip_id, label, dur in mapping:
        lines.append(f"| {name} | {clip_id} | {label} | {dur} |")
    lines.append("")
    with open(os.path.join(dev_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description="Index Nexar clips and build splits.")
    p.add_argument("--root", default="data/nexar", help="Nexar root directory.")
    p.add_argument("--dev-dir", default="data/dev_clips", help="Where to copy dev clips.")
    args = p.parse_args()

    if not os.path.isdir(args.root):
        p.error(f"root not found: {args.root}")

    event_times, note = load_event_times(args.root)
    print(f"Annotations: {note}")

    records = build_records(args.root, event_times)
    n_pos = sum(1 for r in records if r["label"] == 1)
    n_neg = sum(1 for r in records if r["label"] == 0)
    print(f"Found {len(records)} clips: {n_pos} positive, {n_neg} negative.")
    if not records:
        print("No videos found under positive/ or negative/. "
              "Place clips there and re-run.")

    counts = assign_splits(records)
    print(f"Splits: dev={counts['dev']} eval={counts['eval']} "
          f"train_traj={counts['train_traj']} unused={counts['unused']}")
    for name in ("dev", "eval"):
        want = (DEV_NEG + DEV_POS) if name == "dev" else (EVAL_NEG + EVAL_POS)
        if counts[name] < want:
            print(f"  ! {name}: only {counts[name]} clips available (wanted {want}).")

    out_csv = os.path.join(args.root, "index.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(records)
    print(f"Wrote {out_csv}")

    mapping = copy_dev_clips(records, args.dev_dir)
    if mapping:
        write_dev_readme(args.dev_dir, mapping)
        print(f"Copied {len(mapping)} dev clips -> {args.dev_dir} (+ README.md)")


if __name__ == "__main__":
    main()
