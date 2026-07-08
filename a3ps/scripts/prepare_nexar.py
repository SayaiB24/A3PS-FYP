#!/usr/bin/env python
"""Index Nexar clips from an Excel label table, probe them, and build splits.

Nexar clips are labelled by an Excel/CSV (`id, time_of_event, time_of_alert,
target`), not by folder. This script reads that table, matches each row to a
video file by `id`, labels it from `target` (1=positive, 0=negative), pulls the
event/alert times, probes the video with OpenCV, and writes a master index.csv
plus stratified dev/eval/train_traj splits.

Layout (input — flat pool + table, NOT pre-sorted):

    data/nexar/
      videos/          all clips, flat, named by id (e.g. 00584.mp4)
      labels.xlsx      id, time_of_event, time_of_alert, target
                       (multiple .xlsx/.csv/.json are merged)

Outputs:
  * data/nexar/index.csv  (master lookup; columns below)
  * data/dev_clips/dev01.mp4 .. devNN.mp4  (+ README.md)

Splits (stratified, deterministic):
  * dev        : 10 negatives + 5 positives, the SHORTEST clips
  * eval       : 60 positives + 60 negatives, held out — NEVER used for tuning
  * train_traj : all remaining negatives (for trajectory mining)
  * leftover positives get a blank split (unused)

    python scripts/prepare_nexar.py --root data/nexar
    python scripts/prepare_nexar.py --videos D:/nexar/train D:/nexar/test \
        --annotation D:/nexar/train.xlsx D:/nexar/test.xlsx --root data/nexar
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
TABLE_EXTS = {".xlsx", ".xls", ".csv", ".json"}

ID_KEYS = ["id", "clip_id", "video", "video_id", "filename", "file", "name", "clip"]
EVENT_KEYS = ["time_of_event", "event_time_s", "event_time", "collision_time",
              "accident_time", "t_event"]
ALERT_KEYS = ["time_of_alert", "alert_time", "t_alert"]
TARGET_KEYS = ["target", "label", "y"]

DEV_NEG, DEV_POS = 10, 5
EVAL_NEG, EVAL_POS = 60, 60
SEED = 1234

FIELDS = ["clip_id", "path", "label", "event_time_s", "alert_time_s",
          "duration_s", "fps", "width", "height", "split"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def canon(value):
    """Canonical id: strip, drop a trailing .0, and normalize numeric strings
    so '00584', 584, '584.0' all collapse to '584' (matches zero-padded files)."""
    s = str(value).strip().replace("\\", "/").split("/")[-1]
    s = os.path.splitext(s)[0]
    if s.endswith(".0"):
        s = s[:-2]
    return str(int(s)) if s.lstrip("-").isdigit() else s.lower()


def _pick(fieldnames, candidates):
    lower = {str(f).strip().lower(): f for f in fieldnames if f is not None}
    for c in candidates:
        if c in lower:
            return lower[c]
    for f in fieldnames:
        if f is not None and any(c in str(f).strip().lower() for c in candidates):
            return f
    return None


def _to_seconds(val):
    try:
        if val is None or val == "":
            return ""
        return round(float(val), 3)
    except (TypeError, ValueError):
        return ""


def find_videos(folder):
    out = []
    if not os.path.isdir(folder):
        return out
    for dp, _, names in os.walk(folder):
        for n in names:
            if os.path.splitext(n)[1].lower() in VIDEO_EXTS:
                out.append(os.path.join(dp, n))
    return sorted(out)


def probe(path):
    try:
        import cv2
    except ImportError:
        return {"duration_s": "", "fps": "", "width": "", "height": ""}
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"duration_s": "", "fps": "", "width": "", "height": ""}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    dur = round(frames / fps, 3) if fps > 0 and frames > 0 else ""
    return {"duration_s": dur, "fps": round(fps, 3) if fps else "",
            "width": w or "", "height": h or ""}


# ---------------------------------------------------------------------------
# annotation table loading (xlsx / csv / json), returns list of row dicts:
#   {"id", "target", "event_time_s", "alert_time_s"}
# ---------------------------------------------------------------------------

def _rows_from_header(header, data_rows):
    id_k = _pick(header, ID_KEYS)
    tgt_k = _pick(header, TARGET_KEYS)
    ev_k = _pick(header, EVENT_KEYS)
    al_k = _pick(header, ALERT_KEYS)
    if id_k is None:
        return []
    idx = {name: i for i, name in enumerate(header)}
    out = []
    for r in data_rows:
        def get(k):
            return r[idx[k]] if (k is not None and idx.get(k) is not None and idx[k] < len(r)) else None
        tgt = get(tgt_k)
        try:
            label = int(float(tgt)) if tgt not in (None, "") else None
        except (TypeError, ValueError):
            label = None
        out.append({
            "id": get(id_k),
            "target": label,
            "event_time_s": _to_seconds(get(ev_k)),
            "alert_time_s": _to_seconds(get(al_k)),
        })
    return out


def load_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        rows = list(wb.active.iter_rows(values_only=True))
        if not rows:
            return []
        header = [str(h).strip() if h is not None else "" for h in rows[0]]
        return _rows_from_header(header, rows[1:])
    if ext == ".csv":
        with open(path, newline="", encoding="utf-8-sig") as fh:
            r = list(csv.reader(fh))
        if not r:
            return []
        return _rows_from_header(r[0], r[1:])
    if ext == ".json":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        recs = data if isinstance(data, list) else data.get("annotations", data.get("clips", []))
        if not recs:
            return []
        header = list(recs[0].keys())
        return _rows_from_header(header, [[rec.get(k) for k in header] for rec in recs])
    return []


def find_tables(root, explicit):
    if explicit:
        return [p for p in explicit if os.path.isfile(p)]
    out = []
    for n in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        if os.path.splitext(n)[1].lower() in TABLE_EXTS and n != "index.csv":
            out.append(os.path.join(root, n))
    return out


# ---------------------------------------------------------------------------
# record building
# ---------------------------------------------------------------------------

def build_records(video_dirs, tables, index_dir):
    # Build a canon-id -> path lookup across all video dirs.
    lookup = {}
    for d in video_dirs:
        for p in find_videos(d):
            lookup.setdefault(canon(p), p)

    # Merge all annotation rows (later tables don't override earlier ids).
    seen, merged = set(), []
    for t in tables:
        for row in load_table(t):
            cid = canon(row["id"]) if row["id"] is not None else None
            if cid and cid not in seen:
                seen.add(cid)
                merged.append((cid, row))

    records, missing = [], []
    for cid, row in merged:
        vid = lookup.get(cid)
        if vid is None:
            missing.append(cid)
            continue
        if row["target"] is None:
            continue
        meta = probe(vid)
        rel = os.path.relpath(vid, start=index_dir).replace("\\", "/")
        records.append({
            "clip_id": cid,
            "path": rel,
            "label": int(row["target"]),
            "event_time_s": row["event_time_s"],
            "alert_time_s": row["alert_time_s"],
            **meta,
        })
    return records, missing, lookup


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------

def _dur(rec):
    d = rec["duration_s"]
    return d if isinstance(d, (int, float)) else float("inf")


def assign_splits(records):
    for r in records:
        r["split"] = ""
    pos = [r for r in records if r["label"] == 1]
    neg = [r for r in records if r["label"] == 0]

    dev_neg = sorted(neg, key=_dur)[:min(DEV_NEG, len(neg))]
    dev_pos = sorted(pos, key=_dur)[:min(DEV_POS, len(pos))]
    dev_ids = {id(r) for r in dev_neg + dev_pos}
    for r in dev_neg + dev_pos:
        r["split"] = "dev"

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

    for r in neg:
        if id(r) not in dev_ids and id(r) not in eval_ids:
            r["split"] = "train_traj"

    return {
        "dev": len(dev_neg) + len(dev_pos),
        "eval": len(eval_neg) + len(eval_pos),
        "train_traj": sum(1 for r in records if r["split"] == "train_traj"),
        "unused": sum(1 for r in records if r["split"] == ""),
    }


# ---------------------------------------------------------------------------
# dev clip export
# ---------------------------------------------------------------------------

def copy_dev_clips(records, dev_dir, index_dir):
    os.makedirs(dev_dir, exist_ok=True)
    dev = [r for r in records if r["split"] == "dev"]
    dev.sort(key=lambda r: (r["label"], _dur(r)))   # negatives first, shortest first
    mapping = []
    for i, r in enumerate(dev, start=1):
        name = f"dev{i:02d}.mp4"
        src = os.path.normpath(os.path.join(index_dir, *r["path"].split("/")))
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(dev_dir, name))
        mapping.append((name, r["clip_id"],
                        "positive" if r["label"] == 1 else "negative", r["duration_s"]))
    return mapping


def write_dev_readme(dev_dir, mapping):
    lines = ["# Dev clips", "",
             "The 15 shortest clips (10 negative + 5 positive), copied and renamed",
             "for daily development. Generated by `scripts/prepare_nexar.py`.", "",
             "| dev file | original clip_id | label | duration_s |",
             "|----------|------------------|-------|------------|"]
    for name, cid, label, dur in mapping:
        lines.append(f"| {name} | {cid} | {label} | {dur} |")
    lines.append("")
    with open(os.path.join(dev_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Index Nexar clips (Excel-labelled) and build splits.")
    p.add_argument("--root", default="data/nexar", help="Output root (index.csv lives here).")
    p.add_argument("--videos", nargs="*", default=None,
                   help="Video dir(s). Default: <root>/videos.")
    p.add_argument("--annotation", nargs="*", default=None,
                   help="Annotation file(s) .xlsx/.csv/.json. Default: auto-detect in <root>.")
    p.add_argument("--dev-dir", default="data/dev_clips", help="Where to copy dev clips.")
    args = p.parse_args()

    os.makedirs(args.root, exist_ok=True)
    video_dirs = args.videos or [os.path.join(args.root, "videos")]
    tables = find_tables(args.root, args.annotation)

    print(f"video dirs : {video_dirs}")
    print(f"annotations: {tables or '(none found)'}")
    if not tables:
        p.error(f"no annotation table found in {args.root} (need .xlsx/.csv/.json)")

    records, missing, lookup = build_records(video_dirs, tables, args.root)
    n_pos = sum(1 for r in records if r["label"] == 1)
    n_neg = sum(1 for r in records if r["label"] == 0)
    print(f"videos found: {len(lookup)} | labelled records: {len(records)} "
          f"({n_pos} positive, {n_neg} negative)")
    if missing:
        print(f"  ! {len(missing)} annotated ids had no matching video, e.g. {missing[:5]}")
    if not records:
        print("No labelled clips matched. Check videos/ and the Excel `id` column.")
        return

    counts = assign_splits(records)
    print(f"splits: dev={counts['dev']} eval={counts['eval']} "
          f"train_traj={counts['train_traj']} unused={counts['unused']}")
    for name, want in (("dev", DEV_NEG + DEV_POS), ("eval", EVAL_NEG + EVAL_POS)):
        if counts[name] < want:
            print(f"  ! {name}: only {counts[name]} available (wanted {want})")

    out_csv = os.path.join(args.root, "index.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(records)
    print(f"wrote {out_csv}")

    mapping = copy_dev_clips(records, args.dev_dir, args.root)
    if mapping:
        write_dev_readme(args.dev_dir, mapping)
        print(f"copied {len(mapping)} dev clips -> {args.dev_dir} (+ README.md)")


if __name__ == "__main__":
    main()
