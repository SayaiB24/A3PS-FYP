#!/usr/bin/env python
"""Regenerate dashboard/clips/manifest.json from the clip folders on disk.

The pipeline (run_pipeline.py) writes each clip to its own folder but never
touches the manifest, so the dashboard's clip dropdown goes stale after a
batch run. This scans the clips directory for folders that contain a readable
events.json and rewrites the manifest with their ids.

    # list every processed clip
    python scripts/update_manifest.py

    # list only clips that actually produced events (skip empty/clean runs)
    python scripts/update_manifest.py --min-events 1

    # list only clips that have a rendered video (skip eval-only clips)
    python scripts/update_manifest.py --require-video

Note on the numeric clip ids: ``eval_anticipation.py --run`` keys clips by
their original Nexar ``clip_id`` from index.csv (e.g. ``234``), not by dev
name, and writes them with ``render=False`` -- so those folders hold an
events.json but no annotated.mp4 and play blank in the dashboard. Keep eval
runs on their default ``--clips-dir eval/anticipation`` so they never land in
dashboard/clips, and/or pass ``--require-video`` here to keep them out of the
dropdown.

Clips are ordered by event count (most events first), then by id, so the most
demo-worthy clips sit at the top of the dropdown.
"""

import argparse
import json
import os
import sys


def meta_clip_id(clip_dir: str):
    """The clip_id a clip's own meta.json claims, or None."""
    try:
        with open(os.path.join(clip_dir, "meta.json"), "r", encoding="utf-8") as fh:
            return str(json.load(fh).get("clip_id") or "") or None
    except (OSError, ValueError):
        return None


def find_aliases(clips_dir: str, names):
    """Directories that are duplicates of another directory in the same tree.

    Running the pipeline on a dev clip but writing the output to a directory
    named after the original Nexar clip_id (which is what
    ``eval_anticipation.py --run`` does, since it keys off index.csv) leaves two
    directories holding the SAME footage -- e.g. ``1118/`` and ``dev04/``, whose
    meta.json both say ``clip_id: dev04``. The dropdown then lists every clip
    twice.

    A directory is an alias when its meta.json names a *different* directory
    that also exists here; that named directory is the canonical one and wins.
    Returns {alias_name: canonical_name}.
    """
    present = set(names)
    aliases = {}
    for name in names:
        cid = meta_clip_id(os.path.join(clips_dir, name))
        if cid and cid != name and cid in present:
            aliases[name] = cid
    return aliases


def count_events(events_json_path: str):
    """Return the number of events in a clip's events.json, or None if unreadable."""
    try:
        with open(events_json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    events = data.get("events", []) if isinstance(data, dict) else []
    return len(events)


def clip_stats(events_json_path: str):
    """(n_events, mean_actors_per_frame) for a clip, or (None, None).

    Mean actor density is what decides whether a clip is worth putting in front
    of someone: several dev clips were filmed at empty junctions and track
    essentially nothing (dev01 averages 0.00 actors/frame), so they play as a
    blank overlay and look like a broken dashboard rather than a quiet road.
    """
    try:
        with open(events_json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None, None
    frames = data.get("frames") or []
    events = data.get("events") or []
    if not frames:
        return len(events), 0.0
    total = sum(len(f.get("tracks") or ()) for f in frames)
    return len(events), total / len(frames)


def risk_label(clip_dir: str, name: str):
    """Dropdown label for a Phase IV comparison clip, or None if it isn't one.

    Reads the small risk.json (not the huge events.json) and summarises the
    contrast that makes the clip worth opening: whether it is a positive or
    negative, and how each system judged it.
    """
    path = os.path.join(clip_dir, "risk.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            r = json.load(fh)
    except (OSError, ValueError):
        return None
    kind = "pos" if r.get("label") == 1 else "neg"
    learned = (r.get("learned") or {}).get("verdict") or "?"
    old = (r.get("threshold_system") or {}).get("verdict") or "?"
    if learned == old:
        contrast = f"both {learned}"
    else:
        contrast = f"learned {learned} / old {old}"
    return f"{name} · {kind} · {contrast}"


# Dropdown groups, in the order they should appear.
G_PHASE4 = "Phase IV — learned vs threshold system"
G_TRAFFIC = "Old system — clips with traffic"
G_SPARSE = "Old system — sparse / near-empty"

# Below this mean actors/frame a clip has almost nothing to show.
SPARSE_ACTORS = 1.0


def main() -> None:
    p = argparse.ArgumentParser(description="Rebuild the dashboard clip manifest.")
    p.add_argument(
        "--clips-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "dashboard", "clips"),
        help="Directory holding one folder per clip (default: dashboard/clips).",
    )
    p.add_argument(
        "--min-events", type=int, default=0,
        help="Only list clips with at least this many events (default: 0 = all).",
    )
    p.add_argument(
        "--require-video", action="store_true",
        help="Only list clips that have a playable video (annotated.mp4 or "
             "raw.mp4). Filters out clips produced by eval_anticipation.py "
             "--run, which are rendered with render=False.",
    )
    p.add_argument(
        "--keep-aliases", action="store_true",
        help="Keep duplicate clip folders that hold the same footage as another "
             "folder (e.g. 1118/ alongside dev04/). Off by default: aliases are "
             "dropped so each clip appears once, under its canonical name.",
    )
    args = p.parse_args()

    clips_dir = os.path.abspath(args.clips_dir)
    if not os.path.isdir(clips_dir):
        p.error(f"clips dir not found: {clips_dir}")

    all_names = sorted(n for n in os.listdir(clips_dir)
                       if os.path.isdir(os.path.join(clips_dir, n)))
    aliases = {} if args.keep_aliases else find_aliases(clips_dir, all_names)

    clips = []       # {id, group, label, sort} that pass the filter
    skipped = []     # (id, reason) for transparency
    for name in all_names:
        clip_dir = os.path.join(clips_dir, name)
        if name in aliases:
            skipped.append((name, f"duplicate of {aliases[name]}"))
            continue

        has_video = any(os.path.isfile(os.path.join(clip_dir, v))
                        for v in ("annotated.mp4", "raw.mp4"))

        # Phase IV comparison clips carry risk.json and deliberately ship WITHOUT
        # events.json (that file is 20-50 MB and only the overlay needs it), so
        # they must be recognised before the events.json requirement below.
        label = risk_label(clip_dir, name)
        if label is not None:
            if args.require_video and not has_video:
                skipped.append((name, "no video"))
                continue
            # Positives first (they carry the useful-vs-too-early contrast), then
            # numerically by id so the order is stable and not "1004 before 488".
            is_pos = " · pos · " in label
            clips.append({"id": name, "group": G_PHASE4, "label": label,
                          "sort": (0, 0 if is_pos else 1,
                                   int(name) if name.isdigit() else 0, name)})
            continue

        events_path = os.path.join(clip_dir, "events.json")
        if not os.path.isfile(events_path):
            skipped.append((name, "no events.json and no risk.json"))
            continue
        n, actors = clip_stats(events_path)
        if n is None:
            skipped.append((name, "unreadable events.json"))
            continue
        if n < args.min_events:
            skipped.append((name, f"{n} events < min {args.min_events}"))
            continue
        if args.require_video and not has_video:
            skipped.append((name, "no video (eval-only clip)"))
            continue

        sparse = actors < SPARSE_ACTORS
        if sparse:
            desc = "empty" if actors < 0.05 else f"{actors:.1f} actors/frame"
            clips.append({"id": name, "group": G_SPARSE,
                          "label": f"{name} · {desc}", "sort": (2, -actors, name)})
        else:
            clips.append({
                "id": name, "group": G_TRAFFIC,
                "label": f"{name} · {actors:.1f} actors/frame · {n} events",
                "sort": (1, -actors, name)})

    # Phase IV clips first, then busiest overlay clips, then the sparse ones --
    # so the dropdown opens on something worth looking at.
    clips.sort(key=lambda c: c["sort"])
    for c in clips:
        c.pop("sort", None)

    manifest_path = os.path.join(clips_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        # Grouped/labelled form. app.js also accepts a plain array of ids, so an
        # older manifest keeps working.
        json.dump({"clips": clips}, fh, indent=1)

    print(f"Wrote {manifest_path}")
    print(f"  {len(clips)} clip(s) listed (min-events={args.min_events}):")
    group = None
    for c in clips:
        if c["group"] != group:
            group = c["group"]
            print(f"    [{group}]")
        print(f"      {c['label']}")
    if skipped:
        print(f"  {len(skipped)} skipped:")
        for name, reason in skipped:
            print(f"    {name:16s} ({reason})")


if __name__ == "__main__":
    main()
