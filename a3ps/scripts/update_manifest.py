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

    clips = []       # (id, n_events) that pass the filter
    skipped = []     # (id, reason) for transparency
    for name in all_names:
        clip_dir = os.path.join(clips_dir, name)
        if name in aliases:
            skipped.append((name, f"duplicate of {aliases[name]}"))
            continue
        events_path = os.path.join(clip_dir, "events.json")
        if not os.path.isfile(events_path):
            skipped.append((name, "no events.json"))
            continue
        n = count_events(events_path)
        if n is None:
            skipped.append((name, "unreadable events.json"))
            continue
        if n < args.min_events:
            skipped.append((name, f"{n} events < min {args.min_events}"))
            continue
        if args.require_video and not any(
            os.path.isfile(os.path.join(clip_dir, v))
            for v in ("annotated.mp4", "raw.mp4")
        ):
            skipped.append((name, "no video (eval-only clip)"))
            continue
        clips.append((name, n))

    # Most events first, then alphabetically -- demo-worthy clips at the top.
    clips.sort(key=lambda c: (-c[1], c[0]))
    ids = [name for name, _ in clips]

    manifest_path = os.path.join(clips_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(ids, fh)

    print(f"Wrote {manifest_path}")
    print(f"  {len(ids)} clip(s) listed (min-events={args.min_events}):")
    for name, n in clips:
        print(f"    {name:16s} {n} events")
    if skipped:
        print(f"  {len(skipped)} skipped:")
        for name, reason in skipped:
            print(f"    {name:16s} ({reason})")


if __name__ == "__main__":
    main()
