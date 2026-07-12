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

Clips are ordered by event count (most events first), then by id, so the most
demo-worthy clips sit at the top of the dropdown.
"""

import argparse
import json
import os
import sys


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
    args = p.parse_args()

    clips_dir = os.path.abspath(args.clips_dir)
    if not os.path.isdir(clips_dir):
        p.error(f"clips dir not found: {clips_dir}")

    clips = []       # (id, n_events) that pass the filter
    skipped = []     # (id, reason) for transparency
    for name in sorted(os.listdir(clips_dir)):
        clip_dir = os.path.join(clips_dir, name)
        if not os.path.isdir(clip_dir):
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
