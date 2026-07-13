"""TEMPORARY, illustration-only variant of llm_client.py with the facts-only
constraint removed from the system prompt.

Purpose: reproduce ONE deliberately-unconstrained narrative, for a genuine
before/after hallucination-mitigation example in the paper (Section VII.B),
without waiting for a real hallucination to occur naturally in production
output. Prints its result next to the real, constrained SYSTEM_PROMPT's
output on the SAME event, for direct comparison.

*** Do NOT use this in the real pipeline. *** It exists only to generate one
comparison narrative; delete this file (or just stop importing it) once
you've captured the before/after example for the paper.

    python -m a3ps.explain.llm_client_permissive_test dashboard/clips/ped_crossing_01
    python -m a3ps.explain.llm_client_permissive_test dashboard/clips/ped_crossing_01 --overwrite

This reuses llm_client.enrich_events()'s real keyframe extraction, facts
assembly, and retry logic verbatim (via its ``system_prompt`` override) --
nothing about HOW the request is made changes, only the prompt's wording, so
the comparison isolates the prompt as the one variable that matters.
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from a3ps.explain.llm_client import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    enrich_events,
)

# The ONLY difference from the real pipeline: no facts-only constraint, no
# instruction against inventing objects/numbers -- deliberately permissive,
# so it can hallucinate, for a controlled before/after comparison.
PERMISSIVE_SYSTEM_PROMPT = (
    "You are the explanation module of a driving-safety system. Write a "
    "vivid, detailed 2-3 sentence narrative describing the situation to the "
    "driver, in an engaging and dramatic way."
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="[TEMPORARY/illustration only] Generate one permissive-prompt "
                    "narrative alongside the real constrained one, for the paper's "
                    "before/after hallucination-mitigation example.")
    p.add_argument("clip_dir", help="Clip directory containing events.json + raw.mp4.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run even if explanation_llm is already populated "
                        "(needed since the real run may have already enriched these events).")
    args = p.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys "
              "and export it first.")
        sys.exit(1)

    print("=" * 70)
    print("REAL (constrained, facts-only) SYSTEM_PROMPT:")
    print(SYSTEM_PROMPT)
    print("=" * 70)
    real_clip = enrich_events(args.clip_dir, model=args.model,
                               max_tokens=args.max_tokens, overwrite=args.overwrite)

    print("\n" + "=" * 70)
    print("PERMISSIVE (deliberately unconstrained) SYSTEM_PROMPT -- TEST ONLY:")
    print(PERMISSIVE_SYSTEM_PROMPT)
    print("=" * 70)
    permissive_clip = enrich_events(args.clip_dir, model=args.model,
                                     max_tokens=args.max_tokens, overwrite=True,
                                     system_prompt=PERMISSIVE_SYSTEM_PROMPT)

    # Capture the pairing BEFORE restoring, since the restore step below
    # overwrites explanation_llm back to the constrained narrative on disk.
    permissive_by_id = {e.event_id: (e.explanation_llm, e.type, e.t) for e in permissive_clip.events}
    real_by_id = {e.event_id: e.explanation_llm for e in real_clip.events}

    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE (same events, two prompts) -- copy the pair you want "
          "for the paper's before/after example:")
    print("=" * 70)
    for event_id, (perm_text, etype, t) in permissive_by_id.items():
        print(f"\nevent {event_id} ({etype}, t={t:.2f}s):")
        print(f"  [constrained] {real_by_id.get(event_id)}")
        print(f"  [permissive]  {perm_text}")

    # Self-cleaning: restore the real, constrained narrative on disk so this
    # illustration run never leaves a hallucination-prone string sitting in
    # the events.json the actual dashboard/report would read.
    enrich_events(args.clip_dir, model=args.model, max_tokens=args.max_tokens,
                  overwrite=True, system_prompt=SYSTEM_PROMPT)
    print(f"\n{args.clip_dir}/events.json restored to the real, constrained "
          f"narrative (the permissive one above was for illustration only).")


if __name__ == "__main__":
    main()
