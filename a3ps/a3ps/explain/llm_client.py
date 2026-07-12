"""MLLM enrichment via the Groq API (STRETCH, Week 3).

Runs OFFLINE after the pipeline, never in the live loop. For each event in a
clip's ``events.json`` it grabs the keyframe from ``raw.mp4`` at ``event.t``,
downsizes it to 768 px wide, and sends the structured event facts + the image
to Groq's OpenAI-compatible chat API using a vision-capable Llama model. The
2-3 sentence, scene-aware narrative is stored in ``event.explanation_llm`` and
written back to ``events.json``.

    python -m a3ps.explain.llm_client dashboard/clips/ped_crossing_01
    python -m a3ps.explain.llm_client dashboard/clips/ped_crossing_01 --dry-run

Groq has a free tier (get a key at https://console.groq.com/keys, set
GROQ_API_KEY); if API access is a problem this whole step can be cut -- the
deterministic templates (``explanation_template``) already satisfy the XAI
requirement.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from a3ps.common.schema import ClipResult, Event  # noqa: E402

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
except ImportError:
    pass  # python-dotenv not installed -- GROQ_API_KEY must come from the environment

DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_MAX_TOKENS = 300          # ~200 requested; a little headroom for 2-3 sentences
IMAGE_WIDTH = 768
IMAGE_MEDIA_TYPE = "image/jpeg"

SYSTEM_PROMPT = (
    "You are the explanation module of a driving-safety system. Using ONLY the "
    "structured facts provided and what is visible in the image, write 2-3 "
    "sentences explaining the intervention to the driver. Do not invent objects "
    "or numbers not present in the facts."
)


# ---------------------------------------------------------------------------
# keyframe extraction
# ---------------------------------------------------------------------------

def _extract_keyframe(cap, t: float, fps: float, max_width: int = IMAGE_WIDTH) -> Optional[bytes]:
    """Return the frame at time ``t`` (seconds) as JPEG bytes, downsized to
    ``max_width`` wide (aspect preserved). None if the frame can't be read."""
    import cv2

    if fps and fps > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t * fps))))
    else:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t * 1000.0))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None

    h, w = frame.shape[:2]
    if w > max_width:
        new_h = max(1, int(round(h * max_width / w)))
        frame = cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return None
    return buf.tobytes()


# ---------------------------------------------------------------------------
# structured facts
# ---------------------------------------------------------------------------

def _frame_context(clip: ClipResult, event: Event) -> Tuple[List[str], Optional[List[float]]]:
    """Look up the active context flags and the actor's BEV velocity for an event."""
    flags: List[str] = []
    velocity: Optional[List[float]] = None
    for fr in clip.frames:
        if fr.frame_idx != event.frame_idx:
            continue
        flags = list((fr.context or {}).get("flags", []) or [])
        for tr in fr.tracks:
            if tr.id == event.actor_id:
                velocity = tr.velocity_bev
                break
        break
    return flags, velocity


def _build_facts(event: Event, flags: List[str], velocity: Optional[List[float]]) -> Dict[str, Any]:
    return {
        "action": event.type,
        "actor_class": event.actor_cls,
        "actor_id": event.actor_id,
        "collision_prob": event.collision_prob,
        "ttc_s": event.ttc_s,
        "threshold": event.threshold,
        "context_flags": flags,
        "actor_velocity_bev": velocity,
    }


def _facts_block(facts: Dict[str, Any]) -> str:
    return "Structured facts:\n" + json.dumps(facts, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Groq call (OpenAI-compatible chat completions, with retries)
# ---------------------------------------------------------------------------

def _is_retryable(exc: Exception) -> bool:
    """Transient API failures worth retrying (rate limit / 5xx / connection)."""
    name = type(exc).__name__
    if name in {"RateLimitError", "APIConnectionError", "APITimeoutError",
                "InternalServerError", "APIError"}:
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status >= 500


def _call_once(client, model: str, max_tokens: int, image_b64: str, facts_text: str) -> str:
    data_url = f"data:{IMAGE_MEDIA_TYPE};base64,{image_b64}"
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": facts_text},
                ],
            },
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def _call_with_retries(client, model, max_tokens, image_b64, facts_text, max_retries=4) -> str:
    delay = 1.0
    for attempt in range(max_retries + 1):
        try:
            return _call_once(client, model, max_tokens, image_b64, facts_text)
        except Exception as exc:  # noqa: BLE001 - re-raised unless transient
            if attempt >= max_retries or not _is_retryable(exc):
                raise
            time.sleep(min(delay, 30.0))
            delay *= 2.0
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

def enrich_events(
    clip_dir: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    dry_run: bool = False,
    overwrite: bool = False,
    client: Any = None,
    max_retries: int = 4,
) -> ClipResult:
    """Enrich every event in ``<clip_dir>/events.json`` with an LLM narrative.

    Reads the keyframe for each event from ``<clip_dir>/raw.mp4``, calls the
    Groq API, and writes the result into ``event.explanation_llm``. Events
    already enriched are skipped unless ``overwrite`` is set. With ``dry_run``
    the facts + keyframe are prepared and printed but no API call is made and
    nothing is written. ``client`` may be injected (tests); otherwise a
    ``groq.Groq`` client is created lazily from ``GROQ_API_KEY``.
    """
    import cv2

    events_path = os.path.join(clip_dir, "events.json")
    raw_path = os.path.join(clip_dir, "raw.mp4")
    if not os.path.isfile(events_path):
        raise FileNotFoundError(f"no events.json in {clip_dir}")
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(f"no raw.mp4 in {clip_dir}")

    clip = ClipResult.load_json(events_path)
    if not clip.events:
        print(f"{clip_dir}: no events to enrich.")
        return clip

    cap = cv2.VideoCapture(raw_path)
    if not cap.isOpened():
        raise IOError(f"cannot open {raw_path}")
    fps = float(clip.meta.get("fps") or cap.get(cv2.CAP_PROP_FPS) or 30.0)

    if client is None and not dry_run:
        from groq import Groq
        client = Groq()

    enriched = 0
    skipped = 0
    try:
        for event in clip.events:
            if event.explanation_llm and not overwrite:
                skipped += 1
                continue

            frame_bytes = _extract_keyframe(cap, event.t, fps)
            if frame_bytes is None:
                print(f"  ! event {event.event_id}: no frame at t={event.t:.2f}s; skipping")
                continue
            image_b64 = base64.standard_b64encode(frame_bytes).decode("ascii")

            flags, velocity = _frame_context(clip, event)
            facts_text = _facts_block(_build_facts(event, flags, velocity))

            if dry_run:
                print(f"\n[dry-run] event {event.event_id} ({event.type}, "
                      f"t={event.t:.2f}s, keyframe {len(frame_bytes)} B JPEG)")
                print(facts_text)
                continue

            event.explanation_llm = _call_with_retries(
                client, model, max_tokens, image_b64, facts_text, max_retries)
            enriched += 1
            print(f"  event {event.event_id} ({event.type}): {event.explanation_llm}")
    finally:
        cap.release()

    if not dry_run and enriched:
        clip.save_json(events_path)
        print(f"\n{clip_dir}: enriched {enriched}, skipped {skipped} "
              f"(already done) -> {events_path}")
    elif dry_run:
        print(f"\n{clip_dir}: dry-run only, nothing written.")
    else:
        print(f"\n{clip_dir}: nothing to enrich (skipped {skipped}).")
    return clip


def main() -> None:
    p = argparse.ArgumentParser(
        description="Offline MLLM enrichment of a clip's events.json (via Groq).")
    p.add_argument("clip_dir", help="Clip directory containing events.json + raw.mp4.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--dry-run", action="store_true",
                   help="Prepare facts + keyframes and print them; no API call, no write.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-enrich events that already have explanation_llm.")
    args = p.parse_args()

    if not args.dry_run and not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set. Use --dry-run, or get a free key at "
              "https://console.groq.com/keys and export GROQ_API_KEY to enrich for real.")
        sys.exit(1)

    enrich_events(
        args.clip_dir,
        model=args.model,
        max_tokens=args.max_tokens,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
