"""Tests for offline MLLM enrichment — file/keyframe/facts/skip logic, no network.

The Groq client is injected as a fake (OpenAI-compatible chat.completions
shape), so these run with no API key and no network. They cover everything
except the actual model call: keyframe extraction + 768px downsize,
structured-fact assembly, write-back, skip-already-enriched, overwrite, and
dry-run.
"""

import base64
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

cv2 = pytest.importorskip("cv2")

from a3ps.common.schema import (  # noqa: E402
    ClipResult, Event, FrameRecord, Prediction, TrackState,
)
from a3ps.explain import llm_client  # noqa: E402


# ---- fake Groq client (OpenAI-compatible chat.completions shape) ----------

class _Message:
    def __init__(self, text):
        self.content = text


class _Choice:
    def __init__(self, text):
        self.message = _Message(text)


class _Completion:
    def __init__(self, text):
        self.choices = [_Choice(text)]


class _Completions:
    def __init__(self, text):
        self._text = text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Completion(self._text)


class _Chat:
    def __init__(self, text):
        self.completions = _Completions(text)


class _FakeClient:
    def __init__(self, text="A pedestrian is crossing into your path; braking engaged."):
        self.chat = _Chat(text)


# ---- fixtures --------------------------------------------------------------

def _write_video(path, width=1280, height=720, fps=10, n_frames=30):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened(), "could not open VideoWriter (codec missing?)"
    for i in range(n_frames):
        frame = np.full((height, width, 3), (i * 5 % 255, 40, 90), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _make_clip(clip_dir):
    os.makedirs(clip_dir, exist_ok=True)
    _write_video(os.path.join(clip_dir, "raw.mp4"))

    def _frame(idx, t):
        pred = Prediction(horizon_s=4.0, dt=0.2, mean_img=[[0.0, 0.0]],
                          collision_prob=0.82, ttc_s=1.1)
        tr = TrackState(id=4, cls="person", bbox=[0, 0, 10, 10],
                        centroid_img=[5.0, 5.0], velocity_bev=[-1.2, -0.3],
                        prediction=pred)
        return FrameRecord(frame_idx=idx, t=t, tracks=[tr],
                           context={"flags": ["crosswalk_ahead"]})

    events = [
        Event(event_id=1, frame_idx=10, t=1.0, type="ALERT", actor_id=4,
              actor_cls="person", collision_prob=0.62, threshold=0.65, ttc_s=1.6),
        Event(event_id=2, frame_idx=20, t=2.0, type="VIRTUAL_BRAKE", actor_id=4,
              actor_cls="person", collision_prob=0.82, threshold=0.65, ttc_s=1.0),
    ]
    clip = ClipResult(meta={"fps": 10}, frames=[_frame(10, 1.0), _frame(20, 2.0)],
                      events=events)
    clip.save_json(os.path.join(clip_dir, "events.json"))
    return clip


# ---- tests -----------------------------------------------------------------

def test_keyframe_downsized_to_768(tmp_path):
    clip_dir = str(tmp_path / "clip")
    _make_clip(clip_dir)
    cap = cv2.VideoCapture(os.path.join(clip_dir, "raw.mp4"))
    try:
        jpeg = llm_client._extract_keyframe(cap, t=1.0, fps=10)
    finally:
        cap.release()
    assert jpeg is not None
    img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == 768                      # width downsized from 1280


def test_enrich_writes_explanation_and_sends_image_and_facts(tmp_path):
    clip_dir = str(tmp_path / "clip")
    _make_clip(clip_dir)
    fake = _FakeClient()

    llm_client.enrich_events(clip_dir, client=fake, model="meta-llama/llama-4-scout-17b-16e-instruct")

    # Both events enriched and persisted.
    reloaded = ClipResult.load_json(os.path.join(clip_dir, "events.json"))
    assert all(e.explanation_llm for e in reloaded.events)
    assert len(fake.chat.completions.calls) == 2

    call = fake.chat.completions.calls[0]
    assert call["model"] == "meta-llama/llama-4-scout-17b-16e-instruct"
    system_msg = next(m for m in call["messages"] if m["role"] == "system")
    user_msg = next(m for m in call["messages"] if m["role"] == "user")
    assert system_msg["content"] == llm_client.SYSTEM_PROMPT
    content = user_msg["content"]
    image_block = next(b for b in content if b["type"] == "image_url")
    text_block = next(b for b in content if b["type"] == "text")
    # Image is a valid base64 JPEG <= 768 wide, sent as a data: URI.
    url = image_block["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    raw = base64.standard_b64decode(url.split(",", 1)[1])
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] <= 768
    # Facts carry the real context + actor data (nothing invented downstream).
    assert "crosswalk_ahead" in text_block["text"]
    assert "actor_velocity_bev" in text_block["text"]
    assert "\"actor_id\": 4" in text_block["text"]


def test_skip_already_enriched_then_overwrite(tmp_path):
    clip_dir = str(tmp_path / "clip")
    _make_clip(clip_dir)
    llm_client.enrich_events(clip_dir, client=_FakeClient())

    # Second run: everything already has explanation_llm -> no API calls.
    fake2 = _FakeClient()
    llm_client.enrich_events(clip_dir, client=fake2)
    assert len(fake2.chat.completions.calls) == 0

    # With --overwrite it re-enriches.
    fake3 = _FakeClient("Updated narrative.")
    llm_client.enrich_events(clip_dir, client=fake3, overwrite=True)
    assert len(fake3.chat.completions.calls) == 2
    reloaded = ClipResult.load_json(os.path.join(clip_dir, "events.json"))
    assert all(e.explanation_llm == "Updated narrative." for e in reloaded.events)


def test_dry_run_makes_no_calls_and_no_writes(tmp_path):
    clip_dir = str(tmp_path / "clip")
    _make_clip(clip_dir)
    fake = _FakeClient()

    llm_client.enrich_events(clip_dir, client=fake, dry_run=True)

    assert len(fake.chat.completions.calls) == 0
    reloaded = ClipResult.load_json(os.path.join(clip_dir, "events.json"))
    assert all(e.explanation_llm is None for e in reloaded.events)


def test_enrich_events_default_system_prompt_is_the_constrained_one(tmp_path):
    clip_dir = str(tmp_path / "clip")
    _make_clip(clip_dir)
    fake = _FakeClient()
    llm_client.enrich_events(clip_dir, client=fake)
    system_msg = next(m for m in fake.chat.completions.calls[0]["messages"] if m["role"] == "system")
    assert system_msg["content"] == llm_client.SYSTEM_PROMPT


def test_enrich_events_honors_system_prompt_override(tmp_path):
    # llm_client_permissive_test.py depends on this override actually reaching
    # the API call -- pin it here so a future refactor can't silently drop it.
    clip_dir = str(tmp_path / "clip")
    _make_clip(clip_dir)
    fake = _FakeClient()
    permissive = "Write a dramatic narrative with no factual constraints."

    llm_client.enrich_events(clip_dir, client=fake, system_prompt=permissive)

    for call in fake.chat.completions.calls:
        system_msg = next(m for m in call["messages"] if m["role"] == "system")
        assert system_msg["content"] == permissive
        assert system_msg["content"] != llm_client.SYSTEM_PROMPT


def test_is_retryable_classification():
    class RateLimitError(Exception):
        pass

    class BadRequestError(Exception):
        pass

    server = BadRequestError()
    server.status_code = 503
    client = BadRequestError()
    client.status_code = 400

    assert llm_client._is_retryable(RateLimitError())     # by class name
    assert llm_client._is_retryable(server)               # 5xx
    assert not llm_client._is_retryable(client)           # 4xx
    assert not llm_client._is_retryable(ValueError("x"))
