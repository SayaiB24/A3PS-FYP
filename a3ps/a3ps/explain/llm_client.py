"""Async Anthropic API enrichment (STRETCH).

Wraps the deterministic template with a richer natural-language explanation.
The Anthropic SDK is imported lazily; if unavailable or unconfigured, callers
should fall back to the deterministic template.
"""

from __future__ import annotations

import os
from typing import Optional

from a3ps.common.schema import Event
from a3ps.explain.templates import explain as template_explain

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class LLMEnricher:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_tokens: int = 200,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def enrich(self, event: Event) -> str:
        """Return an LLM-enriched explanation, or the template on failure."""
        base = template_explain(event)
        if not self.api_key:
            return base
        try:
            client = self._ensure_client()
            resp = await client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "You are a driving-safety assistant. Rewrite this "
                            "collision alert as one calm, clear sentence for the "
                            f"driver:\n\n{base}"
                        ),
                    }
                ],
            )
            return resp.content[0].text.strip()
        except Exception:
            return base
