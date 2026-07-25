"""Mock LLM backend for testing without API keys."""
from __future__ import annotations

import json

from react_review.llm.base import LLMBackend


class MockLLMBackend(LLMBackend):
    """Returns pre-configured JSON responses for testing.

    Args:
        responses: Optional mapping of prompt substrings to response dicts.
                   If no match is found, returns a generic acknowledgement.
        max_concurrency: Concurrency cap inherited from ``LLMBackend``.
            Defaults to 1024 because the mock makes no real network calls
            and we don't want the throttle to slow down test runs.
    """

    def __init__(
        self,
        responses: dict[str, dict] | None = None,
        max_concurrency: int = 1024,
    ) -> None:
        super().__init__(max_concurrency=max_concurrency)
        self._responses = responses or {}

    @property
    def model_id(self) -> str:
        return "mock-llm-v1"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        # The semaphore is wide-open in the mock backend — wrapping is
        # cheap and keeps the call shape identical to the real backends.
        async with self._sem:
            for key, response in self._responses.items():
                if key in prompt:
                    return json.dumps(response)
            return json.dumps({"status": "ok", "message": "mock response"})
