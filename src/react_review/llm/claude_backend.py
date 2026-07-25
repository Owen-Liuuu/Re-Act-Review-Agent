"""Anthropic Claude LLM backend — connects to Claude 3.5 / Claude 4 / etc.

TODO: Fill in the actual API call implementation.
"""
from __future__ import annotations

from react_review.core.config import LLMSettings
from react_review.llm.base import LLMBackend


class ClaudeBackend(LLMBackend):
    """LLM backend using the Anthropic API.

    Args:
        settings: LLM configuration (api_key, model, temperature, etc.).
    """

    def __init__(self, settings: LLMSettings) -> None:
        super().__init__(max_concurrency=settings.max_concurrency)
        self._settings = settings
        # TODO: Initialise the Anthropic client here
        # Example:
        #   from anthropic import AsyncAnthropic
        #   self._client = AsyncAnthropic(api_key=settings.api_key)
        # When implementing complete(), wrap the actual API call in
        # ``async with self._sem:`` to honour the configured concurrency cap.

    @property
    def model_id(self) -> str:
        return self._settings.model

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        """Send a prompt to Claude and return the response text.

        TODO: Implement the actual API call. Example:

            message = await self._client.messages.create(
                model=self._settings.model,
                max_tokens=self._settings.max_tokens,
                temperature=self._settings.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        """
        raise NotImplementedError(
            "Claude backend not yet implemented. "
            "Fill in the complete() method with your Anthropic API call."
        )
