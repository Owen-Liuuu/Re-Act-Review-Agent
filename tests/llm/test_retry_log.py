"""Mid-retry is debug; only exhaustion warns."""
from __future__ import annotations

from react_review.llm.base import LLMBackend


class _Dummy(LLMBackend):
    @property
    def model_id(self) -> str:
        return "dummy"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        return ""


class _Resp:
    status_code = 429
    headers: dict[str, str] = {}


def test_mid_retry_is_debug_exhaustion_is_warning(monkeypatch):
    logged: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "react_review.llm.base.logger.debug",
        lambda event, **kw: logged.append(("debug", event)),
    )
    monkeypatch.setattr(
        "react_review.llm.base.logger.warning",
        lambda event, **kw: logged.append(("warning", event)),
    )
    backend = _Dummy()
    backend._log_rate_limited(_Resp(), attempt=0, delay=2.0)
    backend._log_rate_limited(_Resp(), attempt=4, delay=0, exhausted=True)
    backend._log_transient_retry(attempt=0, delay=2.0, detail="timeout")
    backend._log_transient_retry(attempt=4, delay=0, detail="timeout", exhausted=True)
    assert logged == [
        ("debug", "llm_rate_limited_retry"),
        ("warning", "llm_rate_limited_exhausted"),
        ("debug", "llm_network_error_retry"),
        ("warning", "llm_network_error_exhausted"),
    ]
