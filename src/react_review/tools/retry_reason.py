"""Classify why the collector asked extract_source_value again.

``collector.py`` is inside the evidence-adequacy hash boundary, so this module
wraps the extract tool instead of editing the retry loop. Every retry is one of
four reasons; there is no ``unknown`` bucket. The counts are the same events as
``repeated_attempts``: a wrap that skips the model (transport already exhausted)
increments neither.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from react_review.core.enums import CollectionOutcome

RETRY_REASONS = (
    "call_failed",
    "not_found",
    "low_confidence",
    "disagreement",
)

_ACCEPT_CONFIDENCE = 0.7
_last_extract: ContextVar[Any] = ContextVar("extract_last_result", default=None)


def _field(result: Any, name: str, default: Any = "") -> Any:
    if result is None:
        return default
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def is_call_failed(result: Any) -> bool:
    """Whether this extract result is a transport/provider failure, not a miss."""
    if result is None:
        return False
    if bool(_field(result, "permanent_failure", False)):
        return True
    error = str(_field(result, "error", "") or "")
    if error.strip():
        return True
    reason = str(_field(result, "not_found_reason", "") or "")
    return "extraction call failed" in f"{error} {reason}".lower()


def classify_retry_reason(result: Any) -> str:
    """Why the next extract attempt is happening, from the previous result.

    Raises if the previous result cannot be attributed to one of the four
    reasons — that is a bug, not a fifth category.
    """
    if result is None:
        raise ValueError("a retry has no previous extract result to classify")
    if is_call_failed(result):
        return "call_failed"
    found = bool(_field(result, "found", False))
    agreement = _field(result, "agreement", None)
    if agreement is False:
        return "disagreement"
    confidence = _field(result, "confidence", None)
    if confidence is not None and float(confidence) < _ACCEPT_CONFIDENCE:
        return "low_confidence"
    if not found:
        return "not_found"
    raise ValueError(
        "a retry of a found extract has no disagreement or low-confidence signal")


def retries_from_processing_records(records: list) -> dict[str, int]:
    """Replay extract observations from a package; one count per retry."""
    counts = {name: 0 for name in RETRY_REASONS}
    for rec in records or []:
        last = None
        steps = rec.get("steps") if isinstance(rec, dict) else getattr(rec, "steps", [])
        for step in steps or []:
            tool = step.get("tool") if isinstance(step, dict) else getattr(step, "tool", "")
            if tool != "extract_source_value":
                continue
            obs = (step.get("observation") if isinstance(step, dict)
                   else getattr(step, "observation", None)) or {}
            if last is not None:
                counts[classify_retry_reason(last)] += 1
            last = obs
    return counts


def _rewrite_call_failed_outcome(result):
    """A failed call is not 'the paper omitted the value' (F7)."""
    item = result.source_item
    if item.collection_outcome is not CollectionOutcome.MISSING_SOURCE:
        return result
    reasons = []
    for record in item.reasons:
        if getattr(record, "code", "") == CollectionOutcome.MISSING_SOURCE.value:
            reasons.append(record.model_copy(
                update={"code": CollectionOutcome.EXTRACTION_FAILED.value}))
        else:
            reasons.append(record)
    item = item.model_copy(update={
        "collection_outcome": CollectionOutcome.EXTRACTION_FAILED,
        "reasons": reasons,
    })
    agent_run = result.record
    final = agent_run.final
    if isinstance(final, dict) and final.get("collection_outcome") == "missing_source":
        final = dict(final)
        final["collection_outcome"] = "extraction_failed"
        copied = []
        for record in final.get("reasons") or []:
            if isinstance(record, dict) and record.get("code") == "missing_source":
                copied.append({**record, "code": "extraction_failed"})
            else:
                copied.append(record)
        if "reasons" in final:
            final["reasons"] = copied
        agent_run = agent_run.model_copy(update={"final": final})
    return result.model_copy(update={"source_item": item, "record": agent_run})


def bind_retry_attribution(collector):
    """Record retry_reason, and do not re-ask a transport that already gave up.

    ``collector.py`` retries whenever ``found`` is false. A call that ``_post_chat``
    already exhausted cannot succeed by asking again; repeating it is what made
    2.8 attempts per claim. The wrap returns the exhausted result instead of
    calling the model, so ``repeated_attempts`` does not count a retry that did
    not happen. The collector still loops; the extra iterations are free.

    A call failure is ``extraction_failed``, not ``missing_source``: the latter
    accuses the paper of omitting a value that was never asked.
    """
    inner = collector._extract
    original = collector.collect

    class _Attributed:
        async def run(self, payload):
            previous = _last_extract.get()
            if payload.attempt > 0:
                reason = classify_retry_reason(previous)
                if reason == "call_failed":
                    return previous
                telemetry = collector._telemetry
                if telemetry is not None:
                    telemetry.record_retry(reason)
            result = await inner.run(payload)
            _last_extract.set(result)
            return result

    async def collect(review_item, *args, **kwargs):
        token = _last_extract.set(None)
        try:
            result = await original(review_item, *args, **kwargs)
        finally:
            last = _last_extract.get()
            _last_extract.reset(token)
        if last is not None and is_call_failed(last) and not bool(
                _field(last, "found", False)):
            return _rewrite_call_failed_outcome(result)
        return result

    collector._extract = _Attributed()
    collector.collect = collect
    return collector
