"""Visible traces for failures that still return empty.

Callers keep their empty return values. This module only records that the
empty result was a failure, not a genuine miss. Logger lines are deduped by
``(event, error_type)`` so a hot path (OpenAlex 429) does not flood the
console; HITL warning lists are the caller's and are not deduped.
"""
from __future__ import annotations

from collections import Counter

import structlog

logger = structlog.get_logger(__name__)

_counts: Counter[tuple[str, str]] = Counter()
_records: list[dict[str, object]] = []


def reset() -> None:
    """Clear per-process counts. Tests call this; a production run is one process."""
    _counts.clear()
    _records.clear()


def records() -> list[dict[str, object]]:
    return list(_records)


def note(
    event: str,
    *,
    error_type: str,
    message: str,
    **fields: object,
) -> str:
    """Log the first ``(event, error_type)``; later hits only increment ``count``.

    Always returns ``message`` so a caller can hang it on a step's warnings.
    """
    key = (event, error_type)
    _counts[key] += 1
    count = _counts[key]
    record = {
        "event": event, "error_type": error_type, "message": message,
        "count": count, **fields,
    }
    _records.append(record)
    if count == 1:
        logger.warning(event, error_type=error_type, message=message,
                       count=count, **fields)
    return message


def trace(
    notes: list[str] | None,
    event: str,
    *,
    error_type: str,
    message: str,
    **fields: object,
) -> str:
    """``note`` plus, when given, append the line to a HITL warnings list."""
    note(event, error_type=error_type, message=message, **fields)
    if notes is not None:
        notes.append(message)
    return message
