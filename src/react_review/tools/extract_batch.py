"""Asking a paper once for everything it says about one field.

The single-target tool asks one question and keeps one answer, and the model
enumerates the other arms anyway — so three claims about one table cost three
readings of it, and the three answers may disagree with each other for no reason
except that they were three separate acts of reading. This asks once.

Two things about the shape of failure here are deliberate.

*Retry stays inside the contract.* A batch that fails in transport, or comes
back as something that is not JSON, or is JSON without the top-level shape, is
retried — under the SAME question, the same profile, a new attempt number, so
the recording of each try is its own cache entry. What it never does is fall
back to the single-target path: re-asking one claim at a time under another
contract would put half a run's answers under a profile the artifact does not
name, and would make the cost of batching impossible to measure, since every
fallback quietly adds the calls the batch was supposed to save.

*A bad line is not a bad batch.* One unusable reading is dropped by the parser
with its reason, and the rest are kept. Retrying the whole prompt for it would
spend a call to re-derive the same refusal, and would let one malformed line
cost every claim in the group.

Replay is strict. There are no v5 recordings, and a miss must stop the run
rather than silently reach for the model or drop to an older profile: an
artifact that says `replay` has to mean it.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import structlog

from react_review.llm.base import LLMBackend, parse_llm_response
from react_review.schemas.batch import BatchExecutionId, BatchQuestionId
from react_review.tools.batch_parse import BatchReading, parse_batch
from react_review.tools.batch_prompt import aggregation_applies, build_batch_prompt
from react_review.tools.extraction_cache import (
    ExtractionCache,
    ExtractionCacheMiss,
    extraction_cache_key,
)
from react_review.tools.extraction_profile import BATCH_PROFILE_NAME, prompt_version

logger = structlog.get_logger(__name__)

#: How a single batch may fail, and what each costs. The first three are worth
#: another try under the same contract; the fourth is not, because asking again
#: produces the same refusal at the same price.
TRANSPORT = "transport"
NOT_JSON = "not_json"
BAD_SHAPE = "bad_shape"
ENTRY_LEVEL = "entry_level"

RETRYABLE = (TRANSPORT, NOT_JSON, BAD_SHAPE)


@dataclass
class BatchAttempt:
    """One try, whether it was served from a recording, and what it cost."""

    attempt: int
    cache_key: str
    served_from_cache: bool = False
    failure: str = ""                    # "" when the attempt produced a reading
    detail: str = ""


@dataclass
class BatchRecord:
    """One reading of one paper, kept ONCE however many claims consume it.

    Claims reference it by execution id. Copying the response onto every claim
    would multiply it by the group size and, worse, make it impossible to show
    that several claims really did come from a single act of reading.

    ``model_payload`` is the DECODED JSON the cache stores, not the model's
    literal bytes — the cache has never held those, and calling this the raw
    response would describe a provenance nobody has.
    """

    question: BatchQuestionId
    execution: BatchExecutionId | None = None
    model_payload: dict | None = None
    reading: BatchReading | None = None
    attempts: list[BatchAttempt] = field(default_factory=list)
    failure: str = ""
    detail: str = ""

    @property
    def execution_id(self) -> str:
        return self.execution.identity() if self.execution else ""

    @property
    def usable(self) -> bool:
        return self.reading is not None and not self.reading.batch_error

    @property
    def served_from_cache(self) -> bool:
        return bool(self.attempts) and self.attempts[-1].served_from_cache

    def summary(self) -> str:
        head = f"{len(self.attempts)} attempt(s)"
        if self.failure:
            return f"{head}; failed: {self.failure} — {self.detail}"
        return f"{head}; {self.reading.summary() if self.reading else 'no reading'}"


class ExtractSourceBatchTool:
    """One prompt, every reading of one field, under the v5 contract."""

    name = "extract_source_batch"

    def __init__(self, backend: LLMBackend | None, *,
                 cache: ExtractionCache | None = None, cache_mode: str = "live",
                 max_attempts: int = 3, telemetry=None) -> None:
        if cache_mode not in {"live", "record", "replay"}:
            raise ValueError("cache_mode must be live, record, or replay")
        if cache_mode in {"record", "replay"} and cache is None:
            raise ValueError(f"{cache_mode} extraction requires a cache")
        if cache_mode != "replay" and backend is None:
            raise ValueError("a live or recording run needs a backend")
        self._backend = backend
        self._cache = cache
        self._cache_mode = cache_mode
        self._telemetry = telemetry
        self._max_attempts = max(1, max_attempts)

    def build_prompt(self, *, target_shape: str, field_type: str, concept: str,
                     raw_label: str, concept_variants: str, unit_hint: str,
                     paper_text: str, research_context: str,
                     timepoint_label: str = "") -> str:
        return build_batch_prompt(
            target_shape=target_shape, context=research_context, concept=concept,
            raw_label=raw_label, field_type=field_type,
            concept_variants=concept_variants, unit_hint=unit_hint,
            paper_text=paper_text, timepoint_label=timepoint_label)

    async def read(self, *, question: BatchQuestionId, prompt: str,
                   document: str) -> BatchRecord:
        """Ask once, retrying only what another try could fix."""
        record = BatchRecord(question=question)
        aggregable = aggregation_applies(question.target_shape, question.field_type)

        for attempt in range(self._max_attempts):
            key = self._key(prompt, attempt)
            payload, failure, detail, cached = await self._one_attempt(prompt, key)
            record.attempts.append(BatchAttempt(
                attempt=attempt, cache_key=key, served_from_cache=cached,
                failure=failure, detail=detail))
            if failure:
                if failure not in RETRYABLE or attempt == self._max_attempts - 1:
                    record.failure, record.detail = failure, detail
                    return record
                continue

            reading = parse_batch(payload, document,
                                  target_shape=question.target_shape,
                                  aggregable=aggregable)
            if reading.batch_error:
                # The response decoded but is not a batch. Another try may well
                # produce one; the parser has already said precisely why.
                if attempt == self._max_attempts - 1:
                    record.model_payload = payload
                    record.failure, record.detail = BAD_SHAPE, reading.batch_error
                    return record
                record.attempts[-1].failure = BAD_SHAPE
                record.attempts[-1].detail = reading.batch_error
                continue

            record.model_payload = payload
            record.reading = reading
            return record
        return record

    async def _one_attempt(self, prompt: str, key: str):
        """One call or one cache read. Never both, and never a silent fallback."""
        try:
            recorded = (self._cache.get(key)
                        if self._cache_mode in {"record", "replay"} else None)
            if recorded is not None:
                return recorded, "", "", True
            if self._cache_mode == "replay":
                # A run that says replay must mean it. Reaching for the model
                # here, or dropping to an older profile, would produce an
                # artifact whose own label is false.
                raise ExtractionCacheMiss(
                    "no recorded batch for this question and attempt")
            if self._telemetry is not None:
                self._telemetry.attempt(self.name)
            assert self._backend is not None
            raw = await self._backend.complete(prompt)
            payload = parse_llm_response(raw, self._backend.model_id)
            if not isinstance(payload, dict):
                return None, NOT_JSON, "the response did not decode to an object", False
            if self._cache_mode == "record" and self._cache is not None:
                self._cache.put(key, payload, model_id=self._backend.model_id)
            return payload, "", "", False
        except ExtractionCacheMiss:
            raise
        except Exception as exc:                     # transport, decode, provider
            logger.warning("extract_source_batch_failed", error=str(exc)[:160])
            return None, TRANSPORT, f"{type(exc).__name__}: {exc}"[:300], False

    def _key(self, prompt: str, attempt: int) -> str:
        """The address a recording lives at.

        Unchanged from the single-target contract on purpose: model, prompt
        version, prompt bytes, attempt and seed. The QUESTION id says what was
        asked and is not this — a cache entry is about words sent to a model,
        and two runs that send the same words may share it whatever they intend
        to do with the answer.
        """
        model_id = ((self._backend.model_id if self._backend is not None else "")
                    or (self._cache.model_id if self._cache is not None else "")
                    or "replay")
        return extraction_cache_key(
            model_id=model_id,
            prompt_version=prompt_version(BATCH_PROFILE_NAME),
            prompt=prompt, attempt=attempt)


def prompt_sha256(prompt: str) -> str:
    """What the question id records, so a reworded prompt is a new question."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest().upper()
