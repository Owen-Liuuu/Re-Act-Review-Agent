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

from react_review.llm.base import LLMBackend, LLMError, parse_llm_response
from react_review.tools.base import ToolStage
from react_review.schemas.batch import (
    BatchExecutionId,
    BatchQuestionId,
    BatchReadingRecord,
    ExcerptProvenance,
)
from react_review.tools.batch_parse import BatchReading, parse_batch
from react_review.tools.batch_prompt import aggregation_applies, build_batch_prompt
from react_review.tools.batch_split import (
    BATCH_LOCATE_VERSION,
    BATCH_TRANSCRIBE_VERSION,
    build_batch_transcribe_prompt,
    format_passages,
    merge_located_and_transcribed,
    parse_locate,
)
from react_review.tools.extraction_cache import (
    ExtractionCache,
    ExtractionCacheMiss,
    extraction_cache_key,
)
from react_review.schemas.telemetry import BATCH_EXTRACTION
from react_review.tools.extraction_profile import (
    BATCH_PROFILE_NAME,
    prompt_version,
)

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
    #: What was SENT — which regions of the paper, chosen by which selector.
    #: Set by the caller that did the windowing, because the tool is handed a
    #: document and cannot know what was left out of it.
    excerpt: ExcerptProvenance | None = None
    attempts: list[BatchAttempt] = field(default_factory=list)
    failure: str = ""
    detail: str = ""
    #: Paper labels transcribed per located index. Kept off BatchEntry because
    #: schemas/batch.py is inside the frozen evaluator boundary.
    field_names: dict = field(default_factory=dict)

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

    def persistent(self) -> BatchReadingRecord:
        """The form an artifact keeps, which is not the form a run holds.

        What the working object contains may change freely; what a written
        record promises may not, so the two are separate shapes and this is the
        one place that maps between them.
        """
        reading = self.reading
        return BatchReadingRecord(
            question_id=self.question.identity(),
            execution_id=self.execution_id,
            study_id=self.question.study_id,
            field_type=self.question.field_type,
            target_shape=self.question.target_shape,
            claim_ids=(self.execution.claim_ids() if self.execution else []),
            attempts=len(self.attempts),
            served_from_cache=self.served_from_cache,
            failure=self.failure, detail=self.detail,
            parse_errors=[str(r.get("reason", "")) for r in
                          (reading.rejected if reading else [])],
            usable_readings=len(reading.usable) if reading else 0,
            rejected_readings=len(reading.rejected) if reading else 0,
            model_payload=self.model_payload,
            excerpt_provenance=self.excerpt)


class ExtractSourceBatchTool:
    """One prompt, every reading of one field, under the v5 contract."""

    name = "extract_source_batch"
    #: Registered beside the single-target extractor. Not a `Tool` subclass:
    #: its entry point takes a question and a prompt rather than one typed
    #: payload, and pretending otherwise would mean inventing a payload shape
    #: that nothing sends.
    stage = ToolStage.EXTRACT

    def __init__(self, backend: LLMBackend | None, *,
                 locate_backend: LLMBackend | None = None,
                 transcribe_backend: LLMBackend | None = None,
                 cache: ExtractionCache | None = None, cache_mode: str = "live",
                 max_attempts: int = 3, telemetry=None) -> None:
        if cache_mode not in {"live", "record", "replay"}:
            raise ValueError("cache_mode must be live, record, or replay")
        if cache_mode in {"record", "replay"} and cache is None:
            raise ValueError(f"{cache_mode} extraction requires a cache")
        if cache_mode != "replay" and backend is None:
            raise ValueError("a live or recording run needs a backend")
        self._backend = backend
        self._locate_backend = locate_backend if locate_backend is not None else backend
        self._transcribe_backend = (
            transcribe_backend if transcribe_backend is not None else backend)
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
                # A retry costs a call, and the run's own repeated-attempt
                # counter is where that has always been recorded. Leaving it out
                # would make a batch that failed twice look as cheap as one that
                # answered first time.
                if (failure in RETRYABLE and attempt < self._max_attempts - 1
                        and self._telemetry is not None):
                    self._telemetry.repeated_attempts += 1
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
                if self._telemetry is not None:
                    self._telemetry.repeated_attempts += 1
                record.attempts[-1].failure = BAD_SHAPE
                record.attempts[-1].detail = reading.batch_error
                continue

            record.model_payload = payload
            record.reading = reading
            return record
        return record

    async def read_split(self, *, question: BatchQuestionId, locate_prompt: str,
                         document: str) -> BatchRecord:
        """Locate every reading, then transcribe every quote — two calls, not N."""
        record = BatchRecord(question=question)
        locate_payload, locate_ok = await self._retry_payload(
            record, locate_prompt, version=BATCH_LOCATE_VERSION,
            backend=self._locate_backend)
        if not locate_ok:
            return record
        located, nothing = parse_locate(locate_payload, document)
        if not located:
            record.model_payload = locate_payload if isinstance(locate_payload, dict) else None
            record.reading = BatchReading(nothing_reported_reason=nothing)
            return record

        transcribe_prompt = build_batch_transcribe_prompt(
            concept=question.concept or question.raw_field_name or question.field_type,
            raw_label=question.raw_field_name or question.concept,
            field_type=question.field_type, unit_hint=question.unit_hint,
            passages=format_passages(located))
        transcribe_payload, transcribe_ok = await self._retry_payload(
            record, transcribe_prompt, version=BATCH_TRANSCRIBE_VERSION,
            backend=self._transcribe_backend)
        if not transcribe_ok:
            return record
        reading, field_names = merge_located_and_transcribed(
            located, transcribe_payload, document,
            target_shape=question.target_shape)
        if reading.batch_error:
            record.model_payload = (transcribe_payload
                                    if isinstance(transcribe_payload, dict) else None)
            record.failure, record.detail = BAD_SHAPE, reading.batch_error
            record.reading = reading
            return record
        record.model_payload = transcribe_payload if isinstance(transcribe_payload, dict) else None
        record.reading = reading
        record.field_names = field_names
        return record

    async def _retry_payload(self, record: BatchRecord, prompt: str, *,
                             version: str, backend) -> tuple[object | None, bool]:
        """Retry transport / not-JSON under one prompt version."""
        for attempt in range(self._max_attempts):
            key = self._key(prompt, attempt, version=version, backend=backend)
            payload, failure, detail, cached = await self._one_attempt(
                prompt, key, backend=backend)
            record.attempts.append(BatchAttempt(
                attempt=attempt, cache_key=key, served_from_cache=cached,
                failure=failure, detail=detail))
            if failure:
                if (failure in RETRYABLE and attempt < self._max_attempts - 1
                        and self._telemetry is not None):
                    self._telemetry.repeated_attempts += 1
                if failure not in RETRYABLE or attempt == self._max_attempts - 1:
                    record.failure, record.detail = failure, detail
                    return None, False
                continue
            return payload, True
        return None, False

    async def _one_attempt(self, prompt: str, key: str, *, backend=None):
        """One call or one cache read. Never both, and never a silent fallback."""
        # An attempt is an attempt whether or not a recording answers it. The
        # single-target tool has always counted it before the cache lookup, and
        # `backend_requests` is the counter that means "the model was asked" —
        # counting only real calls here made a replayed run look like it did no
        # work at all.
        if self._telemetry is not None:
            self._telemetry.attempt(self.name)
        try:
            recorded = (self._cache.get(key)
                        if self._cache_mode in {"record", "replay"} else None)
            if recorded is not None:
                self._record_cache(hits=1)
                return recorded, "", "", True
            # Counted where the lookup failed, and BEFORE the refusal below: a
            # replay miss is the most interesting miss there is, and raising
            # first meant it was the one miss nothing recorded.
            self._record_cache(misses=1)
            if self._cache_mode == "replay":
                # A run that says replay must mean it. Reaching for the model
                # here, or dropping to an older profile, would produce an
                # artifact whose own label is false.
                raise ExtractionCacheMiss(
                    "no recorded batch for this question and attempt")
            who = backend if backend is not None else self._backend
            assert who is not None
            raw = await who.complete(prompt)
            try:
                payload = parse_llm_response(raw, who.model_id)
            except LLMError as exc:
                # The model answered; it just did not answer in JSON. Recording
                # that as a transport failure would make a formatting problem
                # indistinguishable from the network being down, and the two
                # call for different responses even though both are retried.
                return None, NOT_JSON, f"{type(exc).__name__}: {exc}"[:300], False
            if not isinstance(payload, dict):
                return None, NOT_JSON, "the response did not decode to an object", False
            if self._cache_mode == "record" and self._cache is not None:
                self._cache.put(key, payload, model_id=who.model_id)
            return payload, "", "", False
        except ExtractionCacheMiss:
            raise
        except Exception as exc:                     # transport, decode, provider
            logger.warning("extract_source_batch_failed", error=str(exc)[:160])
            return None, TRANSPORT, f"{type(exc).__name__}: {exc}"[:300], False

    def _record_cache(self, *, hits: int = 0, misses: int = 0) -> None:
        """Only the batch stage's bucket.

        The accuracy harness folds each cache's own totals into the global
        counters when a run ends, so touching those here would double every
        number it reports.
        """
        if self._telemetry is None or self._cache_mode == "live":
            return
        self._telemetry.record_stage_cache(BATCH_EXTRACTION, hits=hits,
                                           misses=misses)

    def _key(self, prompt: str, attempt: int, *, version: str | None = None,
             backend=None) -> str:
        """The address a recording lives at.

        Unchanged from the single-target contract on purpose: model, prompt
        version, prompt bytes, attempt and seed. The QUESTION id says what was
        asked and is not this — a cache entry is about words sent to a model,
        and two runs that send the same words may share it whatever they intend
        to do with the answer.
        """
        who = backend if backend is not None else self._backend
        model_id = ((who.model_id if who is not None else "")
                    or (self._cache.model_id if self._cache is not None else "")
                    or "replay")
        return extraction_cache_key(
            model_id=model_id,
            prompt_version=version or prompt_version(BATCH_PROFILE_NAME),
            prompt=prompt, attempt=attempt)


def prompt_sha256(prompt: str) -> str:
    """What the question id records, so a reworded prompt is a new question."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest().upper()
