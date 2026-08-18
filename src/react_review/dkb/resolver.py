"""DKB Resolver — the single owner of field normalization.

The Parser does RAW extraction only and knows no domain knowledge; it hands a raw
field name here. The Resolver runs the cascade and returns a ``ResolvedField`` with
an explicit STATUS:

    cache / DKB-1 deterministic match  → authoritative   (a firm, known concept)
    DKB-2 retrieval + LLM fallback     → candidate        (provisional — NOT firm)
    nothing                            → unresolved

A miss never becomes a confirmed field_type: it is a CANDIDATE the run uses
tentatively, surfaced for human review and collected as a proposal. In audit mode
(``write_back=False``) candidates are NOT merged into the KB — that is the separate
developer "learn" step (promotion + curation). The per-run cache still dedupes so
the same raw name isn't sent to the LLM twice within one run.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
import structlog

from react_review.dkb.agent import AgentClassification, KnowledgeAgent, KnowledgeAgentError
from react_review.dkb.base import KnowledgeBase
from react_review.dkb.retrieval import KeywordRetriever
from react_review.dkb.schema import KnowledgeEntry
from react_review.dkb.verify import verify_candidate
from react_review.normalize.numeric import primary_number
from react_review.normalize.units import normalize_unit, unit_kind
from react_review.schemas.reason import ReasonRecord
from react_review.schemas.resolution import FieldResolutionRecord, ResolutionAttempt

logger = structlog.get_logger(__name__)


class ResolvedField(FieldResolutionRecord):
    """The outcome of resolving one raw field name.

    The Resolver PROVIDES knowledge (concept + scope); the Parser APPLIES scope.
    """

    # Kept optional only for backwards-compatible construction in callers that
    # create a last-resort unresolved result.  Normal Resolver output always has
    # the deterministic hash produced by :func:`resolution_key`.
    resolution_key: str = ""

    @property
    def resolved(self) -> bool:
        return self.field_type is not None

    @property
    def provisional(self) -> bool:
        return self.status == "candidate"


def resolution_key(raw_name: str, unit: str, context: str, modality: str = "") -> str:
    """Every signal that can change the answer must be part of the key.

    Modality included: one ambiguous header can resolve to a length in a study
    that measured it one way and a volume in a study that measured it another, so
    caching on the name alone would give every later row the first row's answer —
    silently.
    """
    material = json.dumps({
        "raw_field_name": raw_name.strip().lower(),
        "unit": unit.strip().lower(),
        "research_context": context.strip().lower(),
        "modality": modality.strip().lower(),
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _context_hash(context: str) -> str:
    return hashlib.sha256((context or "").encode("utf-8")).hexdigest()


def _candidate_resolution_key(
    raw_name: str, unit: str, context: str, value: object,
) -> str:
    """Stable identity for an LLM question after deterministic resolution misses.

    The current candidate prompt does not consume a row's full modality/context
    string, so putting that string (which includes author and values) in the key
    creates one supposedly "concept-level" decision per row.  The observed value
    kind *does* affect the deterministic self-contract, so it remains in the key.
    """
    value_kind = "missing" if value is None else (
        "numeric" if primary_number(value) is not None else "text")
    material = json.dumps({
        "raw_field_name": raw_name.strip().lower(),
        "unit": unit.strip().lower(),
        "research_context": context.strip().lower(),
        "observed_value_kind": value_kind,
        "path": "candidate",
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _unit_signature(unit: str) -> str:
    kind = unit_kind(unit)
    return f"kind:{kind}" if kind != "unknown" else f"unit:{normalize_unit(unit)}"


def _classification_signature(c: AgentClassification) -> tuple:
    """Fields whose agreement actually says two attempts mean the same thing.

    A known concept has a closed canonical vocabulary, so its field_type must
    match exactly.  A new concept has no canonical name yet; comparing its
    invented snake_case spelling would turn harmless naming drift into a false
    disagreement, so only its structural contract is compared.
    """
    attempt = c.attempt or ResolutionAttempt(field_type=c.field_type, is_new=c.is_new)
    structural = (
        c.is_new,
        (attempt.value_type or "").strip().lower(),
        _unit_signature(attempt.default_unit),
        (attempt.scope or "cohort").strip().lower(),
        tuple(sorted(set(c.grounded_on))),
    )
    return structural if c.is_new else (c.field_type, *structural)


@dataclass
class _SampledCandidate:
    candidate: AgentClassification | None = None
    attempts: list[ResolutionAttempt] = field(default_factory=list)
    candidate_names: list[str] = field(default_factory=list)
    consensus_count: int = 0
    stability: str = "unstable"
    errors: list[str] = field(default_factory=list)


class FieldResolver:
    """Resolve raw field names to canonical concepts. Owns all domain knowledge."""

    def __init__(
        self,
        kb: KnowledgeBase,
        agent: KnowledgeAgent | None = None,
        *,
        backend=None,
        cache: dict[str, ResolvedField | str] | None = None,
        write_back: bool = False,
    ) -> None:
        self._kb = kb
        if agent is None and backend is not None:      # convenience: build a default agent
            agent = KnowledgeAgent(backend, KeywordRetriever(kb))
        self._agent = agent
        # Full records, not just field_type strings: a cache hit must retain the
        # checks, reasons and model-attempt provenance that justified the mapping.
        # Legacy string values remain readable for callers with an old in-memory
        # cache, but every new write stores a ResolvedField.
        self._cache: dict[str, ResolvedField | str] = cache if cache is not None else {}
        self._records: dict[str, ResolvedField] = {}
        self._write_back = write_back
        # Candidate concepts proposed this session (fuel for the developer learn step).
        self.proposals: list[KnowledgeEntry] = []

    @property
    def kb(self) -> KnowledgeBase:
        return self._kb

    @property
    def records(self) -> list[ResolvedField]:
        """Unique decisions made this run, in first-seen order."""
        return [r.model_copy(deep=True) for r in self._records.values()]

    def _status_of(self, field_type: str) -> str:
        e = self._kb.entries.get(field_type)
        return "candidate" if (e and e.status == "provisional") else "authoritative"

    def _scope_of(self, field_type: str) -> str:
        e = self._kb.entries.get(field_type)
        return e.scope if e else "cohort"

    @staticmethod
    def _seen(status: str, field_type: str | None) -> dict:
        return {
            "statuses_seen": [status],
            "field_types_seen": [field_type] if field_type else [],
        }

    def _remember(self, result: ResolvedField, *, cache: bool = False) -> ResolvedField:
        self._records[result.resolution_key] = result.model_copy(deep=True)
        if cache:
            self._cache[result.resolution_key] = result.model_copy(deep=True)
        return result

    def _from_cache(self, key: str, base: dict) -> ResolvedField | None:
        """Return one full cached decision without erasing its provenance."""
        if key not in self._cache:
            return None
        cached = self._cache[key]
        if isinstance(cached, ResolvedField):
            cumulative = cached.model_copy(
                deep=True, update={"cache_hits": cached.cache_hits + 1})
            self._cache[key] = cumulative.model_copy(deep=True)
            self._records[key] = cumulative.model_copy(deep=True)
            # Per-call return says this row was one cache hit so parser-level
            # aggregation can sum hits without double-counting.
            return cached.model_copy(deep=True, update={"cache_hits": 1})
        ft = cached
        status = self._status_of(ft) if ft else "unresolved"
        result = ResolvedField(
            **base, field_type=ft or None, status=status,
            scope=self._scope_of(ft) if ft else "cohort", source="legacy_cache",
            checks={"legacy_cache": True}, cache_hits=1,
            **self._seen(status, ft or None),
        )
        self._records[key] = result.model_copy(deep=True)
        return result

    async def _sample_candidate(
        self, raw_field_name: str, unit: str, research_context: str, modality: str,
    ) -> _SampledCandidate:
        """Ask twice with different seeds; use a third only to break a split.

        Agreement can reject an unstable proposal, but it never upgrades a
        proposal beyond ``candidate``.  Model confidence is not consulted.
        """
        assert self._agent is not None
        samples: list[AgentClassification] = []
        attempts: list[ResolutionAttempt] = []
        errors: list[str] = []

        async def ask(seed: int) -> None:
            try:
                answer = await self._agent.classify(
                    raw_field_name, unit, research_context, modality, seed=seed)
                samples.append(answer)
                if answer.attempt is not None:
                    attempts.append(answer.attempt)
            except Exception as exc:                          # noqa: BLE001
                logger.warning("dkb_agent_failed", raw=raw_field_name, seed=seed,
                               error=str(exc)[:160])
                errors.append(str(exc)[:200])
                if isinstance(exc, KnowledgeAgentError):
                    attempts.append(exc.attempt)

        await ask(42)
        await ask(43)
        counts = Counter(_classification_signature(c) for c in samples)
        if not counts or counts.most_common(1)[0][1] < 2:
            await ask(44)
            counts = Counter(_classification_signature(c) for c in samples)

        names = list(dict.fromkeys(c.field_type for c in samples))
        if not counts:
            return _SampledCandidate(
                attempts=attempts, candidate_names=names, errors=errors)
        signature, count = counts.most_common(1)[0]
        if count < 2:
            return _SampledCandidate(
                attempts=attempts, candidate_names=names,
                consensus_count=count, errors=errors)

        members = [c for c in samples if _classification_signature(c) == signature]
        chosen = members[0]
        if chosen.is_new:
            # New names are proposals, not ground truth. Choose a deterministic
            # spelling and retain every alternative as a synonym for curation.
            canonical = min(c.field_type for c in members)
            chosen = next(c for c in members if c.field_type == canonical)
            if chosen.entry is not None:
                synonyms = list(dict.fromkeys([
                    *chosen.entry.synonyms, *(c.field_type for c in members)]))
                chosen = chosen.model_copy(deep=True, update={
                    "field_type": canonical,
                    "entry": chosen.entry.model_copy(
                        deep=True, update={"field_type": canonical, "synonyms": synonyms}),
                })
        return _SampledCandidate(
            candidate=chosen, attempts=attempts, candidate_names=names,
            consensus_count=count, stability="stable", errors=errors)

    async def resolve(
        self, raw_field_name: str, unit: str = "",
        modality: str = "", research_context: str = "", value: object = None,
    ) -> ResolvedField:
        key = resolution_key(raw_field_name, unit, research_context, modality)
        base = dict(
            resolution_key=key, raw_field_name=raw_field_name, unit=unit,
            modality=modality,
            research_context_sha256=_context_hash(research_context),
        )

        # 1. cache (per-run; dedupes repeat names without re-hitting the LLM)
        cached_result = self._from_cache(key, base)
        if cached_result is not None:
            return cached_result

        # 2. DKB-1 deterministic match (synonym + unit/modality disambiguation)
        ft = self._kb.resolve(raw_field_name, unit, modality)
        if ft:
            status = self._status_of(ft)
            return self._remember(ResolvedField(
                **base, field_type=ft, status=status, scope=self._scope_of(ft),
                source="deterministic", checks={"deterministic_match": True},
                **self._seen(status, ft),
            ), cache=True)

        # The model fallback does not consume the row's full context string.
        # Collapse rows that ask the same model question, while preserving the
        # observed value kind that the deterministic self-contract consumes.
        key = _candidate_resolution_key(raw_field_name, unit, research_context, value)
        base.update({"resolution_key": key, "modality": ""})
        cached_result = self._from_cache(key, base)
        if cached_result is not None:
            return cached_result

        # 3. DKB-2 retrieval + LLM → a CANDIDATE (never a firm field_type on a miss)
        if self._agent is None:
            return self._remember(ResolvedField(
                **base, checks={"deterministic_match": False, "agent_available": False},
                reasons=[ReasonRecord(
                    code="concept_unresolved", stage="field_resolution",
                    message=f"column {raw_field_name!r} did not map to a known concept")],
                **self._seen("unresolved", None),
            ))
        sampled = await self._sample_candidate(
            raw_field_name, unit, research_context, modality)
        if sampled.candidate is None:
            all_failed = not sampled.candidate_names
            reason = ("all cross-seed classification attempts failed" if all_failed else
                      "cross-seed classifications had no two matching structural answers")
            if sampled.errors:
                reason += "; " + "; ".join(sampled.errors)
            result = ResolvedField(
                **base, source="retrieval_llm", stability="unstable",
                consensus_count=sampled.consensus_count,
                candidate_names=sampled.candidate_names,
                checks={"deterministic_match": False,
                        "agent_call": bool(sampled.candidate_names),
                        "resampling_consistency": False},
                attempts=sampled.attempts,
                reasons=[ReasonRecord(
                    code=("concept_resolution_exception" if all_failed else
                          "candidate_unstable"),
                    stage="field_resolution",
                    source=("exception" if all_failed else "llm"), message=reason)],
                **self._seen("unresolved", None),
            )
            # A semantic split is reproducible for these fixed seeds; an API or
            # parse error may be transient and must not poison the run cache.
            return self._remember(result, cache=not sampled.errors)
        c = sampled.candidate

        # Cross-seed agreement only says the mapping is stable. It remains a
        # hypothesis and becomes usable as a candidate only after deterministic
        # self-contract / known-concept unit and range checks also pass.
        verdict = verify_candidate(
            c.field_type, kb=self._kb, unit=unit, value=value,
            is_new=(c.entry is not None), confidence=c.confidence,
            grounded_on=c.grounded_on, proposal=c.entry,
        )
        attempts = sampled.attempts
        proposal = c.entry.model_dump(mode="json") if c.entry is not None else None
        attempt_reasons = [ReasonRecord(
            code="candidate_attempt_error", stage="field_resolution",
            source="exception", message=message)
            for message in sampled.errors]
        checks = {
            "deterministic_match": False, "agent_call": True,
            "resampling_consistency": True,
            "all_attempts_succeeded": not sampled.errors,
            **verdict.checks}
        if not verdict.ok:
            logger.info("dkb_candidate_rejected", raw=raw_field_name,
                        field_type=c.field_type, reason=verdict.reason)
            result = ResolvedField(
                **base, source="retrieval_llm", grounded_on=c.grounded_on,
                confidence=c.confidence, checks=checks, attempts=attempts,
                proposal=proposal, stability="stable",
                consensus_count=sampled.consensus_count,
                candidate_names=sampled.candidate_names,
                reasons=[ReasonRecord(
                    code="candidate_rejected", stage="field_resolution",
                    message=verdict.reason), *attempt_reasons],
                **self._seen("unresolved", None),
            )
            # Range and observed-value-type failures are value-dependent and
            # must not poison another row with the same field question.
            cache_rejection = (
                verdict.checks.get("range", True)
                and verdict.checks.get("observed_value_type", True))
            return self._remember(result, cache=cache_rejection)

        if c.entry is not None:
            self.proposals.append(c.entry)          # collect the proposal for the learn step
            if self._write_back:                    # only the developer/learn path mutates the KB
                self._kb.add(c.entry)
        scope = c.entry.scope if c.entry is not None else self._scope_of(c.field_type)
        return self._remember(ResolvedField(
            **base, field_type=c.field_type, status="candidate", scope=scope,
            source="retrieval_llm", grounded_on=c.grounded_on,
            confidence=c.confidence, checks=checks, attempts=attempts,
            proposal=proposal, stability="stable",
            consensus_count=sampled.consensus_count,
            candidate_names=sampled.candidate_names,
            reasons=attempt_reasons,
            **self._seen("candidate", c.field_type),
        ), cache=True)
