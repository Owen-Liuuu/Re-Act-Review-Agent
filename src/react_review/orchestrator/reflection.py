"""Reflection Decider (Proposal E6): accept / retry / escalate, in pure Python.

After each stage the orchestrator asks the decider what to do next based on
external signals — retrieval success, dual-LLM agreement, and a per-value
confidence — rather than letting an LLM decide the control flow. Keeping this
deterministic and threshold-driven is the point (BCS "restricts agent freedom";
confidence comes from external signals, Kadavath et al.).

MVP note: confidence is a placeholder until P4 (decision 7), so it will often be
``None`` — the decider then leans on retrieval / agreement and, absent any
negative signal, ACCEPTs.
"""
from __future__ import annotations

from pydantic import BaseModel

from react_review.core.enums import ReflectionDecision


class ReflectionSignals(BaseModel):
    """The evidence the decider weighs.

    Attributes:
        confidence: per-value confidence in [0, 1], or None when unavailable.
        attempt: how many attempts have already been made (0 = first try).
        retrieval_ok: whether the source / value was successfully obtained.
        agreement: dual-LLM agreement — True / False / None (single model).
    """

    confidence: float | None = None
    attempt: int = 0
    retrieval_ok: bool = True
    agreement: bool | None = None


class ReflectionOutcome(BaseModel):
    decision: ReflectionDecision
    reason: str = ""


class ReflectionDecider:
    """Threshold-based accept/retry/escalate routing."""

    def __init__(
        self,
        *,
        accept_threshold: float = 0.7,
        escalate_threshold: float = 0.4,
        max_attempts: int = 3,
    ) -> None:
        if not (0.0 <= escalate_threshold <= accept_threshold <= 1.0):
            raise ValueError(
                "require 0 <= escalate_threshold <= accept_threshold <= 1"
            )
        self._accept = accept_threshold
        self._escalate = escalate_threshold
        self._max_attempts = max(1, max_attempts)

    def decide(self, signals: ReflectionSignals) -> ReflectionOutcome:
        can_retry = signals.attempt + 1 < self._max_attempts

        # 1. Could not obtain the evidence — retry (switch source) if allowed.
        if not signals.retrieval_ok:
            if can_retry:
                return self._out(ReflectionDecision.RETRY, "retrieval failed; retry / switch source")
            return self._out(ReflectionDecision.ESCALATE, "retrieval failed and retries exhausted")

        # 2. The two extractors disagreed — re-extract if allowed.
        if signals.agreement is False:
            if can_retry:
                return self._out(ReflectionDecision.RETRY, "dual-LLM disagreement; re-extract")
            return self._out(ReflectionDecision.ESCALATE, "dual-LLM disagreement and retries exhausted")

        # 3. Evidence obtained and models agree (or single model). Weigh confidence.
        conf = signals.confidence
        if conf is None:
            return self._out(ReflectionDecision.ACCEPT, "no confidence signal; no negative signal → accept")
        if conf >= self._accept:
            return self._out(ReflectionDecision.ACCEPT, f"confidence {conf:.2f} ≥ {self._accept:.2f}")
        if conf < self._escalate:
            return self._out(ReflectionDecision.ESCALATE, f"confidence {conf:.2f} < {self._escalate:.2f}")
        # Middling confidence — retry to improve it, else escalate.
        if can_retry:
            return self._out(ReflectionDecision.RETRY, f"confidence {conf:.2f} in review band; retry")
        return self._out(ReflectionDecision.ESCALATE, f"confidence {conf:.2f} in review band; retries exhausted")

    @staticmethod
    def _out(decision: ReflectionDecision, reason: str) -> ReflectionOutcome:
        return ReflectionOutcome(decision=decision, reason=reason)
