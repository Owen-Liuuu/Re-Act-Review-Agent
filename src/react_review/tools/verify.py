"""Verify stage: reference existence + metadata check (wraps CrossRefVerifier).

Currently one combined tool (the reused ``ReferenceVerifier.verify`` resolves
the reference AND validates its metadata in one call). A P2 refinement may
split this into ``resolve_reference`` + ``validate_metadata`` when the Collector
needs them as separate ReAct actions.
"""
from __future__ import annotations

from react_review.steps.paper_verification.interfaces import ReferenceVerifier
from react_review.steps.paper_verification.schemas import (
    ReferenceEntry,
    ReferenceVerificationResult,
)
from react_review.tools.base import Tool, ToolStage


class VerifyReferenceTool(Tool):
    """Verify a reference exists and its metadata agrees (CrossRef)."""

    name = "verify_reference"
    stage = ToolStage.VERIFY
    input_model = ReferenceEntry
    output_model = ReferenceVerificationResult

    def __init__(self, verifier: ReferenceVerifier) -> None:
        self._verifier = verifier

    async def run(self, payload: ReferenceEntry) -> ReferenceVerificationResult:
        return await self._verifier.verify(payload)
