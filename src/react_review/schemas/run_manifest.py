"""What a run executed as, and what it was decided by — as a record.

Pure data, kept with the other schemas so an EvidencePackage can carry it
without the schema layer depending on the tool layer. The RULES live in
:mod:`react_review.run_profile`; this is only the account of them, written
beside the results — including the partial package a stopped run leaves behind,
which is exactly when a reader most needs to know what the run was doing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_serializer

from react_review.contracts import ContractError, one_of, sha256_file

EXTRACTION_MODES = ("live", "record", "replay")
SEMANTIC_MODES = ("off", "cache-only", "on")


class ExecutionMode(BaseModel):
    """How the model was reached. Free to vary under one contract."""

    extraction_mode: str = "live"
    extraction_cache: Path | None = None
    semantic_mode: str = "on"
    semantic_cache: Path | None = None

    def validate_modes(self) -> "ExecutionMode":
        one_of(self.extraction_mode, EXTRACTION_MODES, field="extraction mode")
        one_of(self.semantic_mode, SEMANTIC_MODES, field="semantic mode")
        if self.extraction_mode in ("record", "replay") and self.extraction_cache is None:
            raise ContractError(
                f"{self.extraction_mode} extraction needs an extraction cache path")
        if self.semantic_mode == "cache-only" and self.semantic_cache is None:
            raise ContractError("cache-only semantic comparison needs a cache path")
        return self

    def identity(self) -> dict[str, Any]:
        return {
            "extraction_mode": self.extraction_mode,
            "extraction_cache": str(self.extraction_cache or ""),
            "semantic_mode": self.semantic_mode,
            "semantic_cache": str(self.semantic_cache or ""),
        }


class RunManifest(BaseModel):
    """What actually ran: the resolved contract, the mode, and the inputs.

    Written beside the results — including the partial package a stopped run
    leaves behind, which is exactly the case where a reader most needs to know
    what the run was doing when it stopped.
    """

    contract: dict[str, Any] = Field(default_factory=dict)
    execution: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, str] = Field(default_factory=dict)
    # cli | parsed | default — where the research context actually came from.
    context_source: str = "default"
    # Filled once the run finishes; a partial manifest records the cache PATH and
    # what has been written so far, not a hash of a file still being appended to.
    extraction_cache_sha256: str = ""
    semantic_cache_sha256: str = ""
    #: The aggregation policy and the evaluator that cleared it, resolved once
    #: at startup. Present only for a run that batches — a run that never
    #: aggregated anything must not name an evaluator, or it attributes its
    #: answers to code that never ran. Omitted from the serialised form when
    #: empty, so every manifest already written is unchanged.
    aggregation_runtime: dict[str, Any] = Field(default_factory=dict)
    complete: bool = False

    @model_serializer(mode="wrap")
    def _omit_unused_runtime(self, handler):
        body = handler(self)
        if not body.get("aggregation_runtime"):
            body.pop("aggregation_runtime", None)
        return body

    @staticmethod
    def runtime_of(runtime) -> dict[str, Any]:
        """The bound policy-and-identity, as a manifest records it."""
        if runtime is None:
            return {}
        who = runtime.evaluator
        return {
            "policy_id": runtime.policy.policy_id,
            "policy_sha256": runtime.policy.sha256,
            "evaluator_id": who.evaluator_id,
            "evaluator_version": who.evaluator_version,
            "evaluator_hash": who.evaluator_hash,
            "git_commit": who.git_commit,
            "git_commit_matches_evaluator": who.git_commit_matches_evaluator,
            "status": who.status,
            "release_eligible": runtime.release_eligible,
        }

    @classmethod
    def of(cls, contract: RunContractProfile, execution: ExecutionMode, *,
           inputs: dict[str, str] | None = None,
           context_source: str = "default") -> "RunManifest":
        return cls(contract=contract.identity(), execution=execution.identity(),
                   inputs=dict(inputs or {}), context_source=context_source)

    def finalise(self, execution: ExecutionMode) -> "RunManifest":
        """Add the cache content hashes, once nothing more will be written."""
        for attr, key in (("extraction_cache", "extraction_cache_sha256"),
                          ("semantic_cache", "semantic_cache_sha256")):
            path = getattr(execution, attr)
            if path is not None and Path(path).is_file():
                setattr(self, key, sha256_file(path))
        self.complete = True
        return self

    def same_contract_as(self, other: "RunManifest") -> bool:
        """Whether two runs asked the same question, whatever mode they ran in."""
        return (self.contract == other.contract
                and self.inputs == other.inputs
                and self.context_source == other.context_source)
