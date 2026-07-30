"""Citation resolvers: the interface + an offline stub.

A resolver turns a :class:`ReferenceQuery` into candidate works from ONE online
service (CrossRef / OpenAlex / Europe PMC). Live network clients land in step 2;
this step ships the Protocol + an offline :class:`StaticResolver` so the
reconciler and its confidence gate are fully unit-testable without the network.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from react_review.tools.search.models import CandidateWork, ReferenceQuery


@runtime_checkable
class CitationResolver(Protocol):
    """Resolve a citation to candidate works from one service."""

    name: str

    async def resolve(self, query: ReferenceQuery) -> list[CandidateWork]:
        ...


class StaticResolver:
    """Offline resolver returning preset candidates (tests + mock catalogue)."""

    def __init__(self, name: str, candidates: list[CandidateWork] | None = None) -> None:
        self.name = name
        self._candidates = candidates or []

    async def resolve(self, query: ReferenceQuery) -> list[CandidateWork]:
        return list(self._candidates)
