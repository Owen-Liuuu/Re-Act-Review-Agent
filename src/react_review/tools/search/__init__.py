"""Search tools: search-strategy validation + reference reconciliation.

``SearchPubMedTool`` / ``CountResultsTool`` reproduce a review's literature search
(legacy Step 1). ``ResolveReferenceTool`` + ``ReferenceReconciler`` resolve a
citation with no printed DOI to a GATED DOI via online services — Agent 1's
"search" capability. Live network clients are step 2; this package ships the
reconciler + the confidence gate + an offline stub resolver, fully unit-tested.
"""
from react_review.tools.search.clients import CitationResolver, StaticResolver
from react_review.tools.search.gate import (
    DEFAULT_THRESHOLD,
    ReferenceMatch,
    evaluate,
    score_match,
)
from react_review.tools.search.live_clients import (
    CrossRefResolver,
    EuropePMCResolver,
    OpenAlexResolver,
)
from react_review.tools.search.models import (
    CandidateWork,
    ReferenceQuery,
    ResolvedReference,
    ResolveReferenceInput,
    ResolveReferenceResult,
)
from react_review.tools.search.reconciler import ReferenceReconciler
from react_review.tools.search.resolve_reference import ResolveReferenceTool
from react_review.tools.search.validation import CountResultsTool, SearchPubMedTool

__all__ = [
    # legacy search-strategy validation
    "SearchPubMedTool", "CountResultsTool",
    # reference reconciliation
    "CitationResolver", "StaticResolver",
    "CrossRefResolver", "OpenAlexResolver", "EuropePMCResolver",
    "score_match", "evaluate", "ReferenceMatch", "DEFAULT_THRESHOLD",
    "CandidateWork", "ReferenceQuery", "ResolvedReference",
    "ResolveReferenceInput", "ResolveReferenceResult",
    "ReferenceReconciler", "ResolveReferenceTool",
]
