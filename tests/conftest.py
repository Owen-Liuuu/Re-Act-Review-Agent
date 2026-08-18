"""Shared test fixtures for react_review tests."""
from __future__ import annotations

import pytest

from react_review.core.config import AppConfig
from react_review.steps.paper_verification.schemas import ReferenceEntry


@pytest.fixture
def sample_config() -> AppConfig:
    """Provide a default test configuration."""
    return AppConfig()


@pytest.fixture
def sample_reference() -> ReferenceEntry:
    """Provide a sample reference entry."""
    return ReferenceEntry(
        title="Effectiveness of immunotherapy in advanced lung cancer",
        authors=["Zhang Y", "Li X"],
        journal="Journal of Clinical Oncology",
        year=2023,
        doi="10.1234/test-doi-001",
    )


@pytest.fixture
def sample_references(sample_reference: ReferenceEntry) -> list[ReferenceEntry]:
    """Provide a list of sample references."""
    return [
        sample_reference,
        ReferenceEntry(
            title="Chemotherapy outcomes in breast cancer",
            authors=["Smith A"],
            journal="The Lancet Oncology",
            year=2022,
            doi="10.1234/test-doi-002",
        ),
    ]


# --- mid-phase, the evaluator legitimately matches no published manifest -----

def requires_frozen_evaluator() -> None:
    """Skip what is only true of a checkout that IS a published evaluator.

    While `configs/aggregation/evaluators/PENDING.json` exists, the boundary
    files are being changed and `evaluator_readiness` refuses to bind a runtime
    — which is the gate working, not a broken test. A test that resolves a
    runtime therefore cannot pass mid-phase, and the two ways to make it pass
    are to weaken the gate or to regenerate a published manifest at an
    intermediate commit. Both are the failure the marker exists to prevent, so
    the test steps aside instead and comes back when the freeze lands.
    """
    import pytest

    from react_review.contracts import repo_root

    if (repo_root() / "configs/aggregation/evaluators/PENDING.json").exists():
        pytest.skip("the evaluator is declared unfrozen in PENDING.json")
