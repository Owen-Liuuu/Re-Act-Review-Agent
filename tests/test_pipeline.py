"""Integration test: run the full pipeline with mock implementations."""
from __future__ import annotations

import pytest

from react_review.core.config import AppConfig
from react_review.pipeline.factory import create_pipeline
from react_review.pipeline.schemas import PipelineRunResult, StudentReviewInput


@pytest.mark.asyncio
async def test_full_pipeline_with_mocks(sample_student_input: StudentReviewInput):
    """The pipeline should run all 4 steps with mock implementations."""
    config = AppConfig(mock_mode=True)
    pipeline = create_pipeline(config)

    result = await pipeline.run(sample_student_input)

    assert isinstance(result, PipelineRunResult)
    assert result.run_id
    assert result.completed_at is not None

    # Step 1: search validation ran
    assert result.search_result is not None
    assert result.search_result.is_reproducible is True

    # Step 2: paper verification ran for each selected paper
    assert len(result.verification_results) == len(sample_student_input.selected_papers)

    # Step 3: extraction ran (2 papers x 2 extractors = 4 tables)
    assert len(result.extracted_tables) == 4

    # Step 4: comparison and report
    assert result.report is not None
    assert result.report.run_id == result.run_id


@pytest.mark.asyncio
async def test_pipeline_with_subset_steps(sample_student_input: StudentReviewInput):
    """The pipeline should support running only selected steps."""
    config = AppConfig(
        mock_mode=True,
        enabled_steps=["search_validation"],
    )
    pipeline = create_pipeline(config)
    result = await pipeline.run(sample_student_input)

    # Only step 1 should have run
    assert result.search_result is not None
    assert len(result.verification_results) == 0
    assert len(result.extracted_tables) == 0
    assert result.report is None


@pytest.mark.asyncio
async def test_pipeline_result_serializable(sample_student_input: StudentReviewInput):
    """PipelineRunResult should be JSON-serialisable."""
    config = AppConfig(mock_mode=True)
    pipeline = create_pipeline(config)
    result = await pipeline.run(sample_student_input)

    data = result.model_dump(mode="json")
    assert isinstance(data, dict)
    assert "run_id" in data
    assert "all_flags" in data


def test_factory_creates_real_pipeline():
    """Factory should create a real pipeline when mock_mode is False."""
    config = AppConfig(mock_mode=False)
    pipeline = create_pipeline(config)
    assert pipeline is not None


def test_factory_rejects_unknown_llm_provider():
    """Factory should raise ValueError for unknown LLM provider."""
    config = AppConfig(mock_mode=False, llm={"provider": "nonexistent", "model": "x"})
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_pipeline(config)
