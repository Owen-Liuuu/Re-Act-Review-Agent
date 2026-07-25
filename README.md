# ReAct-Review

An autonomous agentic pipeline for clinical evidence synthesis and audit.

ReAct-Review (a) extracts structured evidence tables from clinical publications and
(b) cross-validates already-published systematic reviews against their source papers.
A deterministic Python orchestrator sequences the stages; specialized bounded ReAct
agents (Evidence Collector, Evidence Auditor, Judge/Arbiter) operate over a shared,
typed tool catalogue.

## Status

**P0 — scaffolding.** This project was migrated from the earlier `lit_inspector`
prototype (lifted, renamed, dead code dropped, test baseline restored). The internal
layout still mirrors the prototype's step-based structure; the ReAct restructure into
`tools/`, `agents/`, and `orchestrator/` happens in P1.

## Layout (current, P0)

```
src/react_review/
  core/        config, logging, exceptions, enums
  llm/         backend ABC + retry engine + provider adapters
  pipeline/    orchestrator, factory, schemas
  steps/       search_validation, paper_verification, data_extraction,
               table_comparison, pdf_parsing, reporting
tests/         unit + integration (mock-mode) tests
eval/          benchmark material (ground-truth work, later phases)
```

## Quick start

```bash
pip install -e ".[dev]"
pytest                       # run the test baseline
python -m react_review --config configs/config.example.yaml   # mock-mode smoke run
```

## Reuse provenance

The following prototype modules carried over as high-value reusable assets:
four-tier full-text retrieval, CrossRef verification + confidence scoring,
field-level comparison primitives + per-concept tolerance table, the LLM retry
engine, and the DOCX report renderer.
