"""Render Review Extraction questions. Contract hashes live in configs/."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any

from react_review.contracts import ContractError, read_json_object, repo_root

RENDERER_IDENTITY = "react_review.review_extraction.render.v1"

REVIEW_LENS_V1 = Path("configs/prompt_contracts/review_lens_v1.json")
EVIDENCE_LOCALIZE_V1 = Path("configs/prompt_contracts/evidence_localize_v1.json")
EVIDENCE_LOCALIZE_V2 = Path("configs/prompt_contracts/evidence_localize_v2.json")
CLAIM_ORIGIN_V1 = Path("configs/prompt_contracts/claim_origin_v1.json")

PROMPT_VERSIONS = {
    "review_lens_v1": "review-lens-v1",
    "evidence_localize_v1": "evidence-localize-v1",
    "evidence_localize_v2": "evidence-localize-v2",
    "claim_origin_v1": "claim-origin-v1",
}

_LENS = """You compress the FRONT MATTER of a systematic review into a short ruler.
Later steps will use ONLY this ruler — they will not see the abstract again.

Copy the review's own words. Do not invent a disease, population, comparison,
or outcome that FRONT MATTER does not state. Empty string / empty list when a
field is not stated. If there is no structured abstract, use the first
paragraph and say so in difficulties.

Limits (hard): lens_one_line ≤ 40 words; domain ≤ 12 words; population ≤ 20
words; comparison ≤ 20 words; outcomes = 3 to 8 short labels.

{{"lens_one_line": "population + comparison + outcomes in ≤40 words",
  "domain": "short clinical domain",
  "population": "who was studied",
  "comparison": "intervention vs comparator",
  "outcomes": ["label", "label"],
  "not_audit_focus": ["pooled GRADE", "search strategy"],
  "difficulties": ["no structured abstract; used opening paragraph"]}}

## FRONT MATTER
{front_matter}

Return JSON only."""

_LOCALIZE = """You are listing the evidence-chain displays in a systematic review.

The review's compressed ruler:
{lens}

A display is evidence_chain=true only when it attributes raw per-study numbers
(copied from an included paper) to a named included study, and those numbers
relate to the ruler's population / comparison / outcomes.

evidence_chain=false for pooled OR, GRADE, PRISMA flow, search strategy, and
any figure or table unrelated to the ruler. Do not use a caption regex. Judge
from the ruler plus the caption / nearby text in the RESULTS WINDOW.

kind is pdf_table (a typeset table in the PDF text), forest_plot (RevMan-style
per-study forest, including when only the caption is in the text layer), or
other.

{{"displays": [
  {{"display_id": "table_1",
    "kind": "pdf_table | forest_plot | other",
    "caption": "verbatim caption",
    "page_hint": "printed page or empty",
    "evidence_chain": true,
    "reason": "why this is / is not on the evidence chain, naming the ruler"}}
]}}

## RESULTS WINDOW
{results_window}

Return JSON only."""

_LOCALIZE_V2 = """You are listing the evidence-chain displays in a systematic review.

The review's compressed ruler:
{lens}

A display is evidence_chain=true only when it attributes raw per-study numbers
(copied from an included paper) to a named included study, and those numbers
relate to the ruler's population / comparison / outcomes.

A display is also on the evidence chain when it attributes per-study
characteristics — sample size per arm, population descriptors, study design,
country, year — to named included studies, even when it reports none of the
ruler's outcomes. Those columns carry the arm labels and the study identities
that later steps depend on.

evidence_chain=false for pooled OR, GRADE, PRISMA flow, search strategy, and
any figure or table unrelated to the ruler. Do not use a caption regex. Judge
from the ruler plus the caption / nearby text in the RESULTS WINDOW.

kind is pdf_table (a typeset table in the PDF text), forest_plot (RevMan-style
per-study forest, including when only the caption is in the text layer), or
other.

{{"displays": [
  {{"display_id": "table_1",
    "kind": "pdf_table | forest_plot | other",
    "caption": "verbatim caption",
    "page_hint": "printed page or empty",
    "evidence_chain": true,
    "reason": "why this is / is not on the evidence chain, naming the ruler"}}
]}}

## RESULTS WINDOW
{results_window}

Return JSON only."""

_ORIGIN = """You label each column (or cell) of one already-captured display:
is this value copied from a source paper, computed by the review, or bibliographic?

The review's compressed ruler:
{lens}

source_paper: a raw count or characteristic the included paper itself reports
(sample size, events, age, country, …) written against a named study.
review_computed: derived from other cells in this review (OR, Weight, diamond /
pooled row, GRADE, I²).
bibliographic: identifiers, years, citations — not a measurement to fetch.
If a column is unrelated to the ruler, prefer bibliographic or review_computed
over source_paper. Never mark a pooled / Total (Wald) / diamond row as
source_paper.

Label columns by their printed header path. Add a row index only when one row
differs from the rest of its column (a pooled footer).

{{"labels": [
  {{"table_id": "{table_id}",
    "column_path": "exact header path",
    "row": null,
    "value_source": "source_paper | review_computed | bibliographic",
    "outcome": "which ruler outcome this display is about, or empty",
    "reason": "short"}}
]}}

## DISPLAY
caption: {caption}
column paths: {column_paths}
sample rows: {sample_rows}
pooled / footer rows: {pooled_rows}
footnotes: {footnotes}

Return JSON only."""

PROMPT_TEMPLATES = {
    "review_lens_v1": _LENS,
    "evidence_localize_v1": _LOCALIZE,
    "evidence_localize_v2": _LOCALIZE_V2,
    "claim_origin_v1": _ORIGIN,
}
_CONTRACT_PATHS = {
    "review_lens_v1": REVIEW_LENS_V1,
    "evidence_localize_v1": EVIDENCE_LOCALIZE_V1,
    "evidence_localize_v2": EVIDENCE_LOCALIZE_V2,
    "claim_origin_v1": CLAIM_ORIGIN_V1,
}


def _placeholders(template: str) -> set[str]:
    return {name for _, name, _, _ in Formatter().parse(template) if name}


def sha256_rendered_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest().upper()


def render_extraction_prompt(profile: str, **values: str) -> str:
    try:
        template = PROMPT_TEMPLATES[profile]
    except KeyError:
        raise ContractError(
            f"unknown review extraction prompt {profile!r} "
            f"(known: {', '.join(sorted(PROMPT_TEMPLATES))})") from None
    needed = _placeholders(template)
    missing = needed - values.keys()
    if missing:
        raise ContractError(
            f"{profile} is missing prompt values: {', '.join(sorted(missing))}")
    return template.format(**{key: values[key] for key in needed})


@dataclass(frozen=True)
class ExtractionPromptContract:
    prompt_id: str
    prompt_version: str
    rendered_prompt_sha256: str
    renderer_identity: str
    hash_algorithm: str
    fixture_inputs: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, profile_or_path: str | Path) -> "ExtractionPromptContract":
        if isinstance(profile_or_path, str) and profile_or_path in _CONTRACT_PATHS:
            path = repo_root() / _CONTRACT_PATHS[profile_or_path]
        else:
            candidate = Path(profile_or_path)
            if not candidate.is_absolute():
                if candidate.suffix != ".json":
                    raise ContractError(
                        f"unknown review extraction prompt {str(profile_or_path)!r} "
                        f"(known: {', '.join(sorted(_CONTRACT_PATHS))})")
                candidate = repo_root() / candidate
            path = candidate
        body = read_json_object(path, kind="review extraction prompt contract")
        prompt_id = str(body.get("prompt_id") or "")
        if prompt_id not in PROMPT_TEMPLATES:
            raise ContractError(
                f"unknown review extraction prompt {prompt_id!r} in {path}")
        fixture_inputs = body.get("fixture_inputs")
        needed = _placeholders(PROMPT_TEMPLATES[prompt_id])
        if not isinstance(fixture_inputs, dict) or set(fixture_inputs) != needed:
            raise ContractError(
                f"review extraction contract {path} must pin fixture_inputs "
                f"{sorted(needed)}")
        return cls(
            prompt_id=prompt_id,
            prompt_version=str(body.get("prompt_version") or ""),
            rendered_prompt_sha256=str(body.get("rendered_prompt_sha256") or "").upper(),
            renderer_identity=str(body.get("renderer_identity") or ""),
            hash_algorithm=str(body.get("hash_algorithm") or ""),
            fixture_inputs=dict(fixture_inputs),
            path=path,
        )

    def drifts(self) -> list[str]:
        found: list[str] = []
        if self.prompt_version != PROMPT_VERSIONS[self.prompt_id]:
            found.append(
                f"{self.prompt_id}: version is {self.prompt_version!r}, expected "
                f"{PROMPT_VERSIONS[self.prompt_id]!r}")
        if self.renderer_identity != RENDERER_IDENTITY:
            found.append(
                f"{self.prompt_id}: renderer is {self.renderer_identity!r}, expected "
                f"{RENDERER_IDENTITY!r}")
        if self.hash_algorithm != "sha256-rendered-utf8-v1":
            found.append(f"{self.prompt_id}: unsupported hash algorithm")
        rendered = render_extraction_prompt(self.prompt_id, **self.fixture_inputs)
        actual = sha256_rendered_prompt(rendered)
        if actual != self.rendered_prompt_sha256:
            found.append(
                f"{self.prompt_id}: rendered prompt is {actual[:16]}, "
                f"published as {self.rendered_prompt_sha256[:16]}")
        return found
