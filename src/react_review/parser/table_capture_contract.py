"""Versioned contracts for the exact TableCapture question sent to a model.

The contract boundary is the rendered UTF-8 prompt.  Source comments and
refactors are deliberately outside it; a model-visible byte change requires a
new profile and contract file.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from react_review.contracts import ContractError, read_json_object, repo_root


TABLE_CAPTURE_V1 = Path("configs/prompt_contracts/table_capture_v1.json")
TABLE_CAPTURE_V2 = Path("configs/prompt_contracts/table_capture_v2.json")
DEFAULT_TABLE_CAPTURE_PROFILE = "table_capture_v1"
RENDERER_IDENTITY = "react_review.table_capture.render.v1"
PROMPT_VERSIONS = {
    "table_capture_v1": "table-capture-v1",
    "table_capture_v2": "table-capture-v2",
}


_CAPTURE_V1 = """You are a systematic-review methodologist transcribing the tables of a review
so a colleague can check them against the original PDF.

Transcribe every DATA table: the characteristics-of-included-studies table, any
outcome/effect table, any risk-of-bias table. Skip pure layout or navigation tables.

TRANSCRIBE — do not interpret:
- Copy every cell EXACTLY as printed, including "NR", "NA", "—", "not reported",
  "not reached", and blanks. An empty cell stays an empty string.
- Do NOT rename headers, do NOT standardise units, do NOT convert numbers, do NOT
  reorder or drop columns — including columns whose meaning is unclear to you.
- Keep multi-level headers as SEPARATE header rows. If a header spans several
  columns, put it once and leave the columns it spans empty on that row.
- Every data row must have the same number of cells as the widest header row.
- If part of a table is unreadable, still transcribe what you can and say what
  went wrong in "difficulties". Never invent a value to fill a gap.

Also state the review's research question in one line.

{{"research_context": "one line: population + exposure/intervention + outcome",
  "tables": [
    {{"table_id": "table_1",
      "caption": "the table's printed caption",
      "role": "characteristics | outcomes | quality | other",
      "header_rows": [["Study","Country","EAT",""],["","","T1DM","Control"]],
      "rows": [["Ahmad 2022","Egypt","6.60 ± 0.71","3.83 ± 0.35"]],
      "footnotes": ["values are mean ± SD unless stated"],
      "row_axis_columns": ["Study"],
      "shape_notes": "one row per study; the cohort split is a column pair",
      "cohort_labels_seen": ["T1DM","Control"],
      "extraction_confidence": 0.0,
      "difficulties": ["the last column was cut off in the text layer"]}}
  ]}}

## REVIEW TEXT
{text}

Return JSON only."""


_CAPTURE_V2 = """You are a systematic-review methodologist transcribing tables
from a review so another person can compare the transcription with the PDF.

Transcribe every DATA table, including:
- characteristics of included studies;
- outcomes or effect estimates;
- risk-of-bias or quality assessments.

Skip tables used only for page layout or navigation.

STRICT TRANSCRIPTION RULES:
- Use only information present in REVIEW TEXT.
- Never introduce a disease, intervention, outcome, cohort, arm, unit, author,
  study, or value that is not present in REVIEW TEXT.
- Copy every cell exactly as printed, including "NR", "NA", "—",
  "not reported", "not reached", and blank cells.
- Do not rename headers, standardise units, convert numbers, infer missing values,
  reorder columns, merge columns, split one printed row into several rows, or
  convert cohort columns into cohort rows.
- Preserve multi-level headers as separate header rows.
- If a header spans multiple columns, write it once and use empty strings for
  the remaining cells in that header row.
- Every data row must contain the same number of cells as the widest header row.
- cohort_labels_seen must contain only exact cohort/arm labels visibly present
  in the table header, rows, caption, or footnotes. Use [] when none are printed.
- row_axis_columns must contain only exact printed header names.
- If table structure is uncertain, preserve the visible text and describe the
  uncertainty in difficulties. Never repair the table by guessing.

RESEARCH CONTEXT:
- Summarise the population, intervention/exposure, comparator, and outcome only
  when they are stated in REVIEW TEXT.
- Do not infer a clinical topic from the output schema.
- Use an empty string when the research question cannot be established.

Return exactly one JSON object with this structure.
Text inside angle brackets describes a value and must not be copied literally:

{{
  "research_context": "<string or empty string>",
  "tables": [
    {{
      "table_id": "<stable id such as table_1>",
      "caption": "<exact printed caption or empty string>",
      "role": "<characteristics, outcomes, quality, or other>",
      "header_rows": [["<exact cell text>"]],
      "rows": [["<exact cell text>"]],
      "footnotes": ["<exact printed footnote>"],
      "row_axis_columns": ["<exact printed header>"],
      "shape_notes": "<description of visible table geometry>",
      "cohort_labels_seen": ["<exact printed cohort or arm label>"],
      "extraction_confidence": 0.0,
      "difficulties": ["<specific transcription uncertainty>"]
    }}
  ]
}}

Do not output the angle-bracket placeholders.
Use empty strings or empty arrays when the corresponding information is absent.

## REVIEW TEXT
{text}

Return JSON only."""


PROMPT_TEMPLATES = {
    "table_capture_v1": _CAPTURE_V1,
    "table_capture_v2": _CAPTURE_V2,
}
_CONTRACT_PATHS = {
    "table_capture_v1": TABLE_CAPTURE_V1,
    "table_capture_v2": TABLE_CAPTURE_V2,
}


def sha256_rendered_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest().upper()


def render_table_capture_prompt(profile: str = DEFAULT_TABLE_CAPTURE_PROFILE,
                                *, text: str) -> str:
    try:
        template = PROMPT_TEMPLATES[profile or DEFAULT_TABLE_CAPTURE_PROFILE]
    except KeyError:
        raise ContractError(
            f"unknown table capture prompt profile {profile!r} "
            f"(known: {', '.join(sorted(PROMPT_TEMPLATES))})") from None
    return template.format(text=text)


@dataclass(frozen=True)
class TableCapturePromptContract:
    prompt_id: str
    prompt_version: str
    rendered_prompt_sha256: str
    renderer_identity: str
    hash_algorithm: str
    fixture_inputs: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, profile_or_path: str | Path) -> "TableCapturePromptContract":
        if isinstance(profile_or_path, str) and profile_or_path in _CONTRACT_PATHS:
            path = repo_root() / _CONTRACT_PATHS[profile_or_path]
        else:
            candidate = Path(profile_or_path)
            if not candidate.is_absolute():
                if candidate.suffix != ".json":
                    raise ContractError(
                        f"unknown table capture prompt profile {str(profile_or_path)!r} "
                        f"(known: {', '.join(sorted(_CONTRACT_PATHS))})")
                candidate = repo_root() / candidate
            path = candidate
        body = read_json_object(path, kind="table capture prompt contract")
        prompt_id = str(body.get("prompt_id") or "")
        if prompt_id not in PROMPT_TEMPLATES:
            raise ContractError(
                f"unknown table capture prompt profile {prompt_id!r} in {path}")
        fixture_inputs = body.get("fixture_inputs")
        if not isinstance(fixture_inputs, dict) or set(fixture_inputs) != {"text"}:
            raise ContractError(
                f"table capture prompt contract {path} must pin fixture_inputs.text")
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
        rendered = render_table_capture_prompt(
            self.prompt_id, **self.fixture_inputs)
        actual = sha256_rendered_prompt(rendered)
        if actual != self.rendered_prompt_sha256:
            found.append(
                f"{self.prompt_id}: rendered prompt is {actual[:16]}, "
                f"published as {self.rendered_prompt_sha256[:16]}")
        return found
