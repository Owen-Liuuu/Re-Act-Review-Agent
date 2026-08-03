"""Deterministically assess checklist coverage without inventing claims."""
from __future__ import annotations

import re
from collections import defaultdict

from react_review.checklist.schema import (
    Checklist,
    ChecklistApplication,
    ChecklistAssessment,
    ChecklistEvidence,
    ChecklistGap,
    ChecklistItem,
)
from react_review.normalize.numeric import parse_numeric
from react_review.schemas.evidence import ReviewDataItem
from react_review.schemas.table import CapturedTableSet


def _norm(value: object) -> str:
    return re.sub(r"[\W_]+", " ", str(value or "").lower(), flags=re.UNICODE).strip()


def _contains(haystack: str, needles: list[str]) -> bool:
    text = f" {_norm(haystack)} "
    return any(f" {_norm(needle)} " in text for needle in needles if _norm(needle))


def _selectors(item: ChecklistItem) -> list[str]:
    return [*item.aliases, *(field_type.replace("_", " ") for field_type in item.field_types)]


def _components_match(item: ChecklistItem, claim: ReviewDataItem) -> bool:
    if not item.required_components:
        return True
    found = parse_numeric(claim.value).components()
    return set(item.required_components).issubset(found)


def _claim_matches(item: ChecklistItem, claim: ReviewDataItem) -> bool:
    canonical = claim.field_type in item.field_types if item.field_types else False
    verbatim = " ".join(
        (claim.raw_field_name, claim.column_header, claim.field_type.replace("_", " ")))
    alias = _contains(verbatim, item.aliases)
    return (canonical or alias) and claim.value not in (None, "") and _components_match(item, claim)


def _table_text(tables: CapturedTableSet) -> str:
    parts: list[str] = []
    for table in tables.tables:
        parts.extend((table.caption, *table.footnotes, *table.column_paths()))
        parts.extend(cell for row in table.rows for cell in row)
    return "\n".join(parts)


def _excerpt(text: str, needles: list[str], width: int = 120) -> str:
    normal = text.lower()
    for needle in needles:
        pos = normal.find(needle.lower())
        if pos >= 0:
            lo, hi = max(0, pos - width // 2), min(len(text), pos + len(needle) + width // 2)
            return " ".join(text[lo:hi].split())
    return ""


def _expected_targets(
    item: ChecklistItem, review_items: list[ReviewDataItem], study_ids: list[str],
) -> list[tuple[str, str]]:
    studies = list(dict.fromkeys(
        [sid for sid in study_ids if sid]
        + [claim.study_id for claim in review_items if claim.study_id]))
    if item.scope == "review":
        return [("", "-")]
    if item.scope == "per_study":
        return [(study_id, "-") for study_id in studies] or [("", "-")]
    cohorts = list(dict.fromkeys(
        (claim.study_id, claim.group) for claim in review_items
        if claim.study_id and claim.group not in ("", "-")))
    return cohorts or [("", "-")]


def _target_for(item: ChecklistItem, claim: ReviewDataItem) -> tuple[str, str]:
    if item.scope == "review":
        return "", "-"
    if item.scope == "per_study":
        return claim.study_id, "-"
    return claim.study_id, claim.group


def apply_checklist(
    checklist: Checklist,
    review_items: list[ReviewDataItem],
    captured_tables: CapturedTableSet,
    *,
    review_text: str = "",
    study_ids: list[str] | None = None,
) -> ChecklistApplication:
    """Assess coverage; never manufacture a review value for a missing item.

    ``presence`` questions inspect review text/tables only and never become
    ReviewDataItems. Other questions are covered only by an existing concrete
    study/value claim, which already follows the normal Collector/Auditor path.
    """
    table_text = _table_text(captured_tables)
    assessments: list[ChecklistAssessment] = []
    gaps: list[ChecklistGap] = []
    known_studies = study_ids or []

    for item in checklist.items:
        expected = _expected_targets(item, review_items, known_studies)
        evidence_by_target: dict[tuple[str, str], list[ChecklistEvidence]] = defaultdict(list)

        if item.value_kind == "presence":
            needles = _selectors(item)
            if "review_text" in item.where and _contains(review_text, needles):
                evidence_by_target[("", "-")].append(ChecklistEvidence(
                    source="review_text", excerpt=_excerpt(review_text, needles)))
            elif "review_table" in item.where and _contains(table_text, needles):
                evidence_by_target[("", "-")].append(ChecklistEvidence(
                    source="captured_table", excerpt=_excerpt(table_text, needles)))
        elif "review_table" in item.where:
            for claim in review_items:
                if not _claim_matches(item, claim):
                    continue
                evidence_by_target[_target_for(item, claim)].append(ChecklistEvidence(
                    source="review_item", study_id=claim.study_id, group=claim.group,
                    field_type=claim.field_type, table_id=claim.table_id,
                    cell_ref=claim.cell_ref, excerpt=str(claim.value)))

        # Review-level presence has exactly one target. A presence question at a
        # narrower scope cannot prove per-study coverage from a global mention,
        # so it remains a gap for every target instead of being over-claimed.
        covered = [target for target in expected if evidence_by_target.get(target)]
        missing = [target for target in expected if not evidence_by_target.get(target)]
        for study_id, group in missing:
            if not item.required:
                continue
            target = (f" for study {study_id}" if study_id else "")
            if group not in ("", "-"):
                target += f" / cohort {group}"
            gaps.append(ChecklistGap(
                checklist_id=item.id, question=item.question, scope=item.scope,
                study_id=study_id, group=group,
                reason=f"required checklist item was not found{target}"))

        n_expected, n_found = len(expected), len(covered)
        if not missing:
            status = "covered"
        elif covered:
            status = "partial"
        elif item.required:
            status = "missing_required"
        else:
            status = "missing_optional"
        evidence = [entry for target in expected for entry in evidence_by_target.get(target, [])]
        assessments.append(ChecklistAssessment(
            checklist_id=item.id, question=item.question, required=item.required,
            scope=item.scope, value_kind=item.value_kind, status=status,
            expected=n_expected, found=n_found, evidence=evidence,
            reason=("all expected targets covered" if not missing else
                    f"{len(missing)} of {n_expected} expected target(s) missing"),
        ))

    return ChecklistApplication(
        name=checklist.name, version=checklist.version,
        source_file=checklist.source_file, sha256=checklist.sha256,
        items=[item.model_copy(deep=True) for item in checklist.items],
        assessments=assessments, gaps=gaps,
    )


def annotate_checklist_claims(
    checklist: Checklist, review_items: list[ReviewDataItem],
) -> list[ReviewDataItem]:
    """Mark concrete matching claims without duplicating or inventing rows.

    A cell can satisfy several coverage questions (for example an HR with a CI).
    The first matching non-presence item owns the routing identity; every match
    remains visible in ``ChecklistAssessment.evidence``.
    """
    annotated: list[ReviewDataItem] = []
    for claim in review_items:
        owner = next((item for item in checklist.items
                      if item.value_kind != "presence"
                      and "review_table" in item.where
                      and _claim_matches(item, claim)), None)
        if owner is None or claim.checklist_id:
            annotated.append(claim)
        else:
            annotated.append(claim.model_copy(update={
                "origin": "checklist", "checklist_id": owner.id}))
    return annotated


def render_checklist(application: ChecklistApplication) -> str:
    """Compact terminal rendering; the full typed payload is journalled."""
    lines = [
        f"  checklist {application.name} v{application.version}: "
        f"{len(application.assessments)} item(s), {len(application.gaps)} required gap(s)",
        f"  source: {application.source_file}",
        f"  sha256: {application.sha256}",
    ]
    for result in application.assessments:
        lines.append(
            f"    [{result.status}] {result.checklist_id}: "
            f"{result.found}/{result.expected} target(s) covered")
    return "\n".join(lines)
