"""Deterministic, claim-level evidence adequacy before value comparison.

The model may propose a value and quote.  This evaluator decides only from the
saved document and structured claim whether that proposal is anchored to the
right field, target and stated scope.  It never asks an LLM whether evidence is
"good enough".
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from react_review.contracts import ContractError, one_of, read_json_object, repo_root, sha256_file
from react_review.normalize.cohorts import distinguishing_tokens
from react_review.normalize.numeric import parse_numeric
from react_review.schemas.adequacy import (
    AdequacyEvaluatorIdentity,
    AdequacyStatus,
    AxisResult,
    AxisStatus,
    EvidenceAdequacy,
    EvidenceAnchor,
)
from react_review.schemas.evidence import ReviewDataItem, SourceEvidenceItem
from react_review.steps.data_extraction.schemas import DocumentScope, PaperDocument


AXES = ("value", "field", "target", "population", "timepoint", "analysis_set")
DEFAULT_POLICY = "configs/evidence_adequacy/policy_v1.json"
EVALUATOR_ID = "evidence_adequacy"
EVALUATOR_VERSION = "1.0.0"
EVALUATOR_DIR = "configs/evidence_adequacy/evaluators"
REGISTRY = "configs/evidence_adequacy/registry_v1.json"
HASH_ALGORITHM = "sha256-path-lf-v1"


@dataclass(frozen=True)
class EvidenceAdequacyPolicy:
    policy_id: str
    sha256: str
    required_axes: tuple[str, ...]
    max_context_chars: int
    max_field_distance_chars: int


@lru_cache(maxsize=8)
def load_evidence_adequacy_policy(
    path: str | Path = DEFAULT_POLICY,
) -> EvidenceAdequacyPolicy:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = repo_root() / resolved
    body = read_json_object(resolved, kind="evidence adequacy policy")
    axes = tuple(str(axis) for axis in (body.get("required_axes") or []))
    if not axes or any(axis not in AXES for axis in axes):
        raise ContractError("evidence adequacy policy declares invalid required_axes")
    rules = body.get("rules") or {}
    one_of(rules.get("metadata_only"), ("insufficient",), field="metadata_only rule")
    one_of(
        rules.get("unknown_document_scope"), ("unknown",),
        field="unknown_document_scope rule",
    )
    limits = body.get("limits") or {}
    context = int(limits.get("max_context_chars") or 0)
    distance = int(limits.get("max_field_distance_chars") or 0)
    if context < 200 or distance < 20:
        raise ContractError("evidence adequacy policy limits are too small to be usable")
    return EvidenceAdequacyPolicy(
        policy_id=str(body.get("policy_id") or ""),
        sha256=sha256_file(resolved),
        required_axes=axes,
        max_context_chars=context,
        max_field_distance_chars=distance,
    )


def _document_sha256(text: str) -> str:
    canonical = "\n".join((text or "").splitlines()).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest().upper()


def _fold(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text or "").casefold()
    folded = folded.replace("−", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", folded).strip()


def _find_all(text: str, phrase: str) -> list[tuple[int, int]]:
    """Case-insensitive literal spans in original coordinates."""
    if not phrase:
        return []
    return [(match.start(), match.end()) for match in re.finditer(
        re.escape(phrase), text, flags=re.IGNORECASE
    )]


def _whole_word_spans(text: str, phrase: str) -> list[tuple[int, int]]:
    if not phrase:
        return []
    return [(match.start(), match.end()) for match in re.finditer(
        rf"(?<![A-Za-z0-9]){re.escape(phrase)}(?![A-Za-z0-9])",
        text, flags=re.IGNORECASE,
    )]


def _block_bounds(text: str, start: int, end: int, limit: int) -> tuple[int, int]:
    left = text.rfind("\n\n", 0, start)
    left = 0 if left < 0 else left + 2
    right = text.find("\n\n", end)
    right = len(text) if right < 0 else right
    if right - left <= limit:
        return left, right
    centre = (start + end) // 2
    left = max(left, centre - limit // 2)
    right = min(right, left + limit)
    left = max(0, right - limit)
    return left, right


def _decimal_places(raw: str) -> list[int]:
    return [len(match.group(1) or "") for match in re.finditer(
        r"[-+]?\d+(?:[.,](\d+))?", raw or ""
    )]


def _coarser_compatible(review_value: object, source_value: object) -> bool:
    review = parse_numeric(review_value)
    source = parse_numeric(source_value)
    if review.primary is None or source.primary is None:
        return False
    pairs = [(review.primary, source.primary)]
    if review.spread_kind == source.spread_kind == "sd" and (
        review.spread is not None and source.spread is not None
    ):
        pairs.append((review.spread, source.spread))
    review_places = _decimal_places(str(review_value))
    source_places = _decimal_places(str(source_value))
    if len(source_places) < len(pairs) or len(review_places) < len(pairs):
        return False
    strictly_coarser = False
    for index, (precise, coarse) in enumerate(pairs):
        places = source_places[index]
        if places < review_places[index]:
            strictly_coarser = True
        if round(precise, places) != round(coarse, places):
            return False
    return strictly_coarser


def _label_tokens(text: str) -> set[str]:
    tokens = distinguishing_tokens(text)
    return {token[:-1] if token.endswith("s") and len(token) > 3 else token
            for token in tokens}


def _label_score(one: str, other: str) -> float:
    left, right = _label_tokens(one), _label_tokens(other)
    if not left or not right:
        return 0.0
    shared = len(left & right)
    if not shared:
        return 0.0
    return 2 * shared / (len(left) + len(right))


def _map_cohorts(
    review_cohorts: dict[str, list[str]], source_labels: list[str],
) -> dict[str, str]:
    """Resolve unique labels, then one remaining pair by bijection."""
    mapping: dict[str, str] = {}
    used: set[str] = set()
    candidates: list[tuple[float, str, str]] = []
    for key, variants in review_cohorts.items():
        for label in source_labels:
            score = max((_label_score(variant, label) for variant in variants),
                        default=0.0)
            if score:
                candidates.append((score, key, label))
    for score, key, label in sorted(candidates, reverse=True):
        if key not in mapping and label not in used:
            rivals = [candidate for candidate in candidates
                      if candidate[1] == key and candidate[0] == score]
            if len(rivals) == 1:
                mapping[key] = label
                used.add(label)
    remaining_keys = [key for key in review_cohorts if key not in mapping]
    remaining_labels = [label for label in source_labels if label not in used]
    if len(review_cohorts) == len(source_labels) and (
        len(remaining_keys) == len(remaining_labels) == 1
    ):
        mapping[remaining_keys[0]] = remaining_labels[0]
    return mapping


def _field_phrases(review: ReviewDataItem, variants: list[str]) -> list[str]:
    phrases: list[str] = []
    for phrase in (
        *variants,
        review.raw_field_name,
        review.column_header,
        review.field_type.replace("_", " "),
    ):
        cleaned = str(phrase or "").strip()
        if cleaned and cleaned not in phrases:
            phrases.append(cleaned)
    if review.field_type == "age" and "aged" not in phrases:
        phrases.append("aged")
    return phrases


class _Assessment:
    def __init__(self, document: PaperDocument, policy: EvidenceAdequacyPolicy) -> None:
        self.document = document
        self.text = document.full_text or ""
        self.policy = policy
        self.anchors: list[EvidenceAnchor] = []
        self.quote_start = -1
        self.quote_end = -1
        self.context_start = 0
        self.context_end = 0

    def add_anchor(self, kind: str, start: int, end: int) -> int:
        index = len(self.anchors)
        self.anchors.append(EvidenceAnchor(
            kind=kind,
            text=self.text[start:end],
            start=start,
            end=end,
            context=self.text[self.context_start:self.context_end],
            context_start=self.context_start,
            context_end=self.context_end,
        ))
        return index

    def locate_quote(self, quote: str) -> AxisResult | None:
        spans = _find_all(self.text, quote)
        if not spans:
            return AxisResult(
                status=AxisStatus.FAIL,
                reason="the supporting quote is not a contiguous span of the document",
            )
        if len(spans) > 1:
            return AxisResult(
                status=AxisStatus.UNKNOWN,
                reason="the supporting quote occurs more than once in the document",
            )
        self.quote_start, self.quote_end = spans[0]
        self.context_start, self.context_end = _block_bounds(
            self.text, self.quote_start, self.quote_end,
            self.policy.max_context_chars,
        )
        anchor = self.add_anchor("quote", self.quote_start, self.quote_end)
        return AxisResult(
            status=AxisStatus.PASS, reason="quote is anchored in the document",
            matched_phrases=[quote], anchor_indices=[anchor],
        )


class EvidenceAdequacyEvaluator:
    """Apply a frozen policy to one review/source/document triple."""

    def __init__(
        self,
        policy: EvidenceAdequacyPolicy,
        identity: AdequacyEvaluatorIdentity | None = None,
    ) -> None:
        self.policy = policy
        self.identity = identity or AdequacyEvaluatorIdentity(
            policy_id=policy.policy_id,
            policy_sha256=policy.sha256,
            evaluator_id="evidence_adequacy",
            evaluator_status="unavailable",
        )

    @classmethod
    def development(cls, path: str | Path = DEFAULT_POLICY):
        return cls(load_evidence_adequacy_policy(path))

    @classmethod
    def resolve(
        cls,
        *,
        policy_id: str = "evidence_adequacy_v1",
        evaluator_version: str = EVALUATOR_VERSION,
        root: Path | None = None,
    ) -> "EvidenceAdequacyEvaluator":
        """Resolve policy and evaluator identity together, failing closed."""
        base = root or repo_root()
        identity, policy_path = evaluator_readiness(
            policy_id=policy_id, evaluator_version=evaluator_version, root=base
        )
        policy = load_evidence_adequacy_policy(base / policy_path)
        if policy.policy_id != identity.policy_id or (
            policy.sha256 != identity.policy_sha256
        ):
            raise ContractError(
                "the adequacy policy loaded is not the one readiness cleared"
            )
        return cls(policy, identity)

    def assess(
        self,
        review: ReviewDataItem,
        source: SourceEvidenceItem,
        document: PaperDocument,
        *,
        field_variants: list[str] | None = None,
        cohort_variants: dict[str, list[str]] | None = None,
    ) -> EvidenceAdequacy:
        scope = source.document_scope
        if scope is DocumentScope.UNKNOWN:
            scope = document.document_scope
        required = list(self.policy.required_axes)
        if review.group not in {"", "-", "all"}:
            required.append("target")
        if review.population_scope is not None and review.population_scope.stated:
            required.append("population")
        if review.timepoint_label or review.timepoint not in {"", "single"}:
            required.append("timepoint")
        if (review.population_scope is not None
                and review.population_scope.axis_stated("analysis_set")):
            required.append("analysis_set")
        required = [axis for axis in AXES if axis in set(required)]

        axes = {
            axis: AxisResult(status=AxisStatus.NOT_REQUIRED)
            for axis in AXES
        }
        state = _Assessment(document, self.policy)
        reasons: list[str] = []

        if scope is DocumentScope.METADATA_ONLY:
            return EvidenceAdequacy(
                status=AdequacyStatus.INSUFFICIENT,
                document_scope=scope,
                document_sha256=_document_sha256(state.text),
                required_axes=required,
                axis_results=axes,
                reason_codes=["metadata_only"],
                evaluator=self.identity,
            )

        if scope is DocumentScope.UNKNOWN:
            for axis in required:
                axes[axis] = AxisResult(
                    status=AxisStatus.UNKNOWN,
                    reason="document retrieval scope is unknown",
                )
            return EvidenceAdequacy(
                status=AdequacyStatus.UNKNOWN,
                document_scope=scope,
                document_sha256=_document_sha256(state.text),
                required_axes=required,
                axis_results=axes,
                reason_codes=["document_scope_unknown"],
                evaluator=self.identity,
            )

        quote_result = state.locate_quote(source.source_quote or "")
        if quote_result is None:
            quote_result = AxisResult(status=AxisStatus.FAIL, reason="no quote")

        # Value axis: quote anchoring, exact value anchoring, uniqueness and
        # lower-precision compatibility are separate facts.
        if quote_result.status is not AxisStatus.PASS:
            axes["value"] = quote_result
            reasons.append(
                "quote_unanchored" if quote_result.status is AxisStatus.FAIL
                else "quote_attribution_ambiguous"
            )
            value_spans: list[tuple[int, int]] = []
        else:
            local_spans = _find_all(source.source_quote, str(source.source_value or ""))
            value_spans = [
                (state.quote_start + start, state.quote_start + end)
                for start, end in local_spans
            ]
            if not source.source_value or not value_spans:
                axes["value"] = AxisResult(
                    status=AxisStatus.FAIL,
                    reason="the extracted source value is not printed in its quote",
                )
                reasons.append("value_unanchored")
            elif len(value_spans) != 1:
                indices = [state.add_anchor("value", start, end)
                           for start, end in value_spans]
                axes["value"] = AxisResult(
                    status=AxisStatus.UNKNOWN,
                    reason="the same source value occurs more than once in the quote",
                    matched_phrases=[state.text[start:end] for start, end in value_spans],
                    anchor_indices=indices,
                )
                reasons.append("value_attribution_ambiguous")
            elif _fold(str(review.value)) != _fold(str(source.source_value)) and (
                _coarser_compatible(review.value, source.source_value)
            ):
                start, end = value_spans[0]
                axes["value"] = AxisResult(
                    status=AxisStatus.FAIL,
                    reason=("the source gives a compatible but coarser rounded "
                            "statistic and cannot verify the claim's exact precision"),
                    matched_phrases=[state.text[start:end]],
                    anchor_indices=[state.add_anchor("value", start, end)],
                )
                reasons.append("source_value_coarser_than_claim")
            else:
                start, end = value_spans[0]
                axes["value"] = AxisResult(
                    status=AxisStatus.PASS,
                    reason="source value is uniquely anchored in the quote",
                    matched_phrases=[state.text[start:end]],
                    anchor_indices=[state.add_anchor("value", start, end)],
                )

        # Field axis: the requested field phrase must be near the selected
        # value. This is what prevents an age number supporting a BMI claim.
        phrases = _field_phrases(review, field_variants or [])
        context = state.text[state.context_start:state.context_end]
        field_candidates: list[tuple[int, int, str]] = []
        for phrase in phrases:
            for start, end in _whole_word_spans(context, phrase):
                field_candidates.append((
                    state.context_start + start,
                    state.context_start + end,
                    phrase,
                ))
        if not field_candidates:
            axes["field"] = AxisResult(
                status=AxisStatus.FAIL,
                reason="no requested field phrase is bound to the selected value",
            )
            reasons.append("field_mismatch")
        elif len(value_spans) == 1:
            value_start = value_spans[0][0]
            start, end, _ = min(
                field_candidates,
                key=lambda candidate: min(
                    abs(candidate[1] - value_start), abs(candidate[0] - value_start)
                ),
            )
            distance = min(abs(end - value_start), abs(start - value_start))
            if distance > self.policy.max_field_distance_chars:
                axes["field"] = AxisResult(
                    status=AxisStatus.UNKNOWN,
                    reason="field phrase is too far from the selected value",
                )
                reasons.append("field_binding_unknown")
            else:
                anchor = state.add_anchor("field", start, end)
                axes["field"] = AxisResult(
                    status=AxisStatus.PASS,
                    reason="requested field phrase is bound to the selected value",
                    matched_phrases=[state.text[start:end]],
                    anchor_indices=[anchor],
                )
        else:
            axes["field"] = AxisResult(
                status=AxisStatus.UNKNOWN,
                reason="field cannot be assigned until the value occurrence is unique",
            )
            reasons.append("field_binding_unknown")

        if "target" in required:
            axes["target"] = self._assess_target(
                review, source, state, value_spans, cohort_variants or {}, reasons
            )

        if "timepoint" in required:
            phrase = review.timepoint_label or review.timepoint
            spans = _whole_word_spans(context, phrase)
            if spans:
                start, end = spans[0]
                start, end = state.context_start + start, state.context_start + end
                axes["timepoint"] = AxisResult(
                    status=AxisStatus.PASS,
                    reason="claim timepoint is printed in the evidence block",
                    matched_phrases=[state.text[start:end]],
                    anchor_indices=[state.add_anchor("timepoint", start, end)],
                )
            else:
                axes["timepoint"] = AxisResult(
                    status=AxisStatus.UNKNOWN,
                    reason="claim timepoint is not established by the evidence block",
                )
                reasons.append("timepoint_unknown")

        self._assess_population_axes(review, source, axes, required, reasons)

        statuses = [axes[axis].status for axis in required]
        if AxisStatus.FAIL in statuses:
            status = AdequacyStatus.INSUFFICIENT
        elif AxisStatus.UNKNOWN in statuses:
            status = AdequacyStatus.UNKNOWN
        else:
            status = AdequacyStatus.SUFFICIENT
        return EvidenceAdequacy(
            status=status,
            document_scope=scope,
            document_sha256=_document_sha256(state.text),
            required_axes=required,
            axis_results=axes,
            reason_codes=list(dict.fromkeys(reasons)),
            evidence_anchors=state.anchors,
            evaluator=self.identity,
        )

    def _assess_target(
        self,
        review: ReviewDataItem,
        source: SourceEvidenceItem,
        state: _Assessment,
        value_spans: list[tuple[int, int]],
        cohort_variants: dict[str, list[str]],
        reasons: list[str],
    ) -> AxisResult:
        if source.target_check in {
            "ambiguous", "not_reported", "direction_inverted", "inconsistent",
            "unsupported", "protocol_error",
        }:
            status = (AxisStatus.FAIL if source.target_check in {
                "direction_inverted", "inconsistent", "unsupported",
            } else AxisStatus.UNKNOWN)
            reasons.append("target_mismatch" if status is AxisStatus.FAIL
                           else "target_binding_unknown")
            return AxisResult(status=status, reason=source.target_reason)
        if len(value_spans) != 1:
            reasons.append("target_binding_unknown")
            return AxisResult(
                status=AxisStatus.UNKNOWN,
                reason="target cannot be assigned until the value occurrence is unique",
            )

        target_label = source.assigned_arm_label
        if not target_label:
            target_label = _map_cohorts(
                cohort_variants, list(source.cohorts_seen)
            ).get(review.group, "")
        if not target_label:
            reasons.append("target_binding_unknown")
            return AxisResult(
                status=AxisStatus.UNKNOWN,
                reason="the review cohort cannot be mapped uniquely to a source label",
            )

        context = state.text[state.context_start:state.context_end]
        target_local = _find_all(context, target_label)
        if not target_local:
            reasons.append("target_binding_unknown")
            return AxisResult(
                status=AxisStatus.UNKNOWN,
                reason="the mapped source cohort label is not in the evidence block",
            )
        target_spans = [(state.context_start + a, state.context_start + b)
                        for a, b in target_local]
        target_start, target_end = min(
            target_spans,
            key=lambda span: min(abs(span[1] - value_spans[0][0]),
                                 abs(span[0] - value_spans[0][0])),
        )

        if source.assigned_arm_label and source.target_check in {"ok", "reassigned"}:
            bound = True
        else:
            all_labels = list(dict.fromkeys(source.cohorts_seen))
            label_spans: list[tuple[int, int, str]] = []
            for label in all_labels:
                for start, end in _find_all(context, label):
                    label_spans.append((state.context_start + start,
                                        state.context_start + end, label))
            value_start = value_spans[0][0]
            preceding = [span for span in label_spans if span[1] <= value_start]
            nearest = max(preceding, key=lambda span: span[1]) if preceding else None
            bound = bool(nearest and nearest[2] == target_label)

        anchor = state.add_anchor("target", target_start, target_end)
        if bound:
            return AxisResult(
                status=AxisStatus.PASS,
                reason="selected value is bound to the requested source cohort",
                matched_phrases=[state.text[target_start:target_end]],
                anchor_indices=[anchor],
            )
        reasons.append("target_mismatch")
        return AxisResult(
            status=AxisStatus.FAIL,
            reason="selected value is bound to a different source cohort",
            matched_phrases=[state.text[target_start:target_end]],
            anchor_indices=[anchor],
        )

    @staticmethod
    def _assess_population_axes(review, source, axes, required, reasons) -> None:
        review_scope = review.population_scope
        source_scope = source.population_scope
        for axis, attr in (("population", "basis"),
                           ("analysis_set", "analysis_set")):
            if axis not in required:
                continue
            if source_scope is None:
                axes[axis] = AxisResult(
                    status=AxisStatus.UNKNOWN,
                    reason=f"source {axis} is not established by the evidence",
                )
                reasons.append(f"{axis}_unknown")
            elif getattr(review_scope, attr) != getattr(source_scope, attr):
                axes[axis] = AxisResult(
                    status=AxisStatus.FAIL,
                    reason=(f"review {axis} {getattr(review_scope, attr)!r} differs "
                            f"from source {getattr(source_scope, attr)!r}"),
                )
                reasons.append(f"{axis}_mismatch")
            else:
                axes[axis] = AxisResult(
                    status=AxisStatus.PASS,
                    reason=f"review and source {axis} agree",
                )


def _canonical_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def _hash_sources(
    paths: list[str], root: Path,
) -> tuple[str, dict[str, str]]:
    digest = hashlib.sha256()
    per_file: dict[str, str] = {}
    for relative in sorted(paths):
        body = _canonical_bytes(root / relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(body)).encode("ascii"))
        digest.update(b"\0")
        digest.update(body)
        per_file[relative] = "sha256:" + hashlib.sha256(body).hexdigest()
    return "sha256:" + digest.hexdigest(), per_file


def _git(args: list[str], root: Path, *, status: bool = False):
    try:
        done = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)[:120]
    if status:
        return done.stdout, done.returncode != 0
    return done.stdout, (done.stderr.strip() if done.returncode else "")


def evaluator_readiness(
    *,
    policy_id: str,
    evaluator_version: str,
    root: Path | None = None,
) -> tuple[AdequacyEvaluatorIdentity, str]:
    """Return an attributable identity or an explicitly non-release one."""
    base = root or repo_root()
    registry_path = base / REGISTRY
    registry = read_json_object(registry_path, kind="evidence adequacy registry")
    entry = (registry.get("policies") or {}).get(policy_id) or {}
    policy_path = str(entry.get("file") or "")
    policy_hash = (
        sha256_file(base / policy_path)
        if policy_path and (base / policy_path).is_file() else ""
    )
    manifest_path = (
        base / EVALUATOR_DIR / f"evidence_adequacy_{evaluator_version}.json"
    )
    manifest = read_json_object(manifest_path, kind="evidence adequacy evaluator")
    if manifest.get("hash_algorithm") != HASH_ALGORITHM:
        raise ContractError("evidence adequacy evaluator hash algorithm changed")
    if manifest.get("evaluator_id") != EVALUATOR_ID or (
        manifest.get("evaluator_version") != evaluator_version
    ):
        raise ContractError("evidence adequacy evaluator manifest identity mismatch")
    source_files = manifest.get("source_files") or {}
    if not isinstance(source_files, dict) or not source_files:
        raise ContractError("evidence adequacy evaluator manifest lists no sources")
    computed, per_file = _hash_sources(list(source_files), base)

    def identity(status: str, reason: str = "", *, commit: str = "",
                 matches: bool = False) -> AdequacyEvaluatorIdentity:
        release = (
            status == "registered" and matches and len(commit) == 40
            and bool(policy_hash) and bool(computed)
        )
        return AdequacyEvaluatorIdentity(
            policy_id=policy_id,
            policy_sha256=policy_hash,
            evaluator_id=EVALUATOR_ID,
            evaluator_version=evaluator_version,
            evaluator_hash=computed,
            git_commit=commit,
            git_commit_matches_evaluator=matches,
            evaluator_status=status,
            release_eligible=release,
        )

    if not entry or not policy_hash:
        return identity("unregistered", "policy is absent from the registry"), policy_path
    if policy_hash != str(entry.get("sha256") or ""):
        raise ContractError("frozen evidence adequacy policy hash does not match registry")
    if entry.get("status") != "active" or entry.get("formal_results") is not True:
        return identity("unregistered", "policy is not active"), policy_path
    if computed != manifest.get("evaluator_hash"):
        raise ContractError("evidence adequacy evaluator source hash changed")
    for relative, published in source_files.items():
        if per_file.get(relative) != published:
            raise ContractError(f"{relative} does not match its published hash")
    if evaluator_version not in (entry.get("evaluators") or []):
        return identity("unregistered", "policy/evaluator pair is not registered"), policy_path

    paths = sorted({
        *source_files,
        REGISTRY,
        policy_path,
        manifest_path.relative_to(base).as_posix(),
    })
    tracked, error = _git(["ls-files", "--error-unmatch", *paths], base)
    if tracked is None:
        return identity("unavailable", error), policy_path
    if error:
        return identity("unregistered", "evaluator files are not tracked"), policy_path
    commit, error = _git(["rev-parse", "HEAD"], base)
    commit = (commit or "").strip()
    if len(commit) != 40:
        return identity("unavailable", error), policy_path
    _, dirty = _git(["diff", "--quiet", "HEAD", "--", *paths], base, status=True)
    if dirty:
        return identity(
            "unregistered", "evaluator files differ from HEAD",
            commit=commit, matches=False,
        ), policy_path
    return identity("registered", commit=commit, matches=True), policy_path
