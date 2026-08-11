"""What the aggregation evaluator DOES, as a vector that can be compared.

A version number is a claim about behaviour, and until now it was decided by
reading the diff and forming an opinion. That is exactly the judgement this
project keeps finding to be unreliable: "only provenance wiring" is what
somebody believes before they discover the total moved from 944 to 945.

So the decision is made against a frozen corpus and a frozen vector. Run this
with ``--emit`` before touching the evaluator, ``--compare`` after, and the
version follows from the answer:

    identical  -> PATCH  (1.6.0 -> 1.6.1)
    different  -> MINOR  (1.6.0 -> 1.7.0)

The vector deliberately EXCLUDES prose. A reason's wording is provenance: it
should improve without costing a version. What it includes is everything that
changes what a reader would do — the status, the number, which object was
chosen, the scope it was verified at, the axes it was held to, and the counts it
was built from — plus the STAGE that refused, which is structural rather than
textual.

Case identity is checked before values. A corpus that gained, lost or repeated a
case is not a corpus this baseline describes, and comparing the intersection
would let a case quietly disappear on the day it started failing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from react_review.normalize.cohorts import parse_comparison          # noqa: E402
from react_review.normalize.population import PopulationScope        # noqa: E402
from react_review.tools.batch_parse import parse_batch               # noqa: E402
from react_review.tools.batch_project import project_claim           # noqa: E402

#: Bump only when the SHAPE of the vector changes. A baseline written under an
#: older vector version cannot be compared with one written under a newer, and
#: pretending otherwise is how a comparison comes to pass by omission.
BEHAVIOR_VECTOR_VERSION = 1
GENERATOR_ID = "aggregation_behavior"

CORPUS = ROOT / "eval/baselines/aggregation_corpus.json"
BASELINE = ROOT / "eval/baselines/aggregation_behavior_baseline.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest().upper()


def _scope(declared: str) -> PopulationScope | None:
    if not declared:
        return None
    return PopulationScope.parse(declared)


def behaviour_of(case: dict) -> dict:
    """One case's behaviour vector. Structural facts only — never prose."""
    reading = parse_batch(case["response"], case["document"],
                          target_shape=case["target_shape"],
                          aggregable=bool(case.get("aggregable")))
    claim = case["claim"]
    projection = project_claim(
        reading,
        target_shape=case["target_shape"],
        review_labels=claim.get("review_labels") or {},
        cohort_key=claim.get("cohort_key") or "",
        comparison=(parse_comparison(claim["comparison"])
                    if claim.get("comparison") else None),
        requested_scope=_scope(claim.get("requested_scope") or ""),
        required_axes=list(claim.get("required_axes") or []),
        timepoint_label=claim.get("timepoint_label") or "",
        field_type=case.get("field_type") or "",
    )
    scope = projection.verified_scope
    entry = projection.entry
    return {
        "projection_status": projection.status,
        "released": projection.released,
        "value": projection.value,
        "selected_entry": (entry.quote[:60] if entry is not None else ""),
        "aggregation_set": projection.aggregation_set,
        "aggregation_status": projection.aggregation_status,
        "verified_scope": (f"{scope.basis}/{scope.analysis_set}" if scope else ""),
        "required_axes": list(projection.required_axes),
        "cohort_counts": [[c.arm_label, c.count] for c in projection.cohort_counts],
        "n_candidates": len(projection.candidates),
        "n_parse_errors": len(projection.aggregation_errors),
        "n_unrelated_rejections": len(projection.unrelated_rejections),
        "refusal_stage": _refusal_stage(projection, reading),
    }


def _refusal_stage(projection, reading) -> str:
    """WHICH layer refused, derived from structure rather than from wording."""
    if projection.released:
        return ""
    if projection.status == "batch_failed":
        return "response"
    if reading.aggregation_errors and not reading.aggregation_sets:
        return "parse"
    return {"scope_unresolved": "scope", "timepoint_unresolved": "timepoint",
            "contradictory": "contradiction", "ambiguous": "ambiguity",
            "not_reported": "not_reported", "unsupported": "unsupported",
            }.get(projection.status, projection.status)


def emit(corpus: dict, commit: str) -> dict:
    ids = [c["case_id"] for c in corpus["cases"]]
    if len(ids) != len(set(ids)):
        raise SystemExit("the corpus repeats a case id")

    # A case that does not do what it says is worse frozen than absent, because
    # the baseline then defends the wrong behaviour. The first draft of
    # `explicit_totals_disagree` quoted a population sentence its document did
    # not contain: the aggregation never parsed, the printed total won by
    # default, and the case recorded `ok` while claiming to prove a
    # contradiction is refused.
    vectors, wrong = {}, []
    for case in corpus["cases"]:
        vector = behaviour_of(case)
        vectors[case["case_id"]] = vector
        for key, wanted in (case.get("expect") or {}).items():
            if vector.get(key) != wanted:
                wrong.append(f"{case['case_id']}.{key}: expected {wanted!r}, "
                             f"got {vector.get(key)!r}")
    if wrong:
        raise SystemExit(
            "the corpus does not demonstrate what it declares, so freezing it "
            "would freeze the wrong behaviour:\n    " + "\n    ".join(wrong))
    return {
        "schema_version": 1,
        "behavior_vector_version": BEHAVIOR_VECTOR_VERSION,
        "baseline_commit": commit,
        "generator_id": GENERATOR_ID,
        "generator_sha256": _sha256(Path(__file__)),
        "corpus_id": corpus["corpus_id"],
        "corpus_sha256": _sha256(CORPUS),
        "case_count": len(ids),
        "case_ids": sorted(ids),
        "cases": vectors,
    }


def compare(baseline: dict, corpus: dict) -> list[str]:
    """Every way the two can fail to describe the same thing."""
    problems: list[str] = []
    if baseline.get("behavior_vector_version") != BEHAVIOR_VECTOR_VERSION:
        problems.append(
            f"baseline was written under vector version "
            f"{baseline.get('behavior_vector_version')} and this generator "
            f"produces {BEHAVIOR_VECTOR_VERSION}; they are not comparable")
        return problems
    if baseline.get("corpus_sha256") != _sha256(CORPUS):
        problems.append("the corpus has changed since the baseline was written, "
                        "so the baseline does not describe these cases")
    if baseline.get("generator_sha256") != _sha256(Path(__file__)):
        problems.append("this generator is not the one that wrote the baseline, "
                        "so a difference could be in the measurement")

    ids = [c["case_id"] for c in corpus["cases"]]
    if len(ids) != len(set(ids)):
        problems.append("the corpus repeats a case id")
    was, now = set(baseline.get("case_ids") or []), set(ids)
    if len(baseline.get("case_ids") or []) != baseline.get("case_count"):
        problems.append("the baseline's own case_count does not match its case_ids")
    for missing in sorted(was - now):
        problems.append(f"case {missing} is in the baseline and not in the corpus")
    for added in sorted(now - was):
        problems.append(f"case {added} is in the corpus and not in the baseline")
    if problems:
        return problems                  # comparing values now would be dishonest

    for case in corpus["cases"]:
        before = baseline["cases"][case["case_id"]]
        after = behaviour_of(case)
        for key in sorted(set(before) | set(after)):
            if before.get(key) != after.get(key):
                problems.append(
                    f"{case['case_id']}.{key}: {before.get(key)!r} -> "
                    f"{after.get(key)!r}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", action="store_true",
                        help="write the baseline for the current working tree")
    parser.add_argument("--commit", default="",
                        help="the commit the baseline describes (required to emit)")
    parser.add_argument("--compare", action="store_true",
                        help="compare the current tree against the baseline")
    args = parser.parse_args()

    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    if args.emit:
        if len(args.commit) != 40:
            raise SystemExit("--commit must be the full 40-character commit")
        BASELINE.write_text(
            json.dumps(emit(corpus, args.commit), indent=2, ensure_ascii=False)
            + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {BASELINE.relative_to(ROOT)} for {args.commit[:12]} "
              f"({corpus['case_count']} cases)")
        return 0

    if args.compare:
        problems = compare(json.loads(BASELINE.read_text(encoding="utf-8")), corpus)
        if not problems:
            print(f"behaviour identical to the baseline "
                  f"({corpus['case_count']} cases) — a PATCH is defensible")
            return 0
        print(f"{len(problems)} behavioural difference(s) — this is at least a "
              f"MINOR:")
        for line in problems:
            print("   ", line)
        return 1

    parser.error("choose --emit or --compare")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
