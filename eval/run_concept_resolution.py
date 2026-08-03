"""Offline acceptance runner for DKB field resolution.

The EAT benchmark resolves every field deterministically and therefore cannot
exercise a new-concept change.  This runner drives synthetic questions through
the real KnowledgeAgent -> verifier -> FieldResolver path using recorded model
responses.  It costs no tokens, is deterministic, and can run in CI.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from react_review.dkb import FieldResolver, KnowledgeBase  # noqa: E402
from react_review.llm.base import LLMBackend               # noqa: E402

DEFAULT_FIXTURE = ROOT / "eval" / "fixtures" / "new_concepts.json"
DEFAULT_KB = ROOT / "configs" / "knowledge.seed.json"


class FixtureBackend(LLMBackend):
    """Return one recorded classification while counting actual calls."""

    def __init__(
        self, response: dict[str, Any], responses_by_seed: dict[str, dict] | None = None,
    ) -> None:
        super().__init__()
        self.response = response
        self.responses_by_seed = responses_by_seed or {}
        self.calls = 0
        self.seeds: list[int] = []

    @property
    def model_id(self) -> str:
        return "concept-resolution-fixture"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.calls += 1
        self.seeds.append(seed)
        payload = self.responses_by_seed.get(str(seed), self.response)
        return json.dumps(payload, ensure_ascii=False)


def _matches(actual: Any, expected: Any) -> bool:
    return actual == expected


async def run_fixture(
    fixture_path: Path = DEFAULT_FIXTURE, kb_path: Path = DEFAULT_KB,
) -> list[dict[str, Any]]:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8-sig"))
    results: list[dict[str, Any]] = []
    for case in fixture.get("cases", []):
        backend = FixtureBackend(
            case.get("response") or {}, case.get("responses_by_seed"))
        resolver = FieldResolver(KnowledgeBase.from_json(kb_path), backend=backend)
        repeat = max(1, int(case.get("repeat") or 1))
        for _ in range(repeat):
            await resolver.resolve(
                case["raw_field_name"], unit=case.get("unit", ""),
                modality=case.get("modality", ""),
                research_context=case.get("research_context", ""),
                value=case.get("value"),
            )
        record = resolver.records[-1]
        actual = {
            "status": record.status,
            "field_type": record.field_type,
            "source": record.source,
            "llm_calls": backend.calls,
            "cache_hits": record.cache_hits,
            "attempts": len(record.attempts),
            "stability": record.stability,
            "consensus_count": record.consensus_count,
            "candidate_names": record.candidate_names,
            "proposal": record.proposal is not None,
            "checks": record.checks,
            "reasons": [r.model_dump(mode="json") for r in record.reasons],
            "resolution_key": record.resolution_key,
        }
        expected = case.get("expected") or {}
        failures: list[str] = []
        for name, value in expected.items():
            if name == "checks":
                for check, wanted in value.items():
                    if not _matches(actual["checks"].get(check), wanted):
                        failures.append(
                            f"checks.{check}: expected {wanted!r}, "
                            f"got {actual['checks'].get(check)!r}")
            elif not _matches(actual.get(name), value):
                failures.append(
                    f"{name}: expected {value!r}, got {actual.get(name)!r}")
        results.append({"id": case["id"], "ok": not failures,
                        "failures": failures, **actual})
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Offline acceptance check for new-concept field resolution.")
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--kb", type=Path, default=DEFAULT_KB)
    ap.add_argument("--json", action="store_true", help="print full JSON results")
    args = ap.parse_args(argv)

    if args.json:
        # Keep stdout machine-readable even when structlog is configured to
        # print the Resolver's informational events there.
        with contextlib.redirect_stdout(io.StringIO()):
            results = asyncio.run(run_fixture(args.fixture, args.kb))
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        results = asyncio.run(run_fixture(args.fixture, args.kb))
        for result in results:
            mark = "ok" if result["ok"] else "FAIL"
            print(f"[{mark}] {result['id']}: {result['status']} -> "
                  f"{result['field_type'] or 'UNRESOLVED'} "
                  f"({result['llm_calls']} LLM call(s))")
            for failure in result["failures"]:
                print(f"       {failure}")
    failed = [r for r in results if not r["ok"]]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
