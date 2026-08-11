"""The pinned hashes are checked, not merely written down.

Twice now a contract has been edited in place while the document above it said
that never happens, and once a published hash turned out never to have been
computed from the file at all — it shared a twelve-character prefix with the
real one and differed after, which is the shape a partly-copied value has and no
content change can produce.

Both survived because nothing compared the document to the files. A governance
rule that only exists in prose is a rule until the first time somebody is in a
hurry. This is the same rule, in a form that fails the build.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from react_review.contracts import repo_root, sha256_file

DOC = Path("docs/acceptance/gate_versions.md")

#: Every contract whose bytes are frozen, and the hash the document publishes.
#: Changing one of these values is a decision, not a fix: it means a file that
#: was supposed to be immutable has moved.
PINNED = {
    "configs/gates/cross_domain_v1.json": "AE182D0097A67A18",
    "configs/gates/cross_domain_v2.json": "E29CF4F803BA0F8A",
    "configs/aggregation/safe_sum_v1.json": "1C99DCE79E4FDD3A",
    "configs/aggregation/safe_sum_v2.json": "5FED9271920DF0A4",
    "configs/aggregation/safe_sum_v3.json": "93E381F2ED633E06",
    "configs/aggregation/safe_sum_v4.json": "FE1B925C28FA7558",
    "configs/aggregation/registry.json": "0F16E3F1228BC4E9",
    "configs/aggregation/safe_sum_v5.json": "DAEB6715F812E88E",
    "configs/aggregation/registry_v2.json": "3F593BB1097A3CA0",
    "configs/aggregation/evaluators/safe_aggregation_1.4.0.json": "3B4912D2A0596CEF",
    "configs/aggregation/evaluators/safe_aggregation_1.5.0.json": "0C5274C554AF1813",
}


@pytest.mark.parametrize("path,expected", sorted(PINNED.items()))
def test_a_frozen_contract_still_has_the_bytes_it_was_published_with(path, expected):
    assert sha256_file(repo_root() / path)[:16].upper() == expected


@pytest.mark.parametrize("path,expected", sorted(PINNED.items()))
def test_the_governance_document_publishes_the_hash_the_file_actually_has(
        path, expected):
    """The failure that started this: a pin nobody ever compared to a file."""
    text = (repo_root() / DOC).read_text(encoding="utf-8")
    rows = dict(re.findall(r"`(configs/\S+?\.json)`\s*\|\s*`([0-9A-F]{16})`", text))
    assert path in rows, f"{path} is frozen but {DOC} does not publish its hash"
    assert rows[path] == expected


def test_contract_files_are_pinned_to_lf_so_the_hashes_are_reproducible():
    """A Windows checkout renormalises to CRLF and changes every hash."""
    attributes = (repo_root() / ".gitattributes").read_text(encoding="utf-8")
    for directory in {Path(p).parent.as_posix() for p in PINNED}:
        assert f"{directory}/*.json text eol=lf" in attributes, directory
    for path in PINNED:
        assert b"\r\n" not in (repo_root() / path).read_bytes(), path


def test_a_policy_may_not_declare_a_rule_that_nothing_reads(tmp_path):
    """A file that appears to control behaviour it does not control is worse
    than a file that says nothing."""
    import json

    from react_review.contracts import ContractError
    from react_review.tools.safe_aggregation import load_aggregation_policy

    body = json.loads((repo_root() / "configs/aggregation/safe_sum_v5.json"
                       ).read_text(encoding="utf-8"))
    body["requirements"]["require_something_nobody_implements"] = True
    path = tmp_path / "invented.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ContractError, match="nothing reads"):
        load_aggregation_policy(str(path))


def test_an_invariant_may_not_be_switched_off(tmp_path):
    import json

    from react_review.contracts import ContractError
    from react_review.tools.safe_aggregation import load_aggregation_policy

    body = json.loads((repo_root() / "configs/aggregation/safe_sum_v5.json"
                       ).read_text(encoding="utf-8"))
    body["invariants"]["match_on_both_population_axes"] = False
    path = tmp_path / "disabled.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ContractError, match="not a switch"):
        load_aggregation_policy(str(path))
