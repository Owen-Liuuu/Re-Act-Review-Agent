"""Shared primitives for hash-pinned contract files.

A contract file says which rules a run was made under. It is only worth
anything if the files it names cannot change behind its back, so every loader
here verifies a declared SHA-256 before believing a declaration, and every
failure is an exception rather than a default.

Kept apart from both the eval and the runtime loaders because they now need the
same rules: a benchmark contract that verified hashes one way and a runtime
contract that verified them another would be two standards wearing one name.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """A contract file that must not be used as given."""


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def read_json_object(path: Path, *, kind: str) -> dict[str, Any]:
    """Read a contract file, refusing anything that is not a JSON object."""
    if not path.is_file():
        raise ContractError(f"{kind} does not exist: {path}")
    try:
        body = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"{kind} is not valid JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise ContractError(f"{kind} must be a JSON object: {path}")
    return body


def verify_declared_hash(body: dict[str, Any], key: str, path: Path, *,
                         kind: str = "file") -> str:
    """A file a contract points at must be the one it was hashed against.

    Without this the pin is decorative: the path would still resolve after the
    file it names had been edited, and the contract would be describing rules
    that no longer exist.
    """
    if not path.is_file():
        raise ContractError(f"the contract points at a missing {kind}: {path}")
    declared = str(body.get(key) or "").upper()
    if not declared:
        raise ContractError(f"the contract does not declare {key}")
    actual = sha256_file(path)
    if declared != actual:
        raise ContractError(
            f"{path.name} does not match the {key} recorded in the contract "
            f"(contract {declared[:16]}…, file {actual[:16]}…)")
    return actual


def repo_root() -> Path:
    """Where run profiles and configs live, derived from the installed package.

    Paths inside a run profile resolve against the profile's OWN directory, so a
    profile is self-contained wherever it sits. A benchmark's reference to a run
    profile resolves against this root instead: run profiles live with the code,
    and a benchmark directory must not be able to redefine production rules by
    shipping its own.
    """
    return Path(__file__).resolve().parents[2]


def one_of(value: object, allowed: tuple[str, ...], *, field: str) -> str:
    text = str(value or "")
    if text not in allowed:
        raise ContractError(
            f"unknown {field} {text!r} (known: {', '.join(allowed)})")
    return text
