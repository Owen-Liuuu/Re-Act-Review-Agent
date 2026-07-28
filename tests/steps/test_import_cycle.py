"""Regression: data_extraction <-> paper_verification must not import-cycle.

The cycle only manifests when one side is the FIRST import in a fresh
interpreter (in-process tests prime the import graph and hide it), so each case
runs in its own subprocess.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

_MODULES_FIRST = [
    "react_review.steps.data_extraction.schemas",
    "react_review.steps.paper_verification.interfaces",
    "react_review.steps.paper_verification.schemas",
]


@pytest.mark.parametrize("module", _MODULES_FIRST)
def test_no_circular_import(module):
    r = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"importing {module} first failed:\n{r.stderr}"
