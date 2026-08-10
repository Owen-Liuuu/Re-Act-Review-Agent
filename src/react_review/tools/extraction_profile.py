"""Which extraction prompt contract a request runs under.

The extraction replay cache is keyed on the prompt version AND the prompt's own
SHA-256, so a recorded benchmark stays replayable only while both are byte-for-
byte what they were. A new contract therefore has to be a SEPARATE profile
rather than an edit to the existing prompt: ``legacy_v3`` must keep emitting the
exact bytes Phase 6 recorded, whatever ``targeted_v4`` grows into.

The profile is carried explicitly on the request. It is deliberately not
inferred from whether cohorts happen to be populated: an inferred profile would
make a frozen benchmark's replayability depend on how a caller filled in an
optional field, which is the kind of accident this module exists to prevent.
"""
from __future__ import annotations

LEGACY_V3 = "extract-source-v3-scoped-cohort-counts"
TARGETED_V4 = "extract-source-v4-targeted-components"
#: One batched reading of one field, in one shape. Its prompt lives in
#: tools/batch_prompt.py and asks for EVERY reading rather than one target.
BATCH_V5 = "extract-source-v5-batch"

#: profile name -> the ``prompt_version`` recorded in the replay cache key.
PROMPT_VERSIONS = {
    "legacy_v3": LEGACY_V3,
    "targeted_v4": TARGETED_V4,
    "targeted_v5_batch": BATCH_V5,
}
DEFAULT_PROFILE = "legacy_v3"


def prompt_profile(payload: object) -> str:
    """The profile a request runs under — the single place this is decided."""
    name = str(getattr(payload, "extraction_profile", "") or DEFAULT_PROFILE)
    if name not in PROMPT_VERSIONS:
        raise ValueError(
            f"unknown extraction profile {name!r} "
            f"(known: {', '.join(sorted(PROMPT_VERSIONS))})")
    return name


def prompt_version(profile: str) -> str:
    """The cache-key version string for a profile."""
    try:
        return PROMPT_VERSIONS[profile]
    except KeyError:
        raise ValueError(
            f"unknown extraction profile {profile!r} "
            f"(known: {', '.join(sorted(PROMPT_VERSIONS))})") from None
