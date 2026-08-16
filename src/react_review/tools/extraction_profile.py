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
#: ``targeted_v4``'s question with its worked examples written as cohort
#: placeholders instead of one review's disease and arm names. The rules are the
#: same rules; what changes is that a review from another field no longer reads
#: its own answer in the instructions. Kept as a separate profile because v4's
#: recordings must stay reachable, so the two can be scored against each other
#: rather than swapped on the belief that neutral wording must be better.
TARGETED_V6 = "extract-source-v6-targeted-neutral-examples"

#: profile name -> the ``prompt_version`` recorded in the replay cache key.
PROMPT_VERSIONS = {
    "legacy_v3": LEGACY_V3,
    "targeted_v4": TARGETED_V4,
    "targeted_v5_batch": BATCH_V5,
    "targeted_v6": TARGETED_V6,
}
#: The profile name a batch request runs under, so the batch tool need not
#: repeat the string that keys its own recordings.
BATCH_PROFILE_NAME = "targeted_v5_batch"
DEFAULT_PROFILE = "legacy_v3"
#: Profiles whose prompt renders the enumerate-then-assign sections, so the
#: audit rather than the model decides which arm an answer belongs to. One place
#: to add a profile to, instead of a string comparison per call site.
_TARGETED_PROFILES = frozenset({"targeted_v4", "targeted_v6"})


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


def uses_targeted_sections(profile: str) -> bool:
    """Whether this profile asks the model to enumerate arms for the audit.

    An unknown name is refused rather than treated as "not targeted": silently
    falling back would render the legacy body under a profile nobody defined and
    record it in that profile's namespace.
    """
    if profile not in PROMPT_VERSIONS:
        raise ValueError(
            f"unknown extraction profile {profile!r} "
            f"(known: {', '.join(sorted(PROMPT_VERSIONS))})")
    return profile in _TARGETED_PROFILES
