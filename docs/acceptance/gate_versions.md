# Acceptance gate versions

A pre-registration that edits its own history is worth nothing. Each version is
a file that never changes after it is written; a correction is a new version
with a written reason, and the old one stays exactly as it was.

Hashes are of the file's bytes with LF line endings, which `.gitattributes` now
enforces. They are checked by `tests/test_contract_pins.py`, so a pin that stops
matching fails the suite instead of sitting in a document nobody recomputes.

| Version | File | SHA-256 (first 16) | Status |
| --- | --- | --- | --- |
| v1 | `configs/gates/cross_domain_v1.json` | `AE182D0097A67A18` | superseded; results withdrawn |
| v2 | `configs/gates/cross_domain_v2.json` | `E29CF4F803BA0F8A` | current; **provisional** |
| v1 | `configs/aggregation/safe_sum_v1.json` | `1C99DCE79E4FDD3A` | superseded; never applied to a result |
| v2 | `configs/aggregation/safe_sum_v2.json` | `5FED9271920DF0A4` | superseded; never applied to a result |
| v3 | `configs/aggregation/safe_sum_v3.json` | `93E381F2ED633E06` | superseded; never applied to a result |
| v4 | `configs/aggregation/safe_sum_v4.json` | `FE1B925C28FA7558` | current |
| — | `configs/aggregation/registry.json` | `0F16E3F1228BC4E9` | which policy/evaluator pairs may produce a formal result |

## A policy version is not enough — the code that applies it also has an identity

Every wrong total found in this phase came from a policy that read correctly and
code that did not enforce it. A result recording only `safe_sum_v3` therefore
claims a reproducibility it does not have: the same policy under two commits of
the evaluator gave different answers, and nothing in the artifact said which one
ran.

From `safe_sum_v4` the evaluator is versioned separately, in
`configs/aggregation/evaluators/`, with a hash over the files whose behaviour it
is. `configs/aggregation/registry.json` records which pairs may produce a formal
result. A run resolves both once at startup and records
`policy_hash`, `evaluator_hash` and the full 40-character commit; a working copy
whose evaluator files differ from HEAD still runs, and is marked not
release-eligible. Unrelated files in the working copy — slide decks, scratch
output — are not consulted, because they say nothing about which code decided.

Versioning is semantic and the rule is about OUTCOMES, not about how large the
change felt: anything that can move a claim between `derived`, `rejected`,
`protocol_error` and `not_applicable` is at least a MINOR, even when the author
is confident it is only a bug fix.

## The v2 pin was wrong, and the pins were not reproducible

Two faults, found by review on 2026-08-11.

The published v2 pin, `7F3FACB0B53CD9DF`, does not match `cross_domain_v2.json`
and cannot have matched any version of it: it shares its first twelve characters
with the file's hash and differs after, which no change of content can produce.
It was not a stale value. It was a partly-copied one, written by me in `d9d1039`
— which is worse than a wrong number, because the matching prefix makes it look
verified. Nothing was published under v2; the gate has returned NOT ESTIMABLE
throughout.

It happened again while this section was being written: the `safe_sum_v2` row
was first filled in with a hash that had never been computed from anything. The
test below caught it within the minute, which is the whole argument for having
the test rather than the rule.

And once more in the next review round, where the reflex was to edit `v2` in
place because the commit publishing it had just been sent back for rework. That
reasoning is exactly the exception the rule exists to refuse — a version is not
provisional because its author is still unhappy with it. v2 is restored to its
`8b21e68` bytes and the corrections are `safe_sum_v3`. Three versions in two
days is not version inflation; it is three genuinely different sets of rules,
and the numbers are the only honest record of which one a total was derived
under.

The reason nobody noticed the first time is the second fault: no test compared
the pin to the file, and the file's hash was not reproducible anyway. The repository had no
`.gitattributes`, so a Windows checkout renormalised these files to CRLF and
changed every hash — `cross_domain_v1.json` happened to keep its LF endings and
kept matching, which is luck, not integrity. Contract files are now pinned to
LF and the pins are asserted in the suite.

The gate FILES were not edited. v1 and v2 keep the thresholds they were written
with; only the recorded hash of v2, and the line endings of the working copy,
have changed.

## `safe_sum_v1` was edited in place — the same failure, again

`configs/aggregation/safe_sum_v1.json` was published in `fd0c104` and then
altered in the following commit while its `version` field still read 1. That is
precisely the mistake this document was written to record the first time, made
again one phase later, which says the rule was not load-bearing anywhere in the
process. It is now: `tests/test_contract_pins.py` fails if either policy's bytes
move.

v1 has been restored to its `fd0c104` bytes and the stricter rules published as
`safe_sum_v2.json`. No result was ever produced under v1 — the batch path it
governs is not yet wired into any run — so there is nothing to withdraw, only a
version to retire.

## Why v1's results were withdrawn — and why v1 itself was not touched

v1's thresholds were sound. The checker applying them was not: missing safety
numbers were read as zero, two hard gates were hard-coded to zero inside the
checker, and the confidence intervals depended on the order the reports were
named on the command line. Everything reported under v1 is withdrawn; see the
retraction in `cross_domain_gate.md`.

**This is also a governance failure of my own making.** The first correction
edited `cross_domain_v1.json` in place, adding a `revision 1.2` field — while
the same document said, in the section above it, that changing a threshold
requires a new version file and that v1 is not edited. The file has been
restored to its original bytes and the corrections published as v2.

## What v2 changes

- the estimand is fixed as **domain-weighted**, with studies resampled inside
  each domain, and the honesty note that this choice **raised** the observed
  figures is recorded in the file itself;
- a held-out domain is excluded from the pooled estimate and reported alone;
- undefined resamples are reported, and above 5% the interval is refused;
- discrepancies must be spread: a minimum per domain, and a minimum number of
  studies carrying one;
- `source_coverage` becomes `retrieval_coverage`, and `auditable_coverage` is
  added beside it — located is not the same as judgeable;
- `scope_assessable_rate` and `review_burden` are reported;
- every zero-error hard gate names the count it was graded over, so zero out of
  zero is not a pass.

## Rules

1. A version file is never edited after it is written. Not to fix a typo, not
   to add a threshold, not to record a result.
2. A correction is a new version naming what it supersedes and why.
3. Results are attributed to the gate version and hash that produced them.
4. Withdrawn results stay published as withdrawn, with the defect named.
