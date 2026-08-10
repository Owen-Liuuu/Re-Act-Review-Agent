# Acceptance gate versions

A pre-registration that edits its own history is worth nothing. Each version is
a file that never changes after it is written; a correction is a new version
with a written reason, and the old one stays exactly as it was.

| Version | File | SHA-256 (first 16) | Status |
| --- | --- | --- | --- |
| v1 | `configs/gates/cross_domain_v1.json` | `AE182D0097A67A18` | superseded; results withdrawn |
| v2 | `configs/gates/cross_domain_v2.json` | `7F3FACB0B53CD9DF` | current; **provisional** |

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
