# Cross-domain acceptance gate — pre-registration

- Gate file: `configs/gates/cross_domain_v1.json` (hash-pinned, version 1)
- Status: **provisional** — the thresholds are stated in advance, not yet signed off
- Written: 2026-08-10, **before** the evidence that could satisfy it exists
- Applied with: `python eval/check_gate.py <report.json> [--domain NAME]`

## Why this exists

Phase 6B reported 60% on a fifteen-row checkpoint and Phase 7 reported 80% on
the same fifteen rows. Both numbers were true and neither was usable: one row
moves that figure by 6.7 points, and every row came from one paper, so they were
never fifteen independent observations to begin with. The project has therefore
been unable to say what would count as success — which means it has also been
unable to say honestly that it has not achieved it yet.

A threshold chosen after seeing the result is not a threshold. This document and
the file beside it fix the definition first. Applying it today returns **NOT
ESTIMABLE**, and that is the correct current answer.

## What is measured, and on what unit

**The unit of analysis is the study; the estimand is domain-weighted.** Rows
from one paper share an extraction pass, a PDF and a set of arm labels, so
studies are resampled rather than rows (2000 resamples, fixed seed, percentile
method, studies sorted by identity so the answer cannot depend on the order the
reports were named). And because nine EAT studies beside one melanoma study
would otherwise make the headline figure a report on EAT, **each domain
contributes equally**, with its own studies resampled inside it.

Honesty about this choice: it was made because of the 9:1 sample imbalance, and
it **moved the numbers up** — melanoma's small, clean set now counts as much as
EAT's larger, messier one. It is recorded here, and in the gate file, precisely
so that it cannot later look like a choice made after seeing which way it went.

A held-out domain is **excluded from the pooled estimate and reported alone**.
Resamples in which a statistic is undefined are counted and reported; above 5%
of draws the interval is refused rather than computed from the draws that
happened to work. With fewer than two studies in every domain, no interval is
produced at all.

**An absence is not a zero.** Each hard gate has four states: never reported,
reported with nothing graded against it, graded and held, graded and failed.
Only the third is a pass. The two facts a run cannot certify about itself — that
its answer key was not edited after freezing, and that it reproduces from its own
recording — must come from a signed attestation naming and hashing the reports
it covers.

**Coverage is two numbers.** `retrieval_coverage` counts a value located;
`auditable_coverage` counts a value the audit could actually judge (arm
identified, population established, evidence protocol intact). A system can
raise the first by accepting anything; the gap between them is the part of a
review that was reached but not audited.

**Evaluating is not authorising.** The outcome carries `release_eligible`
separately from its status: a provisional gate can be met and still authorise
nothing, because meeting a bar we invented ourselves is not fitness for use.

Three families are kept apart, because collapsing them is what made earlier
numbers unreadable:

| Family | Examples | How it is judged |
| --- | --- | --- |
| **Safety** | silent releases, wrong target released, wrong scope released, review visibility | exact counts; must be 0 (or 1.0). Never averaged, never traded |
| **Capability** | discrepancy recall, precision, source coverage | cluster-bootstrap **lower bound** must clear the bar |
| **Reported only** | label accuracy, scope-assessable rate, review burden | published, never gated |

`label_accuracy` is deliberately not a gate. It mixes all three families into
one number, and a system can raise it by refusing less — the opposite of what
this design is for.

## The gate, version 1

**Sample minimums** (all must hold before accuracy is even asked):

- ≥ 3 domains, one of them **held out of development**
- ≥ 8 studies per domain
- ≥ 100 rows per domain
- ≥ 25 true discrepancies in total
- ≥ 30 numeric, ≥ 30 structured, ≥ 20 semantic rows

**Hard gates** (exact, non-negotiable):

`silent_release_count = 0` · `wrong_target_released_count = 0` ·
`wrong_scope_released_count = 0` · `review_visibility_rate = 1.0` ·
`answer_key_edits_after_freeze = 0` · `record_replay_differences = 0`

**Capability gates** (95% cluster-bootstrap lower bound):

`discrepancy_recall ≥ 0.70` · `discrepancy_precision ≥ 0.50` ·
`source_coverage ≥ 0.70`

Recall is held to a higher bar than precision on purpose: a false flag costs
review time, a missed discrepancy costs correctness.

## RETRACTED 2026-08-10 — what this section claimed, and why it was wrong

An earlier version of this document reported that "all six hard gates pass" and
published a table of confidence intervals. **Both claims are withdrawn.** They
were produced by the first version of `eval/check_gate.py`, which:

1. **Filled missing safety evidence with zero.** Four of the six hard gates had
   no evidence behind them. The EAT report grades no target identity and no
   population scope at all (`gold_rows = 0`), so its zero errors are zero out of
   zero — and `answer_key_edits_after_freeze` and `record_replay_differences`
   were hard-coded to 0 in the checker itself. Only `silent_release_count` and
   `review_visibility_rate` were ever backed by evidence.

2. **Produced intervals that depended on the order of the command line.** The
   same two reports gave a precision lower bound of 0.444 in one order and
   0.500 in the other — either side of the 0.50 bar — because studies were
   resampled by their position rather than by a stable identity.

The two mistakes have the same shape as the defects this project exists to
find: an absence recorded as a zero, and a number whose provenance was not
what it appeared to be. They are written down here rather than quietly fixed
because a pre-registration document that edits its own history is worth
nothing.

### The corrected evaluation (D6-R0/R1/R2, same evidence)

Verdict: **NOT ESTIMABLE**, release **NOT ELIGIBLE**.

| Hard gate | State |
| --- | --- |
| `silent_release_count` = 0 | ok, over 8 expected discrepancies |
| `review_visibility_rate` = 1.0 | ok |
| `wrong_target_released_count` | **nothing graded** — the weakest report grades 0 identity rows |
| `wrong_scope_released_count` | **nothing graded** — same |
| `answer_key_edits_after_freeze` | **no evidence** — needs a signed attestation |
| `record_replay_differences` | **no evidence** — same |

| Capability (domain-weighted) | Point | 95% interval | Per domain | Bar |
| --- | --- | --- | --- | --- |
| discrepancy recall | 0.900 | [0.750, 1.000] | eat 0.800 · melanoma 1.000 | ≥ 0.70 |
| discrepancy precision | 0.900 | [0.500, 1.000] | eat 0.800 · melanoma 1.000 | ≥ 0.50 |
| retrieval coverage | 0.889 | [0.833, 0.925] | eat 0.912 · melanoma 0.867 | ≥ 0.70 |
| auditable coverage | 0.823 | [0.767, 0.859] | — | ≥ 0.50 |

Blocked on: two domains rather than three; one study in melanoma rather than
eight; both domains under the per-domain row minimum; 8 true discrepancies
rather than 25, and only 4 studies carry one at all; every route under its
minimum; no held-out domain; and four of six hard gates without usable evidence.

Two of these are worth stating plainly. **The capability bars are now met** —
under an estimand fixed before the numbers were seen — and that changes nothing:
the sample cannot support the claim, and four safety gates have no evidence
behind them. And **nothing this project owns can serve as a held-out domain**:
`configs/gates/held_out_register.json` records both existing domains as
development, with an empty availability list.

What has not changed: the gate's definition, the minimums, and the fact that
the current evidence cannot support a cross-domain accuracy claim. Two domains,
one of them a single study, was never going to be enough, and that conclusion
never depended on the arithmetic above.

## Rules that protect the pre-registration

1. Changing a threshold requires a **new version file** with a written reason.
   Version 1 stays in the repository; it is not edited.
2. The gate's SHA-256 is recorded in every result that cites it.
3. A gate evaluated on a domain used during development is not evidence about
   that domain. The held-out requirement exists for exactly this.
4. Re-running after a fix does not re-use the held-out domain. A domain is
   held out once.
5. An answer key is never edited to fit model output. If a key is wrong, the
   correction is published as an overlay with its reason, and the affected
   figures are recomputed under both.

## Still required before this can be called final

- **A clinician's judgement on acceptable missed-discrepancy risk.** The 0.70
  recall bound is an engineering guess; nobody with clinical responsibility has
  said what rate of missed discrepancies is tolerable in a published review.
  Until they do, the gate's status stays `provisional`.
- **Confirmation that any new domain's answer key was built by someone
  qualified in that domain.** The two existing keys were hand-built and
  verified by the project owner; a third domain cannot rest on that alone.
- **A decision on review burden.** Flags per study and minutes per study are
  reported but ungated, because nobody has yet said how much human review makes
  the system worth using.
