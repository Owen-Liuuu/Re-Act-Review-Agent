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

**The unit of analysis is the study, not the row.** Rows from one paper share an
extraction pass, a PDF, and a set of arm labels; one bad read moves many of them
together. Confidence intervals are therefore cluster-bootstrapped over studies
(2000 resamples, fixed seed, percentile method). With fewer than two studies no
interval is produced at all — the gate says so rather than reporting a point
estimate dressed as a measurement.

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

## What it says about today's evidence

Pooling everything that exists — melanoma (1 study, 15 rows) and EAT/T1DM
(9 studies, 57 rows) — gives **NOT ESTIMABLE**, blocked on: 2 domains rather
than 3, 1 study in the melanoma domain rather than 8, 8 true discrepancies
rather than 25, every route below its minimum, and no held-out domain.

All six hard gates pass. The capability numbers, computed anyway for
information:

| Metric | Point | 95% cluster interval | Bar |
| --- | --- | --- | --- |
| discrepancy recall | 0.875 | [0.600, 1.000] | ≥ 0.70 |
| discrepancy precision | 0.875 | [0.444, 1.000] | ≥ 0.50 |
| source coverage | 0.903 | [0.819, 0.983] | ≥ 0.70 |

**Both effect metrics look comfortable and neither clears its bar.** That gap
between a point estimate and what the evidence supports is the single most
useful thing this gate produces, and it is the quantitative form of what the
project has been saying in prose since Phase 6B.

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
