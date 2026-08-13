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
| v4 | `configs/aggregation/safe_sum_v4.json` | `FE1B925C28FA7558` | superseded; never applied to a result |
| v5 | `configs/aggregation/safe_sum_v5.json` | `DAEB6715F812E88E` | current |
| v1 | `configs/aggregation/registry.json` | `0F16E3F1228BC4E9` | superseded; could not grow without breaking its own pin |
| v2 | `configs/aggregation/registry_v2.json` | `3F593BB1097A3CA0` | superseded |
| 1.4.0 | `configs/aggregation/evaluators/safe_aggregation_1.4.0.json` | `3B4912D2A0596CEF` | superseded |
| 1.5.0 | `configs/aggregation/evaluators/safe_aggregation_1.5.0.json` | `0C5274C554AF1813` | superseded |

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

The registry is versioned too. It is pinned by hash, so a single file could
never gain a policy or an evaluator pair without breaking its own immutability
rule — and changing which pairs may publish is a change to what a result means,
which ought to cost a version. `registry_v2.json` also records each policy's
file and bytes, so readiness computes the policy hash itself rather than
accepting one from its caller; handed the string "not-a-hash", the previous
version reported a registered, release-eligible run.

| v3 | `configs/aggregation/registry_v3.json` | `7B439CBDA99D54D4` | superseded |
| 1.6.0 | `configs/aggregation/evaluators/safe_aggregation_1.6.0.json` | `E2514A771F246A1C` | superseded |

| v4 | `configs/aggregation/registry_v4.json` | `83AB58639A50D8EE` | current |
| 1.6.1 | `configs/aggregation/evaluators/safe_aggregation_1.6.1.json` | `A49941F1458D43B4` | current evaluator |

| — | `configs/run_profiles/phase8_batch.json` | `17C65B8C07A45898` | the contract that selects the batch route |
| — | `eval/benchmarks/melanoma_checkpoint_2017/phase8_batch_profile.json` | `8F13C10443B8BEDE` | the benchmark that runs it |

| v5 | `configs/prompt_contracts/batch_v5.json` | `1A02A44DDE55A797` | what `targeted_v5_batch` ASKS |
| v1 | `eval/benchmarks/melanoma_checkpoint_2017/excerpt_gold_v1.json` | `030A04D6332A5E2E` | WHERE the evidence is, for excerpt coverage |

| v5 | `configs/aggregation/registry_v5.json` | `B4BFFCB393E581F2` | current |
| 1.6.2 | `configs/aggregation/evaluators/safe_aggregation_1.6.2.json` | `F01DF1DD72077BA7` | current evaluator |
| — | `configs/run_profiles/phase8_batch_v2.json` | `AD9F2F180F718BEB` | the contract that selects the batch route under 1.6.2 |
| — | `eval/benchmarks/melanoma_checkpoint_2017/phase8_batch_v2_profile.json` | `9275079632081DB6` | the benchmark that runs it |
| v2 | `eval/benchmarks/melanoma_checkpoint_2017/excerpt_gold_v2.json` | `9F67F418E245E656` | WHERE the evidence is, keyed on the batches the run makes |
| — | `eval/benchmarks/melanoma_checkpoint_2017/phase8_batch_v3_profile.json` | `779E9F1C97D5B9A9` | the benchmark that judges against gold v2 |
| v1 | `eval/benchmarks/melanoma_checkpoint_2017/d1_7_expected_plan.json` | `72AD64CBE18447AD` | what a D1-7 recording is expected to ask for |
| v1 | `configs/gates/d1_batch_v1.json` | `B624742F31536742` | superseded; **could not judge the run it was written for** |
| v2 | `configs/gates/d1_batch_v2.json` | `1221EC40F789F04C` | current; **provisional**, capability floor deliberately UNSET |

## d1_batch v1 → v2: a gate that could not express what happened

v1 defines `wrong_released` as a wrong value released WITHOUT review. The D1-7
recording produced MA015: predicted `match`, expected `mismatch`, and
`review_required=True` — wrong AND escalated, which v1 has no term for.

The FAIL reported on the day was not v1's verdict. It came from a throwaway
classifier written beside the result, which called a review-flagged row
`wrong_released` in direct contradiction of the gate's own text, and no
executable v1 classifier existed in the repository at all. The correct record is
**NOT EVALUABLE (protocol error)**, and the earlier FAIL stands only as a record
of what was reported.

v2 adds the fourth state, an executable classifier
(`src/react_review/acceptance_transitions.py`) so a verdict is computed rather
than argued, and the mechanism for a capability floor — because every one of
v1's hard conditions is a prohibition, and a system that refused every row
satisfied all of them.

**The floor's VALUE is deliberately unset.** A draft of v2 set it at 0.8, and
the recording keeps 8 of the baseline's 10 correct rows — exactly 0.8. That
number was chosen after seeing the result and would have been a threshold fitted
to pass it, which is the failure this whole apparatus exists to prevent. So the
mechanism ships and the value does not; like the 0.70 recall bar in
`cross_domain`, it is blocked on a human stating what capability loss is
tolerable. Until then a run can only reach `PASS (PROHIBITIONS ONLY)`, which must
never be read as "the route works".

Applying v2 to the D1-7 recording is a **post-hoc reanalysis** — the gate was
changed after the run, by an author who knew the verdict it would change. It is
published because v1 could not judge the run at all, not because v1's verdict
was inconvenient.

## excerpt_gold v1 → v2: a key that described a reading nobody makes

v1 assumed Larkin's three arm counts would be read in one batch. They are not.
They come from three different review columns — "Intervention arm drug, dose,
n", "Control arm …", "Additional arm …" — and the batch group key includes the
raw field name, so each is its own reading with one claim in it. Batching buys
nothing for that field in this benchmark, which is a fact about the benchmark
and not about the route.

The error was invisible while coverage joined on `(study_id, field_type,
target_shape)`: that triple collapsed all three readings into v1's single entry
and judged them against its witnesses. The join is now `audit_ids ↔ claim_ids`,
checked against study, field and shape, and a duplicate key is refused rather
than resolved. `eval/excerpt_dry_run.py` computes the real grouping with the
real Collector, the real knowledge base and the real selector, without asking a
model — every batch in v2 came from it.

v1 stays exactly as published, and `phase8_batch_v2_profile.json` still names
it: a benchmark profile that changed retroactively would describe runs it never
governed.

## 1.6.2: recording what was SENT is not reading it back

D1-6.2 added `ExcerptProvenance` to `schemas/batch.py` — which regions of a
paper reached the model, and which selector chose them — and repointed
`tools/aggregation_identity.py` at `registry_v5`. Both files are inside the
evaluator boundary, so the tree stopped matching 1.6.1 and said so in
`PENDING.json` rather than by regenerating a published manifest.

The record is written by the caller that did the windowing and is read only
offline, by an answer key. No claim can move between derived, rejected,
protocol_error and not_applicable because of it. That belief did not decide the
version: `eval/aggregation_behavior.py --compare` reported 26 of 26 cases
identical against the frozen corpus, which is what a PATCH means here.

`registry_v5` lists `safe_sum_v5` as cleared by both 1.6.1 and 1.6.2 — the
policy did not change, and runs already made under 1.6.1 stay interpretable.
`phase8_batch.json` and `phase8_batch_profile.json` are untouched; a published
contract that changed retroactively would describe runs it never governed.

## A rendered prompt that changes is a new prompt version

The batch prompt's SHA-256 enters the replay cache key and the
`BatchQuestionId` on every batched answer. A character changed in the template
therefore invalidates every recording ever made under it, and the symptom —
`ExtractionCacheMiss: attempt 1` — reads as a missing recording rather than an
edited prompt. `legacy_v3` has been pinned since Phase 7; the batch contract was
routed to by a frozen benchmark profile while nothing pinned its words, because
that profile pins the run profile, which names route STRINGS.

`configs/prompt_contracts/batch_v5.json` pins the RENDERED prompt for every
branch — each target shape, the timepoint clause present and absent, the
aggregation block present and absent — together with the recording key each
would be written under. Comments, renames and extracted helpers do not change
what is asked and do not fail it.

**The rule.** If a rendered prompt must change, add a new profile in
`tools/extraction_profile.py` and a new contract file for it. Do not update a
hash in an existing contract so that an edited prompt passes: that converts
every recording made under it into a replay miss, with nothing in the artifact
to say why. The contract file's own bytes are pinned in
`tests/test_contract_pins.py`, so editing it is a visible decision rather than a
silent one — which is the same protection the evaluator manifests get, for the
same reason.

## Erratum: what registry_v4 says it changed

`registry_v4.json` and `safe_aggregation_1.6.1.json` describe the D1-5 boundary
change as "the batch schemas, the evidence schemas and the result mapping". The
third is wrong and one is missing. The files inside the evaluator boundary that
actually changed between `f2adc5e` and `ddbeafc` are:

    src/react_review/schemas/batch.py
    src/react_review/schemas/evidence.py
    src/react_review/tools/aggregation_identity.py

`tools/batch_result.py` did not change; `aggregation_identity.py` did, because
its registry pointer moved to `registry_v4`. Neither file is edited to correct
this — that is the rule these documents exist to keep — and neither the hash nor
the 1.6.1 decision is affected: the behaviour comparison is computed from the
code, not from the prose, and reported 26 of 26 identical either way.

An earlier draft of this table also left `registry_v3` marked current beside
`registry_v4`. Only `registry_v4` is current.

## A version is decided by measurement, not by reading the diff

D1-5 edited three files inside the evaluator's hashed boundary and added no new
refusal. "Only provenance wiring" is exactly what somebody believes before
discovering that a total moved from 944 to 945, so the claim is not made by
inspection. `eval/aggregation_behavior.py` runs a frozen 26-case corpus through
the projector and records a behaviour vector — status, value, chosen object,
verified scope, applied axes, component counts, refusal stage — and compares it
to the baseline frozen at `f2adc5e` BEFORE the phase began. Identical means
PATCH; any difference means at least MINOR, whatever the author believes.

It reported 26 of 26 identical, so D1-5 ships as `1.6.1`. Each evaluator version
ships with the registry that authorises it, which is why `registry_v4` accompanies
it: the pointer is part of publishing a version rather than a change within one.

The corpus declares what each case is supposed to demonstrate and the emitter
refuses to freeze a case that does not. The first draft of
`explicit_totals_disagree` quoted a population sentence its document did not
contain, so its aggregation never parsed, the printed total won by default, and
it recorded `ok` while claiming to prove that a contradiction is refused. A
frozen baseline of that would have defended the wrong behaviour.

A policy and the identity that cleared it are ONE object, `AggregationRuntime`.
They were two arguments to the projector, and readiness could clear one policy
while the run applied another — a result naming a policy nobody had checked, and
reporting itself release-eligible. Comparing them at the exit would have closed
that hole and kept the shape of it, so the projector now takes only the bound
pair, and the scope axes come from it too: the flag that let a caller declare
them pre-computed skipped the union this policy exists to require.

`aggregation_identity.py` is inside the hashed evaluator boundary from 1.6.0.
Being only in the clean check caught an uncommitted edit and missed a committed
one — the evaluator hash stayed put and the run went back to clean.

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
