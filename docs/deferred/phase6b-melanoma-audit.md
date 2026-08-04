# Deferred issue archive: Phase 6B melanoma audit

- Status: **deferred by project decision**
- Archived: 2026-08-03
- Discovery commit: `5c344db`
- Benchmark: `melanoma_checkpoint_2017`

## Decision

Phase 6B is preserved as a failed cross-domain accuracy gate. No extraction,
comparison, prompt, or benchmark-answer change is authorized as part of this
archive. The mechanism reached every planned path, but the failures below will
be fixed in a later phase rather than folded into Phase 6B.

The frozen live/record/replay artifacts must not be overwritten. They contain
source-paper quotations and remain under ignored
`output/baselines/melanoma_checkpoint_2017/`. Their hashes are published in
`docs/baselines/melanoma_phase6b_metrics.json`.

## How precise these numbers are

One study, 15 rows. A single row moves label accuracy by 6.7 points, and 9/15
carries a Wilson 95% interval of roughly **[35.7%, 80.2%]**. So "60%" is not a
measurement of cross-domain accuracy — it is far too coarse to rank against the
EAT benchmark or to detect a modest improvement.

What this checkpoint establishes is categorical rather than numeric: every
planned route was reached, no discrepancy was released silently, and the failures
below reproduce from the frozen replay. Read the percentages as "the judgements
are not reliable here yet", not as a score. Any future claim about cross-domain
accuracy needs more studies and more rows — not a better run of this one.

## Preserved observations

- Recorded replay: 15 rows; label accuracy 60%; strict discrepancy precision
  20%; strict recall 33.3%; F1 25%.
- Safety remained visible: zero silent releases and 100% review visibility.
- All expected routes ran: 4 semantic, 4 numeric, and 7 structured rows.
- Live and record differed on two rows: MA005 changed semantic relation/label;
  MA010 changed the completeness of the extracted PFS value.
- The recorded replay had five unexpected label differences: MA005, MA007,
  MA011, MA013, and MA014.
- Semantic-relation differences occurred on MA003, MA005, and MA007.
- The declared confidence-level gap was exposed on MA012. On MA015 it was
  masked by incomplete or wrong extraction and therefore cannot be counted as
  a successful comparator result.

## Deferred problems

### D6B-01 - multi-arm target drift

The directed extractor can select the first nearby treatment arm, PFS value,
or hazard ratio instead of the requested arm/comparison. This is the dominant
cause of the unexpected label differences.

Future work should make the extraction target include the requested cohort or
comparison pair and require the evidence span to anchor those target terms.
Ambiguous or conflicting anchors must return an explicit unresolved outcome,
not the nearest value.

### D6B-02 - incomplete structured extraction

The model sometimes returns only the point estimate even when the source gives
a confidence interval. It can also return wording such as "confidence
interval" that the deterministic parser does not consume as the same structure
as `CI`.

Future work should preserve point estimate, confidence level, lower bound, and
upper bound as explicit source-side components. A missing source component must
remain partial and review-required; it must not receive complete-extraction
credit.

### D6B-03 - semantic relation instability

The live and recorded runs disagreed on MA005. Some rationales also described
one side as more specific while assigning the opposite broader/narrower label.

Future work should deterministically check that the relation label agrees with
the stated direction of specificity. Broader relations remain review-required;
confidence alone is not an acceptance signal.

### D6B-04 - confidence level is not represented

The numeric schema compares interval bounds but does not represent 95% versus
99.5% as a component. This pre-existing frozen gap remains open independently
of extraction quality.

## Re-entry conditions

Work on this archive may resume only as a new, explicitly scoped phase. The
future implementation should:

1. preserve the existing Phase 6B caches and reports unchanged;
2. add offline regression fixtures for wrong-arm selection, partial CI output,
   confidence-level mismatch, and semantic direction inconsistency;
3. make MA005, MA007, MA011, MA013, and MA014 agree with the frozen answer key
   without changing that answer key to fit model output;
4. expose both MA012 and MA015 as genuine confidence-level discrepancies rather
   than allowing extraction errors to mask either one;
5. keep silent releases at zero and review visibility at 100%;
6. run the EAT replay suite to demonstrate no regression;
7. create a new versioned live/record cache instead of overwriting Phase 6B.

Until those conditions are met, the correct project claim is:

> The cross-domain mechanisms were exercised end to end, but cross-domain
> accuracy was not established.

---

## Phase 7 disposition (appended 2026-08-04)

Everything above is the Phase 6B record and is unchanged. This section only
states what a later, separately scoped phase did about it. The Phase 6B
artifacts, caches and metrics were not overwritten; Phase 7 wrote its own under
`phase7_*` names, and a run that does not name the Phase 7 profile still
reproduces the Phase 6 result exactly.

Measurements: `docs/baselines/melanoma_phase7_metrics.json`.

| Issue | Status | What decides it |
| --- | --- | --- |
| D6B-01 multi-arm target drift | largely closed | the request carries the arm or the comparison pair; the model enumerates what the paper reports and deterministic code assigns the target, refusing a tie. `wrong_target_accepted` 3 → 1 |
| D6B-02 incomplete structured extraction | closed here | point estimate, level and both bounds travel as verified components and are consumed by the comparator; an omitted interval the quote states is recorded as incomplete, not credited |
| D6B-03 semantic relation instability | partially closed | a verdict contradicting its own specificity direction is refused (`relation_direction`). A self-consistent but wrong direction remains undetectable — see below |
| D6B-04 confidence level not represented | closed | `ci_level` is a component, compared exactly; MA012 and MA015 are genuine 95%-vs-99.5% discrepancies rather than masked ones |

Re-entry conditions, item by item:

1. **Preserve Phase 6B caches and reports** — done; the Phase 6 replay still runs
   from them under its own profile and returns the same numbers.
2. **Offline regression fixtures** — done, under `tests/fixtures/phase7/`, built
   from the raw responses that actually failed rather than from invented ones.
3. **MA005, MA007, MA011, MA013, MA014 agree with the frozen answer key** —
   **partially**. MA007, MA011 and MA013 now agree. MA005 is refused as
   review-required rather than agreeing, and MA014 is refused because the model
   rewrote its own quote. The answer key was not touched.
4. **MA012 and MA015 exposed as genuine confidence-level discrepancies** — done,
   and this was only possible once the extraction stopped dropping the interval:
   the gap had been hidden by an incomplete answer, not survived by a correct one.
5. **Silent releases zero, review visibility 100%** — held throughout, including
   at the intermediate state where the newly complete intervals had removed the
   accidental review flag but `ci_level` did not yet exist. That state recorded
   two silent releases and was not accepted as a result; it is preserved as
   evidence rather than deleted.
6. **EAT replay shows no regression** — 0 cache misses and identical metrics at
   every step of Phase 7.
7. **New versioned cache instead of overwriting** — done.

### What Phase 7 did NOT establish

- The accuracy gate is not declared passed. Passing it was never a Phase 7
  acceptance target, and 15 rows cannot establish cross-domain accuracy however
  they come out.
- A semantic verdict can be **self-consistent and still wrong**: MA003 labels the
  relation `review_broader` while naming the source as the more specific side,
  which is internally coherent and factually backwards. No deterministic check in
  this design can refute it; 3 of the 4 semantic rows still disagree with the
  Phase 7 relation overlay.
- Two rows are lost to a defect this phase did not fix: the model returned a
  quote it had reworded, and the anchoring check correctly refused it. That is a
  capability loss, not a safety one.
