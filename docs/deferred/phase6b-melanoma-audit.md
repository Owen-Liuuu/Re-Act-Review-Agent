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
