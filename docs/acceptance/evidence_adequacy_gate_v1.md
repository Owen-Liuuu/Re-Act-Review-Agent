# Evidence adequacy transition gate v1

This gate freezes the expected disposition of the 19 mismatch or unit-mismatch
accusations in saved online run `1baf36e59097` before the adequacy evaluator is
implemented. It is a safety transition set, not a claim that two reviews prove
cross-domain generalisation.

The transition CSV uses existing `A###` benchmark identities where a row is in
`audit_template.csv`. The eight review cells outside that numeric benchmark use
the deterministic production claim identities (`A_06`, `B_07`, and so on)
obtained by applying `assign_claim_ids` to the saved captured table. These are
not positional IDs invented by this gate.

## Frozen disposition

- Nine wrong-field or wrong-target bindings — A028, A030, A031, A037, A038,
  A041, A042, A044, A045 — become evidence `insufficient` and
  `not_comparable`.
- Eight group/quality cells bound to unrelated findings — A_06, B_07, C_07,
  D_06, D_07, E_07, F_07, G_07 — also become evidence `insufficient` and
  `not_comparable`.
- A010 and A013 remain `unit_mismatch`: the PMC table supplies both BMI values
  under the printed `kg/m3` header, while the review reports `kg/m2`.

The preimplementation result count therefore changes from 30 match / 12
mismatch / 7 unit mismatch / 28 not comparable to an expected 30 match / 0
mismatch / 2 unit mismatch / 45 not comparable for this saved run. Its overall
verdict is still expected to fail because the two verified unit discrepancies
remain visible.

## Positive abstract evidence

A034 and A035 are explicit counterexamples to a blanket “abstracts are
insufficient” rule. The Iacobellis abstract states that the 15 type 1 diabetes
subjects had age `52.8 ± 12` and BMI `27.8 ± 5.2`. The value, field and T1DM
target are all in the same bounded statement, so both rows must remain match.

The Aslan abstract supplies the corresponding adversarial case: its statement
binds `30.6 ± 10 years` to age in the type 1 diabetes cohort. Reusing that value
for T1DM BMI (A028) fails the field axis; reusing it for control age (A030)
fails the target axis. A direct quote alone is therefore not sufficient.

A041 is narrower: the abstract gives a rounded `31 ± 8` without defining the
statistic, while the claim is the more precise `30.8 ± 7.7`. The rounded text is
compatible but cannot verify the exact statistic and precision, so this gate
marks the value axis insufficient rather than producing an accusation.

## Release failures

The gate fails if any of the following occurs:

- either A034 or A035 becomes not comparable;
- any of the nine hard-gate wrong bindings reaches numeric comparison;
- A010 or A013 is hidden by document scope;
- every abstract row is rejected without checking claim-level bindings;
- a legacy `unknown` document scope reaches `sufficient` from text alone;
- a direct quote alone is treated as sufficient;
- the comparator emits mismatch before adequacy is assessed;
- legacy artifacts gain empty adequacy keys; or
- the policy/evaluator is not registered and pinned by the run manifest.

Validate the frozen transition table with:

```powershell
python eval/check_evidence_adequacy.py
```

After producing a schema-v4 replay artifact, join the frozen expectations to
the actual claim IDs (including positive controls A034/A035) with:

```powershell
python eval/check_evidence_adequacy.py --results path\to\full_accuracy.json
```

The first command validates the precommitted transition contract. The second
is the implementation acceptance check; without a produced `--results` file,
the expected 77-row totals are a preregistered target, not a claim that the
full replay has already run.
