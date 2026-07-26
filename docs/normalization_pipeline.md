# Normalization pipeline

How raw, human-written table content is turned into comparable, matchable data
for the audit. Two tiers (deterministic **syntax** + LLM **semantic**) inside a
**review-driven directed** flow. This is the design; P1 implements Stage 0/3
and a standalone Stage 1 mapper, P2 implements Stage 1/2 in the Collector.

## Stages

```
Review long table (parser / P1 CSV)                    Source PDF
      │                                                    │
      ▼                                                    │
─ Stage 0  Syntax normalization (deterministic; values/units) ─┐
      │                                                    │    │
      ▼                                                    │    │
─ Stage 1  Review-side semantic normalization (LLM + vocab + cache)
      raw_field_name → field_type ; group_raw → canonical group
      hit → use ; miss → LLM maps into / extends the vocabulary
      │                                                    │    │
      ▼  canonical concept becomes the query               │    │
─ Stage 2  Source-side DIRECTED extraction (LLM) ◄─────────┘    │
      find {field_type} for {group} in the source paper,        │
      return value + verbatim quote + location + source's        │
      own field name + source unit                               │
      │                                                          │
      ▼                                                          │
─ Stage 3  Match + tolerance compare (deterministic) ────────────┘
      join (study, group, timepoint, field_type) → 1% rule → label
```

## Per-stage detail

| stage | input | process | output | det / LLM |
|---|---|---|---|---|
| 0 syntax | any value/unit string | primary-number (`6.60 ± 0.71`→6.60), decimal comma, unit norm (`kg/m²`=`kg/m2`), case/punct, DOI prefix | clean number + canonical unit | **deterministic** |
| 1b field_type | raw_field_name + context | cache key=(norm name+context); hit→lookup; miss→LLM maps into controlled vocab (or adds) → write cache+vocab | canonical `field_type` | cache hit=det / miss=LLM |
| 1c group | group_raw + context | same mechanism, separate group vocab | canonical group | same |
| 2 directed extract | (study, group, field_type) + vocab entry + context + source PDF | LLM targeted lookup; maps canonical concept back to the source's wording; returns value+quote+location+unit | source value | **LLM** (dual-LLM, ≤3 retries) |
| 3 compare | normalized review item + source item | units differ→`unit_mismatch`; else rel≤tol→`match` else `mismatch` | `MatchResult` | **deterministic** |

## Why directed (not blind-parallel)

Source extraction is targeted (a precise LLM query with context), not a blind
whole-table dump; the review↔source matching problem disappears; the vocabulary
stays scoped to what the review claims. Matches the architecture PDF's Agent 1
Collector being driven by the "Data Reference List (Claim)". The benchmark's
`audit_template.csv` was built exactly this way.

## Worked examples (benchmark data)

**Ahmad EAT → match**
```
review: ahmad_2022, "T1DM", "EFT/ EAT", "6.60 ± 0.71", mm
 S0  → 6.60, mm
 S1  → field_type=eat_thickness (vocab miss→LLM, writes {eat_thickness, synonyms:[EFT,EAT], unit:mm}); group t1dm
 S2  → Ahmad.pdf directed find → "EFT (mm)" (Table 2), "Diabetic children" (→t1dm synonym), 6.60±0.71, mm
 S3  → both mm; |6.60-6.60|/6.60=0% ≤1% → MATCH
```

**Keles EAT → unit_mismatch**
```
review: keles_2016, "T1DM", "EFT/ EAT", "0.7 (0.6–0.9)", mm (Table 1 column implies mm)
 S1  → eat_thickness, t1dm
 S2  → Keles.pdf → source field "EFT (cm)", value 0.7 (0.6–0.9), unit cm
 S3  → review mm vs source cm → UNIT_MISMATCH (numbers equal, unit differs)
```

## Vocabulary & cache shapes

```json
// field_type vocabulary (controlled + extensible, persisted JSON)
{
  "eat_thickness": {"concept": "epicardial adipose tissue thickness",
                    "value_type": "numeric", "default_unit": "mm",
                    "synonyms": ["EFT", "EAT", "epicardial fat thickness"]},
  "eat_volume":   {"value_type": "numeric", "default_unit": "cm3", "synonyms": [...]},
  "bmi": {...}, "sample_size": {...}
}
// group vocabulary
{"t1dm": {"synonyms": ["T1DM","DM","Diabetic children","Patient"]}, "control": {...}, "all": {...}}
// cache (key MUST include context — "Patient"→t1dm only holds in this topic)
{"hash(EFT/ EAT | ctx=EAT-in-T1DM)": "eat_thickness", ...}
```

## dual-LLM + retry

Both extraction points (Stage 1 review, Stage 2 source) run two LLMs
independently, compare, and re-extract up to 3 rounds on disagreement. This
governs **extraction reliability** — orthogonal to Stage 3's tolerance verdict.

## Scaling note

When the vocabulary grows across domains, retrieve top-K candidate field_types
(keyword / embedding) and pass only those (+ "propose new") to the LLM, instead
of the whole vocabulary. MVP is single-domain, so full vocab is fine.

## P1 vs P2

| stage | P1 (CSV-fed) | P2 (real Collector) |
|---|---|---|
| 0 syntax | ✅ used by compare | ✅ |
| 1 review semantic | CSV already carries field_type; build the mapper standalone and grade it vs benchmark `raw_field_name → field_type` | ✅ live |
| 2 source directed extract | CSV already carries source_value | ✅ live (dual-LLM, PDF) |
| 3 compare | ✅ primary P1 validation (53/0/4 + seeds) | ✅ |
