# Phase 7 offline fixtures

Every file here lets a Phase 7 behaviour be tested **without an LLM**. They are
the recorded evidence of the Phase 6B failures, so a test built on them fails
for the same reason the benchmark did rather than for an invented one.

| File | What it pins |
| --- | --- |
| `larkin_excerpt.txt` | two passages of the source paper — the arm-assignment sentence and the efficacy paragraph. Enough text for the anchoring checks to be real. |
| `wrong_arm_pfs.json` | the extractor was asked for the nivolumab-monotherapy PFS and returned the combination arm's 11.5 months (D6B-01) |
| `wrong_comparison_hr.json` | asked for the combination-vs-nivolumab hazard ratio, returned the combination-vs-ipilimumab 0.42 (D6B-01) |
| `partial_ci_hr.json` | the quote prints `0.42; 99.5% CI, 0.31 to 0.57`, the response returned only `0.42` (D6B-02, and the masking that hides D6B-04) |
| `misquoted_arm_counts.json` | the three arm counts are right, but the display quote misstates the nivolumab arm as 315 where the paper prints 316 |

`response` in each JSON is the **verbatim raw model response** recorded by the
Phase 6E melanoma run (`glm-4.5-flash`), copied out of the run's extraction
cache; `asked_for` and `expected_source_value` restate the question and the
frozen answer key's source value so a test does not have to reach into the
benchmark. The source snippets are short and attributed, matching the published
policy for benchmark answer keys.

These fixtures are inputs, not expectations: what a fixed extractor should
*return* for them is asserted in the tests, not stored here.
