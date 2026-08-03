# Published benchmark summaries

This directory contains reviewable, version-controlled benchmark metadata only.
It intentionally excludes source-paper quotations, full per-row evidence, local
absolute paths, and raw LLM responses.

The complete reports and extraction recordings stay under `output/baselines/`,
which is ignored by Git because every Phase 6 report contains verbatim source
quotes. Their SHA-256 digests are recorded here so a local copy can be verified
without publishing the quoted text.

Phase 6 uses two distinct kinds of evidence:

- **Replay runs** test deterministic code against frozen raw model responses.
- **Live runs** measure model variance and are never selected or repeated until
  a preferred score appears.

`eat_phase5_phase6_metrics.json` is the score history. The Phase 6-0d score is
retained but explicitly invalidated because a missing source value plus a
residual unit accidentally produced `UNIT_MISMATCH`. Phase 6-0e is the first
score using the corrected comparison semantics.

`eat_phase6_manifest.json` records the code/input identities, private-artifact
hashes, acceptance result, and the one-live-run policy.

`melanoma_phase6b_metrics.json` publishes the quote-free result of the frozen
oncology run. It records one independent live observation and one recorded
live run with its deterministic replay. The mechanism reached all four
semantic rows and all expected comparison modes, but the accuracy gate failed:
multi-arm extraction repeatedly selected a neighbouring arm/effect estimate.
