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
The resulting extraction, structured-value, and semantic-relation issues are
intentionally deferred and preserved in
`docs/deferred/phase6b-melanoma-audit.md`; Phase 6B artifacts must not be
overwritten when that work resumes.

`phase6e_acceptance.json` is the final quote-free acceptance record. It ties
the full test suite, both deterministic replays, the package-to-HTML round trip,
artifact hashes, privacy boundary, and intentionally deferred Phase 6B accuracy
failure to the code base accepted at the start of Phase 6E.

`phase6e_release.json` records the final publication rerun on 2026-08-04. It
adds a single independently recorded melanoma live run and its exact offline
replay, the release-artifact hashes, provenance completeness checks, and probe
cleanup. The live sample remains a failed cross-domain accuracy observation;
publishing it does not supersede or conceal the Phase 6B deferred archive.
