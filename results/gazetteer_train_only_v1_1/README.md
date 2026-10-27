# Gazetteer train-only v1.1 results

This directory is the current dictionary result source. `primary_counts.csv`
contains exactly 14 selected train-only rows: two variants for ACL RD-TEC 2.0,
GENIA, CL-RuTerm3, ACTER v1.2, ACTER v1.5, ATE-IT A, and ATE-IT B. It contains
no previous-arm rows. ATE-IT B has no fitted vocabulary.

The release includes six gzip prediction files for ACL, GENIA and CL. Their
paths, row counts and SHA-256 hashes are bound in `selection.json` and
`predictions/manifest.csv`. The other four tasks have numerical summaries in
`primary_counts.csv`; their corpora and predictions are outside this
three-corpus release.

For the six published files, the vocabulary is built only from annotated
`train` rows. Development rows are excluded. Inventory strings use Unicode
`lower()` and whitespace collapse; null `norm` values become empty and are
not inserted. Matching is case-insensitive on the original test text, and
released character offsets preserve that text's positions and case.

`verification.json` records a fresh generation from the prepared train
splits. All six generated outputs matched the released safe-field records in
their original order: **76,380 of 76,380 rows**. Their vocabulary sizes and
full-test TP/FP/FN/F1 matched `primary_counts.csv`. The two CL outputs are
unchanged in decompressed content from the already train-only CL release.
The whole release still has **214** prediction files, now **1,922,280** rows
and **28,685,468** compressed bytes.

All seven analysis commands in the root README were run with the current
predictions and prepared local gold. The full panel comparison with the
updated saved references reported **0 differences** and **0 missing
references**. Profiles, gold agreement, no-witness, nesting, sensitivity and
LLM repeats passed. `scripts/verify_predictions.py`,
`scripts/verify_saved_results.py` and **113 unit tests** passed. The numerical
source hashes are in `results/release_result_sources.json` (49 files).

The local paper TeX and PDF are outside the tracked numerical release and
remain unchanged for a separate author-approved edit. For dictionary values,
use `primary_counts.csv` and the current ACL/GENIA/CL tables under `results/`.

An [all-file recheck](RECHECK.md) distinguishes this active release from the unchanged local paper and dated, ignored audit receipts and Git backups. Seven redundant pre-migration workspaces were removed after verification; see the [cleanup receipt](cleanup_receipt.json).
