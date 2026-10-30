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

The local paper TeX and PDF were updated on 2026-10-27. The ACL and GENIA gazetteer rows now use the current full-precision tables, while the CL rows retain their verified train-only values. The [article update receipt](article_update_receipt.json) and [verification](article_update_verification.json) record the source identities, row checks and PDF build. These local `paper/` files remain outside the tracked numerical release because of the local Git exclusion.

An [all-file recheck](RECHECK.md) records the earlier pre-article state, followed by a dated article-update entry, and distinguishes the active release from dated, ignored audit receipts and Git backups. Seven redundant pre-migration workspaces were removed after verification; see the [cleanup receipt](cleanup_receipt.json).
