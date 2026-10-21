# Fixed model predictions

The release contains 214 deterministic gzip JSONL files (about 29 MB), selected from the saved outputs used by the article. Corpus text, extracted strings and normalized strings are excluded. No model call is required to use these files.

- `manifest.csv`: files, runs, source/release SHA-256 and row counts.
- `llm_repeat_pairs.csv`: the 25 reported repeat comparisons.
- `export_summary.json`: release counts and field/hash conventions.

See the [ready-to-run calculation workflow](../README.md#recompute-results-from-the-saved-predictions) and [dataset notices](../LICENSES/README.md). Each gzip contains JSONL mention records with document/unit identifiers, half-open character offsets and labels, without corpus text or extracted strings.
