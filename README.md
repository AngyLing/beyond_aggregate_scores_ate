# Beyond Aggregate Scores in Automatic Term Extraction

Code, **214 saved prediction files** and numerical references for **Beyond Aggregate Scores: Comparing System Outputs in Automatic Term Extraction**. The main workflow recomputes the article's results from these fixed predictions.

On 2026-10-20, all **17 calculation blocks** were recomputed on ACL RD-TEC 2.0, GENIA Term Corpus v3.02 and CL-RuTerm3 1.1 / test reference v4. All comparisons passed at absolute tolerance `1e-10`; the maximum difference was `2.22e-16` (floating-point rounding).

A later code reorganization was checked with synthetic fixtures; full-corpus calculations were not repeated.

## Recompute results from the saved predictions

Run the following commands from the repository root in a POSIX shell (for example, bash or zsh) with `mktemp` available. Python **3.11.5**, NumPy **2.4.6** and SciPy **1.17.1** were used for the checked analysis. This workflow uses the included predictions; it needs no GPU, model weights or API keys.

### 1. Install analysis dependencies

```sh
python3.11 -m venv build/venvs/analysis &&
. build/venvs/analysis/bin/activate &&
python scripts/install_dependencies.py analysis
```

The single [requirements.lock](requirements.lock) stores exact versions in separate TOML profiles; use the installer above to select one. It checks the recorded CPython version (including the patch version), installs the selected packages and runs `pip check`. Ensure `python3.11` is **CPython 3.11.5** before creating the analysis environment. Keep this environment active for preparation and all seven calculations. The scripts work directly from the checkout; installing the repository as a Python package is unnecessary.

### 2. Obtain and prepare the corpora

Obtain the exact versions listed in [data/SOURCES.md](data/SOURCES.md). It provides the original sources, required files and SHA-256 checksums. Place the source files here:

```text
data/local/sources/
├── acl-rd-tec-2.0/distribution/annoitation_files/
│   ├── annotator1/
│   └── annotator2/
├── GENIA_term_3.02/GENIAcorpus3.02.xml
└── CL-RuTerm3/
    ├── train.jsonl
    ├── dev.jsonl
    ├── dev_ref.jsonl
    ├── test.jsonl
    └── test_ref_v4.jsonl
```

```sh
python scripts/prepare_data.py --source-root data/local/sources
```

Preparation checks the original input hashes and split counts, then writes `train.jsonl`, `dev.jsonl`, `test.jsonl` and `units.jsonl` under `data/local/processed/{acl,genia,cl}/`. Source text and gold remain in ignored `data/local/`; the preparation command performs no downloads.

### 3. Run all seven calculations

Copy this block as written. It uses the included predictions, creates a fresh output directory on every run and stops on the first failed command. Nesting and sensitivity consume the preceding gold-agreement results.

```sh
(
set -eu
mkdir -p build/analysis
OUT=$(mktemp -d build/analysis/saved-XXXXXXXX)

python scripts/reproduce_panels.py --out "$OUT/panels"
python scripts/reproduce_profiles.py --out "$OUT/profiles"
python scripts/reproduce_gold_agreement.py --output "$OUT/gold"
python scripts/reproduce_no_witness.py --output "$OUT/no_witness"
python scripts/reproduce_nesting_contrasts.py --gold-output "$OUT/gold" --output "$OUT/nesting"
python scripts/reproduce_sensitivity.py --gold-output "$OUT/gold" --output "$OUT/sensitivity" --check-reference
python scripts/reproduce_llm_repeats.py --output "$OUT/llm_repeats"

printf 'Results: %s\n' "$OUT"
)
```

The full bootstrap uses the retained shared draws: **5,000 for ACL**, **5,000 for GENIA**, and **10,000 for CL**. The commands write only to the new output directory; the saved predictions and numerical references are preserved.

| Output directory | Calculations |
| --- | --- |
| `panels/` | Corpus statistics, exact/typed scores, diagnostic profiles, all panel comparisons and four bootstrap targets (A01–A04, A07–A08) |
| `profiles/` | Supervised profile comparisons, closeness screen, seed controls and threshold calibration (A04–A06) |
| `gold/` | Exact/relaxed covered-gold agreement and intervals (A09–A10) |
| `no_witness/` | FP no-witness, within-reference/excess and common-six summaries (A11) |
| `nesting/` | Eight flat/nonflat contrasts and 98.75% intervals (A14) |
| `sensitivity/` | IoU .75, Jaccard, CL equal-optimum assignments and GENIA duplicate removal (A12–A13, A15–A16) |
| `llm_repeats/` | All 25 repeat pairs: 22 comparable pairs and three contrasts (A17) |

The output JSON reports compare calculated values with the saved references. In `panels/comparison_with_saved.json`, `difference_count` and `references_missing` should both be zero; its generic `DIFFERENCES_REPORTED` status does not itself indicate a mismatch. Other commands record their checks in `verification.json`, `manifest.json` or their summary JSON. `COMPUTED` alone means calculation completed, not that all reference comparisons matched. The sensitivity command above explicitly enables the strict reference check.

## Contents

- [predictions/](predictions/): 214 text-free gzip JSONL files, the run/hash manifest and the 25 LLM repeat-pair definitions; about 29 MB, without Git LFS.
- [results/](results/): full-precision numerical references and the three shared document-draw arrays.
- [data/](data/): source links, input identities and split definitions; corpora are obtained separately.
- [scripts/](scripts/): command-line entry points and artifact-maintenance tools. [src/ate_paper/](src/ate_paper/): shared preparation, scoring and calculation modules; experiment-generation programs are also retained.
- [prompts/](prompts/): the prompts used by the saved LLM runs.
- [run_settings.json](run_settings.json): the settings registry for all 214 selected artifacts.
- [tests/](tests/): code checks; [requirements.lock](requirements.lock) contains `analysis`, `supervised-cu128`, `llm-july` and `llm-august` dependency profiles. Install each required profile in its own virtual environment; their NumPy and SDK versions differ.
- [LICENSES/](LICENSES/): retained dataset notices and their scope.

### Dependencies for generating new predictions

Use a separate virtual environment for each required profile. The LLM profiles use **CPython 3.11.5** on Linux or macOS:

```sh
python3.11 -m venv build/venvs/llm-july &&
build/venvs/llm-july/bin/python scripts/install_dependencies.py llm-july

python3.11 -m venv build/venvs/llm-august &&
build/venvs/llm-august/bin/python scripts/install_dependencies.py llm-august
```

The training profile uses **CPython 3.13.15 with GIL**, Linux x86_64, glibc ≥2.28 and the CUDA 12.8 build of PyTorch. Ensure `python3.13` has that version:

```sh
python3.13 -m venv build/venvs/supervised &&
build/venvs/supervised/bin/python scripts/install_dependencies.py supervised-cu128
```

These commands install dependencies only. Training additionally requires a compatible NVIDIA driver/GPU. To inspect any profile without installing it, use `python scripts/install_dependencies.py analysis --show`, replacing `analysis` with the required profile name.

## Optional checks

```sh
python scripts/verify_predictions.py
python scripts/verify_saved_results.py
PYTHONPATH=src python -m unittest discover -s tests -v
```

The first two commands check prediction hashes/schema and saved numerical arithmetic against the retained numeric references. They only read the package and print their reports. The last command runs synthetic tests with the analysis dependencies installed above. These checks do not train models or call a hosted service.

## Interpretation and licenses

The calculations are conditional on fixed saved outputs. Seed SD is descriptive; the document-bootstrap intervals are nominal and do not establish linguistic validity or confirmatory significance. Historical provenance limitations include SciBERT tokenizer/runtime binding, three unavailable original Gemini raw responses and the independence of repeat pair P07. These do not prevent rescoring the included predictions.

Source links and citations are recorded in [data/SOURCES.md](data/SOURCES.md); notices are under [LICENSES/](LICENSES/). These notices apply to their respective materials.
