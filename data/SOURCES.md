# Corpus sources

This repository does not distribute source corpora, converted corpus text, or gold annotations. Obtain the three corpora from the sources below and keep the files under the ignored `data/local/` directory. The content checksums in [sources.json](sources.json) identify the inputs used for the paper independently of download availability.

After placing the files as described below, run:

```sh
python scripts/prepare_data.py --source-root data/local/sources
```

This checks source identities and writes `train.jsonl`, `dev.jsonl`, `test.jsonl`, and `units.jsonl` in `data/local/processed/{acl,genia,cl}/`. The units file contains one text record for every unit across the three splits. Split files contain annotation rows, including negative placeholders for term-free units; duplicates are preserved. These generated files contain corpus text and gold and must remain local. No downloads occur in the preparation command.

## ACL RD-TEC 2.0

- Source used: the [creators' repository](https://github.com/languagerecipes/acl-rd-tec-2.0), version **2.0**.
- Input: `distribution/annoitation_files/annotator1/` and `annotator2/`; the spelling `annoitation_files` is from upstream.
- Local destination: `data/local/sources/acl-rd-tec-2.0/`, preserving the upstream directory layout.
- Identity: **471 XML files** (189 in annotator1 and 282 in annotator2), 641,498 bytes in total. The deterministic tree SHA-256 is `bea0af16cf1d810c5df72b1f5bd719d926c15ab30e335d5e40c7ab9e329297bc`.
- On 2026-10-15, a fresh download from creators' revision [`39142690c8a062704b134e4013c0d5bc1c96e7f9`](https://github.com/languagerecipes/acl-rd-tec-2.0/tree/39142690c8a062704b134e4013c0d5bc1c96e7f9) reproduced the exact 471-file input tree. This is a newly verified acquisition revision.
- A source Git commit was not recorded in the research inputs. The tree digest identifies the local XML input bytes; it is not a claim that an arbitrary current checkout has identical contents. The duplicate views under `double_annotated_files/` are outside this digest and are not converter inputs.
- Citation: Behrang QasemiZadeh and Anne-Kathrin Schumann (2016), [The ACL RD-TEC 2.0: A Language Resource for Evaluating Term Extraction and Entity Recognition Methods](https://aclanthology.org/L16-1294/).

The converter reads both annotator layers. Canonical evaluation prefers `ann1` when available and otherwise chooses the layer with the most annotation rows; it does not merge layers. The source has 300 document identifiers, of which 299 produce sentence units. The paper's fixed document partitions contain 214 train, 25 dev, and 60 test documents.

To check the XML tree, run the following from the repository root after obtaining the corpus. This hashes original file bytes without normalizing line endings. It sorts POSIX paths relative to `annoitation_files`, forms one UTF-8 line per file as `SHA256`, two spaces, relative path, and LF, then hashes the concatenated lines.

```sh
python - data/local/sources/acl-rd-tec-2.0/distribution/annoitation_files <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = sorted(
    (p for layer in ("annotator1", "annotator2")
     for p in (root / layer).rglob("*.xml") if p.is_file()),
    key=lambda p: p.relative_to(root).as_posix(),
)
lines = "".join(
    f"{hashlib.sha256(p.read_bytes()).hexdigest()}  "
    f"{p.relative_to(root).as_posix()}\n"
    for p in files
)
print(len(files), sum(p.stat().st_size for p in files))
digest = hashlib.sha256(lines.encode("utf-8")).hexdigest()
print(digest)
assert len(files) == 471
assert digest == "bea0af16cf1d810c5df72b1f5bd719d926c15ab30e335d5e40c7ab9e329297bc"
PY
```

## GENIA term corpus v3.02

- Source recorded in the paper: the [archived NaCTeM term-corpus page dated 2013-12-15](https://web.archive.org/web/20131215201031/http://www.nactem.ac.uk/genia/genia-corpus/term-corpus).
- Original page URL: [NaCTeM GENIA term corpus](http://www.nactem.ac.uk/genia/genia-corpus/term-corpus). Historical URLs may redirect or be unavailable; the archived page is retained as the provenance reference.
- Version: **3.02**, raw XML `GENIAcorpus3.02.xml`; this is the term corpus with its full label inventory, rather than a five-class JNLPBA conversion.
- Local destination: `data/local/sources/GENIA_term_3.02/GENIAcorpus3.02.xml`.
- Input size: **9,259,109 bytes**.
- Input SHA-256: `c6d6966ea48522e48aa4b268323934897d9336fab3942fb48134ac174d989c5e`.
- The research records preserve the dataset page and extracted XML identity; they do not preserve a verified download-archive checksum or the exact archive retrieval URL.
- Verified reacquisition on 2026-10-15: the [official GENIAcorpus3.02.tgz archive](https://www.nactem.ac.uk/GENIA/current/GENIA-corpus/Term/GENIAcorpus3.02.tgz), 1,651,614 bytes, archive SHA-256 `85b2aede313308c33783ba508d054d0af0ff9c7e48490f0db5168a9f88d74d1c`. Its extracted XML matches the input hash above. This records a new successful download; it does not reconstruct the original retrieval history.
- Citation: Jin-Dong Kim, Tomoko Ohta, Yuka Tateisi, and Jun'ichi Tsujii (2003), [GENIA corpus—a semantically annotated corpus for bio-textmining](https://doi.org/10.1093/bioinformatics/btg1023).

The paper retains all 36 observed labels, classified nested constituents, and contiguous coordination spans. Duplicate source identifiers are disambiguated before the fixed document split. The partitions contain 1,442 train, 158 dev, and 400 test documents. One exact full-document duplicate occurs across train and test; the paper documents its sensitivity analysis. Five additional short boilerplate sentence-text types recur across train/test (35 train and 11 test occurrences), all with empty gold; they are separate from the duplicate abstract. Re-fetching individual PubMed records is not a substitute for obtaining this exact XML, because text and annotation revisions can change offsets.

## CL-RuTerm3 1.1

- Source used: the [official repository](https://github.com/mlsa-iai-msu-lab/cl-ruterm3), pinned to commit [`4fd9befb1e1f7ec11fa8de57aeec7970aacc97ec`](https://github.com/mlsa-iai-msu-lab/cl-ruterm3/tree/4fd9befb1e1f7ec11fa8de57aeec7970aacc97ec).
- Release information: [README at the pinned revision](https://github.com/mlsa-iai-msu-lab/cl-ruterm3/blob/4fd9befb1e1f7ec11fa8de57aeec7970aacc97ec/README.md).
- Version: corpus release **1.1**; test reference revision **v4**. These identify different aspects of the input.
- Local destination: place the five files from upstream `data/` directly in `data/local/sources/CL-RuTerm3/`.
- Citation: Angelina Mamontova, Roman Ischenko, and Konstantin Vorontsov (2025), [RuTermEval-2024: Cross-domain Automatic Term Extraction and Classification in Russian Scientific Texts](https://doi.org/10.28995/2075-7182-2025-23-245-256).

| File in upstream `data/` | SHA-256 |
| --- | --- |
| `train.jsonl` | `a72c86113e6bf731e32fb9d0577d23b2b2a9967308a8d7e35e5b763dd3f30e98` |
| `dev.jsonl` | `c43074bfc96bdf726f05ca6ab271a65d7e634ff560b9fb9f9de3af9e257aadc9` |
| `dev_ref.jsonl` | `d5e1576d8adb259aa79e01098fcbe7efa5e66ed62add5f5b9a113b07e6b175bd` |
| `test.jsonl` | `7ef82a7c54f28653051a3e910256219971927ecb036a784797a171623b1ff870` |
| `test_ref_v4.jsonl` | `6698e4d97d29054c5519761929f23c3768e93f11e42a51a82d7e213f49ba5530` |

All five research input files match these hashes. They were recorded as byte-identical to the pinned upstream revision. The official train/dev/test partitions contain 850/103/317 abstract units; dev is not a holdout drawn from train. Join texts and references by unique `id`, preserving original Unicode text and zero-based half-open character offsets.

Review [dataset notices and access scope](../LICENSES/README.md) before obtaining or reusing the corpora. Source availability and a checksum do not grant reuse rights.
