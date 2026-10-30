# Software, models, and settings

This note covers the selected outputs used in the paper, including LLM repeats, and the final checked analysis. Dataset conversion and string normalization used no third-party NLP package; model tokenization used the libraries listed below.

## Supervised models

| Library | Version used for selected runs |
| --- | --- |
| PyTorch | `2.10.0+cu128`, recorded in 115 of 155 run reports. |
| Transformers | `5.0.0`, recorded in the same 115 reports. |
| tokenizers | `0.22.2`, in the environment linked to the 55 CL runs. |
| NumPy | `2.2.6`, in the same CL environment. |

The remaining 40 GENIA reports do not record library versions. Their PyTorch and Transformers versions are inferred from the same-period environment, so they are not reported here as verified run versions.

| Encoder | Checkpoint ID | Paper reference |
| --- | --- | --- |
| mBERT | `bert-base-multilingual-cased` | `devlin-etal-2019-bert` |
| XLM-R | `xlm-roberta-base` | `conneau2020xlmr` |
| SciBERT | `allenai/scibert_scivocab_uncased` | `beltagy2019scibert` |
| cs_roberta | `allenai/cs_roberta_base` | `gururangan2020dont` |
| PubMedBERT | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | `gu2021pubmedbert` |
| RuBERT | `DeepPavlov/rubert-base-cased` | `kuratov2019rubert` |
| ruRoBERTa | `ai-forever/ruRoberta-large` | `aiforever2022ruroberta` |

Training used fast tokenizers (`use_fast=True`), 256-token windows with a 64-token stride, batch size 8, learning rate `2e-5`, AdamW (β = 0.9/0.999, ε = `1e-8`, weight decay 0.01), no warmup, and no mixed precision. Runs used seeds 42–46; ACL and CL trained for 15 epochs, GENIA for 5. Span models considered candidates up to 10 subword tokens and sampled up to five negatives per positive. The training recipe searched decision thresholds from 0.1–0.9 on development data; the chosen threshold is not directly preserved for 15 GENIA span outputs.

CL encoder revisions: mBERT `3f076fdb1ab68d5b2880cb87a0886f315b8146f8`, XLM-R `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`, RuBERT `4036cab694767a299f2b9e6492909664d9414229`, and ruRoBERTa `5192d064ca6ac67c14c40e017ce41612e010f05f`. Exact pretrained revisions were not retained for ACL or GENIA. For the selected SciBERT outputs, `do_lower_case=False` is a conditional reconstruction, not a saved tokenizer setting.

## Hosted LLMs

The selected requests used `openai`, `google-genai`, `together`, and `requests` clients. Their installed versions were not recorded with the final requests, so no SDK version is assigned to those outputs.

| Model | Served ID | Paper reference |
| --- | --- | --- |
| Gemini 3.1 Flash Lite | `gemini-3.1-flash-lite` | `google-2026-gemini31flashlite` |
| DeepSeek V4 Pro | `deepseek-v4-pro` | `deepseek-2026-v4pro` |
| GLM 5.2 | `zai-org/GLM-5.2` | `zai-2026-glm52` |
| Qwen3 235B A22B | `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`; non-FP8 `Qwen/Qwen3-235B-A22B-Instruct-2507` in repeats only | `qwen-team-2025-qwen3instruct2507` |
| GPT-OSS 120B | `openai/gpt-oss-120b` | `openai-2025-gptoss` |
| Llama 4 Maverick | `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` | `meta-2025-llama4maverick` |
| Mistral Large 3 | `mistral-large-2512` | `mistral-2025-large3` |
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | `anthropic-2025-claudehaiku45` |

Requested settings were temperature 0 and a 32,768-token output limit. Top-p was 1 and seed was 42 except for Anthropic, where both were omitted. Reasoning was disabled where supported; Gemini used `minimal` and GPT-OSS used `low`. The requested batch size was 10. Original raw records are missing for the ACL and CL base Gemini outputs. The selected GENIA output has raw records from a repeat, but an independent record of its original run is unavailable.

## Evaluation

The final checked analysis used NumPy `2.4.6` and SciPy `1.17.1`. SciPy's `linear_sum_assignment` used its default `maximize=False`. No third-party F1 or ROUGE package was used.

## References and library changes

The paper cites the original model works listed above. It does not yet cite the software papers below. Software papers: [PyTorch](https://papers.neurips.cc/paper_files/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html), [Transformers](https://aclanthology.org/2020.emnlp-demos.6/), [NumPy](https://doi.org/10.1038/s41586-020-2649-2), and [SciPy](https://doi.org/10.1038/s41592-019-0686-2). Package sites or repositories: [tokenizers](https://github.com/huggingface/tokenizers), [OpenAI SDK](https://github.com/openai/openai-python), [Google SDK](https://github.com/googleapis/python-genai), [Together SDK](https://github.com/togethercomputer/together-py), and [Requests](https://requests.readthedocs.io/en/latest/).

No changes to the source of these third-party libraries were found in the project.
