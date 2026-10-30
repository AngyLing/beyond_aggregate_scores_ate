# Model Parameters and Computational Resources

## Model parameters

Total parameters include the classifier head. Encoder parameters are shown separately. Span encoders include a pooler; BIO encoders do not. For ACL and GENIA, we treat the latest Hugging Face encoder weights as used, although the exact revisions downloaded on Kaggle were not saved.

### Supervised models: ACL RD-TEC 2.0

| Model and configuration | Encoder parameters | Total parameters |
| --- | ---: | ---: |
| `bert-base-multilingual-cased`, typed BIO (15 labels) | 177,262,848 | 177,274,383 |
| `bert-base-multilingual-cased`, untyped BIO (3 labels) | 177,262,848 | 177,265,155 |
| `xlm-roberta-base`, typed BIO (15 labels) | 277,453,056 | 277,464,591 |
| `xlm-roberta-base`, typed span (8 labels) | 278,043,648 | 279,844,968 |
| `allenai/scibert_scivocab_uncased`, typed BIO (15 labels) | 109,327,872 | 109,339,407 |
| `allenai/scibert_scivocab_uncased`, typed span (8 labels) | 109,918,464 | 111,719,784 |
| `xlm-roberta-base`, untyped span (2 labels) | 278,043,648 | 279,840,354 |
| `bert-base-multilingual-cased`, typed span (8 labels) | 177,853,440 | 179,654,760 |
| `allenai/cs_roberta_base`, typed BIO (15 labels) | 124,055,040 | 124,066,575 |
| `allenai/cs_roberta_base`, typed span (8 labels) | 124,645,632 | 126,446,952 |

### Supervised models: GENIA

| Model and configuration | Encoder parameters | Total parameters |
| --- | ---: | ---: |
| `bert-base-multilingual-cased`, typed BIO (73 labels) | 177,262,848 | 177,318,985 |
| `bert-base-multilingual-cased`, typed span (37 labels) | 177,853,440 | 179,677,061 |
| `bert-base-multilingual-cased`, untyped BIO (3 labels) | 177,262,848 | 177,265,155 |
| `xlm-roberta-base`, typed BIO (73 labels) | 277,453,056 | 277,509,193 |
| `xlm-roberta-base`, typed span (37 labels) | 278,043,648 | 279,867,269 |
| `xlm-roberta-base`, untyped span (2 labels) | 278,043,648 | 279,840,354 |
| `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`, typed BIO (73 labels) | 108,891,648 | 108,947,785 |
| `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`, typed span (37 labels) | 109,482,240 | 111,305,861 |
| `allenai/scibert_scivocab_uncased`, typed BIO (73 labels) | 109,327,872 | 109,384,009 |
| `allenai/scibert_scivocab_uncased`, typed span (37 labels) | 109,918,464 | 111,742,085 |

### Supervised models: CL-RuTerm3

| Model and configuration | Encoder parameters | Total parameters |
| --- | ---: | ---: |
| `bert-base-multilingual-cased`, typed BIO (7 labels) | 177,262,848 | 177,268,231 |
| `bert-base-multilingual-cased`, typed span (4 labels) | 177,853,440 | 179,651,684 |
| `bert-base-multilingual-cased`, untyped BIO (3 labels) | 177,262,848 | 177,265,155 |
| `DeepPavlov/rubert-base-cased`, typed BIO (7 labels) | 177,262,848 | 177,268,231 |
| `DeepPavlov/rubert-base-cased`, typed span (4 labels) | 177,853,440 | 179,651,684 |
| `DeepPavlov/rubert-base-cased`, untyped span (2 labels) | 177,853,440 | 179,650,146 |
| `ai-forever/ruRoberta-large`, typed BIO (7 labels) | 354,310,144 | 354,317,319 |
| `ai-forever/ruRoberta-large`, typed span (4 labels) | 355,359,744 | 358,543,716 |
| `xlm-roberta-base`, typed BIO (7 labels) | 277,453,056 | 277,458,439 |
| `xlm-roberta-base`, typed span (4 labels) | 278,043,648 | 279,841,892 |
| `xlm-roberta-base`, untyped span (2 labels) | 278,043,648 | 279,840,354 |

### Hosted models

The published counts describe model architectures. The exact weights used by providers for past API requests are unknown. The non-FP8 Qwen model was used only in repeat runs.

| Model and API identifier | Total parameters | Active parameters per token |
| --- | ---: | ---: |
| Gemini 3.1 Flash-Lite, `gemini-3.1-flash-lite` | Not disclosed | Not disclosed |
| DeepSeek V4 Pro, `deepseek-v4-pro` | 1.6 trillion | 49 billion |
| GLM 5.2, `zai-org/GLM-5.2` | 744 billion | 40 billion |
| Qwen3 235B A22B, `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` | 235 billion | 22 billion |
| Qwen3 235B A22B, `Qwen/Qwen3-235B-A22B-Instruct-2507` | 235 billion | 22 billion |
| GPT-OSS 120B, `openai/gpt-oss-120b` | 116.83 billion | 5.13 billion |
| Llama 4 Maverick, `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` | 400 billion | 17 billion |
| Mistral Large 3, `mistral-large-2512` | 675 billion | 41 billion |
| Claude Haiku 4.5, `claude-haiku-4-5-20251001` | Not disclosed | Not disclosed |

The two deterministic gazetteer methods have no trainable parameters.

## Total computational budget

The paper uses 155 selected supervised outputs: 50 ACL, 50 GENIA, and 55 CL-RuTerm3. The table also includes earlier baseline runs and reruns, but excludes the CL pilot. GPU-hours sum training-job time across GPUs; two jobs running at once count as two GPU-hours per hour.

| Baseline-panel training | Fits | GPU-hours |
| --- | ---: | ---: |
| ACL, selected 15-epoch panel | 50 | 4.70, logged |
| GENIA, two configurations of the selected panel | 10 | 5.20, logged |
| GENIA, remaining eight configurations of a reported full September rerun | 40, conditional | approximately 16, estimated |
| CL-RuTerm3, selected 15-epoch panel | 55 | 18.33, logged |
| ACL, earlier five-epoch panel | 35 | approximately 1–2, estimated |
| GENIA, earlier nine-configuration panel | 45 | approximately 19, estimated |
| CL-RuTerm3, earlier five-epoch panel | 50 | approximately 5–6, estimated |
| CL-RuTerm3, two extra full repeats of `ruroberta_span`, seed 45 | 2 | approximately 0.5, estimated |
| CL-RuTerm3, six configurations rerun after a batch-size change | 30 | approximately 4, estimated |
| Reruns after an epoch-selection fix: ACL 5, GENIA 5, CL 10 | 20 | 4.03, logged |
| **Supervised training total (10 + 40 GENIA scenario)** | **337, conditional** | **approximately 78.7, estimated** |

Of the 337 conditional fits, 135 have logged times totaling 32.27 GPU-hours. The remaining approximately 46.4 GPU-hours are estimated. There is no complete log for the reported September GENIA rerun, and we cannot link it to the selected outputs. If its two logged configurations were trained again later, the total would be **347 fits and approximately 83.9 GPU-hours**.

Across five logged sessions, two-GPU time between the first and last log entries was approximately 30.89 GPU-hours. Training jobs totaled 28.23 GPU-hours, leaving approximately 2.65 GPU-hours within those intervals. Time outside the logs is unknown.

The paper also uses **53 hosted LLM outputs** (24 main and 29 repeats); provider GPU time is unknown. Time for earlier CPU training, unlogged or interrupted runs, and six CPU gazetteer outputs was not recorded. Therefore, the GPU-hour estimates are not a complete compute total.

## Computing infrastructure

| Experiment group | Compute environment | Memory | Parallelism |
| --- | --- | --- | --- |
| Selected ACL and CL-RuTerm3 training, plus two logged GENIA configurations | Kaggle Notebooks set to two NVIDIA T4 GPUs | 16 GB GDDR6 per T4 (nominal) | Up to two jobs at once, one per GPU; seeds ran one after another |
| Earlier GPU baseline runs and reruns | Kaggle Notebooks set to two NVIDIA T4 GPUs | 16 GB GDDR6 per T4 (nominal) | Up to two jobs at once |
| Earlier CPU training and gazetteers | Author's MacBook Pro (`MacBookPro17,1`), Apple M1, eight CPU cores (four performance, four efficiency); no GPU used | 16 GB RAM | Not recorded |
| Hosted LLM outputs | Providers' infrastructure | Not available | Not available |

Logs show two GPU workers, but not the GPU model or available memory. T4 comes from notebook settings, and 16 GB is its nominal capacity. No execution log was found for the conditional 40 GENIA fits.
