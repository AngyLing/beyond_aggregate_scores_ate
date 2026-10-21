#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ate_paper.experiments.llm_run_paths import configure_run_paths


PROMPT_VERSION_DEFAULT = "v1.1"  # shared_system_base (literal-class rule + schema-deferred empty); task prompts unchanged
PROMPT_DIR = Path(__file__).resolve().parents[3] / "prompts"
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_OUTPUT_TOKENS = 32768  # uniform across tasks; high enough that no task truncates (verify max output < cap)
DEFAULT_SEED = 42  # best-effort determinism seed; sent to every provider except Anthropic (no usable seed there)
DEFAULT_QWEN_MODEL = "qwen3.6-plus"
DEFAULT_QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_GENAPI_MODEL = "qwen-3-6-plus"
DEFAULT_GENAPI_BASE_URL = "https://proxy.gen-api.ru/v1"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"  # 2026-09-04: was gemini-2.5-flash (retired, never in the 8-panel)
GEMINI_THINKING_BUDGET = 0  # 0 disables thinking on Gemini 2.5 Flash
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_THINKING = {"type": "disabled"}  # deepseek-v4-pro without thinking
DEFAULT_TOGETHER_BASE_URL = "https://api.together.ai/v1"
DEFAULT_DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
DEFAULT_MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"
# Together hybrid-reasoning models (e.g. GLM-5.x) think by default; turn it off.
TOGETHER_NO_THINKING = {"reasoning": {"enabled": False}}
DEFAULT_OPENAI_COMPATIBLE_TIMEOUT = 180.0






ACL_CLASSES = [
    "Technology and Method",
    "Tool and Library",
    "Language Resource",
    "Language Resource Product",
    "Model",
    "Measures and Measurements",
    "Other",
]
CL_RUTERM3_CLASSES = ["Specific", "Common", "Nomen"]
from ate_paper.experiments.genia_label_maps import (  # single source of truth
    GENIA_CLASSES,
    genia_coarse6,
)
SUPPORTED_TASKS = {"acl_span_class", "genia_span_class", "cl_ruterm3_span_class"}
SENTENCE_GROUPED_SPAN_TASKS = {"acl_span_class", "genia_span_class"}


def default_max_output_tokens(task):
    # Uniform cap across all tasks. It is an inert ceiling (temp 0): it only
    # matters if a call would exceed it, so one high value avoids per-task
    # truncation surprises (ACTER corpus batches reach ~23k). Verify per run
    # that the observed max output stays below it.
    return DEFAULT_MAX_OUTPUT_TOKENS


def load_prompt(name):
    path = PROMPT_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def system_base_prompt(task):
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}")
    return load_prompt("shared_system_base")


def load_dotenv(path=".env"):
    env_path = Path(path)
    if not env_path.exists():
        return []

    loaded = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def normalize_ws(text):
    return re.sub(r"\s+", " ", str(text).strip())


def normalize_lower(text):
    return normalize_ws(text).lower()


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_input_rows(path):
    path = Path(path)
    if path.is_file():
        return read_jsonl(path)
    if path.is_dir():
        rows = []
        for file in sorted(path.rglob("*.jsonl")):
            if file.name == "gold_all.jsonl":
                continue
            rows.extend(read_jsonl(file))
        return rows
    raise FileNotFoundError(f"Input path not found: {path}")


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(row, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def metadata_arm_suffix(meta):
    """Base runs use an empty suffix; each repeat gets its own run_dir."""
    return ""



def resolve_metadata_dir(args, report):
    """Metadata stays in this run, independent of report directory names."""
    if not getattr(args, "run_dir", None):
        raise ValueError("An explicit run_dir is required")
    path = Path(getattr(args, "metadata_dir", None) or Path(args.run_dir) / "metadata").resolve()
    try:
        path.relative_to(Path(args.run_dir).resolve())
    except ValueError:
        raise ValueError("metadata_dir must stay inside run_dir")
    return path



# Fields that differ between two runs of the SAME experiment, or that only one of the
# run/parse writers knows, and so must not count as evidence of DIFFERENT work:
# started_at/completed_at/report vary per invocation; prediction_output/prediction_sha256
# are run OUTPUTS that only parse populates (merged in by merge_prior_metadata), not part
# of the experiment's identity.
_VOLATILE_METADATA_FIELDS = frozenset({
    "started_at", "completed_at", "report", "prediction_output", "prediction_sha256",
})


def merge_prior_metadata(record, out):
    """Fill fields the current writer left null/absent from an existing same-run record.

    A run write and a parse write of one experiment target the same *_metadata.json: the
    run writer knows the decoding params (temperature/top_p/seed/api_model/base_url/
    max_output_tokens) but not the prediction output; parse is the reverse. Without this
    merge the writer that runs last -- parse -- overwrote the run's frozen-decoding
    provenance with nulls (58 committed records ended up decoding-null; see
    reports/history/llm_pipeline_review_2026_08.md). Only null/absent fields are filled, never an
    existing non-null value, so a genuinely different run (e.g. a differing batch size)
    still differs on its own populated fields and is caught by preserve_superseded_metadata.
    """
    out = Path(out)
    if not out.exists():
        return record
    try:
        prior = json.loads(out.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return record
    merged = dict(record)
    for key, value in prior.items():
        if value is not None and merged.get(key) is None:
            merged[key] = value
    return merged


def preserve_superseded_metadata(out, record):
    """Keep any existing run record that describes DIFFERENT work from `record`.

    Re-running one experiment should overwrite its record. Replacing it with a record
    of *different* work is provenance loss. Keep an existing record when settings
    differ, for example a changed batch size.

    Deliberately does NOT raise. This runs after the API calls are already paid for,
    so aborting would discard the record for completed work -- turning a loud
    overwrite into silent loss, the very failure it exists to prevent. Instead the
    old record is moved aside and its new path returned for the caller to warn about.
    """
    out = Path(out)
    if not out.exists():
        return None
    try:
        old = json.loads(out.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None                       # unreadable: let the write replace it
    comparable = lambda d: {k: v for k, v in d.items()
                            if k not in _VOLATILE_METADATA_FIELDS}
    if comparable(old) == comparable(record):
        return None                       # same experiment re-run: overwrite is right
    stamp = re.sub(r"[^0-9]", "", str(old.get("started_at") or ""))[:14] or "unknown"
    kept = out.with_name(f"{out.stem}__superseded_{stamp}.json")
    out.rename(kept)
    return kept


def parse_json_response(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        start = min(
            [index for index in (text.find("{"), text.find("[")) if index != -1],
            default=-1,
        )
        if start == -1:
            raise ValueError(f"Invalid JSON response: {exc}") from exc
        try:
            value, _end = json.JSONDecoder().raw_decode(text[start:])
            return value
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON response: {exc}") from exc


def stable_cache_key(provider, model, task, prompt_version, payload, temperature=None, top_p=None, max_output_tokens=None, prompt_sha=None, seed=None):
    # prompt_sha hashes the ACTUAL system prompt sent, not just its version label:
    # a prompt file edited under the same version would otherwise resume from stale
    # cached responses. seed is part of the request for providers that honour it.
    # Adding these fields changed every key once (2026-08-24, review L-G3): a resume
    # against a pre-existing raw cache misses and re-calls the API for those batches.
    body = json.dumps(
        {
            "provider": provider,
            "model": model,
            "task": task,
            "prompt_version": prompt_version,
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_output_tokens,
            "prompt_sha": prompt_sha,
            "seed": seed,
            "payload": payload,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def load_cache(raw_output):
    cache = {}
    path = Path(raw_output)
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("cache_key") and row.get("parse_status") == "ok":
                cache[row["cache_key"]] = row
    return cache


def _manual_entries_from_json(value):
    if isinstance(value, dict) and isinstance(value.get("responses"), list):
        return value["responses"]
    if isinstance(value, list):
        return value
    return [value]


def _manual_parsed_response(entry):
    if not isinstance(entry, dict):
        raise ValueError("Manual LLM response entries must be JSON objects.")

    if "parsed_response" in entry:
        parsed = entry["parsed_response"]
        text = entry.get("raw_response") or json.dumps(parsed, ensure_ascii=False)
    elif "parsed" in entry:
        parsed = entry["parsed"]
        text = entry.get("raw_response") or entry.get("text") or json.dumps(parsed, ensure_ascii=False)
    elif "raw_response" in entry:
        text = entry["raw_response"]
        parsed = parse_json_response(text)
    elif "text" in entry and len(entry) <= 3:
        text = entry["text"]
        parsed = parse_json_response(text)
    else:
        parsed = entry
        text = json.dumps(entry, ensure_ascii=False)

    return {
        "text": text,
        "parsed": parsed,
        "retry_count": 0,
        "usage": {},
        "batch_id": entry.get("batch_id"),
    }


def load_manual_responses(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manual LLM response file not found: {path}")

    entries = []
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.extend(_manual_entries_from_json(json.loads(line)))
    else:
        with path.open(encoding="utf-8") as f:
            entries.extend(_manual_entries_from_json(json.load(f)))

    ordered = []
    by_batch = {}
    for entry in entries:
        response = _manual_parsed_response(entry)
        batch_id = response.get("batch_id")
        if batch_id is None:
            ordered.append(response)
        else:
            batch_id = int(batch_id)
            if batch_id in by_batch:
                raise ValueError(f"Duplicate manual LLM response for batch_id={batch_id}.")
            by_batch[batch_id] = response
    return ordered, by_batch


def canonical_model_name(provider, model):
    normalized = str(model or "").lower().replace("_", "-").replace(".", "-")
    if provider == "gemini" and "gemini-2-5-flash" in normalized:
        return "Gemini 2.5 Flash"
    if "gemini-3-1-flash-lite" in normalized:
        return "Gemini 3.1 Flash Lite"
    if "qwen-3-6-plus" in normalized or "qwen3-6-plus" in normalized:
        return "Qwen3.6 Plus"
    if "deepseek-v4-pro" in normalized:
        return "DeepSeek V4 Pro"
    if "deepseek-v4-flash" in normalized:
        return "DeepSeek V4 Flash"
    if "glm-5-2" in normalized:
        return "GLM 5.2"
    if "qwen3-235b-a22b-instruct-2507" in normalized:
        return "Qwen3 235B A22B Instruct 2507"
    return model or "unknown"


def result_model_name(provider, model):
    name = canonical_model_name(provider, model)
    if name == "Gemini 2.5 Flash":
        return "gemini_2_5_flash"
    if name == "Gemini 3.1 Flash Lite":
        return "gemini_3_1_flash_lite"
    if name == "Qwen3.6 Plus":
        return "qwen3_6_plus"
    if name == "DeepSeek V4 Pro":
        return "deepseek_v4_pro"
    if name == "DeepSeek V4 Flash":
        return "deepseek_v4_flash"
    if name == "GLM 5.2":
        return "glm_5_2"
    if name == "Qwen3 235B A22B Instruct 2507":
        return "qwen3_235b_a22b_instruct_2507"
    return model_slug(model)


def dataset_name(rows):
    values = sorted({str(row.get("dataset")) for row in rows if row.get("dataset")})
    return ";".join(values) if values else "unknown"


def protocol_record(args, rows, run_datetime):
    return {
        "model_name": canonical_model_name(args.provider, args.model),
        "result_model_name": result_model_name(args.provider, args.model),
        "api_provider": args.provider,
        "date": run_datetime.date().isoformat(),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_output_tokens": args.max_output_tokens,
        "prompt_version": args.prompt_version,
        "prompt_hash": prompt_content_hash(args.task, args.prompt_version, "full"),
        "batch_size": args.batch_size,
        "task": args.task,
        "dataset": dataset_name(rows),
    }


def openai_compatible_chat(api_key, base_url, model, system_prompt, user_payload, temperature, top_p, max_output_tokens, provider_name, extra_body=None, seed=None, json_object=True, omit_top_p=False):
    """OpenAI-compatible chat call.

    json_object=False omits response_format (Anthropic rejects json_object → wants
    json_schema; we rely on the prompt + fence-stripping parser instead).
    omit_top_p=True drops top_p (Anthropic 400s if temperature AND top_p are both sent;
    top_p=1.0 is a no-op anyway).
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(f"The openai package is required for provider={provider_name} OpenAI-compatible API calls.") from exc

    timeout = float(os.environ.get("LLM_API_TIMEOUT", DEFAULT_OPENAI_COMPATIBLE_TIMEOUT))
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    prompt = json.dumps(user_payload, ensure_ascii=False)
    request = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }
    if not omit_top_p:
        request["top_p"] = top_p
    if json_object:
        request["response_format"] = {"type": "json_object"}
    if max_output_tokens is not None:
        request["max_tokens"] = max_output_tokens
    if seed is not None:
        request["seed"] = seed
    if extra_body:
        request["extra_body"] = extra_body
    response = client.chat.completions.create(**request)
    text = response.choices[0].message.content
    usage = response.usage.model_dump() if getattr(response, "usage", None) else {}
    # provenance: endpoint + the backend build that served the call + the seed sent
    usage["base_url"] = base_url
    usage["system_fingerprint"] = getattr(response, "system_fingerprint", None)
    usage["seed"] = seed
    return text, usage


def gemini_usage(usage_metadata):
    """Map a Gemini usage_metadata object/dict to the provider usage shape."""
    if usage_metadata is None:
        return {}

    def field(key):
        if isinstance(usage_metadata, dict):
            return usage_metadata.get(key)
        return getattr(usage_metadata, key, None)

    usage = {}
    for src, dst in (
        ("prompt_token_count", "prompt_tokens"),
        ("candidates_token_count", "completion_tokens"),
        ("total_token_count", "total_tokens"),
    ):
        value = field(src)
        if value is not None:
            usage[dst] = value
    return usage


def together_extra_body(model):
    """Per-model Together reasoning/thinking control.

    - GLM-5.x (hybrid-reasoning): disable thinking.
    - gpt-oss (reasoning-native, cannot fully disable): lowest reasoning_effort.
    - Qwen3 *Instruct* / Llama-4 (non-thinking by design): no override.
    """
    m = str(model or "").lower()
    if "glm" in m:
        return TOGETHER_NO_THINKING
    if "gpt-oss" in m:
        return {"reasoning_effort": "low"}
    return None


def provider_api_call(provider, model, system_prompt, user_payload, schema, temperature, base_url=None, top_p=DEFAULT_TOP_P, max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS, seed=DEFAULT_SEED):
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for provider=openai.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai package is required for provider=openai.") from exc

        client = OpenAI()
        prompt = json.dumps(user_payload, ensure_ascii=False)
        response_format = {"type": "json_object"}
        request = {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "response_format": response_format,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if max_output_tokens is not None:
            request["max_tokens"] = max_output_tokens
        if seed is not None:
            request["seed"] = seed
        response = client.chat.completions.create(**request)
        text = response.choices[0].message.content
        usage = response.usage.model_dump() if getattr(response, "usage", None) else {}
        usage["system_fingerprint"] = getattr(response, "system_fingerprint", None)
        usage["seed"] = seed
        return text, usage

    if provider == "qwen":
        api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("QWEN_API_KEY or DASHSCOPE_API_KEY is required for provider=qwen.")
        qwen_base_url = base_url or os.environ.get("QWEN_BASE_URL") or os.environ.get("DASHSCOPE_BASE_URL") or DEFAULT_QWEN_BASE_URL
        return openai_compatible_chat(api_key, qwen_base_url, model, system_prompt, user_payload, temperature, top_p, max_output_tokens, provider, seed=seed)

    if provider == "genapi":
        api_key = os.environ.get("GEN_API_KEY") or os.environ.get("GENAPI_API_KEY") or os.environ.get("GEN_API_RU_KEY")
        if not api_key:
            raise RuntimeError("GEN_API_KEY, GENAPI_API_KEY, or GEN_API_RU_KEY is required for provider=genapi.")
        genapi_base_url = base_url or os.environ.get("GEN_API_BASE_URL") or os.environ.get("GENAPI_BASE_URL") or DEFAULT_GENAPI_BASE_URL
        return openai_compatible_chat(api_key, genapi_base_url, model, system_prompt, user_payload, temperature, top_p, max_output_tokens, provider, seed=seed)

    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for provider=deepseek.")
        deepseek_base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL") or DEFAULT_DEEPSEEK_BASE_URL
        return openai_compatible_chat(
            api_key, deepseek_base_url, model, system_prompt, user_payload,
            temperature, top_p, max_output_tokens, provider,
            extra_body={"thinking": DEEPSEEK_THINKING}, seed=seed,
        )

    if provider == "together":
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError("TOGETHER_API_KEY is required for provider=together.")
        together_base_url = base_url or os.environ.get("TOGETHER_BASE_URL") or DEFAULT_TOGETHER_BASE_URL
        return openai_compatible_chat(
            api_key, together_base_url, model, system_prompt, user_payload,
            temperature, top_p, max_output_tokens, provider,
            extra_body=together_extra_body(model), seed=seed,
        )

    if provider == "deepinfra":
        api_key = os.environ.get("DEEPINFRA_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPINFRA_API_KEY is required for provider=deepinfra.")
        deepinfra_base_url = base_url or os.environ.get("DEEPINFRA_BASE_URL") or DEFAULT_DEEPINFRA_BASE_URL
        return openai_compatible_chat(
            api_key, deepinfra_base_url, model, system_prompt, user_payload,
            temperature, top_p, max_output_tokens, provider, seed=seed,
        )

    if provider == "mistral":
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY is required for provider=mistral.")
        mistral_base_url = base_url or os.environ.get("MISTRAL_BASE_URL") or DEFAULT_MISTRAL_BASE_URL
        # Mistral rejects the OpenAI `seed` param (422 extra_forbidden); its determinism
        # knob is `random_seed`, sent via the request body (extra_body), not top-level seed.
        return openai_compatible_chat(
            api_key, mistral_base_url, model, system_prompt, user_payload,
            temperature, top_p, max_output_tokens, provider,
            extra_body={"random_seed": seed} if seed is not None else None, seed=None,
        )

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for provider=anthropic.")
        # NOTE: do NOT honour $ANTHROPIC_BASE_URL here — that env var (set by some runtimes,
        # e.g. "https://api.anthropic.com" without /v1/) is for the NATIVE Anthropic SDK, which
        # appends its own path. The OpenAI-compat endpoint needs the explicit /v1/ base, else the
        # OpenAI SDK posts to /chat/completions and 404s. Only an explicit --base-url overrides.
        anthropic_base_url = base_url or DEFAULT_ANTHROPIC_BASE_URL
        # Claude via Anthropic's OpenAI-compat endpoint: no `response_format:json_object`
        # (wants json_schema; we rely on the prompt + fence-stripping parser), cannot send
        # temperature AND top_p together (drop top_p, =1.0 no-op), extended thinking stays
        # off (opt-in), no usable seed. max_tokens required.
        return openai_compatible_chat(
            api_key, anthropic_base_url, model, system_prompt, user_payload,
            temperature, top_p, max_output_tokens, provider,
            json_object=False, omit_top_p=True, seed=None,
        )

    if provider == "gemini":
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY is required for provider=gemini.")
        contents = json.dumps(user_payload, ensure_ascii=False)
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            # Gemini 3.x cannot disable thinking (minimal is the floor) and controls it via
            # thinking_level, not thinking_budget; 2.x disables via thinking_budget=0.
            if "gemini-3" in model:
                thinking_cfg = types.ThinkingConfig(thinking_level="minimal")
            else:
                thinking_cfg = types.ThinkingConfig(thinking_budget=GEMINI_THINKING_BUDGET)
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                max_output_tokens=max_output_tokens,
                response_mime_type="application/json",
                thinking_config=thinking_cfg,
            )
            response = client.models.generate_content(model=model, contents=contents, config=config)
            usage = gemini_usage(getattr(response, "usage_metadata", None))
            usage["model_version"] = getattr(response, "model_version", None)  # resolved snapshot
            usage["seed"] = seed
            return response.text, usage
        except ImportError as exc:
            # No legacy google.generativeai fallback: it sent NO thinking config
            # (silently breaking the frozen decoding protocol for gemini-3 models)
            # and left no SDK marker in usage. Every documented runtime installs
            # google-genai; require it (review L-transport-5).
            raise RuntimeError(
                "The google-genai SDK is required for provider=gemini "
                "(pip install google-genai); the legacy google.generativeai "
                "fallback was removed because it cannot pin the thinking config."
            ) from exc

    raise ValueError(f"Unsupported provider: {provider}")


class LLMCallExhausted(ValueError):
    """Every retry produced unparseable output; carries the last raw response.

    Without this the failed-path raw record stored only str(exc) -- the parser's
    error message -- and the model's actual final response text was lost
    (code review L-parsing-5).
    """

    def __init__(self, message, last_text="", usage=None):
        super().__init__(message)
        self.last_text = last_text
        self.usage = usage or {}


def call_llm(provider, model, system_prompt, user_payload, schema, temperature, max_retries, base_url=None, top_p=DEFAULT_TOP_P, max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS, seed=DEFAULT_SEED):
    last_text = ""
    usage = {}
    for retry in range(max_retries + 1):
        text, usage = provider_api_call(
            provider,
            model,
            system_prompt,
            user_payload,
            schema,
            temperature,
            base_url,
            top_p,
            max_output_tokens,
            seed,
        )
        last_text = text
        try:
            parsed = parse_json_response(text)
            return {
                "text": text,
                "parsed": parsed,
                "provider": provider,
                "model": model,
                "usage": usage,
                "retry_count": retry,
            }
        except ValueError as exc:
            if retry >= max_retries:
                raise LLMCallExhausted(str(exc), last_text=last_text, usage=usage) from exc
    raise LLMCallExhausted("no attempts were made (max_retries < 0)", last_text=last_text, usage=usage)


def task_prompt(task, prompt_version=None, run_mode=None):
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}")
    # cl_ruterm3 uses one base prompt (context-window output) for every run mode;
    # the output format is fixed, not tied to pilot/full.
    return load_prompt(f"{task}_v1")


def output_schema(task):
    if task in SENTENCE_GROUPED_SPAN_TASKS:
        return {"type": "object", "required": ["sentences"]}
    if task == "cl_ruterm3_span_class":
        return {"type": "object", "required": ["terms"]}
    raise ValueError(f"Unsupported task: {task}")



# Only the base recipes used by this paper are included.
NESTED_POLICIES = ("include",)
CLASS_DETAILS = ("labels",)
GENIA_CLASS_SCHEMES = ("full",)
GENIA_COARSE6_CLASSES = ("protein", "DNA", "RNA", "cell_line", "cell_type", "other")


def build_system_prompt(task, prompt_version=None, run_mode=None, ablation=None, shots=0):
    base = {"class_detail": "labels", "nested_policy": "include", "class_scheme": "full"}
    if shots or any(base.get(key) != value for key, value in (ablation or {}).items()):
        raise ValueError("This repository contains only the paper's base prompts.")
    return system_base_prompt(task) + "\n\n" + task_prompt(task, prompt_version, run_mode)


def prompt_content_hash(task, prompt_version=None, run_mode=None, ablation=None):
    """sha256 of the exact, fully-built system prompt (base + task text) sent to the
    model. `prompt_version` alone is a bare label -- it doesn't change when someone edits
    a task prompt file's *content* under the same name, so it can't distinguish two
    different edits (task spec Gate 3: "record prompt version/hash"). This hash can.
    """
    text = build_system_prompt(task, prompt_version, run_mode, ablation)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_annotation_like(row):
    return (
        row.get("unit_type") in {"span_mention", "sentence_term_string", "term_string"}
        and (
            row.get("start") is not None
            or row.get("end") is not None
            or row.get("surface") not in {None, ""}
            or row.get("norm") not in {None, ""}
            or row.get("class_label") not in {None, ""}
        )
    )


def reconstruct_text_units(rows):
    units = {}
    warning = any(is_annotation_like(row) for row in rows)
    for row in rows:
        if not row.get("text"):
            continue
        key = (
            row.get("dataset"),
            row.get("version"),
            row.get("split"),
            row.get("language"),
            row.get("domain"),
            row.get("doc_id"),
            row.get("sent_id"),
            row.get("text"),
        )
        units.setdefault(key, {
            "dataset": row.get("dataset"),
            "version": row.get("version"),
            "split": row.get("split"),
            "language": row.get("language"),
            "domain": row.get("domain"),
            "doc_id": row.get("doc_id"),
            "sent_id": row.get("sent_id"),
            "text": row.get("text"),
        })
    warnings = []
    if warning and units:
        warnings.append(
            "WARNING: text units were reconstructed from gold annotation rows; "
            "text units without gold terms may be missing, so false positives in "
            "empty-gold units cannot be produced and precision may be inflated. "
            "Use complete test text-unit files for fair evaluation."
        )
    return list(units.values()), warnings


ACTER_DOMAIN_MAP = {
    "corp": "corruption",
    "equi": "dressage/equitation",
    "htfl": "heart failure",
    "wind": "wind energy",
}

ACTER_LANGUAGE_MAP = {
    "en": "English",
    "fr": "French",
    "nl": "Dutch",
}














def input_id(row):
    if row.get("chunk_id"):
        return f"{row.get('doc_id')}:{row.get('chunk_id')}"
    return row.get("member_id") or row.get("sent_id") or row.get("doc_id") or ""


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def build_batches(task, rows, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if task in SENTENCE_GROUPED_SPAN_TASKS:
        for batch_id, batch in enumerate(chunks(rows, batch_size), start=1):
            yield batch_id, batch, {"sentences": [
                {"doc_id": row.get("doc_id"), "sent_id": row.get("sent_id"), "text": row.get("text", "")}
                for row in batch
            ]}
    elif task == "cl_ruterm3_span_class":
        for batch_id, row in enumerate(rows, start=1):
            yield batch_id, [row], {"doc_id": row.get("doc_id"), "sent_id": row.get("sent_id"), "text": row.get("text", "")}
    else:
        raise ValueError(f"Unsupported task: {task}")



def exact_surface_offsets(text, surface):
    offsets = []
    if not text or not surface:
        return offsets
    start = 0
    while True:
        index = text.find(surface, start)
        if index == -1:
            break
        offsets.append((index, index + len(surface)))
        start = index + 1
    return offsets


def context_surface_offsets(text, surface, left_context, right_context):
    offsets = exact_surface_offsets(text, surface)
    if left_context is None and right_context is None:
        return offsets
    left_context = "" if left_context is None else str(left_context)[-30:]
    right_context = "" if right_context is None else str(right_context)[:30]
    filtered = []
    for start, end in offsets:
        left_ok = text[:start].endswith(left_context)
        right_ok = text[end:].startswith(right_context)
        if left_ok and right_ok:
            filtered.append((start, end))
    return filtered


def longest_match_terms(terms):
    normalized = sorted({normalize_lower(term) for term in terms if normalize_lower(term)}, key=lambda term: (-len(term), term))
    kept = []
    for term in normalized:
        if any(term != other and term in other for other in kept):
            continue
        kept.append(term)
    return sorted(kept)


def deduplicate(rows, key_func):
    seen = set()
    result = []
    duplicates = 0
    for row in rows:
        key = key_func(row)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        result.append(row)
    return result, duplicates






def iter_sentence_terms(task, parsed):
    """Yield (doc_id, sent_id, term) for span tasks.

    Sentence-batched tasks group terms under each sentence header
    ({"sentences":[{doc_id,sent_id,terms:[...]}]}); single-sentence tasks return
    a flat {"terms":[...]} list, where doc_id/sent_id come from the term itself.
    """
    if task in SENTENCE_GROUPED_SPAN_TASKS:
        for sentence in parsed.get("sentences", []) or []:
            doc_id = sentence.get("doc_id")
            sent_id = sentence.get("sent_id")
            for term in sentence.get("terms", []) or []:
                yield doc_id, sent_id, term
    else:
        for term in parsed.get("terms", []) or []:
            # A bare-string term (surface with no wrapper object) is the same
            # tolerated model quirk postprocess_spans coerces for grouped tasks;
            # .get() on a str would otherwise abort the whole parse run with an
            # uncaught AttributeError (review L-parsing-3).
            if isinstance(term, str):
                term = {"surface": term}
            if not isinstance(term, dict):
                continue
            yield term.get("doc_id"), term.get("sent_id"), term


def tolerant_source_lookup(source_by_id, doc_id, sent_id, stats):
    """Recover the source row when the model returned a mangled sentence header.

    Applied only after the exact (doc_id, sent_id) lookup fails (review
    L-parsing-2; e.g. GLM on GENIA echoed document MEDLINE:92142646 with a
    bare-numeric doc_id, silently emptying the whole document). Two conservative
    rules, each requiring a UNIQUE match within the batch:
    1. same sent_id and a doc_id that is a substring of the real one (or vice
       versa) -- covers dropped/added prefixes like "MEDLINE:";
    2. a sent_id carried by exactly one row of the batch.
    """
    if sent_id is None:
        return None
    same_sent = [row for (_d, s), row in source_by_id.items() if s == sent_id]
    doc = str(doc_id) if doc_id is not None else ""
    if doc:
        affix = []
        for row in same_sent:
            row_doc = str(row.get("doc_id") or "")
            if row_doc and (doc in row_doc or row_doc in doc):
                affix.append(row)
        if len(affix) == 1:
            stats["recovered_source_sentence"] = stats.get("recovered_source_sentence", 0) + 1
            return affix[0]
    if len(same_sent) == 1:
        stats["recovered_source_sentence"] = stats.get("recovered_source_sentence", 0) + 1
        return same_sent[0]
    return None


def postprocess_spans(task, batch_rows, parsed, stats, batch_id=None):
    if task == "acl_span_class":
        allowed = ACL_CLASSES
    elif task == "genia_span_class":
        allowed = GENIA_COARSE6_CLASSES if stats.get("_genia_class_scheme") == "coarse6" else GENIA_CLASSES
    elif task == "cl_ruterm3_span_class":
        allowed = CL_RUTERM3_CLASSES
    else:
        raise ValueError(f"Unsupported task: {task}")
    require_context = bool(stats.get("_require_context"))
    sentence_batched = task in SENTENCE_GROUPED_SPAN_TASKS
    if sentence_batched:
        source_by_id = {(row.get("doc_id"), row.get("sent_id")): row for row in batch_rows}
        fallback_source = batch_rows[0] if len(batch_rows) == 1 else None
    else:
        source_by_id = {}
        fallback_source = batch_rows[0]
    rows = []
    for doc_id, sent_id, item in iter_sentence_terms(task, parsed):
        # Some models emit a term as a bare string instead of {"surface": ...};
        # treat the string as the surface (labeled tasks then drop it for lacking
        # a class label).
        if isinstance(item, str):
            item = {"surface": item}
        if not isinstance(item, dict):
            continue
        surface = item.get("surface")
        class_label = item.get("class_label")
        if class_label not in allowed:
            continue
        if sentence_batched:
            source = source_by_id.get((doc_id, sent_id))
            if source is None and not stats.get("_legacy_exact_headers"):
                source = tolerant_source_lookup(source_by_id, doc_id, sent_id, stats)
            if source is None and fallback_source is not None:
                source = fallback_source
            if source is None:
                stats["unknown_source_sentence"] = stats.get("unknown_source_sentence", 0) + 1
                continue
        else:
            source = fallback_source
        text = source.get("text", "")
        all_offsets = exact_surface_offsets(text, surface)
        if not all_offsets:
            stats["unmatched_surface"] = stats.get("unmatched_surface", 0) + 1
            continue
        if require_context and task == "cl_ruterm3_span_class":
            if "left_context" not in item or "right_context" not in item:
                raise ValueError(f"{task} full-run terms require left_context and right_context.")
            offsets = context_surface_offsets(text, surface, item.get("left_context"), item.get("right_context"))
            if not offsets:
                stats["context_unmatched_surface"] = stats.get("context_unmatched_surface", 0) + 1
                continue
            if len(all_offsets) > 1:
                # "Disambiguated" only when the context actually narrowed the
                # candidates to ONE offset; a context that still matches several
                # is counted separately (review L-grounding-3).
                if len(offsets) == 1:
                    stats["context_disambiguated_surfaces"] = stats.get("context_disambiguated_surfaces", 0) + 1
                else:
                    stats["context_still_ambiguous"] = stats.get("context_still_ambiguous", 0) + 1
        else:
            offsets = all_offsets
        if len(all_offsets) > 1:
            stats["ambiguous_surface_match"] = stats.get("ambiguous_surface_match", 0) + 1
            stats["multiple_match_surfaces"] = stats.get("multiple_match_surfaces", 0) + 1
        for start, end in offsets:
            matched_surface = source.get("text", "")[start:end]
            # Every span task lowercases norm: all span golds carry lowercased
            # norms, and a case-preserving norm can never match one (review L-G1:
            # acl_span_class alone fell to normalize_ws here, depressing every
            # committed ACL type_f1 by +0.17..+0.21 and flipping MentionTypeGap).
            norm = normalize_lower(matched_surface)
            row = {
                "dataset": source.get("dataset"),
                "version": source.get("version"),
                "split": source.get("split"),
                "language": source.get("language"),
                "domain": source.get("domain"),
                "doc_id": source.get("doc_id"),
                "sent_id": source.get("sent_id"),
                "text": source.get("text"),
                "unit_type": "span_mention",
                "surface": matched_surface,
                "norm": norm,
                "start": start,
                "end": end,
                "class_label": class_label,
            }
            if task == "cl_ruterm3_span_class":
                row["is_nomen"] = class_label == "Nomen"
            rows.append(row)
    return deduplicate(rows, lambda row: (row["dataset"], row["doc_id"], row["sent_id"], row["start"], row["end"], row["class_label"]))














def postprocess(task, batch_rows, parsed, stats, batch_id=None):
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}")
    return postprocess_spans(task, batch_rows, parsed, stats, batch_id=batch_id)



def validate_parsed(task, parsed, require_context=False):
    if not isinstance(parsed, dict):
        raise ValueError("Parsed response must be a JSON object.")
    if task in SENTENCE_GROUPED_SPAN_TASKS:
        if not isinstance(parsed.get("sentences"), list):
            raise ValueError(f"{task} response must contain sentences list.")
        for index, sentence in enumerate(parsed.get("sentences", []), start=1):
            if not isinstance(sentence, dict):
                raise ValueError(f"{task} sentences must be objects.")
            if "doc_id" not in sentence or "sent_id" not in sentence:
                raise ValueError(f"{task} sentence {index} must contain doc_id and sent_id.")
            terms = sentence.get("terms")
            if not isinstance(terms, list):
                raise ValueError(f"{task} sentence {index} must contain a terms list.")
            for t_index, item in enumerate(terms, start=1):
                # A bare string item (surface with no wrapper object) is a known,
                # deliberately-tolerated model quirk -- postprocess_spans() coerces it to
                # {"surface": item}. Only a dict whose surface VALUE is a non-string (e.g. a
                # bare number for a measurement) is invalid: reject it here as a clean
                # ValueError, not let it reach exact_surface_offsets()/str.find() downstream
                # and crash with an uncaught TypeError that aborts the whole parse run.
                surface = item.get("surface") if isinstance(item, dict) else item
                if surface is not None and not isinstance(surface, str):
                    raise ValueError(
                        f"{task} sentence {index} term {t_index} has a non-string surface "
                        f"({type(surface).__name__})."
                    )
    list_or_flat_terms_tasks = {"cl_ruterm3_span_class"}
    if task in list_or_flat_terms_tasks and not isinstance(parsed.get("terms"), list):
        raise ValueError(f"{task} response must contain terms list.")
    if task in list_or_flat_terms_tasks:
        for t_index, item in enumerate(parsed.get("terms", []), start=1):
            surface = item.get("surface") if isinstance(item, dict) else item
            if surface is not None and not isinstance(surface, str):
                raise ValueError(
                    f"{task} term {t_index} has a non-string surface ({type(surface).__name__})."
                )
    if task == "cl_ruterm3_span_class" and require_context:
        for index, item in enumerate(parsed.get("terms", []), start=1):
            if not isinstance(item, dict):
                raise ValueError("CL-RuTerm3 terms must be objects.")
            if not isinstance(item.get("left_context"), str) or not isinstance(item.get("right_context"), str):
                raise ValueError(
                    f"CL-RuTerm3 full-run term {index} must contain left_context and "
                    "right_context as strings (empty string is fine; null/missing is not)."
                )



def model_slug(model):
    return re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_").lower()


def raw_record(protocol, model, task, temperature, batch_id, input_ids, prompt_version, cache_key, response, parse_status):
    return {
        "provider": protocol["api_provider"],
        "model": model,
        "model_name": protocol["model_name"],
        "result_model_name": protocol["result_model_name"],
        "api_provider": protocol["api_provider"],
        "date": protocol["date"],
        "task": task,
        "temperature": temperature,
        "top_p": protocol["top_p"],
        "max_output_tokens": protocol["max_output_tokens"],
        "batch_size": protocol["batch_size"],
        "dataset": protocol["dataset"],
        "batch_id": batch_id,
        "input_ids": input_ids,
        "prompt_version": prompt_version,
        "prompt_hash": protocol.get("prompt_hash"),
        "protocol": protocol,
        "cache_key": cache_key,
        "raw_response": response.get("text"),
        "error": response.get("error"),
        "parsed_response": response.get("parsed"),
        "parse_status": parse_status,
        "retry_count": response.get("retry_count", 0),
        "usage": response.get("usage", {}),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_pipeline(args):
    configure_run_paths(args, ("output", "raw_output", "report"))
    if args.task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {args.task}")
    if not hasattr(args, "top_p") or args.top_p is None:
        args.top_p = DEFAULT_TOP_P
    if not hasattr(args, "max_output_tokens") or args.max_output_tokens is None:
        args.max_output_tokens = default_max_output_tokens(args.task)

    rows = read_input_rows(args.input)
    rows, warnings = reconstruct_text_units(rows)

    manual_response_path = getattr(args, "manual_response", None)
    manual_ordered = []
    manual_by_batch = {}
    manual_order_index = 0
    used_manual_batches = set()
    if manual_response_path:
        manual_ordered, manual_by_batch = load_manual_responses(manual_response_path)
        warnings.append("Manual LLM responses were loaded from file; no provider API calls were made.")

    cache = load_cache(args.raw_output)
    output_rows = []
    stats = Counter()
    run_datetime = datetime.now(timezone.utc)
    protocol = protocol_record(args, rows, run_datetime)
    system_prompt = build_system_prompt(args.task, args.prompt_version, "full")
    # Hash the prompt actually sent, so an edited prompt file (same version label)
    # can never resume from stale cached responses (review L-G3).
    system_prompt_sha = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    schema = output_schema(args.task)
    # run_pipeline always builds the "full" prompt (see build_system_prompt call above), so
    # require_context only depends on task -- mirrors run_llm_api.requires_context() /
    # parse_llm_outputs.py's require_context, which additionally gate on run_mode=="full".
    require_context = args.task == "cl_ruterm3_span_class"
    if require_context:
        stats["_require_context"] = 1

    for batch_id, batch_rows, payload in build_batches(args.task, rows, args.batch_size):
        cache_key = stable_cache_key(
            args.provider,
            args.model,
            args.task,
            args.prompt_version,
            payload,
            args.temperature,
            args.top_p,
            args.max_output_tokens,
            prompt_sha=system_prompt_sha,
            seed=getattr(args, "seed", DEFAULT_SEED),
        )
        cached = cache.get(cache_key)

        if args.dry_run:
            response = {"text": json.dumps({"dry_run": True}), "parsed": {}, "retry_count": 0, "usage": {}}
            stats["successful_responses"] += 1
            continue
        elif manual_response_path:
            if batch_id in manual_by_batch:
                response = manual_by_batch[batch_id]
                used_manual_batches.add(batch_id)
            else:
                if manual_order_index >= len(manual_ordered):
                    raise ValueError(f"Manual LLM response file has no response for batch {batch_id}.")
                response = manual_ordered[manual_order_index]
                manual_order_index += 1
            validate_parsed(args.task, response["parsed"], require_context=require_context)
            append_jsonl(
                raw_record(
                    protocol,
                    args.model,
                    args.task,
                    args.temperature,
                    batch_id,
                    [input_id(row) for row in batch_rows],
                    args.prompt_version,
                    cache_key,
                    response,
                    "manual_ok",
                ),
                args.raw_output,
            )
            stats["successful_responses"] += 1
            stats["manual_responses"] += 1
        elif cached and not args.force:
            parsed = cached.get("parsed_response")
            response = {"text": cached.get("raw_response"), "parsed": parsed, "retry_count": cached.get("retry_count", 0), "usage": cached.get("usage", {})}
            stats["successful_responses"] += 1
        else:
            try:
                response = call_llm(
                    args.provider,
                    args.model,
                    system_prompt,
                    payload,
                    schema,
                    args.temperature,
                    args.max_retries,
                    getattr(args, "base_url", None),
                    args.top_p,
                    args.max_output_tokens,
                    getattr(args, "seed", DEFAULT_SEED),
                )
                validate_parsed(args.task, response["parsed"], require_context=require_context)
                append_jsonl(
                    raw_record(
                        protocol,
                        args.model,
                        args.task,
                        args.temperature,
                        batch_id,
                        [input_id(row) for row in batch_rows],
                        args.prompt_version,
                        cache_key,
                        response,
                        "ok",
                    ),
                    args.raw_output,
                )
                stats["successful_responses"] += 1
                stats["retries"] += response.get("retry_count", 0)
            except Exception as exc:
                # Store the model's real final response when one exists (parse
                # exhaustion carries it via LLMCallExhausted); str(exc) alone would
                # replace the response with the parser's error message.
                failed_text = getattr(exc, "last_text", "") or str(exc)
                append_jsonl(
                    raw_record(
                        protocol,
                        args.model,
                        args.task,
                        args.temperature,
                        batch_id,
                        [input_id(row) for row in batch_rows],
                        args.prompt_version,
                        cache_key,
                        {"text": failed_text, "error": str(exc), "parsed": None,
                         "retry_count": args.max_retries,
                         "usage": getattr(exc, "usage", None) or {}},
                        "failed",
                    ),
                    args.raw_output,
                )
                stats["failed_responses"] += 1
                raise

        validate_parsed(args.task, response["parsed"], require_context=require_context)
        rows_now, duplicates = postprocess(args.task, batch_rows, response["parsed"], stats, batch_id=batch_id)
        stats["duplicates_removed"] += duplicates
        output_rows.extend(rows_now)

    if manual_response_path:
        unused_ordered = len(manual_ordered) - manual_order_index
        unused_by_batch = set(manual_by_batch) - used_manual_batches
        if unused_ordered or unused_by_batch:
            raise ValueError(
                "Manual LLM response file contains unused responses: "
                f"ordered={unused_ordered}, batch_id={sorted(unused_by_batch)}"
            )

    if not args.dry_run:
        output_rows, duplicates = deduplicate(output_rows, lambda row: json.dumps(row, sort_keys=True, ensure_ascii=False))
        stats["duplicates_removed"] += duplicates
        write_jsonl(output_rows, args.output)

    stats["parsed_prediction_rows"] = len(output_rows)
    stats["input_rows"] = len(rows)
    stats["batches"] = sum(1 for _ in build_batches(args.task, rows, args.batch_size))
    stats["run_date"] = protocol["date"]
    stats["model_name"] = protocol["model_name"]
    stats["result_model_name"] = protocol["result_model_name"]
    stats["dataset"] = protocol["dataset"]
    write_report(args, stats, warnings)
    return output_rows, stats, warnings


def write_report(args, stats, warnings):
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LLM Baseline Report",
        "",
        "## Frozen Protocol",
        "",
        f"- model_name: {stats.get('model_name', canonical_model_name(args.provider, args.model))}",
        f"- result_model_name: {stats.get('result_model_name', result_model_name(args.provider, args.model))}",
        f"- api_provider: {args.provider}",
        f"- date: {stats.get('run_date', 'unknown')}",
        f"- temperature: {args.temperature}",
        f"- top_p: {args.top_p}",
        f"- max_output_tokens: {args.max_output_tokens}",
        f"- prompt_version: {args.prompt_version}",
        f"- batch_size: {args.batch_size}",
        f"- task: {args.task}",
        f"- dataset: {stats.get('dataset', 'unknown')}",
        "",
        "## Run Summary",
        "",
        f"- provider: {args.provider}",
        f"- model: {args.model}",
        f"- base URL: {getattr(args, 'base_url', None) or 'provider default'}",
        f"- task: {args.task}",
        f"- temperature: {args.temperature}",
        f"- prompt version: {args.prompt_version}",
        f"- input rows: {stats.get('input_rows', 0)}",
        f"- batches: {stats.get('batches', 0)}",
        f"- successful responses: {stats.get('successful_responses', 0)}",
        f"- failed responses: {stats.get('failed_responses', 0)}",
        f"- retries: {stats.get('retries', 0)}",
        f"- manual responses: {stats.get('manual_responses', 0)}",
        f"- parsed prediction rows: {stats.get('parsed_prediction_rows', 0)}",
        f"- unmatched surfaces: {stats.get('unmatched_surface', 0)}",
        f"- unknown source sentences (terms dropped): {stats.get('unknown_source_sentence', 0)}",
        f"- recovered source sentences (tolerant header match): {stats.get('recovered_source_sentence', 0)}",
        f"- ambiguous surface matches: {stats.get('ambiguous_surface_match', 0)}",
        f"- context disambiguated surfaces: {stats.get('context_disambiguated_surfaces', 0)}",
        f"- context still ambiguous (several offsets kept): {stats.get('context_still_ambiguous', 0)}",
        f"- duplicates removed: {stats.get('duplicates_removed', 0)}",
        f"- manual response path: {getattr(args, 'manual_response', None) or 'none'}",
        f"- raw output path: {args.raw_output}",
        f"- prediction output path: {args.output}",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- none")
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--provider", choices=["qwen", "genapi", "gemini", "deepseek", "together", "openai", "manual"])
    parser.add_argument("--model")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL for provider=qwen or provider=genapi.")
    parser.add_argument("--task", choices=sorted(SUPPORTED_TASKS), required=True)
    parser.add_argument("--input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--manual-response", help="JSON or JSONL file with manually saved LLM responses.")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Best-effort determinism seed sent to providers that honour it.")
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="Output token cap; default 32768 (uniform across tasks).")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--prompt-version", default=PROMPT_VERSION_DEFAULT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.input:
        raise SystemExit("--input is required.")
    if args.manual_response:
        args.provider = args.provider or "manual"
        if args.provider == "qwen":
            args.model = args.model or DEFAULT_QWEN_MODEL
        elif args.provider == "genapi":
            args.model = args.model or DEFAULT_GENAPI_MODEL
        else:
            args.model = args.model or "manual-upload"
    elif args.provider == "qwen":
        args.model = args.model or DEFAULT_QWEN_MODEL
    elif args.provider == "genapi":
        args.model = args.model or DEFAULT_GENAPI_MODEL
    elif args.provider == "gemini":
        args.model = args.model or DEFAULT_GEMINI_MODEL
    elif args.provider == "deepseek":
        args.model = args.model or DEFAULT_DEEPSEEK_MODEL
    elif not args.provider or not args.model:
        raise SystemExit("--provider and --model are required unless --manual-response is provided.")

    configure_run_paths(args, ("output", "raw_output", "report"))
    load_dotenv()
    try:
        output_rows, stats, _ = run_pipeline(args)
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    print(f"Prediction rows: {len(output_rows)}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
