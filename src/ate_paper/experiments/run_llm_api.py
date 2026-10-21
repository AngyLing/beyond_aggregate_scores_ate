#!/usr/bin/env python3
import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from ate_paper.experiments.llm_run_paths import configure_run_paths, pin_run_identity

from ate_paper.experiments import make_llm_predictions as llm


MODEL_ALIASES = {
    # Floating alias — the GA gemini-2.5-flash text model exposes NO callable
    # dated snapshot (models.list reports version 001 as metadata, but
    # "gemini-2.5-flash-001" returns 404). So it cannot be pinned by id; we
    # instead record the resolved model_version per call for provenance.
    "gemini_2_5_flash": "gemini-2.5-flash",
    "gemini_3_1_flash_lite": "gemini-3.1-flash-lite",  # dot in version, per Google convention
    "qwen3_6_plus": "qwen-3-6-plus",
    "deepseek_v4_pro": "deepseek-v4-pro",
    "deepseek_v4_flash": "deepseek-v4-flash",
    "glm_5_2": "zai-org/GLM-5.2",
    "qwen3_235b_a22b_instruct_2507": "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",  # Together
    # Same model, same fp8 quantisation, different host. Together stopped serving the FP8
    # id serverlessly on 2026-07-17 (400 model_not_available; dedicated 4xH100 only, 36c/min),
    # so runs that cannot rent a GPU go to DeepInfra, which serves the identical weights at
    # fp8 but omits the suffix from the id (`quantization: fp8` in its models/list metadata).
    # Kept as a separate key: the panel's committed ACL/CL/GENIA predictions came from
    # Together, and that provenance must not be silently rewritten.
    "qwen3_235b_a22b_instruct_2507_deepinfra": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    # cross-vendor expansion:
    "gpt_oss_120b": "openai/gpt-oss-120b",                                          # Together, reasoning-native → reasoning_effort=low
    "llama_4_maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",        # DeepInfra FP8 (clean quant); non-reasoning
    "mistral_large_3": "mistral-large-2512",                                        # Mistral La Plateforme (first-party); non-reasoning (magistral is the reasoning line)
    "claude_haiku_4_5": "claude-haiku-4-5-20251001",                                # Anthropic (first-party, OpenAI-compat endpoint); extended thinking off
}


def api_model_name(model):
    return MODEL_ALIASES.get(model, model)


# Default parallelism for sync runs, tuned per provider from documented limits
# and our pilots (override with --concurrency):
#   deepseek  -> 500 concurrent connections allowed (v4-pro); 16 is safe & fast.
#   together  -> dynamic per-org limits; 16 (on a 429 the run resumes, lower it).
#   gemini    -> paid tier (2.5-flash ~1000+ RPM); 16 is well within limits.
#   qwen/genapi/openai -> limits unknown for this account; stay sequential.
# Qwen3-235B on Together is run via the Batch API, not here.
RECOMMENDED_CONCURRENCY = {
    "deepseek": 16,
    "together": 16,
    "gemini": 16,
    "deepinfra": 16,
    "mistral": 8,
    "anthropic": 8,
    "qwen": 1,
    "genapi": 1,
    "openai": 1,
}


def recommended_concurrency(provider):
    return RECOMMENDED_CONCURRENCY.get(provider, 1)


def batch_ids(batch_dir):
    return sorted(path.name.replace("_meta.json", "") for path in Path(batch_dir).glob("batch_*_meta.json"))


def select_batch_ids(ids, start_batch=None, end_batch=None):
    selected = []
    for batch_id in ids:
        if start_batch and batch_id < start_batch:
            continue
        if end_batch and batch_id > end_batch:
            continue
        selected.append(batch_id)
    return selected


def load_batch(batch_dir, batch_id):
    root = Path(batch_dir)
    prompt = (root / f"{batch_id}_prompt.txt").read_text(encoding="utf-8")
    payload = json.loads((root / f"{batch_id}_input.json").read_text(encoding="utf-8"))
    meta = json.loads((root / f"{batch_id}_meta.json").read_text(encoding="utf-8"))
    return prompt, payload, meta


def approx_tokens(text):
    return max(1, len(str(text)) // 4)


def success_exists(raw_dir, batch_id, args=None):
    raw_path = Path(raw_dir) / f"{batch_id}_raw.json"
    if not raw_path.exists():
        return False
    try:
        row = json.loads(raw_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if row.get("status") != "ok":
        return False
    if args is not None:
        expected = getattr(args, "_run_identity_sha256", None)
        if not expected or row.get("run_identity_sha256") != expected:
            raise ValueError("Cached raw is not bound to these run settings: " + batch_id)
    return True


def pilot_report_exists(model, task, prompt_version, report_dir):
    for path in sorted(Path(report_dir).glob(f"llm_pilot_{model}_{task}*.md")):
        text = path.read_text(encoding="utf-8")
        if f"prompt_version: {prompt_version}" in text and "ready_for_full_run: yes" in text:
            return True
    return False


def warn_or_require_pilot(args, first_meta):
    run_mode = first_meta.get("run_mode")
    prompt_version = first_meta.get("prompt_version", args.prompt_version)
    if run_mode != "full":
        return None
    if pilot_report_exists(args.model, args.task, prompt_version, args.pilot_report_dir):
        return None
    warning = "WARNING: no completed pilot report found for this model/task/prompt version."
    if args.require_pilot:
        raise RuntimeError(warning)
    print(warning)
    return warning


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def usage_counts(prompt, payload, text, usage):
    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
    completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
    batch_total = usage.get("total_tokens")
    usage_source = "provider"
    if not prompt_tokens or not completion_tokens:
        usage_source = "approximate"
        prompt_tokens = prompt_tokens or approx_tokens(prompt + json.dumps(payload, ensure_ascii=False))
        completion_tokens = completion_tokens or approx_tokens(text)
    batch_total = batch_total or prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, batch_total, usage_source


def build_raw_record(args, batch_id, provider_model, meta, text, usage, created_at, prompt, payload, attempt):
    prompt_tokens, completion_tokens, batch_total, usage_source = usage_counts(prompt, payload, text, usage)
    # Flag output-cap saturation at capture time, so truncated/degenerate outputs
    # are distinguishable from JSON-contract failures downstream (L-transport-1).
    provider_completion = (usage or {}).get("completion_tokens") or (usage or {}).get("output_tokens")
    cap_hit = bool(args.max_output_tokens) and bool(provider_completion) and provider_completion >= args.max_output_tokens
    return {
        "status": "ok",
        "run_identity_sha256": getattr(args, "_run_identity_sha256", None),
        "batch_id": batch_id,
        "attempt": attempt,
        "provider": args.provider,
        "model": args.model,
        "api_model": provider_model,
        "task": args.task,
        "run_mode": meta.get("run_mode"),
        "prompt_version": meta.get("prompt_version", args.prompt_version),
        "prompt_hash": meta.get("prompt_hash"),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_output_tokens": args.max_output_tokens,
        "output_cap_hit": cap_hit,
        "created_at": created_at,
        "response_text": text,
        "usage": usage or {},
        "estimated_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": batch_total,
            "source": usage_source,
        },
        "meta": meta,
    }


def requires_context(task, meta):
    return task == "cl_ruterm3_span_class" and meta.get("run_mode") == "full"


def response_validation_error(task, text, meta=None):
    try:
        parsed = llm.parse_json_response(text)
        llm.validate_parsed(task, parsed, require_context=requires_context(task, meta or {}))
    except Exception as exc:
        return exc
    return None


def is_rate_limit_error(exc):
    """True if exc is an HTTP 429 / rate-limit error from any provider path."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 429:
        return True
    if "ratelimit" in type(exc).__name__.lower():
        return True
    text = str(exc)[:200].lower()
    return "429" in text or "rate limit" in text or "rate_limited" in text


def rate_limit_delay(exc, attempt_index, base=2.0, cap=60.0):
    """Backoff seconds for a 429: honour a Retry-After header if present, else
    exponential (base * 2**attempt_index) capped, plus jitter."""
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            retry_after = response.headers.get("retry-after")
        except Exception:  # noqa: BLE001 - headers access is best-effort
            retry_after = None
        if retry_after:
            try:
                return min(cap, float(retry_after))
            except (TypeError, ValueError):
                pass
    return min(cap, base * (2 ** attempt_index)) + random.uniform(0, 1)


def provider_call_with_backoff(args, provider_model, prompt, payload):
    """Wrap the provider call with a DEDICATED 429 backoff, separate from the outer
    JSON-validation retry loop. On a rate-limit error it sleeps (Retry-After or
    exponential) and retries up to --rate-limit-retries times, so a transient 429
    never consumes the JSON-validation budget and never silently drops a batch (the
    failure mode that lost ~85% of the free-tier Mistral sync run). Non-429 errors —
    and a 429 that survives all backoff — are re-raised for the caller's normal
    handling, so providers that never rate-limit are unaffected."""
    retries = max(0, getattr(args, "rate_limit_retries", 8))
    for index in range(retries + 1):
        try:
            return llm.provider_api_call(
                args.provider,
                provider_model,
                prompt,
                payload,
                llm.output_schema(args.task),
                args.temperature,
                args.base_url,
                args.top_p,
                args.max_output_tokens,
                getattr(args, "seed", llm.DEFAULT_SEED),
            )
        except Exception as exc:  # noqa: BLE001 - classify then re-raise if not a 429
            if index < retries and is_rate_limit_error(exc):
                delay = rate_limit_delay(exc, index)
                print(f"  [429] {getattr(args, 'model', '?')} rate-limited; backoff "
                      f"{delay:.1f}s (retry {index + 1}/{retries})", file=sys.stderr)
                time.sleep(delay)
                continue
            raise


def process_batch(args, batch_id, raw_dir, provider_model):
    """Run one batch (with JSON-validation retries) and write its files.

    Returns a per-batch counters dict. Writes only batch-scoped files (distinct
    paths per batch id), so this is safe to run concurrently across batches.
    """
    result = {
        "completed": 0, "failed": 0, "skipped": 0,
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "approximate_usage_batches": 0, "invalid_json_retries": 0,
        "failed_batch_id": None,
    }
    if success_exists(raw_dir, batch_id, args) and not args.overwrite:
        result["skipped"] = 1
        return result

    prompt, payload, meta = load_batch(args.batches, batch_id)
    raw_path = raw_dir / f"{batch_id}_raw.json"
    response_path = raw_dir / f"{batch_id}_response.txt"
    error_path = raw_dir / f"{batch_id}_error.json"
    max_attempts = args.json_validation_retries + 1
    for attempt in range(1, max_attempts + 1):
        created_at = datetime.now(timezone.utc).isoformat()
        call_start = time.monotonic()
        try:
            text, usage = provider_call_with_backoff(args, provider_model, prompt, payload)
            latency_seconds = round(time.monotonic() - call_start, 4)
            raw = build_raw_record(args, batch_id, provider_model, meta, text or "", usage or {}, created_at, prompt, payload, attempt)
            raw["latency_seconds"] = latency_seconds
            if raw.get("output_cap_hit"):
                print(f"WARNING [{batch_id}] completion_tokens reached max_output_tokens "
                      f"({args.max_output_tokens}) -- output may be truncated or degenerate",
                      file=sys.stderr)
            write_json(raw_dir / f"{batch_id}_attempt_{attempt}_raw.json", raw)
            (raw_dir / f"{batch_id}_attempt_{attempt}_response.txt").write_text(text or "", encoding="utf-8")

            validation_error = response_validation_error(args.task, text or "", meta)
            if validation_error is not None:
                result["invalid_json_retries"] += 1
                write_json(raw_dir / f"{batch_id}_attempt_{attempt}_error.json", {
                    "batch_id": batch_id,
                    "attempt": attempt,
                    "provider": args.provider,
                    "model": args.model,
                    "error_type": type(validation_error).__name__,
                    "error_message": str(validation_error),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "meta": meta,
                })
                if attempt < max_attempts:
                    continue

            response_path.write_text(raw["response_text"], encoding="utf-8")
            write_json(raw_path, raw)
            estimated = raw["estimated_usage"]
            result["input_tokens"] += estimated["prompt_tokens"]
            result["output_tokens"] += estimated["completion_tokens"]
            result["total_tokens"] += estimated["total_tokens"]
            if estimated["source"] == "approximate":
                result["approximate_usage_batches"] += 1
            result["completed"] += 1
            return result
        except Exception as exc:
            if attempt < max_attempts:
                result["invalid_json_retries"] += 1
                write_json(raw_dir / f"{batch_id}_attempt_{attempt}_error.json", {
                    "batch_id": batch_id,
                    "attempt": attempt,
                    "provider": args.provider,
                    "model": args.model,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "created_at": created_at,
                    "meta": meta,
                })
                continue
            result["failed"] += 1
            result["failed_batch_id"] = batch_id
            error = {
                "batch_id": batch_id,
                "provider": args.provider,
                "model": args.model,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "created_at": created_at,
                "meta": meta,
            }
            write_json(error_path, error)
            write_json(raw_path, {**error, "status": "failed"})
            return result
    return result


def run_api(args):
    configure_run_paths(args, ('raw_output', 'report', 'metadata_dir', 'pilot_report_dir'))
    llm.load_dotenv()
    if not hasattr(args, "json_validation_retries"):
        args.json_validation_retries = 1
    if not hasattr(args, "rate_limit_retries"):
        args.rate_limit_retries = 8
    if getattr(args, "max_output_tokens", None) is None:
        args.max_output_tokens = llm.default_max_output_tokens(args.task)
    all_ids = batch_ids(args.batches)
    ids = select_batch_ids(all_ids, getattr(args, "start_batch", None), getattr(args, "end_batch", None))
    if not ids:
        raise ValueError(f"No batch metadata files found in {args.batches} for the requested range")
    raw_dir = Path(args.raw_output)
    raw_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    warnings = []

    _, _, first_meta = load_batch(args.batches, ids[0])
    warning = warn_or_require_pilot(args, first_meta)
    if warning:
        warnings.append(warning)

    provider_model = api_model_name(args.model)
    pin_run_identity(args, "sync", provider_model)
    concurrency = max(1, getattr(args, "concurrency", 1) or 1)
    stop_on_first_failure = getattr(args, "stop_on_first_failure", False)

    if concurrency == 1 or stop_on_first_failure:
        # Sequential path; required to honour stop_on_first_failure deterministically.
        results = []
        for batch_id in ids:
            res = process_batch(args, batch_id, raw_dir, provider_model)
            results.append(res)
            if res["failed"] and stop_on_first_failure:
                warnings.append(f"Stopped after first failed batch: {res['failed_batch_id']}")
                break
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            results = list(executor.map(
                lambda batch_id: process_batch(args, batch_id, raw_dir, provider_model),
                ids,
            ))

    totals = {key: 0 for key in (
        "completed", "failed", "skipped", "input_tokens", "output_tokens",
        "total_tokens", "approximate_usage_batches", "invalid_json_retries",
    )}
    for res in results:
        for key in totals:
            totals[key] += res[key]

    return finalize_run(
        args,
        first_meta,
        started_at,
        totals["completed"],
        totals["failed"],
        totals["skipped"],
        len(ids),
        totals["input_tokens"],
        totals["output_tokens"],
        totals["total_tokens"],
        totals["approximate_usage_batches"],
        totals["invalid_json_retries"],
        warnings,
    )


def finalize_run(args, first_meta, started_at, completed, failed, skipped, number_of_batches, input_tokens, output_tokens, total_tokens, approximate_usage_batches, invalid_json_retries, warnings):
    completed_at = datetime.now(timezone.utc)
    pilot_scale = max(1, number_of_batches)
    estimated_full_tokens = total_tokens
    if first_meta.get("run_mode") == "pilot" and first_meta.get("pilot_frac"):
        estimated_full_tokens = int(total_tokens / float(first_meta["pilot_frac"]))

    report = write_report(
        args,
        first_meta,
        number_of_batches,
        completed,
        failed,
        skipped,
        input_tokens,
        output_tokens,
        total_tokens,
        estimated_full_tokens,
        approximate_usage_batches,
        invalid_json_retries,
        warnings,
    )
    write_metadata(args, first_meta, started_at, completed_at, report)
    return {
        "number_of_batches": number_of_batches,
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "report": str(report),
    }


def write_report(args, meta, number_of_batches, completed, failed, skipped, input_tokens, output_tokens, total_tokens, estimated_full_tokens, approximate_usage_batches, invalid_json_retries, warnings):
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    run_mode = meta.get("run_mode", "unknown")
    lines = [
        "# LLM API Run Report",
        "",
        f"- model: {args.model}",
        f"- provider: {args.provider}",
        f"- task: {args.task}",
        f"- run_mode: {run_mode}",
        f"- prompt_version: {meta.get('prompt_version', args.prompt_version)}",
        f"- temperature: {args.temperature}",
        f"- top_p: {args.top_p}",
        f"- max_output_tokens: {args.max_output_tokens}",
        f"- batch_size: {meta.get('batch_size', 'unknown')}",
        f"- requested_batch_size: {meta.get('requested_batch_size', meta.get('batch_size', 'unknown'))}",
        f"- effective_items_per_batch: {meta.get('effective_items_per_batch', 'unknown')}",
        f"- number of batches: {number_of_batches}",
        f"- number of completed batches: {completed}",
        f"- number of failed batches: {failed}",
        f"- number of skipped batches: {skipped}",
        f"- input token count: {input_tokens if input_tokens else 'token_usage_unavailable'}",
        f"- output token count: {output_tokens if output_tokens else 'token_usage_unavailable'}",
        f"- total token count: {total_tokens if total_tokens else 'token_usage_unavailable'}",
        f"- approximate token usage batches: {approximate_usage_batches}",
        f"- invalid JSON/schema retries: {invalid_json_retries}",
        f"- estimated full-run token count from pilot: {estimated_full_tokens if estimated_full_tokens else 'token_usage_unavailable'}",
    ]
    if meta.get("batching_note"):
        lines.extend(["", "## Batching Note", "", f"- {meta['batching_note']}"])
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- none")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_metadata(args, meta, started_at, completed_at, report):
    # Keep generation and parse records in the explicitly selected run.
    arm = llm.metadata_arm_suffix(meta)
    meta_dir = llm.resolve_metadata_dir(args, report)
    out = meta_dir / f"{llm.model_slug(args.model)}_{args.task}_{meta.get('run_mode', 'unknown')}_{meta.get('prompt_version', 'unknown')}{arm}_metadata.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "model": args.model,
        "provider": args.provider,
        "task": args.task,
        "ablation": meta.get("ablation"),
        "shots": meta.get("shots"),
        "run_mode": meta.get("run_mode"),
        "prompt_version": meta.get("prompt_version", args.prompt_version),
        "prompt_hash": meta.get("prompt_hash"),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_output_tokens": args.max_output_tokens,
        "seed": getattr(args, "seed", None),
        "api_model": api_model_name(args.model),
        "base_url": getattr(args, "base_url", None),
        "pilot_frac": meta.get("pilot_frac"),
        "pilot_seed": meta.get("pilot_seed"),
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "batch_dir": args.batches,
        "raw_output_dir": args.raw_output,
        "prediction_output": None,
        "report": str(report),
    }
    kept = llm.preserve_superseded_metadata(out, record)
    if kept:
        print(f"[metadata] {out.name} already held a record of DIFFERENT work; "
              f"kept it as {kept.name} rather than overwriting it")
    write_json(out, record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--provider", choices=["gemini", "qwen", "genapi", "deepseek", "together", "deepinfra", "mistral", "anthropic", "openai"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", choices=sorted(llm.SUPPORTED_TASKS), required=True)
    parser.add_argument("--batches", required=True)
    parser.add_argument("--raw-output", required=True)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=llm.DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=llm.DEFAULT_SEED,
                        help="Best-effort determinism seed sent to providers that honour it.")
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="Output token cap; default 32768 (uniform across tasks).")
    parser.add_argument("--prompt-version", default=llm.PROMPT_VERSION_DEFAULT)
    parser.add_argument("--base-url")
    parser.add_argument("--report", required=True)
    parser.add_argument("--metadata-dir")
    parser.add_argument("--pilot-report-dir")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-pilot", action="store_true")
    parser.add_argument("--json-validation-retries", type=int, default=1)
    parser.add_argument("--rate-limit-retries", type=int, default=8,
                        help="Dedicated 429 backoff retries (exponential + Retry-After), "
                             "separate from --json-validation-retries. Needed for the "
                             "rate-limited Mistral free tier; harmless elsewhere.")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Number of batches to request in parallel (threaded). "
                             "Default is per-provider (deepseek/gemini/together 16, else 1). "
                             "Ignored when --stop-on-first-failure is set.")
    parser.add_argument("--start-batch", help="First batch id to process, for example batch_0138")
    parser.add_argument("--end-batch", help="Last batch id to process, for example batch_0200")
    parser.add_argument("--stop-on-first-failure", action="store_true")
    args = parser.parse_args()
    if args.concurrency is None:
        args.concurrency = recommended_concurrency(args.provider)
        print(f"Using per-provider concurrency={args.concurrency} for provider={args.provider}")
    result = run_api(args)
    print(f"Completed: {result['completed']}")
    print(f"Failed: {result['failed']}")
    print(f"Report: {result['report']}")


if __name__ == "__main__":
    main()
