#!/usr/bin/env python3
"""Run prepared LLM batches through the Anthropic Message Batches API (~50% cost).

A different submission path from run_llm_api.py: one async batch per arm at half the
per-token price (most batches finish < 1h). It produces the *same* per-batch raw_*.json
files as the sync/Together/Mistral paths, so parse_llm_outputs.py consumes the output
unchanged.

Anthropic specifics (mirrors the sync `anthropic` branch, which uses the OpenAI-compat
endpoint; the Batch API is NATIVE Messages only):
  * NO file upload — the requests are POSTed INLINE as {"requests":[{custom_id, params}]}.
  * Auth is `x-api-key` + `anthropic-version` headers (not Bearer).
  * `system` is a top-level param; the user JSON payload is the single user message.
  * NO response_format json_object (Claude rejects it) -> rely on the prompt; Claude may
    wrap JSON in ```json fences, which parse_llm_outputs already strips.
  * temperature only (Claude 400s on temperature+top_p together) -> top_p omitted.
  * no seed (Anthropic ignores it); extended thinking left off.
  * The native base is hardcoded to https://api.anthropic.com/v1 and $ANTHROPIC_BASE_URL
    is deliberately NOT honoured (that env var, e.g. "https://api.anthropic.com" without
    /v1, is for the native SDK path shape and 404s the OpenAI-compat calls; see the sync
    branch note).

custom_id == batch_id (matches Anthropic's ^[A-Za-z0-9_-]{1,64}$).

Resume / no-double-submit: the created batch id is written to <raw_output>/_batch_job.json
IMMEDIATELY after creation, and `--batch-id ID` resumes a submitted batch (poll + fetch
results) WITHOUT re-submitting. `--list` prints recent account batches.

"""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from ate_paper.experiments.llm_run_paths import configure_run_paths, pin_run_identity, pin_batch_selection

from ate_paper.experiments import make_llm_predictions as llm
from ate_paper.experiments import run_llm_api as runner

ANTHROPIC_API = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"


def _headers(api_key):
    return {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


def build_params(model, prompt, payload, max_output_tokens, temperature):
    """One native Messages request body: system prompt + JSON payload, temperature
    only (no top_p / response_format / seed), same content as the sync path."""
    return {
        "model": model,
        "max_tokens": max_output_tokens,
        "temperature": temperature,
        "system": prompt,
        "messages": [
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    }


def build_requests(batch_ids, batches_dir, model, max_output_tokens, temperature):
    reqs = []
    for batch_id in batch_ids:
        prompt, payload, _meta = runner.load_batch(batches_dir, batch_id)
        reqs.append({"custom_id": batch_id,
                     "params": build_params(model, prompt, payload, max_output_tokens, temperature)})
    return reqs


def extract_result(entry):
    """Return (ok, text, usage) from an Anthropic batch result line."""
    result = entry.get("result") or {}
    if result.get("type") != "succeeded":
        return False, "", {}
    message = result.get("message") or {}
    text = "".join(
        block.get("text", "")
        for block in (message.get("content") or [])
        if block.get("type") == "text"
    )
    raw_usage = message.get("usage") or {}
    usage = {}
    if raw_usage.get("input_tokens") is not None:
        usage["input_tokens"] = raw_usage["input_tokens"]
    if raw_usage.get("output_tokens") is not None:
        usage["output_tokens"] = raw_usage["output_tokens"]
    return bool(text), text, usage


# ------------------------------- HTTP calls -------------------------------

def create_batch(api_key, reqs):
    import requests
    resp = requests.post(f"{ANTHROPIC_API}/messages/batches", headers=_headers(api_key),
                         json={"requests": reqs}, timeout=600)
    resp.raise_for_status()
    return resp.json()


def get_batch(api_key, batch_id):
    import requests
    resp = requests.get(f"{ANTHROPIC_API}/messages/batches/{batch_id}",
                        headers=_headers(api_key), timeout=120)
    resp.raise_for_status()
    return resp.json()






def fetch_results(api_key, results_url, out_path):
    import requests
    resp = requests.get(results_url, headers=_headers(api_key), timeout=1200, stream=True)
    resp.raise_for_status()
    rows = {}
    with open(out_path, "w", encoding="utf-8") as f:
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            f.write(line + "\n")
            entry = json.loads(line)
            rows[entry.get("custom_id")] = entry
    return rows


def poll_until_ended(api_key, batch_id, poll_interval, max_wait):
    waited = 0
    batch = get_batch(api_key, batch_id)
    while batch.get("processing_status") != "ended":
        if waited >= max_wait:
            raise TimeoutError(f"Batch {batch_id} still {batch.get('processing_status')} after {max_wait}s")
        counts = batch.get("request_counts") or {}
        print(f"  status={batch.get('processing_status')} "
              f"ok={counts.get('succeeded', 0)} err={counts.get('errored', 0)} "
              f"proc={counts.get('processing', 0)} (waited {waited}s)")
        time.sleep(poll_interval)
        waited += poll_interval
        batch = get_batch(api_key, batch_id)
    print(f"Batch ended: {batch.get('request_counts')}")
    return batch


# ------------------------------ orchestration ------------------------------

def postprocess(args, results, raw_dir, provider_model):
    valid = set(runner.batch_ids(args.batches))
    requests_path = Path(raw_dir) / "_batch_requests.json"
    submitted = ([row["custom_id"] for row in json.loads(requests_path.read_text(encoding="utf-8"))]
                 if requests_path.exists() else sorted(set(results) & valid))
    submitted = [key for key in submitted if key in valid]
    completed = failed = 0
    input_tokens = output_tokens = total_tokens = approximate = 0
    for batch_id in submitted:
        prompt, payload, meta = runner.load_batch(args.batches, batch_id)
        created_at = datetime.now(timezone.utc).isoformat()
        result_row = results.get(batch_id)
        ok, text, usage = extract_result(result_row) if result_row else (False, "", {})
        if not ok:
            failed += 1
            detail = ((result_row or {}).get("result") or {}).get("error") or "errored/expired/canceled"
            error = {
                "batch_id": batch_id,
                "provider": args.provider,
                "model": args.model,
                "error_type": "batch_error",
                "error_message": json.dumps(detail, ensure_ascii=False)[:1000],
                "created_at": created_at,
                "meta": meta,
            }
            runner.write_json(raw_dir / f"{batch_id}_error.json", error)
            runner.write_json(raw_dir / f"{batch_id}_raw.json", {**error, "status": "failed"})
            continue
        raw = runner.build_raw_record(args, batch_id, provider_model, meta, text, usage, created_at, prompt, payload, 1)
        runner.write_json(raw_dir / f"{batch_id}_raw.json", raw)
        (raw_dir / f"{batch_id}_response.txt").write_text(raw["response_text"], encoding="utf-8")
        estimated = raw["estimated_usage"]
        input_tokens += estimated["prompt_tokens"]
        output_tokens += estimated["completion_tokens"]
        total_tokens += estimated["total_tokens"]
        if estimated["source"] == "approximate":
            approximate += 1
        completed += 1
    return completed, failed, len(submitted), input_tokens, output_tokens, total_tokens, approximate


def run_anthropic_batch(args):
    configure_run_paths(args, ("raw_output", "report", "metadata_dir"))
    llm.load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for the Message Batches API.")

    if getattr(args, "max_output_tokens", None) is None:
        args.max_output_tokens = llm.default_max_output_tokens(args.task)

    all_ids = runner.batch_ids(args.batches)
    ids = runner.select_batch_ids(all_ids, getattr(args, "start_batch", None), getattr(args, "end_batch", None))
    if not ids:
        raise ValueError(f"No batch metadata files found in {args.batches} for the requested range")
    raw_dir = Path(args.raw_output)
    raw_dir.mkdir(parents=True, exist_ok=True)
    _, _, first_meta = runner.load_batch(args.batches, ids[0])
    provider_model = runner.api_model_name(args.model)
    pin_run_identity(args, "anthropic_batch", provider_model)
    pin_batch_selection(args, ids, getattr(args, "batch_id", None))
    started_at = datetime.now(timezone.utc)

    job_path = raw_dir / "_batch_job.json"
    batch_id = None
    if args.batch_id:
        batch_id = args.batch_id
        print(f"Resuming batch {batch_id} (no re-submit)")
    elif job_path.exists() and not args.overwrite:
        # Auto-resume: a batch was already submitted for this arm (idempotent — this is
        # how parallel --submit-only then poll works, and how a killed poller recovers
        # without double-spending).
        existing = json.loads(job_path.read_text(encoding="utf-8"))
        if existing.get("id"):
            batch_id = existing["id"]
            print(f"Auto-resuming already-submitted batch {batch_id} (from {job_path})")

    if batch_id is None:
        pending = [b for b in ids if not (runner.success_exists(raw_dir, b, args) and not args.overwrite)]
        if not pending:
            print("All selected batches already have ok raw; nothing to submit.")
            return runner.finalize_run(args, first_meta, started_at, 0, 0, len(ids), len(ids), 0, 0, 0, 0, 0, [])
        reqs = build_requests(pending, args.batches, provider_model, args.max_output_tokens, args.temperature)
        runner.write_json(raw_dir / "_batch_requests.json", reqs)
        print(f"Submitting {len(reqs)} requests inline to the Message Batches API ...")
        batch = create_batch(api_key, reqs)
        batch_id = batch["id"]
        # Persist the batch id BEFORE polling so a killed process can resume via --batch-id.
        runner.write_json(job_path,
                          {"id": batch_id, "processing_status": batch.get("processing_status"),
                           "submitted": len(reqs)})
        print(f"Batch: {batch_id}  (id saved to {job_path})")

    if args.submit_only:
        print(f"--submit-only: batch {batch_id} left processing (poll later to fetch).")
        return None

    batch = poll_until_ended(api_key, batch_id, args.poll_interval, args.max_wait)
    results_url = batch.get("results_url")
    if not results_url:
        raise RuntimeError(f"Batch {batch_id} ended without a results_url: {batch.get('request_counts')}")
    results = fetch_results(api_key, results_url, raw_dir / "_batch_results.jsonl")
    runner.write_json(raw_dir / "_batch_job.json",
                      {"id": batch_id, "processing_status": batch.get("processing_status"),
                       "request_counts": batch.get("request_counts"), "results_url": results_url})

    completed, failed, submitted, it, ot, tt, approx = postprocess(args, results, raw_dir, provider_model)
    skipped = len(ids) - submitted
    return runner.finalize_run(args, first_meta, started_at, completed, failed, skipped,
                               len(ids), it, ot, tt, approx, 0, [])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", default="claude_haiku_4_5")
    parser.add_argument("--task", choices=sorted(llm.SUPPORTED_TASKS))
    parser.add_argument("--batches")
    parser.add_argument("--raw-output")
    parser.add_argument("--report")
    parser.add_argument("--metadata-dir")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=llm.DEFAULT_TOP_P,
                        help="Recorded for provenance; NOT sent (Claude rejects temperature+top_p).")
    parser.add_argument("--seed", type=int, default=llm.DEFAULT_SEED,
                        help="Recorded for provenance; Anthropic ignores seed.")
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="Output token cap; default 32768 (uniform across tasks).")
    parser.add_argument("--prompt-version", default=llm.PROMPT_VERSION_DEFAULT)
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start-batch")
    parser.add_argument("--end-batch")
    parser.add_argument("--batch-id", help="Resume an already-submitted batch (no re-submit).")
    parser.add_argument("--submit-only", action="store_true",
                        help="Submit (or reuse) the batch and return without polling; poll later to fetch.")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--max-wait", type=int, default=86400)
    args = parser.parse_args()
    for required in ("task", "batches", "raw_output", "report"):
        if getattr(args, required) is None:
            parser.error(f"--{required.replace('_', '-')} is required")
    result = run_anthropic_batch(args)
    if result is not None:
        print(f"Completed: {result['completed']}")
        print(f"Failed: {result['failed']}")
        print(f"Report: {result['report']}")


if __name__ == "__main__":
    main()
