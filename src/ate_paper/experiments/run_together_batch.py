#!/usr/bin/env python3
"""Run prepared LLM batches through the Together AI Batch API.

A different submission path from run_llm_api.py (one async batch job at ~50%
cost, best-effort within 24h) but it produces the *same* per-batch raw_*.json
files, so parse_llm_outputs.py consumes the output unchanged.

Each prepared batch (batch_0001 ...) is one JSONL line whose "custom_id" is the
batch_id, so results (returned in arbitrary order) map straight back and any
per-line failure is identified by batch_id in the error file. Structured output
(grouped JSON v2) is requested via response_format; GLM thinking-off is applied
via together_extra_body (Qwen3 Instruct is already non-thinking).

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

TERMINAL_STATES = {"COMPLETED", "FAILED", "EXPIRED", "CANCELLED"}


def build_request_body(model, prompt, payload, max_output_tokens, temperature, top_p, seed=None):
    """One Together chat-completions request body: same system prompt, JSON
    payload, grouped-JSON structured output, GLM thinking-off, and the seed as
    the sync path."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "max_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "response_format": {"type": "json_object"},
    }
    if seed is not None:
        body["seed"] = seed
    extra = llm.together_extra_body(model)
    if extra:
        body.update(extra)
    return body


def extract_from_output_line(line):
    """Return (ok, text, usage) from a Together batch output line."""
    resp = line.get("response") or {}
    body = resp.get("body") or {}
    choices = body.get("choices") or []
    text = ""
    if choices:
        text = (choices[0].get("message") or {}).get("content") or ""
    usage = {}
    raw_usage = body.get("usage") or {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if raw_usage.get(key) is not None:
            usage[key] = raw_usage[key]
    if body.get("system_fingerprint") is not None:
        usage["system_fingerprint"] = body.get("system_fingerprint")
    # A 200 with EMPTY content must count as a failure (and so get re-run), not be
    # written as an ok raw -- matches the Mistral/Anthropic runners.
    return resp.get("status_code") == 200 and bool(text), text, usage


def write_input_jsonl(path, batch_ids, batches_dir, provider_model, max_output_tokens, temperature, top_p, seed=None):
    with open(path, "w", encoding="utf-8") as f:
        for batch_id in batch_ids:
            prompt, payload, _meta = runner.load_batch(batches_dir, batch_id)
            body = build_request_body(provider_model, prompt, payload, max_output_tokens, temperature, top_p, seed)
            f.write(json.dumps({"custom_id": batch_id, "body": body}, ensure_ascii=False) + "\n")


def download_file(client, file_id, out_path):
    with client.files.with_streaming_response.content(id=file_id) as response:
        with open(out_path, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)


def load_keyed_jsonl(path):
    rows = {}
    if not Path(path).exists():
        return rows
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row.get("custom_id")] = row
    return rows


def wait_for_job(client, batch, poll_interval, max_wait):
    waited = 0
    while batch.status not in TERMINAL_STATES:
        if waited >= max_wait:
            raise TimeoutError(f"Batch {batch.id} still {batch.status} after {max_wait}s")
        print(f"  status={batch.status} (waited {waited}s)")
        time.sleep(poll_interval)
        waited += poll_interval
        batch = client.batches.retrieve(batch.id)
    print(f"Batch finished: {batch.status}")
    return batch



def run_together_batch(args):
    configure_run_paths(args, ('raw_output', 'report', 'metadata_dir'))
    llm.load_dotenv()
    if not os.environ.get("TOGETHER_API_KEY"):
        raise RuntimeError("TOGETHER_API_KEY is required for the Together Batch API.")
    if getattr(args, "max_output_tokens", None) is None:
        args.max_output_tokens = llm.default_max_output_tokens(args.task)
    from together import Together

    all_ids = runner.batch_ids(args.batches)
    ids = runner.select_batch_ids(all_ids, getattr(args, "start_batch", None), getattr(args, "end_batch", None))
    if not ids:
        raise ValueError(f"No batch metadata files found in {args.batches} for the requested range")
    raw_dir = Path(args.raw_output)
    raw_dir.mkdir(parents=True, exist_ok=True)

    provider_model = runner.api_model_name(args.model)
    pin_run_identity(args, "together_batch", provider_model)
    pin_batch_selection(args, ids, getattr(args, "job_id", None))
    pending = [b for b in ids if not (runner.success_exists(raw_dir, b, args) and not args.overwrite)]
    skipped = len(ids) - len(pending)
    started_at = datetime.now(timezone.utc)
    _, _, first_meta = runner.load_batch(args.batches, ids[0])
    warnings = []

    completed = failed = 0
    input_tokens = output_tokens = total_tokens = approximate = 0

    job_path = raw_dir / "_batch_job.json"
    existing = json.loads(job_path.read_text(encoding="utf-8")) if job_path.exists() and not args.overwrite else {}
    resume_id = getattr(args, "job_id", None) or existing.get("id")
    if pending or resume_id:
        client = Together(api_key=os.environ["TOGETHER_API_KEY"])
        input_path = raw_dir / "_batch_input.jsonl"
        if resume_id:
            batch = client.batches.retrieve(resume_id)
            # Resume the exact submitted set, independent of newly completed raw files.
            if not input_path.exists():
                raise ValueError("Cannot resume without this run's _batch_input.jsonl")
            pending = list(load_keyed_jsonl(input_path))
            if not set(pending).issubset(set(ids)):
                raise ValueError("Resume selection must contain every previously submitted batch")
            skipped = len(ids) - len(pending)
        else:
            write_input_jsonl(input_path, pending, args.batches, provider_model, args.max_output_tokens,
                              args.temperature, args.top_p, getattr(args, "seed", llm.DEFAULT_SEED))
            uploaded = client.files.upload(file=str(input_path), purpose="batch-api", check=False)
            created = client.batches.create(input_file_id=uploaded.id, endpoint=args.endpoint)
            batch = getattr(created, "job", created)
            runner.write_json(job_path, {"id": batch.id, "status": batch.status})
        print(f"Batch job: {batch.id}")
        if getattr(args, "submit_only", False):
            print("Submitted; run the same command later to collect this job.")
            return None
        batch = wait_for_job(client, batch, args.poll_interval, args.max_wait)
        runner.write_json(job_path, {"id": batch.id, "status": batch.status})
        if batch.status != "COMPLETED":
            raise RuntimeError(f"Batch job did not complete: {batch.status}")

        download_file(client, batch.output_file_id, raw_dir / "_batch_output.jsonl")
        results = load_keyed_jsonl(raw_dir / "_batch_output.jsonl")
        errors = {}
        if getattr(batch, "error_file_id", None):
            download_file(client, batch.error_file_id, raw_dir / "_batch_errors.jsonl")
            errors = load_keyed_jsonl(raw_dir / "_batch_errors.jsonl")

        for batch_id in pending:
            prompt, payload, meta = runner.load_batch(args.batches, batch_id)
            created_at = datetime.now(timezone.utc).isoformat()
            row = results.get(batch_id)
            ok, text, usage = extract_from_output_line(row) if row else (False, "", {})
            if not ok:
                failed += 1
                detail = errors.get(batch_id) or (row.get("response") if row else None) or "missing from batch results"
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
            raw = runner.build_raw_record(args, batch_id, provider_model, meta, text or "", usage, created_at, prompt, payload, 1)
            runner.write_json(raw_dir / f"{batch_id}_raw.json", raw)
            (raw_dir / f"{batch_id}_response.txt").write_text(raw["response_text"], encoding="utf-8")
            estimated = raw["estimated_usage"]
            input_tokens += estimated["prompt_tokens"]
            output_tokens += estimated["completion_tokens"]
            total_tokens += estimated["total_tokens"]
            if estimated["source"] == "approximate":
                approximate += 1
            completed += 1

    return runner.finalize_run(
        args, first_meta, started_at, completed, failed, skipped, len(ids),
        input_tokens, output_tokens, total_tokens, approximate, 0, warnings,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", default="qwen3_235b_a22b_instruct_2507")
    parser.add_argument("--task", choices=sorted(llm.SUPPORTED_TASKS), required=True)
    parser.add_argument("--batches", required=True)
    parser.add_argument("--raw-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--metadata-dir")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=llm.DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=llm.DEFAULT_SEED,
                        help="Best-effort determinism seed sent to Together that honours it.")
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="Output token cap; default 32768 (uniform across tasks).")
    parser.add_argument("--prompt-version", default=llm.PROMPT_VERSION_DEFAULT)
    parser.add_argument("--provider", default="together")
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start-batch")
    parser.add_argument("--end-batch")
    parser.add_argument("--job-id", help="Resume a previously submitted job")
    parser.add_argument("--submit-only", action="store_true")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--max-wait", type=int, default=86400)
    args = parser.parse_args()
    result = run_together_batch(args)
    if result is not None:
        print(f"Completed: {result['completed']}")
        print(f"Failed: {result['failed']}")
        print(f"Report: {result['report']}")


if __name__ == "__main__":
    main()
