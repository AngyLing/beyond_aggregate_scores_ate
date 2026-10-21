#!/usr/bin/env python3
"""Run prepared LLM batches through the Mistral Batch API (native REST, ~50% cost).

A different submission path from run_llm_api.py: one async batch job per arm at
half the per-token price and — crucially — a queue that is SEPARATE from the sync
rate limit, so it does not 429 (the sync mistral run lost ~85% of requests to
HTTP 429 because run_llm_api has no 429 backoff). It produces the *same* per-batch
raw_*.json files as the sync/Together paths, so parse_llm_outputs.py consumes the
output unchanged.

Design (mirrors run_together_batch.py):
  * custom_id == batch_id, so results (returned in arbitrary order) map straight back.
  * The model is a JOB-level parameter (Mistral sets it at create time), NOT per line.
  * Mistral's determinism knob is `random_seed` (it 422s on the OpenAI `seed`); it
    accepts response_format json_object + top_p (same as the sync mistral branch).
  * No new SDK dependency — raw HTTP via `requests`.

Resume / no-double-submit: the created job id is written to
<raw_output>/_batch_job.json IMMEDIATELY after creation, and `--job-id JOB`
resumes a submitted job (poll + download + post-process) WITHOUT re-submitting, so
a killed process never double-spends. `--list` prints account batch jobs.

Docs: https://docs.mistral.ai/capabilities/batch/
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

MISTRAL_API = "https://api.mistral.ai/v1"
TERMINAL_STATES = {"SUCCESS", "FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"}


def _headers(api_key):
    return {"Authorization": f"Bearer {api_key}"}


def build_request_body(prompt, payload, max_output_tokens, temperature, top_p, seed):
    """One Mistral chat body: same system prompt + JSON payload + json_object output
    as the sync path; `random_seed` (not `seed`); model is set at job level."""
    body = {
        "max_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    }
    if seed is not None:
        body["random_seed"] = seed
    return body


def write_input_jsonl(path, batch_ids, batches_dir, max_output_tokens, temperature, top_p, seed):
    with open(path, "w", encoding="utf-8") as f:
        for batch_id in batch_ids:
            prompt, payload, _meta = runner.load_batch(batches_dir, batch_id)
            body = build_request_body(prompt, payload, max_output_tokens, temperature, top_p, seed)
            f.write(json.dumps({"custom_id": batch_id, "body": body}, ensure_ascii=False) + "\n")


def extract_from_output_line(line):
    """Return (ok, text, usage) from a Mistral batch output line
    (response.body.choices[0].message.content, wrapped with a status_code)."""
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
    return resp.get("status_code") == 200 and bool(text), text, usage


def load_keyed_jsonl(path):
    rows = {}
    if not Path(path).exists():
        return rows
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row.get("custom_id")] = row
    return rows


def submitted_ids_from_input(raw_dir):
    """The exact custom_ids uploaded for this job, read from the persisted
    _batch_input.jsonl so post-processing covers the full submitted set (like
    run_together_batch iterating `pending`) even on a --job-id resume."""
    path = Path(raw_dir) / "_batch_input.jsonl"
    ids = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.append(json.loads(line).get("custom_id"))
    return [i for i in ids if i]


# ------------------------------- HTTP calls -------------------------------

def upload_input_file(api_key, path):
    import requests
    with open(path, "rb") as fh:
        resp = requests.post(
            f"{MISTRAL_API}/files",
            headers=_headers(api_key),
            files={"file": (Path(path).name, fh, "application/jsonl")},
            data={"purpose": "batch"},
            timeout=600,
        )
    resp.raise_for_status()
    return resp.json()["id"]


def create_job(api_key, file_id, model, endpoint, metadata):
    import requests
    resp = requests.post(
        f"{MISTRAL_API}/batch/jobs",
        headers={**_headers(api_key), "Content-Type": "application/json"},
        json={"input_files": [file_id], "model": model, "endpoint": endpoint, "metadata": metadata},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def get_job(api_key, job_id):
    import requests
    resp = requests.get(f"{MISTRAL_API}/batch/jobs/{job_id}", headers=_headers(api_key), timeout=120)
    resp.raise_for_status()
    return resp.json()


def list_jobs(api_key):
    import requests
    resp = requests.get(f"{MISTRAL_API}/batch/jobs", headers=_headers(api_key), timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", data) if isinstance(data, dict) else data


def download_file(api_key, file_id, out_path):
    import requests
    resp = requests.get(f"{MISTRAL_API}/files/{file_id}/content", headers=_headers(api_key), timeout=1200, stream=True)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)


def _progress(job):
    done = job.get("succeeded_requests", 0) or 0
    failed = job.get("failed_requests", 0) or 0
    total = job.get("total_requests", 0) or 0
    return f"{done + failed}/{total} (ok={done} fail={failed})"


def poll_until_terminal(api_key, job_id, poll_interval, max_wait):
    waited = 0
    job = get_job(api_key, job_id)
    while job.get("status") not in TERMINAL_STATES:
        if waited >= max_wait:
            raise TimeoutError(f"Batch {job_id} still {job.get('status')} after {max_wait}s")
        print(f"  status={job.get('status')} {_progress(job)} (waited {waited}s)")
        time.sleep(poll_interval)
        waited += poll_interval
        job = get_job(api_key, job_id)
    print(f"Batch finished: {job.get('status')} {_progress(job)}")
    return job


# ------------------------------ orchestration ------------------------------

def postprocess(args, job, submitted_ids, raw_dir, provider_model):
    api_key = os.environ["MISTRAL_API_KEY"]
    results = {}
    if job.get("output_file"):
        download_file(api_key, job["output_file"], raw_dir / "_batch_output.jsonl")
        results = load_keyed_jsonl(raw_dir / "_batch_output.jsonl")
    errors = {}
    if job.get("error_file"):
        download_file(api_key, job["error_file"], raw_dir / "_batch_errors.jsonl")
        errors = load_keyed_jsonl(raw_dir / "_batch_errors.jsonl")

    # Iterate the full SUBMITTED set (like run_together_batch iterates `pending`), so
    # every submitted batch gets a raw — a batch missing from both output and error is
    # recorded as failed rather than silently dropped, and `skipped` stays exact.
    # Fall back to whatever came back only if the input sidecar is unavailable.
    valid = set(runner.batch_ids(args.batches))
    submitted = [b for b in submitted_ids if b in valid] or sorted((set(results) | set(errors)) & valid)

    completed = failed = 0
    input_tokens = output_tokens = total_tokens = approximate = 0
    for batch_id in submitted:
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


def run_mistral_batch(args):
    configure_run_paths(args, ('raw_output', 'report', 'metadata_dir'))
    llm.load_dotenv()
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is required for the Mistral Batch API.")


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
    pin_run_identity(args, "mistral_batch", provider_model)
    pin_batch_selection(args, ids, getattr(args, "job_id", None))
    started_at = datetime.now(timezone.utc)

    job_path = raw_dir / "_batch_job.json"
    job_id = None
    submitted_ids = None
    if args.job_id:
        job_id = args.job_id
        submitted_ids = submitted_ids_from_input(raw_dir)
        print(f"Resuming job {job_id} (no re-submit)")
    elif job_path.exists() and not args.overwrite:
        # Auto-resume an already-submitted job (idempotent — enables the parallel
        # --submit-only-then-poll pattern and recovers a killed poller without double-spend).
        existing = json.loads(job_path.read_text(encoding="utf-8"))
        if existing.get("id"):
            job_id = existing["id"]
            submitted_ids = submitted_ids_from_input(raw_dir)
            print(f"Auto-resuming already-submitted job {job_id} (from {job_path})")

    if job_id is None:
        pending = [b for b in ids if not (runner.success_exists(raw_dir, b, args) and not args.overwrite)]
        if not pending:
            print("All selected batches already have ok raw; nothing to submit.")
            return runner.finalize_run(args, first_meta, started_at, 0, 0, len(ids), len(ids), 0, 0, 0, 0, 0, [])
        submitted_ids = pending
        input_path = raw_dir / "_batch_input.jsonl"
        write_input_jsonl(input_path, pending, args.batches, args.max_output_tokens,
                          args.temperature, args.top_p, getattr(args, "seed", llm.DEFAULT_SEED))
        print(f"Uploading {len(pending)} requests to Mistral Files API ...")
        file_id = upload_input_file(api_key, input_path)
        job = create_job(api_key, file_id, provider_model, args.endpoint,
                         {"task": args.task, "arm": raw_dir.name, "batches": str(args.batches)})
        job_id = job["id"]
        # Persist the job id BEFORE polling so a killed process can resume via --job-id.
        runner.write_json(job_path,
                          {"id": job_id, "input_file": file_id, "status": job.get("status"),
                           "total_requests": job.get("total_requests"), "submitted": len(pending)})
        print(f"Batch job: {job_id}  (id saved to {job_path})")

    if args.submit_only:
        print(f"--submit-only: job {job_id} left processing (poll later to fetch).")
        return None

    job = poll_until_terminal(api_key, job_id, args.poll_interval, args.max_wait)
    runner.write_json(raw_dir / "_batch_job.json",
                      {"id": job_id, "status": job.get("status"),
                       "output_file": job.get("output_file"), "error_file": job.get("error_file"),
                       "total_requests": job.get("total_requests"),
                       "succeeded_requests": job.get("succeeded_requests"),
                       "failed_requests": job.get("failed_requests")})
    if job.get("status") != "SUCCESS":
        print(f"WARNING: job terminal status {job.get('status')} (not SUCCESS); "
              f"post-processing whatever partial output exists.")

    completed, failed, submitted, it, ot, tt, approx = postprocess(args, job, submitted_ids, raw_dir, provider_model)
    skipped = len(ids) - submitted
    return runner.finalize_run(args, first_meta, started_at, completed, failed, skipped,
                               len(ids), it, ot, tt, approx, 0, [])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", default="mistral_large_3")
    parser.add_argument("--task", choices=sorted(llm.SUPPORTED_TASKS))
    parser.add_argument("--batches")
    parser.add_argument("--raw-output")
    parser.add_argument("--report")
    parser.add_argument("--metadata-dir")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=llm.DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=llm.DEFAULT_SEED,
                        help="Mistral determinism knob (sent as random_seed).")
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="Output token cap; default 32768 (uniform across tasks).")
    parser.add_argument("--prompt-version", default=llm.PROMPT_VERSION_DEFAULT)
    parser.add_argument("--provider", default="mistral")
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start-batch")
    parser.add_argument("--end-batch")
    parser.add_argument("--job-id", help="Resume an already-submitted job (no re-submit).")
    parser.add_argument("--submit-only", action="store_true",
                        help="Submit (or reuse) the batch job and return without polling; poll later to fetch.")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--max-wait", type=int, default=86400)
    args = parser.parse_args()
    for required in ("task", "batches", "raw_output", "report"):
        if getattr(args, required) is None:
            parser.error(f"--{required.replace('_', '-')} is required")
    result = run_mistral_batch(args)
    if result is not None:
        print(f"Completed: {result['completed']}")
        print(f"Failed: {result['failed']}")
        print(f"Report: {result['report']}")


if __name__ == "__main__":
    main()
