"""Local output paths for an explicitly selected experiment run."""
import hashlib
import json
import os
from pathlib import Path


def configure_run_paths(args, output_fields):
    value = getattr(args, "run_dir", None)
    if not value:
        raise ValueError("--run-dir is required for all experiment outputs")
    root = Path(value).expanduser().resolve()
    args.run_dir = str(root)
    defaults = {"metadata_dir": "metadata", "pilot_report_dir": "pilot_reports", "report": "parse_report.md"}
    for field in output_fields:
        raw = getattr(args, field, None)
        if raw is None and field in defaults:
            raw = root / defaults[field]
        if raw is None:
            raise ValueError("An output path is required: " + field)
        path = Path(raw).expanduser().resolve()
        try:
            path.relative_to(root)
        except ValueError:
            raise ValueError(field + " must stay inside --run-dir: " + str(root))
        if path == root:
            raise ValueError(field + " must name a file or subdirectory inside --run-dir")
        setattr(args, field, str(path))
    return root


def pin_run_identity(args, transport, api_model):
    """Bind reusable raw/job artifacts to the actual prepared request inputs.

    Called before provider operations. Changing prompts, input metadata, model,
    transport or decoding settings requires a fresh run directory.
    """
    root = Path(args.raw_output)
    root.mkdir(parents=True, exist_ok=True)
    batches = []
    for meta_path in sorted(Path(args.batches).glob("batch_*_meta.json")):
        batch_id = meta_path.name[:-len("_meta.json")]
        prompt = (meta_path.parent / (batch_id + "_prompt.txt")).read_text(encoding="utf-8")
        payload = json.loads((meta_path.parent / (batch_id + "_input.json")).read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if meta.get("prompt_hash") != prompt_sha:
            raise ValueError("Prepared prompt differs from its metadata: " + batch_id)
        meta.pop("created_at", None)
        batches.append({"batch_id": batch_id, "prompt_sha256": prompt_sha,
                        "payload": payload, "metadata": meta})
    env_names = {"deepseek": ["DEEPSEEK_BASE_URL"], "together": ["TOGETHER_BASE_URL"],
                 "deepinfra": ["DEEPINFRA_BASE_URL"], "mistral": ["MISTRAL_BASE_URL"],
                 "qwen": ["QWEN_BASE_URL", "DASHSCOPE_BASE_URL"],
                 "genapi": ["GEN_API_BASE_URL", "GENAPI_BASE_URL"],
                 "openai": ["OPENAI_BASE_URL"]}
    endpoint_env = ({name: os.environ.get(name) for name in env_names.get(args.provider, [])}
                    if transport == "sync" and not getattr(args, "base_url", None) else {})
    identity = {"transport": transport, "provider": args.provider,
                "model": args.model, "api_model": api_model, "task": args.task,
                "base_url": getattr(args, "base_url", None), "endpoint_environment": endpoint_env,
                "endpoint": getattr(args, "endpoint", None),
                "prompt_version": args.prompt_version,
                "temperature": args.temperature, "top_p": args.top_p,
                "seed": getattr(args, "seed", None),
                "max_output_tokens": args.max_output_tokens, "batches": batches}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    path = root / "_run_identity.json"
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior.get("sha256") != digest:
            raise ValueError("Prepared batches or run settings changed; use a fresh --run-dir")
    else:
        if any(root.iterdir()):
            raise ValueError("Existing raw/job files have no run identity; use a fresh --run-dir")
        # The digest binds content without duplicating corpus text in the receipt.
        with path.open("x", encoding="utf-8") as stream:
            json.dump({"schema_version": 1, "sha256": digest}, stream)
            stream.write("\n")
    args._run_identity_sha256 = digest
    return digest


def pin_batch_selection(args, batch_ids, explicit_job_id=None):
    """Refuse an unbound or differently selected existing async job."""
    root = Path(args.raw_output)
    job_path = root / "_batch_job.json"
    job = json.loads(job_path.read_text(encoding="utf-8")) if job_path.exists() and not args.overwrite else None
    if explicit_job_id and (not job or job.get("id") != explicit_job_id):
        raise ValueError("--job-id/--batch-id must match this run's saved job receipt")
    path = root / "_batch_selection.json"
    selected = {"run_identity_sha256": args._run_identity_sha256, "batch_ids": list(batch_ids)}
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != selected:
            raise ValueError("Async job batch selection changed; use a fresh --run-dir")
    else:
        if job:
            raise ValueError("Existing async job has no batch selection receipt; use a fresh --run-dir")
        with path.open("x", encoding="utf-8") as stream:
            json.dump(selected, stream)
            stream.write("\n")
