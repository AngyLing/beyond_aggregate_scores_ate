"""Build article experiment commands; execution requires an explicit flag."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from .inputs import DATASETS, paths, prepare_inputs, sha256

ROOT = Path(__file__).resolve().parents[3]
GROUPS = ("supervised", "llm", "llm_repeats", "gazetteer")


def records(root: Path = ROOT) -> list[dict]:
    registry = json.loads((root / "run_settings.json").read_text(encoding="utf-8"))
    result = []
    for section in ("supervised_runs", "llm_artifacts", "gazetteer_runs"):
        for record in registry[section]:
            item = dict(record)
            item["dataset_key"] = next(k for k, v in DATASETS.items() if v == item["dataset"])
            item["group"] = Path(item["release_path"]).parts[-2]
            if item["group"] not in GROUPS or not re.fullmatch(r"[A-Za-z0-9_-]+", item["run"]):
                raise ValueError("Unexpected experiment identifier")
            result.append(item)
    if len(result) != 214 or len({r["release_path"] for r in result}) != 214:
        raise ValueError("The settings registry does not cover the selected 214 artifacts")
    return result


def checked_output_root(path: Path, root: Path = ROOT) -> Path:
    path = path.resolve()
    allowed = root.resolve() / "build/experiments"
    try:
        relative = path.relative_to(allowed)
    except ValueError:
        raise ValueError("--output-root must be under this repository's build/experiments/")
    if not relative.parts:
        raise ValueError("Give a named subdirectory under build/experiments/")
    return path


def command(module: str, *args) -> list[str]:
    return [sys.executable, "-m", "ate_paper.experiments." + module, *map(str, args)]


def plan_record(record: dict, processed_root: Path, output_root: Path,
                scibert_casing: str | None = None, device: str = "cuda") -> dict:
    ds, group = record["dataset_key"], record["group"]
    s = record["accepted_settings"]
    run_dir = output_root / ds / group / record["run"]
    p = paths(processed_root, output_root, ds)
    prediction = run_dir / "predictions.jsonl"
    item = {"dataset": ds, "group": group, "run": record["run"],
            "reference_artifact": record["release_path"], "run_dir": str(run_dir),
            "prediction": str(prediction), "commands": [], "requires": []}
    if group == "supervised":
        if "scibert" in record["configuration"] and scibert_casing is None:
            item["requires"] = ["Explicit --scibert-do-lower-case false|true"]
            return item
        shared = ["--dataset", record["dataset"], "--model-name", s["pretrained_model"],
                  "--epochs", s["epochs"], "--batch-size", s["batch_size"],
                  "--learning-rate", s["learning_rate"], "--seed", record["seed_label"],
                  "--max-length", s["max_length"], "--stride", s["stride"],
                  "--dev-thresholds", ",".join(map(str, s["dev_thresholds"])),
                  "--warmup-ratio", s["warmup_ratio"], "--device", device,
                  "--output", prediction, "--report", run_dir / "training_report.md"]
        if record["pretrained_revision"]:
            shared += ["--model-revision", record["pretrained_revision"]]
        if "scibert" in record["configuration"]:
            shared += ["--scibert-do-lower-case", scibert_casing]
        if s["decoder"] == "BIO":
            args = ["--train-gold", p["train"], "--dev-gold", p["dev"],
                    "--test-text", p["test_units"], "--model-dir", run_dir / "model",
                    "--label-mode", "class" if s["typed"] else "unlabeled",
                    "--report-title", record["configuration"] + " Sequence Labeler Report",
                    "--weight-decay", s["weight_decay"], "--max-grad-norm", s["max_grad_norm"],
                    "--flatten-policy", "longest"]
            item["commands"] = [command("train_mbert_sequence_labeler", *args, *shared)]
        else:
            args = ["--train", p["train" if s["typed"] else "train_untyped"],
                    "--dev", p["dev" if s["typed"] else "dev_untyped"],
                    "--predict", p["test_units"], "--output-dir", run_dir / "model",
                    "--max-span-tokens", s["span_max_tokens"],
                    "--negative-ratio", s["span_negative_ratio"],
                    "--width-embedding-dim", s["span_width_embedding_dim"], "--output-format", "unified"]
            item["commands"] = [command("train_xlmr_span_classifier", *args, *shared)]
    elif group == "gazetteer":
        args = ["--train", p["gazetteer_train"], "--test", p["test_units"],
                "--output", prediction, "--report", run_dir / "report.md",
                "--inventory-scope", s["inventory_scope"], "--min-term-chars", s["min_term_chars"],
                "--min-alnum-chars", s["min_alnum_chars"], "--min-term-tokens", s["min_term_tokens"]]
        for key in ("drop_stopword_terms", "drop_single_letter_terms"):
            if s[key]:
                args.append("--" + key.replace("_", "-"))
        item["commands"] = [command("make_gazetteer_predictions", *args)]
    else:
        from .prepare_llm_batches import batch_output_dir
        for ref in s["prompt_files"]:
            prompt_path = ROOT / ref["path"]
            prompt_path.resolve().relative_to((ROOT / "prompts").resolve())
            if sha256(prompt_path) != ref["sha256"]:
                raise ValueError(f"Prompt differs from the article recipe: {ref['path']}")
        provider, model, task = s["provider"], s["api_model"], s["task"]
        transport, = record["raw_transport"]
        module = "run_llm_api" if transport == "sync" else {
            "together": "run_together_batch", "mistral": "run_mistral_batch",
            "anthropic": "run_anthropic_batch"}[provider]
        batches = batch_output_dir(run_dir / "batches", model, task, "full", s["prompt_version"])
        prepare = command("prepare_llm_batches", "--run-dir", run_dir, "--model", model,
                          "--task", task, "--input", p["test_units"], "--output-dir", run_dir / "batches",
                          "--batch-size", s["requested_batch_size"], "--prompt-version", s["prompt_version"])
        args = ["--run-dir", run_dir, "--provider", provider, "--model", model, "--task", task,
                "--batches", batches, "--raw-output", run_dir / "raw", "--report", run_dir / "generation_report.md",
                "--metadata-dir", run_dir / "metadata", "--temperature", s["temperature"],
                "--max-output-tokens", s["max_output_tokens"], "--prompt-version", s["prompt_version"]]
        if s["top_p"] is not None:
            args += ["--top-p", s["top_p"]]
        if s["request_seed"] is not None:
            args += ["--seed", s["request_seed"]]
        parse_args = ["--run-dir", run_dir, "--provider", provider, "--model", model, "--task", task,
                      "--raw-dir", run_dir / "raw", "--batch-dir", batches,
                      "--output", prediction, "--failed-output", run_dir / "failed_batches.jsonl",
                      "--report", run_dir / "parse_report.md", "--metadata-dir", run_dir / "metadata"]
        if record["historical_parser_status"] == "pre_2026_08_24_header_rule_byte_match":
            parse_args += ["--legacy-exact-headers"]
        item["transport"] = transport
        item["prompt_files"] = [p["path"] for p in s["prompt_files"]]
        item["commands"] = [prepare, command(module, *args), command("parse_llm_outputs", *parse_args)]
    return item


def execution_sources(items: list[dict]) -> dict[str, str]:
    """Bind a resumed run to the actual local programs and prompts it uses."""
    files = set((ROOT / "src/ate_paper/experiments").glob("*.py"))
    files.update({ROOT / "scripts/run_experiments.py", ROOT / "src/ate_paper/prepare/__init__.py"})
    for item in items:
        for relative in item.get("prompt_files", []):
            path = ROOT / relative
            path.resolve().relative_to((ROOT / "prompts").resolve())
            files.add(path)
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(files)}


def execute(items: list[dict], processed_root: Path, output_root: Path,
            registry_sha: str, resume: bool = False,
            planned_sources: dict[str, str] | None = None) -> None:
    if any(item["requires"] for item in items):
        raise ValueError("Resolve the explicit choices listed in the plan before --execute")
    # Compute each item's identity independently: selecting additional runs must
    # not alter an already-recorded run's own code/prompt identity.
    current_sources = execution_sources(items)
    if planned_sources is not None and current_sources != planned_sources:
        raise ValueError("Code or prompts changed after planning; create the plan again")
    specs = []
    completed = set()
    # Validate every selected run before preparing data or starting any child.
    for item in items:
        run_dir = Path(item["run_dir"])
        own_prompts = set(item.get("prompt_files", []))
        own_sources = {p: h for p, h in current_sources.items()
                       if not p.startswith("prompts/") or p in own_prompts}
        spec = {"registry_sha256": registry_sha, "source_sha256": own_sources, **item}
        specs.append(spec)
        try:
            run_dir.resolve().relative_to(output_root)
        except ValueError:
            raise ValueError("Run directory resolves outside --output-root")
        if run_dir.exists() and (run_dir.is_symlink() or any(p.is_symlink() for p in run_dir.rglob("*"))):
            raise ValueError("Existing run directory contains symlinks; use a fresh output tag")
        if run_dir.exists() and not resume:
            raise ValueError(f"Run directory already exists; choose a fresh tag or --resume: {run_dir}")
        saved_plan = run_dir / "plan.json"
        if run_dir.exists():
            if not saved_plan.is_file():
                raise ValueError(f"Existing run has no saved plan; use a fresh output tag: {run_dir}")
            if json.loads(saved_plan.read_text(encoding="utf-8")) != spec:
                raise ValueError(f"Resume settings, code or prompts differ from saved plan: {run_dir}")
        completion = run_dir / "complete.json"
        if completion.exists():
            old = json.loads(completion.read_text(encoding="utf-8"))
            if sha256(Path(item["prediction"])) != old["prediction_sha256"]:
                raise ValueError("Completed predictions changed; use a fresh output directory")
            completed.add(str(run_dir))
    for ds in sorted({item["dataset"] for item in items}):
        prepare_inputs(processed_root, output_root, ds)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for item, spec in zip(items, specs):
        run_dir = Path(item["run_dir"])
        if str(run_dir) in completed:
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        saved_plan = run_dir / "plan.json"
        saved_plan.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        completion = run_dir / "complete.json"
        with (run_dir / "execution.log").open("a", encoding="utf-8") as log:
            for argv in item["commands"]:
                log.write(json.dumps(argv) + "\n")
                log.flush()
                subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        prediction = Path(item["prediction"])
        completion.write_text(json.dumps({"prediction_sha256": sha256(prediction)}, indent=2) + "\n")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List selected artifact identities and exit")
    parser.add_argument("--dataset", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument("--group", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--run", action="append", help="Exact registry run name; repeat to select more")
    parser.add_argument("--processed-root", type=Path, default=ROOT / "data/local/processed")
    parser.add_argument("--output-root", type=Path, default=ROOT / "build/experiments/new-run")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--scibert-do-lower-case", choices=["false", "true"])
    parser.add_argument("--execute", action="store_true", help="Run the displayed programs; can train/download/call paid APIs")
    parser.add_argument("--resume", action="store_true", help="Resume the same saved settings/output directory")
    args = parser.parse_args(argv)
    try:
        chosen = [r for r in records() if r["dataset_key"] in args.dataset and r["group"] in args.group
                  and (not args.run or r["run"] in args.run)]
        if not chosen:
            raise ValueError("No selected artifact matches these filters")
        if args.run and set(args.run) - {r["run"] for r in chosen}:
            raise ValueError("Some requested run names do not match the dataset/group filters")
        if args.list:
            print(json.dumps([{k: r[k] for k in ("dataset_key", "group", "run", "release_path")}
                              for r in chosen], indent=2))
            return
        output_root = checked_output_root(args.output_root)
        processed = args.processed_root.resolve()
        items = [plan_record(r, processed, output_root, args.scibert_do_lower_case, args.device) for r in chosen]
        registry_sha = sha256(ROOT / "run_settings.json")
        source_hashes = execution_sources(items)
        print(json.dumps({"mode": "execute" if args.execute else "plan_only",
                          "registry_sha256": registry_sha, "source_sha256": source_hashes,
                          "runs": items}, indent=2), flush=True)
        if args.execute:
            execute(items, processed, output_root, registry_sha, args.resume, source_hashes)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Experiment command failed: {error}\n")


if __name__ == "__main__":
    main()
