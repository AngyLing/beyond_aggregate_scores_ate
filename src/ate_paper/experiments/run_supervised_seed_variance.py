#!/usr/bin/env python3
"""Run the article's supervised BIO/span baselines over explicit seeds.

Each seed gets its own model bundle, prediction JSONL and report. The portable
article launcher supplies the settings collected in run_settings.json.
No evaluation or data preparation is performed automatically.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from ate_paper.experiments.supervised_tokenizers import add_pretrained_arguments, tokenizer_kwargs


SUPPORTED_MODELS = {
    "mbert_class",
    "mbert_class_bio",
    "mbert_unlabeled",
    "mbert_unlabeled_bio",
    "mbert_span",
    "xlmr_bio",
    "xlmr_unlabeled_bio",
    "xlmr_span_classifier",
    "rubert_bio",
    "rubert_span",
    "xlmr_span_unlabeled",
    "rubert_span_unlabeled",
    "ruroberta_bio",
    "ruroberta_span",
    "pubmedbert_span",
    "pubmedbert_bio",
    "scibert_span",
    "scibert_bio",
    "csroberta_bio",
    "csroberta_span",
}
SUPPORTED_DATASETS = {"ACL RD-TEC 2.0", "CL-RuTerm3", "GENIA"}
# Every model that dispatches to train_xlmr_span_classifier.py (the only trainer
# with --select-horizons). Single source for _dispatch_seed_command and for the
# horizon-aware --skip-existing check below.
SPAN_MODELS = {
    "xlmr_span_classifier", "rubert_span", "pubmedbert_span", "scibert_span",
    "mbert_span", "xlmr_span_unlabeled", "rubert_span_unlabeled", "ruroberta_span",
    "csroberta_span",
}

def safe_dataset_name(dataset):
    mapping = {
        "ACL RD-TEC 2.0": "acl-rd-tec-2",
        "CL-RuTerm3": "cl-ruterm3",
    }
    if dataset in mapping:
        return mapping[dataset]
    return re.sub(r"[^a-z0-9]+", "-", dataset.lower()).strip("-")


def parse_seeds(seed_values):
    seeds = []
    for value in seed_values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                seeds.append(int(part))
    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate seeds are not allowed: {seeds}")
    return seeds


def mbert_module():
    return "ate_paper.experiments.train_mbert_sequence_labeler"


def xlmr_bio_module():
    return "ate_paper.experiments.train_xlmr_bio_sequence_labeler"


def rubert_bio_module():
    return "ate_paper.experiments.train_rubert_bio_sequence_labeler"


def pubmedbert_bio_module():
    return "ate_paper.experiments.train_pubmedbert_bio_sequence_labeler"


def scibert_bio_module():
    return "ate_paper.experiments.train_scibert_bio_sequence_labeler"


def xlmr_module():
    return "ate_paper.experiments.train_xlmr_span_classifier"


def _jsonl_complete(path):
    """Non-empty and every line parses as JSON (a truncated last line fails)."""
    out = Path(path)
    if not (out.exists() and out.stat().st_size > 0):
        return False
    try:
        with open(out, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    json.loads(line)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        # UnicodeDecodeError (a truncated multi-byte UTF-8 char, e.g. Cyrillic
        # CL-RuTerm3 predictions cut off mid-character by an OOM kill) is a
        # ValueError subclass, not OSError -- must be listed explicitly or this
        # exact "incomplete file -> treat as absent, retrain" guard crashes
        # instead of doing what it exists to do.
        return False
    return True


def seed_output_present(paths, extra_outputs=()):
    """True when this seed already produced a non-empty, well-formed prediction file
    (and every file in extra_outputs, e.g. the _h<cap> horizon files).

    A crash during the trainer's final write_jsonl pass can leave a truncated last
    line; size alone would treat that as a finished seed and --skip-existing would
    never retrain it. Parsing every line catches that without needing an atomic-write
    change to the trainers themselves.
    """
    return all(_jsonl_complete(p) for p in [paths["output"], *extra_outputs])


def horizon_output_paths(args, paths):
    """The extra _h<cap>.jsonl files a span run with --select-horizons must have
    written (train_xlmr_span_classifier._with_h_suffix naming). The span trainer
    writes the un-suffixed output FIRST and the horizon files after it, so a kill in
    between leaves a seed that --skip-existing would otherwise treat as finished
    while its _h5/_h10 files are silently missing. Empty for BIO models and when
    the flag is unset."""
    value = getattr(args, "select_horizons", None)
    if not value or args.model not in SPAN_MODELS:
        return []
    caps = sorted({int(x) for x in str(value).split(",") if x.strip()})
    out = str(paths["output"])
    stem = out[:-len(".jsonl")] if out.endswith(".jsonl") else out
    return [Path(f"{stem}_h{cap}.jsonl") for cap in caps]


def seed_run_name(model, dataset, seed):
    return f"{display_model_name(model)}_{safe_dataset_name(dataset)}_seed{seed}"


def display_model_name(model):
    if model == "mbert_class":
        return "mbert_class_bio"
    if model == "mbert_unlabeled":
        return "mbert_unlabeled_bio"
    return model


def bio_label_mode_for_model(model):
    if model in {"mbert_unlabeled", "mbert_unlabeled_bio", "xlmr_unlabeled_bio"}:
        return "unlabeled"
    return "class"


def seed_paths(args, seed):
    name = seed_run_name(args.model, args.dataset, seed)
    return {
        "name": name,
        "output": Path(args.output_dir) / f"{name}.jsonl",
        "model_dir": Path(args.model_dir) / name,
        "report": Path(args.reports_dir) / f"{name}_report.md",
    }


def build_bio_command(args, seed, paths, script_path, label_mode="class"):
    command = [
        sys.executable,
        "-m",
        str(script_path),
        "--train-gold",
        args.train,
        "--dev-gold",
        args.dev,
        "--test-text",
        args.test_text,
        "--output",
        str(paths["output"]),
        "--model-dir",
        str(paths["model_dir"]),
        "--dataset",
        args.dataset,
        "--model-name",
        args.base_model,
        "--label-mode",
        label_mode,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--seed",
        str(seed),
        "--max-length",
        str(args.max_length),
        "--stride",
        str(args.stride),
        "--dev-thresholds",
        args.dev_thresholds,
        "--report",
        str(paths["report"]),
        "--device",
        args.device,
        "--log-every-n-batches",
        str(args.log_every_n_batches),
    ]
    if getattr(args, "amp", False):
        command.append("--amp")
    # ALWAYS pass explicitly, including 0.0 -- `if args.warmup_ratio:` treated the
    # default (0.0, flat LR) as falsy and silently dropped the flag, so the trainer
    # fell back to ITS OWN --warmup-ratio default instead of the value the caller
    # actually asked for. When that trainer default was briefly 0.1 (2026-07-27),
    # this is exactly how scibert_bio/pubmedbert_bio's E1 runs ended up trained at
    # warmup=0.1 while their own run summary (which prints args.warmup_ratio, not
    # what reached the trainer) still correctly reported 0.0. See
    # `reports/supervised/e1_class_granularity_train_side.md` for the incident this caused.
    command.extend(["--warmup-ratio", str(getattr(args, "warmup_ratio", 0.0))])
    # Modern-recipe knobs (all default to the published-panel behavior). The two
    # numeric ones use the same 0-means-off / explicit-default semantics as
    # --warmup-ratio and are ALWAYS passed for the same reason: a conditional
    # pass silently falls back to the trainer's own default, which is exactly
    # how the E1 warmup incident happened.
    command.extend(["--max-grad-norm", str(getattr(args, "max_grad_norm", 0.0))])
    command.extend(["--weight-decay", str(getattr(args, "weight_decay", 0.01))])
    if getattr(args, "no_decay_bias_layernorm", False):
        command.append("--no-decay-bias-layernorm")
    if args.length_balanced_positive_sampling:
        command.append("--length-balanced-positive-sampling")
        command.extend(["--long-span-token-threshold", str(args.long_span_token_threshold)])
        command.extend(["--long-positive-weight", str(args.long_positive_weight)])
    return command


def build_mbert_command(args, seed, paths):
    return build_bio_command(
        args,
        seed,
        paths,
        mbert_module(),
        label_mode=bio_label_mode_for_model(args.model),
    )


def build_xlmr_bio_command(args, seed, paths):
    return build_bio_command(
        args,
        seed,
        paths,
        xlmr_bio_module(),
        label_mode=bio_label_mode_for_model(args.model),
    )


def build_rubert_bio_command(args, seed, paths):
    return build_bio_command(
        args,
        seed,
        paths,
        rubert_bio_module(),
        label_mode=bio_label_mode_for_model(args.model),
    )


def build_pubmedbert_bio_command(args, seed, paths):
    return build_bio_command(
        args,
        seed,
        paths,
        pubmedbert_bio_module(),
        label_mode=bio_label_mode_for_model(args.model),
    )


def build_scibert_bio_command(args, seed, paths):
    return build_bio_command(
        args,
        seed,
        paths,
        scibert_bio_module(),
        label_mode=bio_label_mode_for_model(args.model),
    )


# Report titles for BIO configs that run through the base trainer directly (no
# encoder-named wrapper). The title is cosmetic -- every report also records
# `- model name: <--model-name>` -- but a "mBERT ... Report" header over a
# cs_roberta run would be a provenance trap for the next reader.
BIO_REPORT_TITLES = {
    "csroberta_bio": "cs_roberta_base BIO Sequence Labeler Report",
}


def build_generic_bio_command(args, seed, paths):
    """BIO through train_mbert_sequence_labeler.py itself: the encoder comes from
    --model-name (args.base_model), the header from BIO_REPORT_TITLES."""
    command = build_bio_command(
        args,
        seed,
        paths,
        mbert_module(),
        label_mode=bio_label_mode_for_model(args.model),
    )
    return command + ["--report-title", BIO_REPORT_TITLES[args.model]]


def build_xlmr_command(args, seed, paths):
    command = [
        sys.executable,
        "-m",
        xlmr_module(),
        "--train",
        args.train,
        "--dev",
        args.dev,
        "--predict",
        args.test_text,
        "--output_dir",
        str(paths["model_dir"]),
        "--model_name",
        args.base_model,
        "--output",
        str(paths["output"]),
        "--report",
        str(paths["report"]),
        "--dataset",
        args.dataset,
        "--max_span_tokens",
        str(args.max_span_tokens),
        "--negative_ratio",
        str(args.negative_ratio),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--seed",
        str(seed),
        "--max_length",
        str(args.max_length),
        "--stride",
        str(args.stride),
        "--dev-thresholds",
        args.dev_thresholds,
        "--device",
        args.device,
        "--output-format",
        "unified",
        "--log-every-n-batches",
        str(args.log_every_n_batches),
    ]
    if getattr(args, "amp", False):
        command.append("--amp")
    # ALWAYS pass explicitly -- see build_bio_command's identical fix for why a
    # conditional `if args.warmup_ratio:` silently drops 0.0 (falsy) and lets the
    # trainer fall back to its own default instead of the caller's request.
    command.extend(["--warmup-ratio", str(getattr(args, "warmup_ratio", 0.0))])
    # Horizon reuse (span trainer only): one long run also predicts best<=cap for each
    # smaller cap, so best<=5 and best<=15 come from one training of the shared early
    # epochs. Pass-through only; absent by default (no behavior change).
    if getattr(args, "select_horizons", None):
        command.extend(["--select-horizons", str(args.select_horizons)])
    if args.length_balanced_positive_sampling:
        command.append("--length-balanced-positive-sampling")
        command.extend(["--long-span-token-threshold", str(args.long_span_token_threshold)])
        command.extend(["--long-positive-weight", str(args.long_positive_weight)])
    if args.threshold is not None:
        command.extend(["--threshold", str(args.threshold)])
    if args.top_k_per_doc is not None:
        command.extend(["--top_k_per_doc", str(args.top_k_per_doc)])
    if getattr(args, "label_list", None):
        command.extend(["--label-list", args.label_list])
    return command


def _dispatch_seed_command(args, seed, paths):
    if args.model in {"mbert_class", "mbert_class_bio", "mbert_unlabeled", "mbert_unlabeled_bio"}:
        return build_mbert_command(args, seed, paths)
    if args.model in {"xlmr_bio", "xlmr_unlabeled_bio"}:
        return build_xlmr_bio_command(args, seed, paths)
    if args.model in {"rubert_bio", "ruroberta_bio"}:
        return build_rubert_bio_command(args, seed, paths)
    if args.model == "pubmedbert_bio":
        return build_pubmedbert_bio_command(args, seed, paths)
    if args.model == "scibert_bio":
        return build_scibert_bio_command(args, seed, paths)
    if args.model in BIO_REPORT_TITLES:
        return build_generic_bio_command(args, seed, paths)
    if args.model in SPAN_MODELS:
        # The span trainer does not implement the modern-recipe knobs yet.
        # Refuse loudly instead of silently training without them -- a span run
        # that LOOKS like the modern recipe but trained without clipping /
        # no-decay groups would poison the pilot's comparison.
        modern = []
        if float(getattr(args, "max_grad_norm", 0.0) or 0.0) > 0:
            modern.append("--max-grad-norm")
        if float(getattr(args, "weight_decay", 0.01)) != 0.01:
            modern.append("--weight-decay")
        if getattr(args, "no_decay_bias_layernorm", False):
            modern.append("--no-decay-bias-layernorm")
        if modern:
            raise ValueError(
                f"{args.model} dispatches to train_xlmr_span_classifier.py, which does not "
                f"support {', '.join(modern)}. The modern-recipe pilot is BIO-only for now; "
                "extend the span trainer first if a span arm is needed."
            )
        return build_xlmr_command(args, seed, paths)
    raise ValueError(f"Unsupported model: {args.model}")


def build_seed_command(args, seed, paths):
    # Fail before a child process/model import if SciBERT casing was not chosen.
    tokenizer_kwargs(args.base_model, getattr(args, "scibert_do_lower_case", None))
    command = _dispatch_seed_command(args, seed, paths)
    if getattr(args, "model_revision", None):
        command.extend(["--model-revision", args.model_revision])
    if getattr(args, "scibert_do_lower_case", None) is not None:
        command.extend(["--scibert-do-lower-case", args.scibert_do_lower_case])
    # optional dev-selection collapse (E1 arm A); both BIO and span trainers accept it
    if getattr(args, "dev_eval_collapse", None):
        command = command + ["--dev-eval-collapse", args.dev_eval_collapse]
    return command


def parse_report_value(report_path, key):
    path = Path(report_path)
    if not path.exists():
        return None
    prefix = f"- {key}:"
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped.split(":", 1)[1].strip().strip("`")
    return None


def evaluation_pred_args(paths_by_seed):
    return " \\\n".join(
        f"    {paths['name']}:{paths['output']}" for _, paths in paths_by_seed
    )


def write_summary(args, seeds, paths_by_seed, summary_path, failed_seeds=None):
    failed_seeds = failed_seeds or []
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Supervised Seed Variance Runs",
        "",
        f"- model: {display_model_name(args.model)}",
        f"- dataset: {args.dataset}",
        f"- base model: {args.base_model}",
        f"- model revision: {getattr(args, 'model_revision', None)}",
        f"- scibert do lower case choice: {getattr(args, 'scibert_do_lower_case', None)}",
        f"- train file: {args.train}",
        f"- dev file: {args.dev}",
        f"- test text file: {args.test_text}",
        f"- seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- epochs: {args.epochs}",
        f"- batch size: {args.batch_size}",
        f"- learning rate: {args.learning_rate}",
        f"- max length: {args.max_length}",
        f"- stride: {args.stride}",
        f"- dev thresholds: {args.dev_thresholds}",
        f"- device: {args.device}",
        f"- amp: {getattr(args, 'amp', False)}",
        f"- warmup ratio: {getattr(args, 'warmup_ratio', 0.0)}",
        f"- max grad norm: {getattr(args, 'max_grad_norm', 0.0)}",
        f"- weight decay: {getattr(args, 'weight_decay', 0.01)}",
        f"- no-decay bias/LayerNorm: {getattr(args, 'no_decay_bias_layernorm', False)}",
        f"- length-balanced positive sampling: {args.length_balanced_positive_sampling}",
        f"- long span token threshold: {args.long_span_token_threshold}",
        f"- long positive weight: {args.long_positive_weight}",
        f"- log every n batches: {args.log_every_n_batches}",
        "",
        "## Runs",
        "",
        "| Seed | Status | Prediction | Model Dir | Report | Prediction Rows |",
        "|---:|---|---|---|---|---:|",
    ]
    for seed, paths in paths_by_seed:
        rows = parse_report_value(paths["report"], "prediction rows written")
        rows = rows if rows is not None else ""
        status = "FAILED" if seed in failed_seeds else "OK"
        lines.append(
            f"| {seed} | {status} | `{paths['output']}` | `{paths['model_dir']}` | "
            f"`{paths['report']}` | {rows} |"
        )

    if failed_seeds:
        lines.extend([
            "",
            "## Failed Seeds",
            "",
            f"{len(failed_seeds)} of {len(seeds)} seed(s) crashed and have NO predictions -- "
            "do not fold them into the mean/std evaluation below. Fix the underlying cause "
            "and rerun with --skip-existing to fill these in without redoing the seeds "
            "marked OK above.",
            "",
        ])
        lines.extend(f"- seed {seed}" for seed in failed_seeds)

    lines.extend([
        "",
        "## Next step",
        "",
        "The run writes predictions, model bundles and reports only. Follow the repository's",
        "reproduction instructions to prepare evaluation inputs and run analysis separately.",
    ])
    Path(summary_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_seed_variance(args):
    if args.dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset: {args.dataset}. "
            f"Supported: {', '.join(sorted(SUPPORTED_DATASETS))}."
        )
    if args.model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {args.model}")
    if args.long_span_token_threshold < 1:
        raise ValueError("--long-span-token-threshold must be >= 1.")
    if args.long_positive_weight < 1:
        raise ValueError("--long-positive-weight must be >= 1.")
    if args.log_every_n_batches < 0:
        raise ValueError("--log-every-n-batches must be >= 0.")

    seeds = parse_seeds(args.seeds)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.reports_dir).mkdir(parents=True, exist_ok=True)

    paths_by_seed = []
    failed_seeds = []
    for seed in seeds:
        paths = seed_paths(args, seed)
        paths_by_seed.append((seed, paths))
        if args.dry_run:
            continue
        if getattr(args, "skip_existing", False) and seed_output_present(
                paths, extra_outputs=horizon_output_paths(args, paths)):
            # Opt-in per-seed resume. Without it an interrupted multi-seed job
            # retrains every seed from scratch on the next attempt, which on a
            # metered GPU is the difference between losing one seed and losing three.
            print(f"Skipping seed {seed}: predictions already at {paths['output']}", flush=True)
            continue
        command = build_seed_command(args, seed, paths)
        # Preserve the installed/staged ate_paper package for child module calls.
        # This source layout has the package under src/; no old project root is used.
        _child_env = dict(os.environ)
        _repo_root = str(Path(__file__).resolve().parents[2])
        _child_env["PYTHONPATH"] = _repo_root + (
            os.pathsep + _child_env["PYTHONPATH"] if _child_env.get("PYTHONPATH") else "")
        try:
            subprocess.run(command, check=True, env=_child_env)
        except subprocess.CalledProcessError as exc:
            # Continue rather than fail-fast, matching run_ate_it_supervised.py: on a
            # multi-seed metered-GPU matrix, one bad seed (transient OOM, a flaky
            # download) shouldn't cost the seeds that would have succeeded after it.
            # --skip-existing already exists precisely to resume past exactly this.
            print(f"!! {paths['name']} failed (exit {exc.returncode}); continuing", flush=True)
            failed_seeds.append(seed)

    summary_path = Path(args.reports_dir) / (
        f"seed_variance_{args.model}_{safe_dataset_name(args.dataset)}_summary.md"
    )
    write_summary(args, seeds, paths_by_seed, summary_path, failed_seeds=failed_seeds)
    if failed_seeds:
        print(f"!! {len(failed_seeds)} seed(s) failed -- see '## Failed Seeds' in {summary_path}", flush=True)
    return paths_by_seed, summary_path, failed_seeds


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run supervised baselines over multiple seeds.")
    parser.add_argument("--model", required=True, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--dataset", required=True, choices=sorted(SUPPORTED_DATASETS))
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--test-text", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", default="build/experiments/supervised/predictions")
    parser.add_argument("--model-dir", default="build/experiments/supervised/models")
    parser.add_argument("--reports-dir", default="build/experiments/supervised/reports")
    parser.add_argument("--select-horizons", default=None,
                        help="Span trainer only: comma-separated epoch caps (e.g. '5,10,15'). "
                             "One long run additionally predicts best<=cap for each cap, "
                             "written as _h<cap> next to --output. Lets best<=5 and best<=15 "
                             "come from one training of the shared early epochs. Default off.")
    parser.add_argument("--seeds", nargs="+", default=["42", "43", "44"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--dev-thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--dev-eval-collapse", default=None,
                        help="collapse dev class labels for threshold/best-epoch "
                             "selection only (e.g. genia_coarse6 for E1 arm A)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--amp", action="store_true",
                        help="Mixed-precision training (pass-through to the BIO/span "
                             "trainers). No-op off CUDA.")
    parser.add_argument("--warmup-ratio", type=float, default=0.0,
                        help="Fraction of total steps for linear LR warmup, then linear "
                             "decay to 0 (pass-through to the BIO/span trainers). Default 0.0 (flat LR, no warmup). Was 0.1 (10 percent, the standard BERT fine-tuning ratio) from 2026-07-27 to 2026-07-28, reverted after concluding the per-entry-point bookkeeping this required (tracking which runs/dates used which default) cost more than a literature value never validated on this project's own data was worth; every already-published result was flat-LR anyway.")
    parser.add_argument("--max-grad-norm", type=float, default=0.0,
                        help="Pass-through to the BIO trainers: clip the global L2 grad "
                             "norm (0 = off, the published-panel behavior). BIO-only; a "
                             "span model with a non-zero value fails loudly.")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Pass-through to the BIO trainers: AdamW weight decay. "
                             "Default 0.01 = torch's AdamW default, which published "
                             "panels already used implicitly. BIO-only.")
    parser.add_argument("--no-decay-bias-layernorm", action="store_true",
                        help="Pass-through to the BIO trainers: exclude bias/LayerNorm "
                             "from weight decay (HF-Trainer convention). OFF by default "
                             "(published panels decayed every parameter). BIO-only.")
    parser.add_argument("--max-span-tokens", type=int, default=10)
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--length-balanced-positive-sampling", action="store_true")
    parser.add_argument("--long-span-token-threshold", type=int, default=4)
    parser.add_argument("--long-positive-weight", type=int, default=3)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--top-k-per-doc", type=int)
    parser.add_argument("--label-list", default=None,
                        help="Span-classifier models only: forwarded to "
                             "train_xlmr_span_classifier.py's --label-list override. "
                             "See that flag's help for why (a training arm that removes "
                             "every example of one class must not silently drop it from "
                             "its own label space -- the E1/E6 softmax-width confound).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip a seed whose prediction file already exists and is non-empty. OFF by "
             "default on purpose: if the training gold changed, a silent skip would reuse "
             "stale predictions. Turn it on for resuming an interrupted GPU run.")
    parser.add_argument("--log-every-n-batches", type=int, default=50)
    add_pretrained_arguments(parser)
    return parser


def main():
    args = build_arg_parser().parse_args()
    _, summary_path, failed_seeds = run_seed_variance(args)
    print(f"Saved seed variance summary: {summary_path}")
    if failed_seeds:
        # Continue-on-error means the loop above already ran every seed it could,
        # but a caller that only checks this process's exit code (e.g.
        # run_e1_kaggle.py's `if rc == 0`) must still be able to tell a partial
        # failure apart from a full success -- returning 0 unconditionally here
        # would silently report a job with failed seeds as fully done.
        sys.exit(1)


if __name__ == "__main__":
    main()
