#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

from ate_paper.experiments.llm_run_paths import configure_run_paths

from ate_paper.experiments import make_llm_predictions as llm


def model_dir_name(model):
    name = str(model)
    if name in {"gemini-2.5-flash", "gemini_2_5_flash"}:
        return "gemini_2_5_flash"
    if name in {"gemini-3.1-flash-lite", "gemini-3-1-flash-lite", "gemini_3_1_flash_lite"}:
        return "gemini_3_1_flash_lite"
    if name in {"qwen-3-6-plus", "qwen3_6_plus", "qwen3.6-plus"}:
        return "qwen3_6_plus"
    if name in {"deepseek-v4-pro", "deepseek_v4_pro"}:
        return "deepseek_v4_pro"
    if name in {"deepseek-v4-flash", "deepseek_v4_flash"}:
        return "deepseek_v4_flash"
    if name in {"zai-org/GLM-5.2", "glm_5_2"}:
        return "glm_5_2"
    if name in {"Qwen/Qwen3-235B-A22B-Instruct-2507-FP8", "qwen3_235b_a22b_instruct_2507"}:
        return "qwen3_235b_a22b_instruct_2507"
    if name in {"openai/gpt-oss-120b", "gpt_oss_120b"}:
        return "gpt_oss_120b"
    if name in {"meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "llama_4_maverick"}:
        return "llama_4_maverick"
    if name in {"mistral-large-2512", "mistral_large_3"}:
        return "mistral_large_3"
    if name in {"claude-haiku-4-5-20251001", "claude_haiku_4_5"}:
        return "claude_haiku_4_5"
    return llm.model_slug(name)


def load_task_rows(args):
    rows = llm.read_input_rows(args.input)
    return llm.reconstruct_text_units(rows)



def deterministic_sample(rows, pilot_frac=0.1, pilot_max_items=None, pilot_seed=42):
    rows = list(rows)
    if not rows:
        return []
    frac_count = max(1, math.ceil(len(rows) * pilot_frac))
    target = frac_count
    if pilot_max_items is not None:
        target = min(target, pilot_max_items)
    target = min(target, len(rows))
    rng = random.Random(pilot_seed)
    indices = sorted(rng.sample(range(len(rows)), target))
    return [rows[index] for index in indices]






def prepare_rows(args, rows):
    if not args.pilot:
        return rows
    return deterministic_sample(rows, args.pilot_frac, args.pilot_max_items, args.pilot_seed)


def source_ids(rows):
    return [llm.input_id(row) for row in rows]


def ablation_arm(class_detail, nested_policy):
    """Path segment for a non-default ablation arm; None for the base arm so the
    default output path is unchanged."""
    if class_detail == "labels" and nested_policy == "include":
        return None
    return f"class_{class_detail}__nested_{nested_policy}"


def batch_output_dir(output_dir, model, task, run_mode, prompt_version, arm=None):
    output_dir = Path(output_dir)
    model_name = model_dir_name(model)
    base = output_dir if output_dir.name == model_name else output_dir / model_name
    base = base / task / run_mode / prompt_version
    return base / arm if arm else base


def write_batches(args):
    configure_run_paths(args, ("output_dir",))
    is_cl = args.task == "cl_ruterm3_span_class"
    is_genia = args.task == "genia_span_class"
    is_acl = args.task == "acl_span_class"
    class_detail = getattr(args, "class_detail", "labels")
    nested_policy = getattr(args, "nested_policy", "include")
    class_scheme = getattr(args, "class_scheme", "full")
    if nested_policy != "include" and not (is_cl or is_genia):
        raise SystemExit("--nested-policy ablation is only defined for cl_ruterm3_span_class and genia_span_class.")
    if class_detail != "labels" and not (is_cl or is_acl):
        raise SystemExit("--class-detail ablation is only defined for cl_ruterm3_span_class and acl_span_class.")
    if class_scheme != "full" and not is_genia:
        raise SystemExit("--class-scheme ablation is only defined for genia_span_class.")
    shots = getattr(args, "shots", 0)
    if shots and not (is_cl or is_genia):
        raise SystemExit("--shots is only defined for cl_ruterm3_span_class and genia_span_class.")
    ablation = None
    arm = None
    if is_cl:
        ablation = {"class_detail": class_detail, "nested_policy": nested_policy}
        arm = ablation_arm(class_detail, nested_policy)
    elif is_acl and class_detail != "labels":
        ablation = {"class_detail": class_detail}
        arm = f"class_{class_detail}"
    elif is_genia and (class_scheme != "full" or nested_policy != "include"):
        ablation = {"class_scheme": class_scheme, "nested_policy": nested_policy}
        parts = []
        if class_scheme != "full":
            parts.append(f"scheme_{class_scheme}")
        if nested_policy != "include":
            parts.append(f"nested_{nested_policy}")
        arm = "__".join(parts)
    if shots:
        arm = f"{arm}__shots_{shots}" if arm else f"shots_{shots}"
    loaded_rows, preparation_warnings = load_task_rows(args)
    rows = prepare_rows(args, loaded_rows)
    run_mode = "pilot" if args.pilot else "full"
    out_dir = batch_output_dir(args.output_dir, args.model, args.task, run_mode, args.prompt_version, arm)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = llm.build_system_prompt(args.task, args.prompt_version, run_mode, ablation=ablation, shots=shots)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    created_at = datetime.now(timezone.utc).isoformat()

    batch_count = 0
    span_single_item_batching = args.task == "cl_ruterm3_span_class"
    for batch_number, batch_rows, payload in llm.build_batches(args.task, rows, args.batch_size):
        batch_id = f"batch_{batch_number:04d}"
        (out_dir / f"{batch_id}_prompt.txt").write_text(prompt, encoding="utf-8")
        (out_dir / f"{batch_id}_input.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        meta = {
            "model": model_dir_name(args.model),
            "api_model": args.model,
            "task": args.task,
            "run_mode": run_mode,
            "batch_id": batch_id,
            "prompt_version": args.prompt_version,
            "prompt_hash": prompt_hash,
            "num_items": len(batch_rows),
            "source_ids": source_ids(batch_rows),
            "created_at": created_at,
            "pilot_seed": args.pilot_seed if args.pilot else None,
            "pilot_frac": args.pilot_frac if args.pilot else None,
            "batch_size": args.batch_size,
            "requested_batch_size": args.batch_size,
            "effective_items_per_batch": len(batch_rows),
            "input_rows": batch_rows,
            "preparation_warnings": preparation_warnings,
        }
        if ablation:
            meta["ablation"] = dict(ablation)
        if shots:
            meta["shots"] = shots
        if span_single_item_batching:
            meta["batching_note"] = (
                "Span-task batching currently uses one text unit per API request. "
                "requested_batch_size is the requested maximum, not the realized batch size."
            )
        (out_dir / f"{batch_id}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        batch_count += 1

    return out_dir, len(rows), batch_count, preparation_warnings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", choices=sorted(llm.SUPPORTED_TASKS), required=True)
    parser.add_argument("--input")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--prompt-version", default=llm.PROMPT_VERSION_DEFAULT)
    parser.add_argument("--nested-policy", choices=list(llm.NESTED_POLICIES), default="include",
                        help="cl_ruterm3 ablation: nested-term inclusion (targets NestedGap).")
    parser.add_argument("--class-detail", choices=list(llm.CLASS_DETAILS), default="labels",
                        help="cl_ruterm3 ablation: class block richness (targets ClassGap).")
    parser.add_argument("--class-scheme", choices=list(llm.GENIA_CLASS_SCHEMES), default="full",
                        help="genia ablation: class-scheme granularity 36 (full) vs 6 (coarse6); targets ClassGap.")
    parser.add_argument("--shots", type=int, choices=[0], default=0,
                        help="Number of demonstrations; only zero-shot extraction is supported.")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-frac", type=float, default=0.1)
    parser.add_argument("--pilot-max-items", type=int)
    parser.add_argument("--pilot-seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input:
        raise SystemExit("--input is required.")
    out_dir, item_count, batch_count, preparation_warnings = write_batches(args)
    print(f"Items: {item_count}")
    print(f"Batches: {batch_count}")
    print(f"Batch dir: {out_dir}")
    for warning in preparation_warnings:
        print(warning)


if __name__ == "__main__":
    main()
