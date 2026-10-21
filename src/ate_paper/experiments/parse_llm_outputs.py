#!/usr/bin/env python3
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ate_paper.experiments.llm_run_paths import configure_run_paths

from ate_paper.experiments import make_llm_predictions as llm


ACTER_CLASS_PRIORITY = {
    "Specific": 0,
    "Common": 1,
    "OOD": 2,
    "NE": 3,
    None: 4,
}


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_failure(path, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_meta(batch_dir, batch_id):
    return json.loads((Path(batch_dir) / f"{batch_id}_meta.json").read_text(encoding="utf-8"))


def raw_files(raw_dir):
    return sorted(path for path in Path(raw_dir).glob("batch_*_raw.json") if "_attempt_" not in path.name)


def expected_batch_ids(batch_dir):
    return sorted(path.name.replace("_meta.json", "") for path in Path(batch_dir).glob("batch_*_meta.json"))


def batch_id_from_raw(path):
    return path.name.replace("_raw.json", "")






def count_invalid_class_labels(task, parsed, genia_scheme=None):
    allowed = set({"acl_span_class": llm.ACL_CLASSES,
                   "genia_span_class": llm.GENIA_CLASSES,
                   "cl_ruterm3_span_class": llm.CL_RUTERM3_CLASSES}[task])
    return sum(1 for _doc, _sent, item in llm.iter_sentence_terms(task, parsed)
               if (item.get("class_label") if isinstance(item, dict) else None) not in allowed)



def parse_raw_outputs(args):
    configure_run_paths(args, ('output', 'failed_output', 'report', 'metadata_dir'))
    started_at = datetime.now(timezone.utc)
    output_rows = []
    stats = Counter()
    if getattr(args, "legacy_exact_headers", False):
        stats["_legacy_exact_headers"] = 1
    failures = []
    first_meta = None
    Path(args.failed_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.failed_output).write_text("", encoding="utf-8")

    raw_by_batch = {batch_id_from_raw(path): path for path in raw_files(args.raw_dir)}
    ids = expected_batch_ids(args.batch_dir) or sorted(raw_by_batch)
    for batch_id in ids:
        raw_path = raw_by_batch.get(batch_id)
        if raw_path is None:
            stats["missing_raw_responses"] += 1
            failure = {"batch_id": batch_id, "failure_type": "missing_raw_response", "message": "No raw response file found for expected batch."}
            append_failure(args.failed_output, failure)
            failures.append(failure)
            continue
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        meta = load_meta(args.batch_dir, batch_id)
        first_meta = first_meta or meta
        batch_rows = meta.get("input_rows", [])
        if raw.get("status") != "ok":
            stats["api_failed_batches"] += 1
            failure = {"batch_id": batch_id, "failure_type": "api_error", "message": raw.get("error_message", "API call failed")}
            append_failure(args.failed_output, failure)
            failures.append(failure)
            continue

        text = raw.get("response_text")
        if text is None:
            response_path = Path(args.raw_dir) / f"{batch_id}_response.txt"
            text = response_path.read_text(encoding="utf-8") if response_path.exists() else ""

        # Flag output-cap saturation so truncation is distinguishable from a plain
        # JSON-contract failure (review L-transport-1: the cap was documented as
        # "inert" while 17 committed batches actually saturated it).
        completion_tokens = (raw.get("usage") or {}).get("completion_tokens") or (raw.get("usage") or {}).get("output_tokens")
        cap = raw.get("max_output_tokens")
        cap_hit = bool(cap) and bool(completion_tokens) and completion_tokens >= cap
        if cap_hit:
            stats["output_cap_hits"] += 1

        try:
            parsed = llm.parse_json_response(text)
        except ValueError as exc:
            stats["failed_json_parses"] += 1
            if cap_hit:
                stats["failed_json_parses_at_cap"] += 1
            failure = {"batch_id": batch_id, "failure_type": "invalid_json_at_output_cap" if cap_hit else "invalid_json", "message": str(exc), "response_preview": text[:500]}
            append_failure(args.failed_output, failure)
            failures.append(failure)
            continue

        require_context = args.task == "cl_ruterm3_span_class" and meta.get("run_mode") == "full"
        genia_scheme = (meta.get("ablation") or {}).get("class_scheme")
        try:
            llm.validate_parsed(args.task, parsed, require_context=require_context)
        except ValueError as exc:
            stats["invalid_schemas"] += 1
            failure = {"batch_id": batch_id, "failure_type": "invalid_schema", "message": str(exc), "parsed_response": parsed}
            append_failure(args.failed_output, failure)
            failures.append(failure)
            continue

        try:
            stats["invalid_class_labels"] += count_invalid_class_labels(args.task, parsed, genia_scheme)
        except (AttributeError, TypeError):
            # Stats-only counter: a malformed structure it cannot walk must not
            # abort the run or drop the batch -- postprocess below is the gate.
            pass
        before_unmatched = stats.get("unmatched_surface", 0)
        before_multiple = stats.get("multiple_match_surfaces", 0)
        before_context_unmatched = stats.get("context_unmatched_surface", 0)
        before_context_disambiguated = stats.get("context_disambiguated_surfaces", 0)
        if require_context:
            stats["_require_context"] = 1
        else:
            stats.pop("_require_context", None)
        if genia_scheme:
            stats["_genia_class_scheme"] = genia_scheme
        else:
            stats.pop("_genia_class_scheme", None)
        try:
            rows_now, duplicates = llm.postprocess(args.task, batch_rows, parsed, stats, batch_id=batch_id)
        except (ValueError, AttributeError, TypeError) as exc:
            # AttributeError/TypeError: model output whose SHAPE validate_parsed
            # tolerates but postprocess cannot walk (e.g. bare-string terms before
            # the coercions landed) used to abort the ENTIRE parse run here; record
            # it as a per-batch failure like any other (review L-parsing-3).
            stats["invalid_schemas"] += 1
            failure = {"batch_id": batch_id, "failure_type": "postprocess_error", "message": str(exc), "parsed_response": parsed}
            append_failure(args.failed_output, failure)
            failures.append(failure)
            continue

        stats["successful_batches"] += 1
        stats["duplicates_removed"] += duplicates
        stats["unmatched_surface_count"] += stats.get("unmatched_surface", 0) - before_unmatched
        stats["multiple_match_surface_count"] += stats.get("multiple_match_surfaces", 0) - before_multiple
        stats["context_unmatched_surface_count"] += stats.get("context_unmatched_surface", 0) - before_context_unmatched
        stats["context_disambiguated_surface_count"] += stats.get("context_disambiguated_surfaces", 0) - before_context_disambiguated
        output_rows.extend(rows_now)

    output_rows, duplicates = llm.deduplicate(output_rows, lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True))
    stats["duplicates_removed"] += duplicates
    stats["prediction_rows"] = len(output_rows)
    write_jsonl(output_rows, args.output)

    report = write_report(args, first_meta or {}, stats)
    write_metadata(args, first_meta or {}, started_at, datetime.now(timezone.utc), report)
    return output_rows, stats, failures


def token_totals(raw_dir):
    totals = Counter()
    for raw_path in raw_files(raw_dir):
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        usage = raw.get("usage") or {}
        estimated = raw.get("estimated_usage") or {}
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
        output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")
        if not prompt_tokens and estimated:
            totals["approximate_batches"] += 1
            prompt_tokens = estimated.get("prompt_tokens")
            output_tokens = output_tokens or estimated.get("completion_tokens")
            total_tokens = total_tokens or estimated.get("total_tokens")
        totals["input"] += prompt_tokens or 0
        totals["output"] += output_tokens or 0
        totals["total"] += total_tokens or 0
    return totals






def write_report(args, meta, stats):
    # The default report is local to the selected run.
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    batches = len(expected_batch_ids(args.batch_dir)) or len(raw_files(args.raw_dir))
    valid_json = batches - stats.get("failed_json_parses", 0) - stats.get("api_failed_batches", 0) - stats.get("missing_raw_responses", 0)
    valid_rate = valid_json / batches if batches else 0
    usage = token_totals(args.raw_dir)
    estimated_full = usage["total"]
    if meta.get("run_mode") == "pilot" and meta.get("pilot_frac") and usage["total"]:
        estimated_full = int(usage["total"] / float(meta["pilot_frac"]))
    ready = (
        stats.get("failed_json_parses", 0) == 0
        and stats.get("invalid_schemas", 0) == 0
        and stats.get("api_failed_batches", 0) == 0
        and stats.get("missing_raw_responses", 0) == 0
    )
    lines = [
        "# LLM Parse Report",
        "",
        f"- model: {args.model}",
        f"- header_rule: {'legacy_exact' if getattr(args, 'legacy_exact_headers', False) else 'current_tolerant'}",
        f"- provider: {args.provider or 'unknown'}",
        f"- task: {args.task}",
        f"- prompt_version: {meta.get('prompt_version', 'unknown')}",
        f"- pilot fraction: {meta.get('pilot_frac')}",
        f"- pilot seed: {meta.get('pilot_seed')}",
        f"- requested_batch_size: {meta.get('requested_batch_size', meta.get('batch_size', 'unknown'))}",
        f"- effective_items_per_batch: {meta.get('effective_items_per_batch', 'unknown')}",
        f"- number of input items: {sum((json.loads(path.read_text(encoding='utf-8')).get('meta') or {}).get('num_items', 0) for path in raw_files(args.raw_dir))}",
        f"- number of batches: {batches}",
        f"- missing raw response count: {stats.get('missing_raw_responses', 0)}",
        f"- valid JSON rate: {valid_rate:.3f}",
        f"- parse failure count: {stats.get('failed_json_parses', 0)}",
        f"- output-cap saturated batches (completion_tokens >= cap): {stats.get('output_cap_hits', 0)}",
        f"- parse failures at the output cap (truncation, not contract): {stats.get('failed_json_parses_at_cap', 0)}",
        f"- schema failure count: {stats.get('invalid_schemas', 0)}",
        f"- unmatched surface count: {stats.get('unmatched_surface', 0)}",
        f"- unknown source sentence count (terms dropped): {stats.get('unknown_source_sentence', 0)}",
        f"- recovered source sentence count (tolerant header match): {stats.get('recovered_source_sentence', 0)}",
        f"- multiple match surface count: {stats.get('multiple_match_surfaces', 0)}",
        f"- context unmatched surface count: {stats.get('context_unmatched_surface', 0)}",
        f"- context disambiguated surface count: {stats.get('context_disambiguated_surfaces', 0)}",
        f"- context still ambiguous count (several offsets kept): {stats.get('context_still_ambiguous', 0)}",
        f"- invalid class label count: {stats.get('invalid_class_labels', 0)}",
        f"- duplicate predictions removed: {stats.get('duplicates_removed', 0)}",
        f"- ambiguous predicted classes: {stats.get('ambiguous_predicted_classes', 0)}",
        f"- ungrounded surface count (list arms): {stats.get('ungrounded_surface', 0)}",
        f"- cross-language ambiguous grounding (list arms): {stats.get('cross_language_ambiguous_grounding', 0)}",
        f"- same-batch class conflict (typed list arm): {stats.get('same_batch_class_conflict', 0)}",
        f"- input token count: {usage['input'] if usage['input'] else 'token_usage_unavailable'}",
        f"- output token count: {usage['output'] if usage['output'] else 'token_usage_unavailable'}",
        f"- total token count: {usage['total'] if usage['total'] else 'token_usage_unavailable'}",
        f"- approximate token usage batches: {usage['approximate_batches']}",
        f"- estimated full-run tokens: {estimated_full if estimated_full else 'token_usage_unavailable'}",
        "- estimated full-run cost: pricing_config_unavailable",
        f"- ready_for_full_run: {'yes' if ready else 'no'}",
        "",
    ]
    if ready:
        lines.append("The pilot is safe to scale to a full run.")
    else:
        lines.append("The pilot is not safe to scale until failures are inspected.")
    if meta.get("batching_note"):
        lines.extend(["", "## Batching Note", "", meta["batching_note"]])
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
        "prompt_version": meta.get("prompt_version"),
        "prompt_hash": meta.get("prompt_hash"),
        "parser_header_rule": "legacy_exact" if getattr(args, "legacy_exact_headers", False) else "current_tolerant",
        "temperature": None,
        "max_output_tokens": None,
        "pilot_frac": meta.get("pilot_frac"),
        "pilot_seed": meta.get("pilot_seed"),
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "batch_dir": args.batch_dir,
        "raw_output_dir": args.raw_dir,
        "prediction_output": args.output,
        # Stamped at parse time so integrity check C6 can verify the prediction
        # file on disk is the one this record describes (review L-validation-3:
        # the field was designed for C6 but never written by any writer).
        "prediction_sha256": (hashlib.sha256(Path(args.output).read_bytes()).hexdigest()
                              if Path(args.output).exists() else None),
        "report": str(report),
    }
    # Fill decoding params (temperature/top_p/seed/api_model/base_url/max_output_tokens)
    # from the run writer's record instead of nulling them out -- parse runs last and
    # would otherwise clobber the run's frozen-decoding provenance.
    record = llm.merge_prior_metadata(record, out)
    kept = llm.preserve_superseded_metadata(out, record)
    if kept:
        print(f"[metadata] {out.name} already held a record of DIFFERENT work; "
              f"kept it as {kept.name} rather than overwriting it")
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--task", choices=sorted(llm.SUPPORTED_TASKS), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--failed-output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--legacy-exact-headers", action="store_true",
                        help="Use historical exact sentence headers; disable tolerant recovery.")
    parser.add_argument("--metadata-dir")
    args = parser.parse_args()
    rows, stats, _ = parse_raw_outputs(args)
    print(f"Prediction rows: {len(rows)}")
    print(f"Parse failures: {stats.get('failed_json_parses', 0)}")


if __name__ == "__main__":
    main()
