"""Exact full-test F1 movement for the article's 25 selected LLM pairs.

Scores exact span sets using precision, recall and F1.
CL is scored against its prepared current test reference.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from ate_paper.predictions import CorpusIndex, read_jsonl, sha256
from .io import SLUGS, read_csv

DECISIONS = {"primary_raw", "primary_documented", "transport_contrast"}
EXPECTED_ALL = {"acl": 8, "genia": 9, "cl": 8}
EXPECTED_COMPARABLE = {"acl": 7, "genia": 9, "cl": 6}
TRANSPORT_IDS = {"P03", "P17", "P21"}
REFERENCE_DIR = "results/llm_repeat_output_agreement_2026_10_13"
TOLERANCE = 1e-12


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def f1_from_sets(gold, pred):
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    return prf(tp, fp, fn)


def span_rows(rows):
    return [
        r for r in rows
        if r.get("unit_type") == "span_mention"
        and r.get("start") is not None
        and r.get("end") is not None
    ]


def span_key(row, class_aware=False):
    key = (row.get("doc_id"), row.get("sent_id"), row.get("start"), row.get("end"))
    if class_aware:
        key += (row.get("class_label"),)
    return key


def span_set(rows, class_aware=False):
    return {span_key(r, class_aware) for r in span_rows(rows)}


def score_output(gold_rows, prediction_rows):
    """Exact detection uses all test gold, irrespective of missing/failed batches."""
    gold, predicted = span_set(gold_rows), span_set(prediction_rows)
    precision, recall, f1 = f1_from_sets(gold, predicted)
    tp = len(gold & predicted)
    return {"gold": len(gold), "predicted": len(predicted), "tp": tp,
            "fp": len(predicted - gold), "fn": len(gold - predicted),
            "precision": precision, "recall": recall, "f1": f1}


def validate_roster(pairs, manifest):
    ids = [row["pair_id"] for row in pairs]
    if len(ids) != 25 or set(ids) != {f"P{i:02d}" for i in range(1, 26)}:
        raise ValueError("Repeat roster must contain each of P01–P25 exactly once")
    public = {entry["release_path"]: entry for entry in manifest}
    if len(public) != len(manifest):
        raise ValueError("Duplicate manifest release_path")
    counts, comparable = Counter(), Counter()
    for row in pairs:
        if row["dataset"] not in SLUGS or row["decision"] not in DECISIONS:
            raise ValueError("Unexpected repeat corpus or decision")
        slug = SLUGS[row["dataset"]]
        counts[slug] += 1
        if row["decision"] != "transport_contrast":
            comparable[slug] += 1
        for side in ("base", "repeat"):
            path = row[f"{side}_path"]
            if path not in public:
                raise ValueError("Repeat path is absent from manifest: " + path)
            entry = public[path]
            if entry["dataset"] != row["dataset"] or entry["group"] not in {"llm", "llm_repeats"}:
                raise ValueError("Repeat references another corpus or a non-LLM output")
    contrasts = {r["pair_id"] for r in pairs if r["decision"] == "transport_contrast"}
    documented = {r["pair_id"] for r in pairs if r["decision"] == "primary_documented"}
    if (counts != EXPECTED_ALL or comparable != EXPECTED_COMPARABLE
            or contrasts != TRANSPORT_IDS or documented != {"P07"}):
        raise ValueError("Repeat population differs from the selected 22 + 3 pairs")
    return public


def summarize(rows):
    result = {}
    for slug in ("acl", "genia", "cl"):
        values = [float(row["absolute_delta_unlabeled_f1"])
                  for row in rows if row["dataset"] == slug]
        result[slug] = {"n_pairs": len(values),
                        "median_abs_delta_unlabeled_f1": statistics.median(values) if values else None,
                        "max_abs_delta_unlabeled_f1": max(values) if values else None}
    return result


def aggregate_pairs(rows):
    comparable = [row for row in rows if row["decision"] != "transport_contrast"]
    transport = [row for row in rows if row["decision"] == "transport_contrast"]
    return {
        "summary": summarize(comparable),
        "all_saved_full_test_pairs": summarize(rows),
        "transport_summary": summarize(transport),
        "sensitivity": {
            "genia_without_documented_raw_missing_pair": summarize(
                [r for r in comparable if r["pair_id"] != "P07"])["genia"],
            "cl_without_llama_format_failure_pair": summarize(
                [r for r in comparable if r["pair_id"] != "P19"])["cl"],
        },
        "transport_contrasts": [{key: row[key] for key in
                                 ("pair_id", "dataset", "absolute_delta_unlabeled_f1")}
                                for row in transport],
    }


def compare_saved(rows, summary, saved_rows, saved_summary):
    """Compare deterministic F1 values and descriptive summaries only in release mode."""
    expected = {r["pair_id"]: r for r in saved_rows}
    if len(expected) != len(saved_rows) or set(expected) != {r["pair_id"] for r in rows}:
        raise ValueError("Saved A17 table has another pair roster")
    checked = 0
    def equal(actual, reference, name):
        nonlocal checked
        if isinstance(actual, dict):
            if set(actual) != set(reference):
                raise ValueError("Saved A17 summary keys differ: " + name)
            for key, value in actual.items():
                equal(value, reference[key], name + "/" + key)
            return
        if isinstance(actual, (int, float)):
            wanted = float(reference)
            ok = math.isfinite(wanted) and (actual == wanted if isinstance(actual, int)
                                            else math.isclose(actual, wanted, rel_tol=0, abs_tol=TOLERANCE))
        else:
            ok = actual == reference
        if not ok:
            raise ValueError(f"Saved A17 result differs at {name}: {actual} != {reference}")
        checked += 1
    for row in rows:
        retained = expected[row["pair_id"]]
        for key in ("dataset", "decision", "model_key", "base_f1", "repeat_f1", "absolute_delta_unlabeled_f1"):
            equal(row[key], retained[key], row["pair_id"] + "/" + key)
    for name in ("summary", "all_saved_full_test_pairs", "sensitivity"):
        equal(summary[name], saved_summary[name], name)
    actual_contrasts = {r["pair_id"]: r for r in summary["transport_contrasts"]}
    retained_contrasts = {r["pair_id"]: r for r in saved_summary["transport_contrasts"]}
    equal(actual_contrasts, retained_contrasts, "transport_contrasts")
    return {"pairs": len(rows), "cells_checked": checked, "absolute_tolerance": TOLERANCE}


def reproduce_llm_repeats(root: Path, data_dir: Path, output_dir: Path,
                          prediction_root: Path | None = None):
    from .io import load_prediction_rows, validate_test_rows, compare_rows

    root, data_dir, output_dir = root.resolve(), data_dir.resolve(), output_dir.resolve()
    prediction_root = prediction_root.resolve() if prediction_root is not None else None
    try:
        relative = output_dir.relative_to(root / "build")
    except ValueError:
        raise ValueError("--output must be inside this repository's build/")
    if not relative.parts:
        raise ValueError("Give a fresh named subdirectory under build/")
    pairs_path, manifest_path = root / "predictions/llm_repeat_pairs.csv", root / "predictions/manifest.csv"
    pairs, manifest = read_csv(pairs_path), read_csv(manifest_path)
    public = validate_roster(pairs, manifest)
    output_dir.mkdir(parents=True, exist_ok=False)
    input_paths = {pairs_path, manifest_path}
    gold_by_dataset, indexes = {}, {}
    for dataset, slug in SLUGS.items():
        path = data_dir / slug / "test.jsonl"
        gold = list(read_jsonl(path))
        validate_test_rows(gold, dataset)
        gold_by_dataset[dataset] = gold
        indexes[dataset] = CorpusIndex.build(gold, dataset)
        input_paths.add(path)
    scores, pair_rows = {}, []
    for pair in pairs:
        dataset = pair["dataset"]
        for side in ("base", "repeat"):
            relative_path = pair[side + "_path"]
            if relative_path not in scores:
                records, path = load_prediction_rows(root, public[relative_path], indexes[dataset], prediction_root)
                scores[relative_path] = score_output(gold_by_dataset[dataset], records)
                input_paths.add(path)
        base, repeat = (scores[pair[side + "_path"]] for side in ("base", "repeat"))
        pair_rows.append({"pair_id": pair["pair_id"], "dataset": SLUGS[dataset],
                          "model_key": pair["model_key"], "decision": pair["decision"],
                          "base_path": pair["base_path"], "repeat_path": pair["repeat_path"],
                          "base_f1": base["f1"], "repeat_f1": repeat["f1"],
                          "absolute_delta_unlabeled_f1": abs(base["f1"] - repeat["f1"]),
                          "gold": base["gold"], "full_test_units": len(indexes[dataset].units),
                          "base_predicted": base["predicted"], "repeat_predicted": repeat["predicted"],
                          "base_tp": base["tp"], "repeat_tp": repeat["tp"],
                          "source": "full_test_selected_predictions"})
    summary = aggregate_pairs(pair_rows)
    table_path = root / REFERENCE_DIR / "limitations_full_repeat_f1.csv"
    summary_path = root / REFERENCE_DIR / "limitations_full_repeat_f1_summary.json"
    reference_rows = read_csv(table_path)
    reference_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    input_paths.update((table_path, summary_path))
    comparison = compare_rows(pair_rows, reference_rows, ("pair_id",),
                              fields=("base_f1", "repeat_f1", "absolute_delta_unlabeled_f1"),
                              tolerance=TOLERANCE)
    checks = (compare_saved(pair_rows, summary, reference_rows, reference_summary)
              if prediction_root is None else None)
    table_output = output_dir / "limitations_full_repeat_f1.csv"
    with table_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    receipt = {
        "schema": "llm_full_repeat_limitations_f1_v3", "status": "PASS" if checks else "COMPUTED",
        "mode": "saved_release" if prediction_root is None else "fresh_predictions",
        "prediction_root": str(prediction_root) if prediction_root is not None else None,
        "data_dir": str(data_dir),
        "metric": "exact unlabeled F1", "population": "22 comparable full-test pairs; 3 sync/batch contrasts separately",
        "scope": "A17: selected P01–P25, all three full test sets", **summary,
        "limitations": {
            "P07": "Historical pair is documented only; original raw does not establish independent execution. Output identity is not proof of independence.",
            "P19": {"historical_base_failed_batches": 27, "historical_repeat_failed_batches": 64,
                    "use": "metadata only; do not exclude units or gold from either full-test denominator"},
            "fresh_outputs": "Historical failure counts describe saved runs, not newly supplied predictions.",
        },
        "interpretation": "Comparable pairs describe movement between selected outputs, not iid generation variance; transport contrasts do not isolate a transport effect.",
        "checks": checks, "comparison_to_reference": comparison,
        "pair_table_sha256": sha256(table_output),
        "inputs": [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
                   for path in sorted(input_paths)],
    }
    (output_dir / "limitations_full_repeat_f1_summary.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt
