#!/usr/bin/env python3
"""Recompute unmatched-prediction proximity for the fixed supervised panels.

Gold defines exact false positives. A witness is any prediction of the other
run in the same text unit at character IoU >= .5; witnesses may be reused.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ate_paper.analysis.io import DATASETS, SLUGS, load_fixed_dataset, read_csv, compare_rows, comparison_summary, base_roster

ROOT = Path(__file__).resolve().parents[1]
TARGET = "fp_no_witness_share"
COMMON_MODELS = frozenset(("mbert_class_bio", "mbert_unlabeled_bio", "mbert_span",
                           "xlmr_bio", "xlmr_span_classifier", "xlmr_span_unlabeled"))


def complete_median(values):
    """Keep a descriptive summary undefined if its required population is incomplete."""
    return statistics.median(values) if values and all(math.isfinite(v) for v in values) else None


def configuration_summary(configs, boots, datasets):
    result = {}
    for dataset in datasets:
        rows = [r for r in configs if r["dataset"] == dataset]
        present = {r[k] for r in rows for k in ("model_a", "model_b")}
        if not COMMON_MODELS <= present:
            raise ValueError("Missing common-six configuration: " + dataset)
        shared = [r for r in rows if {r["model_a"], r["model_b"]} <= COMMON_MODELS]
        share = [r[TARGET + "_mean"] for r in rows]
        excess = [r[TARGET + "_excess"] for r in rows]
        shared_excess = [r[TARGET + "_excess"] for r in shared]
        result[dataset] = {
            "configuration_pairs": len(rows), "median_share": complete_median(share),
            "median_excess": complete_median(excess),
            "defined_share_pairs": sum(math.isfinite(v) for v in share),
            "defined_excess_pairs": sum(math.isfinite(v) for v in excess),
            "undefined_share_pairs": sum(not math.isfinite(v) for v in share),
            "undefined_excess_pairs": sum(not math.isfinite(v) for v in excess),
            "positive_excess": sum(v > 0 for v in excess if math.isfinite(v)),
            "positive_interval_lower": sum(r["excess_ci_low"] is not None and r["excess_ci_low"] > 0
                                           for r in boots if r["dataset"] == dataset),
            "shared_model_count": len(COMMON_MODELS), "shared_configuration_pairs": len(shared),
            "shared_median_excess": complete_median(shared_excess),
            "shared_defined_excess_pairs": sum(math.isfinite(v) for v in shared_excess),
            "shared_undefined_excess_pairs": sum(not math.isfinite(v) for v in shared_excess),
            "shared_positive_excess": sum(v > 0 for v in shared_excess if math.isfinite(v))}
    return result


def receipt(path):
    content = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def near_boundary_mask(keys, prediction_keys, threshold=0.5):
    """Original many-to-one presence screen; exact membership is a fast path."""
    by_sentence = defaultdict(list)
    for key in prediction_keys:
        by_sentence[key[:2]].append(key[2:])
    result = np.zeros(len(keys), dtype=bool)
    for i, key in enumerate(keys):
        if key in prediction_keys:
            result[i] = True
            continue
        start, end = key[2:]
        for other_start, other_end in by_sentence[key[:2]]:
            if other_end <= start or end <= other_start:
                continue
            overlap = max(0, min(end, other_end) - max(start, other_start))
            union = max(end, other_end) - min(start, other_start)
            if union and overlap / union >= threshold:
                result[i] = True
                break
    return result


def compute(dataset, specs, gold, predictions, docs, weights, skip_bootstrap):
    runs = [row["run"] for row in specs]
    false_positives = [predictions[run] - gold for run in runs]
    keys = sorted(set.union(*false_positives))
    lookup = {doc: i for i, doc in enumerate(docs)}
    key_docs = np.array([lookup[key[0]] for key in keys], dtype=np.int64)
    membership = np.array([[key in fp for key in keys] for fp in false_positives], dtype=bool)
    near = np.array([near_boundary_mask(keys, predictions[run]) for run in runs], dtype=bool)

    def counts(mask):
        return np.bincount(key_docs[mask], minlength=len(docs)).astype(np.int32)

    fp_documents = np.stack([counts(mask) for mask in membership])
    pairs = list(itertools.combinations(range(len(runs)), 2))
    nw_a = np.empty((len(pairs), len(docs)), dtype=np.int32)
    nw_b = np.empty_like(nw_a)
    pair_rows = []
    for index, (i, j) in enumerate(pairs):
        a, b = false_positives[i], false_positives[j]
        nw_a[index] = counts(membership[i] & ~near[j])
        nw_b[index] = counts(membership[j] & ~near[i])
        na, nb = int(nw_a[index].sum()), int(nw_b[index].sum())
        pair_rows.append({"dataset": dataset, "run_a": runs[i], "run_b": runs[j],
                          "model_a": specs[i]["model"], "model_b": specs[j]["model"],
                          "within_configuration": specs[i]["model"] == specs[j]["model"],
                          "fp_a": len(a), "fp_b": len(b), "common_fp": len(a & b),
                          "unique_fp_a": len(a - b), "unique_fp_b": len(b - a),
                          "no_witness_fp_a": na, "no_witness_fp_b": nb,
                          TARGET: float((ratio(na, len(a)) + ratio(nb, len(b))) / 2),
                          "unique_fp_no_witness_share": float((ratio(na, len(a - b))
                                                                + ratio(nb, len(b - a))) / 2)})
    groups = defaultdict(list)
    for i, row in enumerate(specs):
        groups[row["model"]].append(i)
    require(all(len(indices) == 5 for indices in groups.values()), "Expected five runs per configuration")
    pair_lookup = {pair: index for index, pair in enumerate(pairs)}
    within = {model: [pair_lookup[i, j] for i, j in itertools.combinations(indices, 2)]
              for model, indices in groups.items()}
    points = np.array([[row[TARGET] for row in pair_rows]])
    comparisons, config_rows = [], []
    for ma, mb in itertools.combinations(sorted(groups), 2):
        indices = [pair_lookup[tuple(sorted((i, j)))] for i in groups[ma] for j in groups[mb]]
        wa, wb = within[ma], within[mb]
        require(len(indices) == 25 and len(wa) == len(wb) == 10, "Wrong fixed-run pair population")
        bb, reference, excess = summarize_cloud(points, indices, wa, wb)
        row = {"dataset": dataset, "model_a": ma, "model_b": mb,
               "group_a": "supervised", "group_b": "supervised", "between_pairs": 25,
               TARGET + "_mean": float(bb[0]), TARGET + "_minimum": float(np.min(points[:, indices])),
               TARGET + "_maximum": float(np.max(points[:, indices])), TARGET + "_valid_pairs": int(np.isfinite(points[:, indices]).sum()),
               TARGET + "_within_a": float(np.mean(points[:, wa])),
               TARGET + "_within_b": float(np.mean(points[:, wb])),
               TARGET + "_within_reference": float(reference[0]), TARGET + "_excess": float(excess[0])}
        for field in ("fp_a", "fp_b", "common_fp", "unique_fp_a", "unique_fp_b",
                      "no_witness_fp_a", "no_witness_fp_b", "unique_fp_no_witness_share"):
            values = []
            for index in indices:
                pair, key = pair_rows[index], field
                if pair["model_a"] != ma and field.endswith(("_a", "_b")):
                    key = field[:-1] + ("b" if field[-1] == "a" else "a")
                values.append(pair[key])
            row[field + "_mean"] = float(np.mean(values))
        comparisons.append((indices, wa, wb))
        config_rows.append(row)
    boot_rows = []
    if not skip_bootstrap:
        i, j = (np.array([pair[k] for pair in pairs]) for k in (0, 1))
        fp_totals = weights @ fp_documents.T.astype(float)
        values = np.empty((len(weights), len(pairs)), dtype=float)
        for start in range(0, len(pairs), 128):
            sl = slice(start, start + 128)
            a = weights @ nw_a[sl].T.astype(float)
            b = weights @ nw_b[sl].T.astype(float)
            values[:, sl] = (ratio(a, fp_totals[:, i[sl]]) + ratio(b, fp_totals[:, j[sl]])) / 2
        for row, (indices, wa, wb) in zip(config_rows, comparisons):
            output = {"dataset": dataset, "model_a": row["model_a"], "model_b": row["model_b"],
                      "target": TARGET, "requested": len(weights), "within_sides": 2}
            clouds = summarize_cloud(values, indices, wa, wb)
            point_values = summarize_cloud(points, indices, wa, wb)
            for name, cloud, point in zip(("between", "within_reference", "excess"), clouds, point_values):
                output[name + "_point"] = float(point[0])
                output.update({name + "_" + key: value for key, value in interval(cloud).items()})
            boot_rows.append(output)
    print(dataset, "computed", len(pairs), "run pairs", flush=True)
    return pair_rows, config_rows, boot_rows


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare_saved(name, rows, keys, datasets, fresh=False):
    original = []
    inputs = set()
    for dataset in datasets:
        path = ROOT / (f"results/v1_profile_verified_document_analysis_{name}.csv"
                       if dataset != "CL-RuTerm3" else
                       f"results/cl_ruterm3_1_1_output_agreement_2026_10_13/{name}.csv")
        original.extend(row for row in read_csv(path) if row["dataset"] == dataset)
        inputs.add(path)
    if fresh:
        models = {dataset: {r["model"] for r in base_roster(ROOT, dataset, {"supervised"})[0]}
                  for dataset in datasets}
        selected = [r for r in original if r["model_a"] in models[r["dataset"]]
                    and r["model_b"] in models[r["dataset"]]]
        return compare_rows(rows, selected, keys, fields=tuple(rows[0])), inputs
    expected = {tuple(row[key] for key in keys): row for row in original}
    cells = 0
    require(len(rows) == len({tuple(row[key] for key in keys) for row in rows}), "Duplicate result identity")
    for row in rows:
        identity = tuple(row[key] for key in keys)
        source = expected[identity]
        for field, value in row.items():
            saved = source[field]
            if str(value) == saved or (value is None and saved == ""):
                cells += 1
                continue
            try:
                a, b = float(value), float(saved)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name}/{identity}/{field}: {value!r} != {saved!r}") from error
            require((math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=0, abs_tol=1e-10),
                    f"{name}/{identity}/{field}: {value!r} != {saved!r}")
            cells += 1
    return {"rows": len(rows), "cells": cells, "all_saved_values_match": True}, inputs


def main():
    global np, interval, ratio, summarize_cloud
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/local/processed")
    parser.add_argument("--output", type=Path, default=ROOT / "build/no_witness",
                        help="New output directory (must not exist)")
    parser.add_argument("--dataset", action="append", choices=tuple(SLUGS.values()))
    parser.add_argument("--prediction-root", type=Path, help="New run_experiments.py output root")
    parser.add_argument("--skip-bootstrap", action="store_true")
    args = parser.parse_args()
    try:
        import numpy as np
        from ate_paper.analysis.statistics import interval, ratio, summarize_cloud
    except ImportError as error:
        parser.error("Install the analysis extra: python -m pip install '.[analysis]' (" + str(error) + ")")
    output = args.output.resolve()
    require(not output.exists(), "Use a new output directory: " + str(output))
    datasets = tuple(dataset for dataset in DATASETS if not args.dataset or SLUGS[dataset] in args.dataset)
    for dataset in datasets:
        require((args.data_dir / SLUGS[dataset] / "test.jsonl").is_file(), "Missing prepared local gold: " + SLUGS[dataset])
    pairs, configs, boots, input_paths = [], [], [], set()
    for dataset in datasets:
        specs, gold, predictions, docs, weights, sources = load_fixed_dataset(ROOT, args.data_dir, dataset, args.prediction_root)
        p, c, b = compute(dataset, specs, gold, predictions, docs, weights, args.skip_bootstrap)
        pairs.extend(p)
        configs.extend(c)
        boots.extend(b)
        input_paths.update(sources)
    results = {"run_pairs": (pairs, ("dataset", "run_a", "run_b")),
               "configuration_pairs": (configs, ("dataset", "model_a", "model_b"))}
    if boots:
        results["bootstrap"] = boots, ("dataset", "model_a", "model_b", "target")
    checks = {}
    for name, (rows, keys) in results.items():
        checks[name], sources = compare_saved(name, rows, keys, datasets, fresh=args.prediction_root is not None)
        input_paths.update(sources)
    summary = configuration_summary(configs, boots, datasets)
    output.mkdir(parents=True)
    for name, (rows, _) in results.items():
        with (output / (name + ".csv")).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    result_status = "PASS" if args.prediction_root is None else "COMPUTED"
    record = {"status": result_status,
              "input_mode": "published_predictions" if args.prediction_root is None else "new_predictions",
                "prediction_root": str(args.prediction_root.resolve()) if args.prediction_root is not None else None,
              "target": TARGET, "fixed_supervised_runs": sum(50 if d != "CL-RuTerm3" else 55 for d in datasets),
              "checks_against_saved_results": checks, "summary": summary,
              "bootstrap": "Skipped" if args.skip_bootstrap else "Saved shared document draws; nominal percentile 95%; fixed runs",
              "inputs": [receipt(path) for path in sorted(input_paths)],
              "implementation": [receipt(Path(__file__)), receipt(ROOT / "src/ate_paper/analysis/io.py"),
                                 receipt(ROOT / "src/ate_paper/analysis/statistics.py")],
              "interpretation": "Formal output proximity; any other-run output may witness many unmatched predictions; no semantic adjudication"}
    (output / "manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result_status, "checks": comparison_summary(checks), "summary": summary,
                      "output": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
