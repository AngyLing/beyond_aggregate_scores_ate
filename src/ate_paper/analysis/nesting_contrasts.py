"""Paired-document flat/nonflat contrasts for relaxed gold-ID agreement."""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import math
from pathlib import Path

from ate_paper.predictions import read_jsonl, sha256
from ate_paper.io import read_csv as rows, file_receipt
from ate_paper.analysis.io import SLUGS, DRAW_NAMES, nesting_categories, span_set, compare_rows

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "build/nesting_contrasts"
SOURCE = ROOT / "build/gold_agreement"
GOLD_DIR = ROOT / "data/local/processed"
PAIRS = {
    "GENIA": (("mbert_class_bio", "mbert_span"), ("xlmr_bio", "xlmr_span_classifier"),
              ("scibert_bio", "scibert_span"), ("pubmedbert_bio", "pubmedbert_span")),
    "CL-RuTerm3": (("mbert_class_bio", "mbert_span"), ("xlmr_bio", "xlmr_span_classifier"),
                   ("rubert_bio", "rubert_span"), ("ruroberta_bio", "ruroberta_span")),
}
GOLD = {dataset: GOLD_DIR / SLUGS[dataset] / "test.jsonl" for dataset in PAIRS}
ARRAYS = {dataset: ROOT / "results/document_draws" / (DRAW_NAMES[dataset] + ".npz")
          for dataset in PAIRS}


def vector(keys, lookup):
    return np.bincount([lookup[x[0]] for x in keys], minlength=len(lookup)).astype(float)


def selected_datasets(runs, requested=None):
    """Inherit nested corpus availability from the verified upstream roster."""
    available = {row["dataset"] for row in runs}
    selected = tuple(dataset for dataset in PAIRS
                     if dataset in available and (not requested or SLUGS[dataset] in requested))
    if requested and set(requested) - {SLUGS[d] for d in selected}:
        raise ValueError("Requested nested corpus is absent from the gold producer")
    if not selected:
        raise ValueError("No GENIA/CL nesting corpus is available in the gold producer")
    return selected


def main():
    global np, OUT, SOURCE, GOLD, ratio
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=GOLD_DIR,
                        help="Prepared local corpus root with genia/test.jsonl and cl/test.jsonl")
    parser.add_argument("--gold-output", type=Path, default=SOURCE,
                        help="Completed reproduce_gold_agreement.py output directory")
    parser.add_argument("--output", type=Path, default=OUT,
                        help="New contrast output directory (must not exist)")
    parser.add_argument("--dataset", action="append", choices=("genia", "cl"),
                        help="Default: available nested corpora from the gold-producer roster")
    args = parser.parse_args()
    try:
        import numpy as np
        from ate_paper.analysis.statistics import ratio
    except ImportError as error:
        parser.error("Install the analysis extra first: python -m pip install '.[analysis]' (" + str(error) + ")")
    OUT, SOURCE = args.output.resolve(), args.gold_output.resolve()
    if OUT.exists():
        raise FileExistsError("Use a new output directory: " + str(OUT))
    producer = json.loads((SOURCE / "manifest.json").read_text())
    if producer["status"] not in ("PASS", "COMPUTED"):
        raise ValueError("Gold-agreement producer is not PASS")
    fresh = producer.get("input_mode") == "new_predictions"
    saved_outputs = {Path(row["path"]).name: row for row in producer["outputs"]}
    for name in ("run_scores.csv", "configuration_pairs.csv", "assignments_05.csv.gz"):
        path = SOURCE / name
        if sha256(path) != saved_outputs[name]["sha256"]:
            raise ValueError("Gold-producer output hash differs: " + name)
    runs = rows(SOURCE / "run_scores.csv")
    datasets = selected_datasets(runs, args.dataset)
    GOLD = {dataset: args.data_dir.resolve() / SLUGS[dataset] / "test.jsonl" for dataset in datasets}
    selected_arrays = {dataset: ARRAYS[dataset] for dataset in datasets}
    producer_inputs = {}
    for entry in producer["inputs"]:
        path = Path(entry["path"])
        path = path if path.is_absolute() else ROOT / path
        producer_inputs[path.resolve()] = entry["sha256"]
    for path in (*GOLD.values(), *selected_arrays.values()):
        if producer_inputs.get(path.resolve()) != sha256(path):
            raise ValueError("Gold/draw input differs from the gold producer: " + str(path))
    configs = {(r["dataset"], frozenset((r["model_a"], r["model_b"]))): r
               for r in rows(SOURCE / "configuration_pairs.csv")}
    matched = {(r["dataset"], r["run"]): set() for r in runs}
    with gzip.open(SOURCE / "assignments_05.csv.gz", "rt", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["doc_id"], row["sent_id"] or None,
                   int(row["gold_start"]), int(row["gold_end"]))
            matched.setdefault((row["dataset"], row["run"]), set()).add(key)
    output = []
    for dataset in datasets:
        model_pairs = PAIRS[dataset]
        gold = span_set(read_jsonl(GOLD[dataset]), dataset)
        flat = {key for key, category in nesting_categories(gold).items() if category == "flat"}
        with np.load(ARRAYS[dataset], allow_pickle=False) as loaded:
            docs = list(map(str, loaded["doc_ids"]))
            weights = loaded["weights"].astype(float)
        lookup = {doc: i for i, doc in enumerate(docs)}
        group = {}
        for model in {model for pair in model_pairs for model in pair}:
            group[model] = [r["run"] for r in runs if r["dataset"] == dataset and r["model"] == model]
            assert len(group[model]) == 5
        for ma, mb in model_pairs:
            pair = configs[dataset, frozenset((ma, mb))]
            union_runs = group[ma] + group[mb]
            record = {"dataset": dataset, "model_a": ma, "model_b": mb,
                      "between_pairs": 25, "within_pairs_per_side": 10}
            clouds = {}
            for label, scope in (("flat", flat), ("nonflat", gold - flat)):
                run_vec = {run: vector(matched[dataset, run] & scope, lookup) for run in union_runs}
                pair_cloud = {}
                pair_point = {}
                for a, b in itertools.combinations(union_runs, 2):
                    aa, bb = weights @ run_vec[a], weights @ run_vec[b]
                    common_vec = vector(matched[dataset, a] & matched[dataset, b] & scope, lookup)
                    common = weights @ common_vec
                    pair_cloud[frozenset((a, b))] = ratio(np.minimum(aa, bb) - common,
                                                          np.minimum(aa, bb))
                    denom_point = min(run_vec[a].sum(), run_vec[b].sum())
                    pair_point[frozenset((a, b))] = float(ratio(denom_point - common_vec.sum(), denom_point))
                between = np.mean(np.stack([pair_cloud[frozenset((a, b))]
                                            for a in group[ma] for b in group[mb]]), axis=0)
                within_a = np.mean(np.stack([pair_cloud[frozenset(x)]
                                             for x in itertools.combinations(group[ma], 2)]), axis=0)
                within_b = np.mean(np.stack([pair_cloud[frozenset(x)]
                                             for x in itertools.combinations(group[mb], 2)]), axis=0)
                excess = between - np.maximum(within_a, within_b)
                clouds[label] = between, excess
                between_point = float(np.mean([pair_point[frozenset((a, b))]
                                               for a in group[ma] for b in group[mb]]))
                within_a_point = float(np.mean([pair_point[frozenset(x)]
                                                for x in itertools.combinations(group[ma], 2)]))
                within_b_point = float(np.mean([pair_point[frozenset(x)]
                                                for x in itertools.combinations(group[mb], 2)]))
                for metric, value in (("between", between), ("excess", excess)):
                    point = float(pair[f"{label}_05_{metric}"])
                    recalculated = (between_point if metric == "between" else
                                    between_point - float(np.maximum(within_a_point, within_b_point)))
                    same = ((math.isnan(recalculated) and math.isnan(point))
                            or math.isclose(recalculated, point, rel_tol=0, abs_tol=1e-10))
                    if not same:
                        raise AssertionError("Flat/nonflat point differs from gold-producer configuration")
                    record[f"{label}_{metric}"] = recalculated
            for metric, position in (("between", 0), ("excess", 1)):
                difference = clouds["nonflat"][position] - clouds["flat"][position]
                valid = difference[np.isfinite(difference)]
                low, high = (np.quantile(valid, [0.00625, 0.99375])
                             if len(valid) else (np.nan, np.nan))
                record.update({f"delta_{metric}": record[f"nonflat_{metric}"] - record[f"flat_{metric}"],
                               f"delta_{metric}_ci_low": float(low),
                               f"delta_{metric}_ci_high": float(high),
                               f"delta_{metric}_valid": len(valid)})
            output.append(record)
    accepted_path = ROOT / "results/relaxed_gold_id_agreement_2026_10_13/nested_contrasts.csv"
    accepted = {(row["dataset"], row["model_a"], row["model_b"]): row
                for row in rows(accepted_path) if row["dataset"] in datasets}
    if len(output) != len(accepted) or len(output) != sum(len(PAIRS[d]) for d in datasets):
        raise ValueError("Incomplete selected nesting contrasts")
    comparison = compare_rows(output, list(accepted.values()), ("dataset", "model_a", "model_b"))
    checked = 0
    for row in output if not fresh else []:
        source = accepted[row["dataset"], row["model_a"], row["model_b"]]
        for field, value in row.items():
            if str(value) == source[field]:
                checked += 1
                continue
            try:
                same = math.isclose(float(value), float(source[field]), rel_tol=0, abs_tol=1e-10)
            except ValueError:
                same = False
            if not same:
                raise ValueError(f"Nested contrast {row['dataset']}/{row['model_a']}/{field}: {value!r} != {source[field]!r}")
            checked += 1
    OUT.mkdir(parents=True)
    path = OUT / "nested_contrasts.csv"
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, output[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    result_status = "COMPUTED" if fresh else "PASS"
    manifest = {"status": result_status, "datasets": list(datasets),
                "input_mode": "new_predictions" if fresh else "published_predictions",
                "prediction_root": producer.get("prediction_root"),
                "checks_against_saved_results": comparison if fresh else {"rows": len(output), "cells": checked, "all_saved_values_match": True},
                "protocol": "Shared document draws; nonflat minus flat after full-gold exact-priority IoU>=0.5 matching; 98.75% nominal percentile intervals within each four-pair corpus group; fixed runs and post-hoc pair selection",
                "inputs": [file_receipt(path, root=ROOT) for path in (SOURCE / "manifest.json", SOURCE / "assignments_05.csv.gz",
                                                       SOURCE / "run_scores.csv", SOURCE / "configuration_pairs.csv",
                                                       accepted_path, *GOLD.values(), *selected_arrays.values())],
                "implementation": [file_receipt(Path(__file__), root=ROOT),
                                   file_receipt(Path(__file__).resolve().parents[3] / "scripts/reproduce_nesting_contrasts.py", root=ROOT),
                                   file_receipt(Path(__file__).resolve().parents[1] / "io.py", root=ROOT),
                                   file_receipt(Path(__file__).parent / "statistics.py", root=ROOT), file_receipt(ROOT / "src/ate_paper/analysis/io.py", root=ROOT)], "output": file_receipt(path, root=ROOT)}
    (OUT / "nested_contrasts_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result_status, "contrasts": len(output), "cells": checked, "output": str(path)}))
    for row in output:
        print(row["dataset"], row["model_a"], row["model_b"],
              round(row["delta_between"], 4),
              (round(row["delta_between_ci_low"], 4), round(row["delta_between_ci_high"], 4)))


if __name__ == "__main__":
    main()
