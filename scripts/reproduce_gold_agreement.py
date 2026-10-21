#!/usr/bin/env python3
"""Compare covered gold IDs after exact-preserving, one-to-one IoU matching.

All inputs are accepted fixed outputs.  Results are written to a new directory;
existing paper scores and output-agreement artifacts are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ate_paper.analysis.io import load_fixed_dataset, compare_rows, comparison_summary

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build/gold_agreement"
GOLD_DIR = ROOT / "data/local/processed"
ACL_GENIA_ROWS = ROOT / "results/v1_profile_verified_document_analysis_run_scores.csv"
CL_ROWS = ROOT / "results/cl_ruterm3_1_1_output_agreement_2026_10_13/run_scores.csv"
DATASETS = ("ACL RD-TEC 2.0", "GENIA", "CL-RuTerm3")
SLUGS = {"ACL RD-TEC 2.0": "acl", "GENIA": "genia", "CL-RuTerm3": "cl"}
DRAW_NAMES = {"ACL RD-TEC 2.0": "acl_rd_tec_2_0", "GENIA": "genia_3_02", "CL-RuTerm3": "cl_ruterm3_1_1"}
ARRAY_PATHS = {dataset: ROOT / "results/document_draws" / (slug + ".npz")
               for dataset, slug in DRAW_NAMES.items()}
CONFIG_PATHS = {
    DATASETS[0]: ROOT / "results/v1_profile_verified_document_analysis_configuration_pairs.csv",
    DATASETS[1]: ROOT / "results/v1_profile_verified_document_analysis_configuration_pairs.csv",
    DATASETS[2]: ROOT / "results/cl_ruterm3_1_1_output_agreement_2026_10_13/configuration_pairs.csv",
}
TARGETS = ("exact", "relaxed_05", "relaxed_075", "flat_05", "nonflat_05")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Empty result: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})


def receipt(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def vector(keys: set[tuple], lookup: dict[str, int]) -> np.ndarray:
    return np.bincount([lookup[key[0]] for key in keys], minlength=len(lookup)).astype(np.int32)


def by_unit(keys: set[tuple]) -> dict[tuple, set[tuple[int, int]]]:
    result = defaultdict(set)
    for doc, sent, start, end in keys:
        result[doc, sent].add((start, end))
    return result


def nesting_categories(keys):
    sentences = defaultdict(list)
    for key in keys:
        sentences[key[:2]].append(key)
    categories = {}
    for sentence in sentences.values():
        for key in sentence:
            start, end = key[2:]
            inner = any(other != key and other[2] <= start and end <= other[3]
                        for other in sentence)
            outer = any(other != key and start <= other[2] and other[3] <= end
                        for other in sentence)
            categories[key] = ("both" if inner and outer else "inner-only" if inner
                               else "outer-only" if outer else "flat")
    return categories


def span_set(records, dataset):
    return {(str(row["doc_id"]), row.get("sent_id"), row["start"], row["end"])
            for row in records if row.get("dataset") == dataset
            and row.get("unit_type") == "span_mention"
            and row.get("start") is not None and row.get("end") is not None}


def load_dataset(dataset: str, prediction_root=None):
    return load_fixed_dataset(ROOT, GOLD_DIR, dataset, prediction_root)


def match_runs(dataset, specs, gold, predictions, flat, assignment_writer):
    gold_units = by_unit(gold)
    result, ambiguity, score_rows = {}, [], []
    for row in specs:
        run = row["run"]
        pred = predictions[run]
        pred_units = by_unit(pred)
        exact = pred & gold
        matched = {}
        for target, threshold in (("relaxed_05", Fraction(1, 2)),
                                  ("relaxed_075", Fraction(3, 4))):
            covered = set()
            unstable = 0
            for unit in sorted(gold_units.keys() | pred_units.keys()):
                assignments, ambiguous = match_unit(gold_units.get(unit, set()),
                                                    pred_units.get(unit, set()), threshold)
                unstable += ambiguous
                if target == "relaxed_05" and ambiguous:
                    ambiguity.append({"dataset": dataset, "run": run,
                                      "doc_id": unit[0], "sent_id": unit[1],
                                      "ambiguous_gold": ambiguous,
                                      "matched_gold": len(assignments)})
                for g, p, iou, is_exact in assignments:
                    covered.add((unit[0], unit[1], *g))
                    if target == "relaxed_05":
                        assignment_writer.writerow((dataset, run, unit[0], unit[1],
                                                    g[0], g[1], p[0], p[1],
                                                    iou.numerator, iou.denominator,
                                                    int(is_exact)))
            if not exact <= covered:
                raise AssertionError("Exact gold identity lost: " + run)
            matched[target] = covered
            if target == "relaxed_05":
                score_rows.append({"dataset": dataset, "run": run, "model": row["model"],
                                   "exact_tp": len(exact), "relaxed_05_tp": len(covered),
                                   "new_05_tp": len(covered - exact),
                                   "ambiguous_gold": unstable})
            else:
                score_rows[-1]["relaxed_075_tp"] = len(covered)
        matched["exact"] = exact
        matched["flat_05"] = matched["relaxed_05"] & flat
        matched["nonflat_05"] = matched["relaxed_05"] - flat
        result[run] = matched
        print(dataset, run, "exact", len(exact), "relaxed", len(matched["relaxed_05"]), flush=True)
    return result, ambiguity, score_rows


def pair_counts(dataset, specs, matched, gold, flat, docs, check_reference=True):
    lookup = {doc: i for i, doc in enumerate(docs)}
    run_names = [row["run"] for row in specs]
    pair_indices = list(itertools.combinations(range(len(specs)), 2))
    run_counts = {target: np.stack([vector(matched[run][target], lookup) for run in run_names])
                  for target in TARGETS}
    common = {target: np.zeros((len(pair_indices), len(docs)), dtype=np.int32)
              for target in TARGETS}
    resolved_docs = np.zeros((len(pair_indices), len(docs)), dtype=np.int32)
    unique_exact_docs = np.zeros_like(resolved_docs)
    rows = []
    for index, (i, j) in enumerate(pair_indices):
        a, b = run_names[i], run_names[j]
        row = {"dataset": dataset, "run_a": a, "run_b": b,
               "model_a": specs[i]["model"], "model_b": specs[j]["model"],
               "within_configuration": specs[i]["model"] == specs[j]["model"]}
        for target in TARGETS:
            aa, bb = matched[a][target], matched[b][target]
            cc = aa & bb
            common[target][index] = vector(cc, lookup)
            row[target + "_a"] = len(aa)
            row[target + "_b"] = len(bb)
            row[target + "_common"] = len(cc)
            row[target + "_unique_a"] = len(aa - bb)
            row[target + "_unique_b"] = len(bb - aa)
            row[target + "_turnover"] = float(ratio(min(len(aa), len(bb)) - len(cc),
                                                   min(len(aa), len(bb))))
        ae, be = matched[a]["exact"], matched[b]["exact"]
        ar, br = matched[a]["relaxed_05"], matched[b]["relaxed_05"]
        unique_exact = ae ^ be
        resolved = ((ae - be) & br) | ((be - ae) & ar)
        residual = unique_exact - resolved
        transitions = {
            "e0_r0": len(gold - (ar | br)),
            "e0_r1": len((ar - br) - (ae | be)),
            "e0_r2": len((br - ar) - (ae | be)),
            "e0_r3": len((ar & br) - (ae | be)),
            "e1_r1": len((ae - be) - br),
            "e1_r3": len((ae - be) & br),
            "e2_r2": len((be - ae) - ar),
            "e2_r3": len((be - ae) & ar),
            "e3_r3": len(ae & be),
        }
        if sum(transitions.values()) != len(gold):
            raise AssertionError("Gold-state transition counts do not partition gold")
        unique_exact_docs[index] = vector(unique_exact, lookup)
        resolved_docs[index] = vector(resolved, lookup)
        row.update({"unique_exact": len(unique_exact), "resolved": len(resolved),
                    "residual": len(residual),
                    "resolution_fraction": len(resolved) / len(unique_exact) if unique_exact else None,
                    "unique_relaxed": len(ar ^ br),
                    "new_shared": transitions["e0_r3"],
                    "new_unique_a": transitions["e0_r1"],
                    "new_unique_b": transitions["e0_r2"],
                    "flat_gold": len(flat), "gold": len(gold), **transitions})
        rows.append(row)
    # Check all exact pair identities against the previous saved output analysis.
    previous_path = (ROOT / "results/v1_profile_verified_document_analysis_run_pairs.csv"
                     if dataset != "CL-RuTerm3" else
                     ROOT / "results/cl_ruterm3_1_1_output_agreement_2026_10_13/run_pairs.csv")
    old = {(r["dataset"], frozenset((r["run_a"], r["run_b"]))): r
           for r in read_csv(previous_path) if r["dataset"] == dataset}
    for row in rows if check_reference else []:
        source = old[dataset, frozenset((row["run_a"], row["run_b"]))]
        if row["exact_common"] != int(source["common_tp"]):
            raise AssertionError("Exact shared gold IDs differ from previous analysis")
    return rows, pair_indices, run_counts, common, resolved_docs, unique_exact_docs, previous_path


def configuration_summaries(dataset, specs, pair_rows, pair_indices, check_reference=True):
    run_to_index = {row["run"]: i for i, row in enumerate(specs)}
    pair_lookup = {pair: i for i, pair in enumerate(pair_indices)}
    groups = defaultdict(list)
    for row in specs:
        groups[row["model"]].append(run_to_index[row["run"]])
    configs, config_indices = [], []
    old = {(r["model_a"], r["model_b"]): r for r in read_csv(CONFIG_PATHS[dataset])
           if r["dataset"] == dataset and r["group_a"] == r["group_b"] == "supervised"}
    for ma, mb in itertools.combinations(sorted(groups), 2):
        between = [pair_lookup[tuple(sorted((i, j)))] for i in groups[ma] for j in groups[mb]]
        wa = [pair_lookup[i, j] for i, j in itertools.combinations(groups[ma], 2)]
        wb = [pair_lookup[i, j] for i, j in itertools.combinations(groups[mb], 2)]
        assert len(between) == 25 and len(wa) == len(wb) == 10
        row = {"dataset": dataset, "model_a": ma, "model_b": mb,
               "between_run_pairs": 25, "within_a_pairs": 10, "within_b_pairs": 10}
        for target in TARGETS:
            points = np.array([r[target + "_turnover"] for r in pair_rows], dtype=float)[None, :]
            between_value, reference, excess = summarize_cloud(points, between, wa, wb)
            row.update({target + "_between": float(between_value[0]),
                        target + "_within_reference": float(reference[0]),
                        target + "_excess": float(excess[0])})
        row["resolved_mean"] = float(np.mean([pair_rows[i]["resolved"] for i in between]))
        row["residual_mean"] = float(np.mean([pair_rows[i]["residual"] for i in between]))
        row["unique_exact_mean"] = float(np.mean([pair_rows[i]["unique_exact"] for i in between]))
        row["unique_relaxed_mean"] = float(np.mean([pair_rows[i]["unique_relaxed"] for i in between]))
        row["resolution_fraction_mean"] = float(np.mean(
            [np.nan if pair_rows[i]["resolution_fraction"] is None else pair_rows[i]["resolution_fraction"]
             for i in between]))
        source = old[ma, mb]
        if check_reference and (abs(row["exact_between"] - float(source["tp_turnover_mean"])) > 1e-12
                or abs(row["exact_excess"] - float(source["tp_turnover_excess"])) > 1e-12):
            raise AssertionError("Exact configuration turnover differs from previous analysis")
        configs.append(row)
        config_indices.append((between, wa, wb))
    return configs, config_indices


def bootstrap(dataset, weights, pair_indices, run_counts, common, resolved_docs,
              unique_exact_docs, configs, config_indices):
    i = np.array([a for a, _ in pair_indices])
    j = np.array([b for _, b in pair_indices])
    output = []
    for target in ("relaxed_05", "relaxed_075", "flat_05", "nonflat_05"):
        totals = weights @ run_counts[target].T.astype(float)
        values = np.empty((len(weights), len(pair_indices)), dtype=float)
        for start in range(0, len(pair_indices), 128):
            sl = slice(start, start + 128)
            intersection = weights @ common[target][sl].T.astype(float)
            denominator = np.minimum(totals[:, i[sl]], totals[:, j[sl]])
            values[:, sl] = ratio(denominator - intersection, denominator)
        for row, (between, wa, wb) in zip(configs, config_indices):
            bb, _ref, excess = summarize_cloud(values, between, wa, wb)
            for metric, cloud in (("between", bb), ("excess", excess)):
                ci = interval(cloud)
                output.append({"dataset": dataset, "model_a": row["model_a"],
                               "model_b": row["model_b"], "target": target,
                               "metric": metric, "point": row[target + "_" + metric],
                               "valid_draws": ci["valid"],
                               "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]})
        print(dataset, "bootstrap", target, flush=True)
        del values
    u = weights @ unique_exact_docs.T.astype(float)
    resolved = weights @ resolved_docs.T.astype(float)
    rates = ratio(resolved, u)
    for row, (between, _wa, _wb) in zip(configs, config_indices):
        cloud = np.mean(rates[:, between], axis=1)
        ci = interval(cloud)
        output.append({"dataset": dataset, "model_a": row["model_a"],
                       "model_b": row["model_b"], "target": "resolution_fraction",
                       "metric": "between", "point": row["resolution_fraction_mean"],
                       "valid_draws": ci["valid"],
                       "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]})
    return output


def compare_saved(name, rows, keys, datasets, fresh=False):
    expected = [row for row in read_csv(ROOT / "results/relaxed_gold_id_agreement_2026_10_13" / (name + ".csv"))
                if row["dataset"] in datasets]
    if fresh:
        return compare_rows(rows, expected, keys)
    actual = {tuple(row[key] for key in keys): row for row in rows}
    wanted = {tuple(row[key] for key in keys): row for row in expected}
    if len(actual) != len(rows) or actual.keys() != wanted.keys():
        raise ValueError("Different saved/new row identities: " + name)
    cells = 0
    for identity, original in wanted.items():
        current = actual[identity]
        for field, value in original.items():
            if field not in current:
                raise ValueError("Missing field: " + name + "/" + field)
            new = current[field]
            if str(new) == value or (new is None and value == ""):
                cells += 1
                continue
            try:
                a, b = float(new), float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name}/{identity}/{field}: {new!r} != {value!r}") from error
            if not ((math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=0, abs_tol=1e-10)):
                raise ValueError(f"{name}/{identity}/{field}: {new!r} != {value!r}")
            cells += 1
    return {"rows": len(rows), "cells": cells, "all_saved_values_match": True}


def main():
    global OUT, GOLD_DIR, np, match_unit, interval, ratio, summarize_cloud
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=GOLD_DIR,
                        help="Local processed corpus root with <slug>/test.jsonl")
    parser.add_argument("--output", type=Path, default=OUT,
                        help="New output directory (must not exist)")
    parser.add_argument("--dataset", action="append", choices=tuple(SLUGS.values()),
                        help="Repeat to select corpora; default is all three")
    parser.add_argument("--prediction-root", type=Path, help="New run_experiments.py output root")
    parser.add_argument("--skip-bootstrap", action="store_true",
                        help="Recompute matching and point results without interval resampling")
    args = parser.parse_args()
    try:
        import numpy as np
        from ate_paper.analysis.matching import match_unit
        from ate_paper.analysis.statistics import interval, ratio, summarize_cloud
    except ImportError as error:
        parser.error("Install the analysis extra first: python -m pip install '.[analysis]' (" + str(error) + ")")
    GOLD_DIR, OUT = args.data_dir.resolve(), args.output.resolve()
    datasets = tuple(dataset for dataset in DATASETS if not args.dataset or SLUGS[dataset] in args.dataset)
    for dataset in datasets:
        if not (GOLD_DIR / SLUGS[dataset] / "test.jsonl").is_file():
            parser.error("Missing local gold: " + str(GOLD_DIR / SLUGS[dataset] / "test.jsonl"))
    if OUT.exists():
        raise FileExistsError("Use a new output directory: " + str(OUT))
    OUT.mkdir(parents=True)
    all_scores, all_pairs, all_configs, all_boot, all_ambiguity = [], [], [], [], []
    # These accepted results are actively compared with the recomputation.
    input_paths = {ROOT / "results/relaxed_gold_id_agreement_2026_10_13" / (name + ".csv")
                   for name in ("run_scores", "run_pairs", "configuration_pairs")}
    if not args.skip_bootstrap:
        input_paths.add(ROOT / "results/relaxed_gold_id_agreement_2026_10_13/bootstrap.csv")
    assignment_path = OUT / "assignments_05.csv.gz"
    with gzip.open(assignment_path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("dataset", "run", "doc_id", "sent_id", "gold_start", "gold_end",
                         "pred_start", "pred_end", "iou_num", "iou_den", "exact"))
        for dataset in datasets:
            specs, gold, predictions, docs, weights, sources = load_dataset(dataset, args.prediction_root)
            input_paths.update(sources)
            categories = nesting_categories(gold)
            flat = {key for key, category in categories.items() if category == "flat"}
            matched, ambiguity, scores = match_runs(dataset, specs, gold, predictions, flat, writer)
            pairs, pair_indices, run_counts, common, resolved, u_exact, old_pair_path = pair_counts(
                dataset, specs, matched, gold, flat, docs, check_reference=args.prediction_root is None)
            input_paths.add(old_pair_path)
            configs, indices = configuration_summaries(dataset, specs, pairs, pair_indices, check_reference=args.prediction_root is None)
            if not args.skip_bootstrap:
                all_boot.extend(bootstrap(dataset, weights, pair_indices, run_counts, common,
                                          resolved, u_exact, configs, indices))
            all_scores.extend(scores)
            all_pairs.extend(pairs)
            all_configs.extend(configs)
            all_ambiguity.extend(ambiguity)
    checks = {
        "run_scores": compare_saved("run_scores", all_scores, ("dataset", "run"), datasets, fresh=args.prediction_root is not None),
        "run_pairs": compare_saved("run_pairs", all_pairs, ("dataset", "run_a", "run_b"), datasets, fresh=args.prediction_root is not None),
        "configuration_pairs": compare_saved("configuration_pairs", all_configs,
                                              ("dataset", "model_a", "model_b"), datasets, fresh=args.prediction_root is not None),
    }
    if not args.skip_bootstrap:
        checks["bootstrap"] = compare_saved("bootstrap", all_boot,
                                            ("dataset", "model_a", "model_b", "target", "metric"), datasets, fresh=args.prediction_root is not None)
    outputs = [assignment_path]
    for name, rows in (("run_scores", all_scores), ("run_pairs", all_pairs),
                       ("configuration_pairs", all_configs), ("bootstrap", all_boot)):
        if rows:
            path = OUT / (name + ".csv")
            write_csv(path, rows)
            outputs.append(path)
    transition_path = OUT / "transitions.csv"
    transition_fields = ("dataset", "run_a", "run_b", "model_a", "model_b", "gold",
                         "e0_r0", "e0_r1", "e0_r2", "e0_r3", "e1_r1", "e1_r3",
                         "e2_r2", "e2_r3", "e3_r3")
    write_csv(transition_path, [{key: row[key] for key in transition_fields}
                                for row in all_pairs])
    outputs.append(transition_path)
    if all_ambiguity:
        ambiguity_path = OUT / "ambiguous_units.csv"
        write_csv(ambiguity_path, all_ambiguity)
        outputs.append(ambiguity_path)
    protocol = {"status": "EXPLORATORY_FIXED_OUTPUT_GOLD_ID_MATCHING",
                "scope": {"supervised_runs": len(all_scores), "run_pairs": len(all_pairs),
                          "configuration_pairs": len(all_configs)},
                "matching": "Exact offsets anchored; residual one-to-one maximum cardinality, then maximum total character IoU, then lexicographic edge list; within original text unit",
                "thresholds": [1.0, 0.75, 0.5],
                "bootstrap": "Skipped" if args.skip_bootstrap else "Saved shared document multiplicities; fixed runs; nominal 95% percentile intervals",
                "interpretation": "Formal gold coverage, not semantic term validity, profile equivalence, or a causal decoder effect"}
    protocol_path = OUT / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    outputs.append(protocol_path)
    result_status = "PASS" if args.prediction_root is None else "COMPUTED"
    manifest = {"status": result_status,
                "input_mode": "published_predictions" if args.prediction_root is None else "new_predictions",
                "prediction_root": str(args.prediction_root.resolve()) if args.prediction_root is not None else None,
                "checks_against_saved_results": checks,
                "inputs": [receipt(path) for path in sorted(input_paths)],
                "implementation": [receipt(ROOT / "src/ate_paper/analysis/matching.py"),
                                   receipt(ROOT / "src/ate_paper/analysis/statistics.py"),
                                   receipt(ROOT / "src/ate_paper/analysis/io.py"),
                                   receipt(Path(__file__))],
                "outputs": [receipt(path) for path in outputs]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result_status, "scope": protocol["scope"],
                      "bootstrap": protocol["bootstrap"], "checks": comparison_summary(checks),
                      "output": str(OUT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
