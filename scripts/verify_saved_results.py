#!/usr/bin/env python3
"""Check saved numerical evidence against retained numerical references.

This is aggregate verification. It does not reconstruct gold matching or
resample documents, and it does not independently validate saved intervals.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATASETS = ("ACL RD-TEC 2.0", "GENIA", "CL-RuTerm3")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(relative: str) -> list[dict[str, str]]:
    with (RESULTS / relative).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual: float, expected: float, context: str) -> None:
    require(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12),
            f"{context}: {actual} != {expected}")


def decimal(value: float, *, signed: bool = False, places: int = 3) -> str:
    return format(value, f"{'+' if signed else ''}.{places}f").replace("0.", ".")


def check_source_files() -> dict:
    receipt = json.loads((RESULTS / "release_result_sources.json").read_text())
    for item in receipt["files"]:
        path = ROOT / item["path"]
        content = path.read_bytes()
        require(len(content) == item["bytes"], f"Changed size: {item['path']}")
        require(hashlib.sha256(content).hexdigest() == item["sha256"],
                f"Changed numerical source: {item['path']}")
    for path in RESULTS.rglob("*.csv"):
        with path.open(encoding="utf-8", newline="") as stream:
            fields = next(csv.reader(stream))
        forbidden = {"text", "surface", "norm", "sentence", "abstract", "title"} & set(fields)
        require(not forbidden, f"Text-bearing result columns: {path}: {forbidden}")
    return {"files": len(receipt["files"]), "all_source_sha256_match": True,
            "no_text_surface_norm_columns": True}


def check_headlines() -> dict:
    reference = json.loads((RESULTS / "saved_results_verification.json").read_text())["headlines"]
    relaxed = read_csv("relaxed_gold_id_agreement_2026_10_13/configuration_pairs.csv")
    exact = read_csv("v1_profile_verified_document_analysis_configuration_pairs.csv")
    exact += read_csv("cl_ruterm3_1_1_output_agreement_2026_10_13/configuration_pairs.csv")
    relaxed_ci = read_csv("relaxed_gold_id_agreement_2026_10_13/bootstrap.csv")
    exact_ci = read_csv("v1_profile_verified_document_analysis_bootstrap.csv")
    exact_ci += read_csv("cl_ruterm3_1_1_output_agreement_2026_10_13/bootstrap.csv")
    expected = dict(zip(DATASETS, ((45, 15, 9, 44, 41, 14),
                                  (45, 43, 40, 40, 39, 13),
                                  (55, 53, 52, 55, 55, 15))))
    model_sets = [{r[key] for r in relaxed if r["dataset"] == dataset
                   for key in ("model_a", "model_b")} for dataset in DATASETS]
    common_models = set.intersection(*model_sets)
    require(len(common_models) == 6, "The common supervised panel must contain six models")
    summary = {}
    for dataset in DATASETS:
        r = [x for x in relaxed if x["dataset"] == dataset]
        e = [x for x in exact if x["dataset"] == dataset
             and x["group_a"] == x["group_b"] == "supervised"]
        n, relaxed_positive, relaxed_low_positive, fp_positive, fp_low_positive, common_positive = expected[dataset]
        require(len(r) == len(e) == n, f"Wrong supervised pair count: {dataset}")
        keys = {(x["model_a"], x["model_b"]) for x in e}
        rci = [x for x in relaxed_ci if x["dataset"] == dataset
               and x["target"] == "relaxed_05" and x["metric"] == "excess"]
        eci = [x for x in exact_ci if x["dataset"] == dataset
               and x["target"] == "fp_no_witness_share"
               and (x["model_a"], x["model_b"]) in keys]
        require(len(rci) == len(eci) == n, f"Missing interval records: {dataset}")
        counts = (sum(float(x["relaxed_05_excess"]) > 0 for x in r),
                  sum(float(x["ci_low"]) > 0 for x in rci),
                  sum(float(x["fp_no_witness_share_excess"]) > 0 for x in e),
                  sum(float(x["excess_ci_low"]) > 0 for x in eci))
        require(counts == (relaxed_positive, relaxed_low_positive, fp_positive, fp_low_positive),
                f"Headline positive counts changed: {dataset}: {counts}")
        medians = [statistics.median(float(x[field]) for x in source) for source, field in (
            (r, "exact_between"), (r, "exact_excess"),
            (r, "relaxed_05_between"), (r, "relaxed_05_excess"),
            (e, "fp_no_witness_share_mean"), (e, "fp_no_witness_share_excess"))]
        expected_medians = reference[dataset]["main_table_medians"]
        require(len(expected_medians) == len(medians), "Wrong headline reference width")
        for actual, expected_median in zip(medians, expected_medians):
            close(actual, expected_median, "Headline median: " + dataset)
        common = [x for x in e if {x["model_a"], x["model_b"]} <= common_models]
        require(len(common) == 15, f"Wrong common-six pair count: {dataset}")
        require(sum(float(x["fp_no_witness_share_excess"]) > 0 for x in common) == common_positive,
                f"Common-six positive count changed: {dataset}")
        common_median = statistics.median(float(x["fp_no_witness_share_excess"]) for x in common)
        expected_common_median = dict(zip(DATASETS, ("+.040", "+.083", "+.096")))[dataset]
        require(decimal(common_median, signed=True) == expected_common_median,
                f"Common-six median changed: {dataset}")
        summary[dataset] = {"pairs": n, "main_table_medians": medians,
                            "relaxed_excess_positive": counts[0],
                            "relaxed_excess_ci_lower_positive": counts[1],
                            "fp_no_witness_excess_positive": counts[2],
                            "fp_no_witness_excess_ci_lower_positive": counts[3],
                            "resolved_fraction_median": statistics.median(
                                float(x["resolution_fraction_mean"]) for x in r),
                            "common_six_fp_excess_median": common_median,
                            "common_six_fp_excess_positive": common_positive}
    return summary


def check_distance_sensitivity() -> dict:
    rows = read_csv("relaxed_gold_id_agreement_2026_10_13/run_pairs.csv")
    saved = json.loads((RESULTS / "paper_gold_agreement_sensitivity_2026_10_13.json").read_text())["results"]
    checked = 0
    for target, distance in itertools.product(("exact", "relaxed_05", "relaxed_075"),
                                               ("smaller_set", "jaccard")):
        for dataset in DATASETS:
            within: dict[str, list[float]] = defaultdict(list)
            between: dict[tuple[str, str], list[float]] = defaultdict(list)
            for row in rows:
                if row["dataset"] != dataset:
                    continue
                a, b, common = (int(row[f"{target}_{part}"]) for part in ("a", "b", "common"))
                denominator = min(a, b) if distance == "smaller_set" else a + b - common
                require(denominator > 0, "Undefined distance in saved pair counts")
                value = 1 - common / denominator
                if row["within_configuration"] == "True":
                    within[row["model_a"]].append(value)
                else:
                    between[tuple(sorted((row["model_a"], row["model_b"])))].append(value)
            reference = {model: statistics.mean(values) for model, values in within.items()}
            excess = [statistics.mean(values) - max(reference[a], reference[b])
                      for (a, b), values in between.items()]
            expected = saved[f"{target}_{distance}"][dataset]
            close(statistics.median(excess), expected["median_excess"], f"Sensitivity {target}/{distance}/{dataset}")
            require(sum(x > 0 for x in excess) == expected["positive_pairs"]
                    and len(excess) == expected["pairs"], "Sensitivity pair counts differ")
            checked += 1
    return {"distance_threshold_dataset_summaries_checked": checked,
            "genia_duplicate_removal": "Saved estimate only; assignments are not included",
            "raw_matching": "Not rerun"}


def check_calibration_and_nested() -> dict:
    calibration = read_csv("profile_closeness_calibration_2026_10_13/calibration_summary.csv")
    for row, dataset, n, passed in zip(calibration, DATASETS, (100, 100, 110), (0, 1, 9)):
        require(row["dataset"] == dataset, "Calibration corpus order changed")
        require(int(row["within_pairs"]) == n and int(row["original_within_joint_pass"]) == passed,
                f"Within-configuration calibration counts changed: {dataset}")
    contrasts = read_csv("relaxed_gold_id_agreement_2026_10_13/nested_contrasts.csv")
    require(len(contrasts) == 8, "Expected eight selected nesting contrasts")
    for row in contrasts:
        close(float(row["nonflat_between"]) - float(row["flat_between"]),
              float(row["delta_between"]), "Nested contrast arithmetic")
        require(float(row["delta_between_ci_low"]) > 0, "Nested interval no longer positive")
    ranges = {}
    for dataset, expected in (("GENIA", (".094", ".172")), ("CL-RuTerm3", (".098", ".136"))):
        values = [float(x["delta_between"]) for x in contrasts if x["dataset"] == dataset]
        require(len(values) == 4 and (decimal(min(values)), decimal(max(values))) == expected,
                f"Nested contrast range changed: {dataset}")
        ranges[dataset] = [min(values), max(values)]
    return {"within_configuration_joint_passes": [0, 1, 9],
            "nested_intervals_lower_positive": 8, "nested_difference_ranges": ranges}


def check_llm_repeats() -> dict:
    rows = read_csv("llm_repeat_output_agreement_2026_10_13/limitations_full_repeat_f1.csv")
    saved = json.loads((RESULTS / "llm_repeat_output_agreement_2026_10_13/limitations_full_repeat_f1_summary.json").read_text())
    require(len(rows) == 25 and len({x["pair_id"] for x in rows}) == 25, "Expected 25 unique LLM pairs")
    for row in rows:
        close(abs(float(row["base_f1"]) - float(row["repeat_f1"])),
              float(row["absolute_delta_unlabeled_f1"]), f"LLM pair {row['pair_id']}")
    result = {}
    for dataset, count in (("acl", 7), ("genia", 9), ("cl", 6)):
        values = [float(x["absolute_delta_unlabeled_f1"]) for x in rows
                  if x["dataset"] == dataset and x["decision"].startswith("primary_")]
        require(len(values) == count == saved["summary"][dataset]["n_pairs"], "Comparable LLM pair count changed")
        median, maximum = statistics.median(values), max(values)
        close(median, saved["summary"][dataset]["median_abs_delta_unlabeled_f1"], "LLM median")
        close(maximum, saved["summary"][dataset]["max_abs_delta_unlabeled_f1"], "LLM maximum")
        result[dataset] = {"comparable_pairs": count, "median_absolute_f1_delta": median,
                           "maximum_absolute_f1_delta": maximum}
    transports = {x["pair_id"]: float(x["absolute_delta_unlabeled_f1"]) for x in rows
                  if x["decision"] == "transport_contrast"}
    require(set(transports) == {"P03", "P17", "P21"}, "Transport contrasts changed")
    for row in saved["transport_contrasts"]:
        close(transports[row["pair_id"]], row["absolute_delta_unlabeled_f1"], "Transport contrast")
    cl_without_llama = [float(x["absolute_delta_unlabeled_f1"]) for x in rows
                        if x["dataset"] == "cl" and x["decision"].startswith("primary_")
                        and x["model_key"] != "llama_4_maverick"]
    close(max(cl_without_llama), saved["sensitivity"]["cl_without_llama_format_failure_pair"]["max_abs_delta_unlabeled_f1"],
          "CL LLM sensitivity")
    return {"comparable": result, "transport_contrasts": transports,
            "cl_maximum_without_llama": max(cl_without_llama)}


def main() -> None:
    result = {"status": "PASS", "sources": check_source_files(),
              "headlines": check_headlines(),
              "sensitivity": check_distance_sensitivity(),
              "calibration_and_nesting": check_calibration_and_nested(),
              "llm_repeats": check_llm_repeats(),
              "scope": "Saved aggregate arithmetic against retained numerical references; no raw matching or bootstrap"}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
