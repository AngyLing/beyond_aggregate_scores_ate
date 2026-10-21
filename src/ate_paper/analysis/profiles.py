"""Rebuild supervised recall profiles and descriptive screen calibration.

Only the standard library is required. Slice definitions and greedy boundary
matching follow the independently verified ACL/GENIA profile implementations;
CL uses the same slice definition and native greedy boundary convention.
"""
from __future__ import annotations

import heapq
import itertools
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from ate_paper.io import read_csv, write_csv

from ate_paper.predictions import CorpusIndex, inside_root, read_jsonl, sha256

DATASETS = {"ACL RD-TEC 2.0": "acl", "GENIA": "genia", "CL-RuTerm3": "cl"}
UNTYPED = {"mbert_unlabeled_bio", "xlmr_span_unlabeled", "rubert_span_unlabeled"}
LENGTHS = ("1", "2", "3", "4-5", "6+")
FREQUENCIES = ("0", "1", "2-5", "6+")
PR_ORIGINAL = 0.02
PROFILE_ORIGINAL = 0.05


def real_rows(rows):
    return [r for r in rows if r.get("unit_type") == "span_mention"
            and r.get("start") is not None and r.get("end") is not None]


def base_key(row):
    return row.get("doc_id"), row.get("sent_id"), row["start"], row["end"]


def norm_value(row):
    return " ".join(str(row.get("norm") or row.get("surface") or "").split()).lower()


def token_count(row):
    surface = row.get("surface")
    if not surface:
        surface = row.get("text", "")[row["start"]:row["end"]]
    return len(surface.split())


def length_bin(length):
    return str(max(1, length)) if length <= 3 else "4-5" if length <= 5 else "6+"


def frequency_bin(count):
    return "0" if count == 0 else "1" if count == 1 else "2-5" if count <= 5 else "6+"


def nested_categories(keys):
    units = defaultdict(set)
    for key in keys:
        units[key[:2]].add(key)
    result = {}
    for spans in units.values():
        for key in spans:
            parents = children = False
            for other in spans - {key}:
                parents |= other[2] <= key[2] and key[3] <= other[3]
                children |= key[2] <= other[2] and other[3] <= key[3]
            result[key] = ("both" if parents and children else "inner-only" if parents
                           else "outer-only" if children else "flat")
    return result


def describe_gold(train_rows, gold_rows):
    train = {}
    for row in real_rows(train_rows):
        key = base_key(row)
        if key in train and norm_value(train[key]) != norm_value(row):
            raise ValueError(f"Ambiguous duplicate training offset: {key}")
        train[key] = row  # Same offset counts once; last saved row is retained.
    counts = Counter(norm_value(row) for row in train.values())
    annotations = defaultdict(list)
    for row in real_rows(gold_rows):
        annotations[base_key(row)].append(row)
    categories = nested_categories(annotations)
    descriptors, slices = {}, defaultdict(set)
    for key, rows in annotations.items():
        norms = {norm_value(row) for row in rows}
        lengths = {token_count(row) for row in rows}
        if len(norms) != 1 or len(lengths) != 1:
            raise ValueError(f"Ambiguous duplicate gold offset: {key}")
        norm, length = next(iter(norms)), next(iter(lengths))
        n_train = counts[norm]
        lb, fb = length_bin(length), frequency_bin(n_train)
        classes = sorted({str(row.get("class_label")) for row in rows})
        descriptors[key] = {"norm": norm, "train_count": n_train, "length": length,
                            "length_bin": lb, "train_frequency_bin": fb,
                            "nesting": categories[key], "classes": classes}
        for family, value in [("length", lb), ("train_frequency", fb),
                              ("nesting", categories[key]), ("length_x_frequency", lb + "|" + fb)]:
            slices[family, value].add(key)
        for label in classes:
            slices["gold_class", label].add(key)
    for family, values in [("length", LENGTHS), ("train_frequency", FREQUENCIES),
                           ("nesting", ("flat", "inner-only", "outer-only", "both")),
                           ("length_x_frequency", [a + "|" + b for a in LENGTHS for b in FREQUENCIES])]:
        for value in values:
            slices[family, value]
    return descriptors, dict(slices)


def exact_f1(gold, pred):
    return 2 * len(gold & pred) / (len(gold) + len(pred)) if gold or pred else 0.0


def soft_matches(gold_rows, pred_rows):
    """Greedy IoU >= 1/2, ties use descending saved gold/pred indices.

    An exact rational priority queue implements the saved scorer's specified
    ordering independently. Deduplication retains the first occurrence per unit.
    """
    groups = []
    for rows in (gold_rows, pred_rows):
        grouped = defaultdict(dict)
        for row in real_rows(rows):
            key = base_key(row)
            grouped[key[:2]].setdefault(key[2:], None)
        groups.append(grouped)
    count = 0
    for unit in groups[0].keys() & groups[1].keys():
        queue = []
        for gi, (gs, ge) in enumerate(groups[0][unit]):
            for pi, (ps, pe) in enumerate(groups[1][unit]):
                intersection = min(ge, pe) - max(gs, ps)
                union = max(ge, pe) - min(gs, ps)
                if intersection > 0 and 2 * intersection >= union:
                    heapq.heappush(queue, (-Fraction(intersection, union), -gi, -pi))
        used_gold, used_pred = set(), set()
        while queue:
            _, gi, pi = heapq.heappop(queue)
            if gi not in used_gold and pi not in used_pred:
                used_gold.add(gi)
                used_pred.add(pi)
                count += 1
    return count


def nearest_rank(values, quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def distance(a: dict, b: dict, supports: dict) -> dict:
    keys = sorted(a.keys() & b.keys())
    if not keys:
        raise ValueError("No common applicable active profile coordinates")
    deltas = {key: abs(a[key] - b[key]) for key in keys}
    worst = max(keys, key=lambda key: deltas[key])
    family_values = defaultdict(list)
    for key, value in deltas.items():
        family = key.split(":", 1)[0]
        if family == "gap":
            family_values[key].append((value, 1))
        else:
            weight = supports[key]
            if weight is None or weight < 30:
                raise ValueError("Active recall coordinate lacks support >=30")
            family_values[family].append((value, weight))
    family_distance = {family: sum(v * w for v, w in pairs) / sum(w for _v, w in pairs)
                       for family, pairs in family_values.items()}
    return {"maximum": deltas[worst], "weighted_family_maximum": max(family_distance.values()),
            "worst_coordinate": worst, "worst_support": supports[worst],
            "coordinates": len(keys), "family_distances": family_distance}


def pair_record(dataset, left, right, profiles, supports):
    d = distance(profiles[left["run"]], profiles[right["run"]], supports)
    delta_p = abs(float(left["precision"]) - float(right["precision"]))
    delta_r = abs(float(left["recall"]) - float(right["recall"]))
    pr_max = max(delta_p, delta_r)
    return {"dataset": dataset, "model_a": left["model"], "model_b": right["model"],
            "run_a": left["run"], "run_b": right["run"],
            "within_configuration": left["model"] == right["model"],
            "delta_p": delta_p, "delta_r": delta_r, "pr_max": pr_max,
            "maximum": d["maximum"], "weighted_family_maximum": d["weighted_family_maximum"],
            "joint_scale": max(pr_max / PR_ORIGINAL, d["maximum"] / PROFILE_ORIGINAL),
            "worst_coordinate": d["worst_coordinate"], "worst_support": d["worst_support"],
            "coordinates": d["coordinates"]}


def screens(within, cross, means, pr_margin: float, profile_margin: float):
    def passes(row):
        return row["pr_max"] <= pr_margin + 1e-12 and row["maximum"] <= profile_margin + 1e-12
    mean_pass = sum(passes(row) for row in means)
    groups = defaultdict(list)
    for row in cross:
        groups[row["model_a"], row["model_b"]].append(row)
    stable_pass = sum(passes(row) and sum(passes(item) for item in groups[row["model_a"], row["model_b"]]) >= 20
                      for row in means)
    return {"within_joint_pass": sum(passes(row) for row in within),
            "within_pr_pass": sum(row["pr_max"] <= pr_margin + 1e-12 for row in within),
            "within_profile_pass": sum(row["maximum"] <= profile_margin + 1e-12 for row in within),
            "mean_configuration_pairs_pass": mean_pass,
            "stable_configuration_pairs_pass": stable_pass}


def gap_context(gold_rows: list[dict], predicted_rows: list[dict],
                descriptors: dict, typed: bool) -> dict:
    gold, predicted = ({base_key(row) for row in real_rows(rows)}
                       for rows in (gold_rows, predicted_rows))
    gold_typed, predicted_typed = (
        {base_key(row) + (row.get("class_label"),) for row in real_rows(rows)}
        for rows in (gold_rows, predicted_rows))
    matched = gold & predicted
    strict = exact_f1(gold, predicted)
    typed_f1 = exact_f1(gold_typed, predicted_typed)
    soft_tp = soft_matches(gold_rows, predicted_rows)
    soft_f1 = 2 * soft_tp / (len(gold) + len(predicted)) if gold or predicted else 0.0
    inner = {key for key, value in descriptors.items()
             if value["nesting"] in ("inner-only", "both")}
    outer = {key for key, value in descriptors.items()
             if value["nesting"] in ("outer-only", "both")}
    nested_gap = (len(matched & outer) / len(outer) - len(matched & inner) / len(inner)
                  if inner and outer else None)
    return {
        "gold_offsets": len(gold), "prediction_offsets": len(predicted),
        "exact_tp": len(matched), "strict_f1": strict,
        "typed_f1_arithmetic": typed_f1, "typed_applicable": typed,
        "soft_matches": soft_tp, "soft_f1": soft_f1,
        "inner_support": len(inner), "outer_support": len(outer),
        "BoundaryGap": soft_f1 - strict,
        "NestedGap": nested_gap,
        "ClassGap": strict - typed_f1 if typed else None,
    }


def rebuild_dataset(root: Path, gold_dir: Path, dataset: str,
                    public_manifest: list[dict], prediction_root: Path | None = None) -> tuple:
    slug = DATASETS[dataset]
    train_path, test_path = (gold_dir / slug / f"{split}.jsonl"
                             for split in ("train", "test"))
    train_rows, gold_rows = list(read_jsonl(train_path)), list(read_jsonl(test_path))
    if any(row.get("dataset") != dataset for row in train_rows + gold_rows):
        raise ValueError(f"Local corpus dataset mismatch: {dataset}")
    if any(row.get("split") != split for split, rows in
           (("train", train_rows), ("test", gold_rows)) for row in rows):
        raise ValueError(f"Local corpus split mismatch: {dataset}")
    from .io import validate_test_rows
    validate_test_rows(gold_rows, dataset)
    index = CorpusIndex.build(gold_rows, dataset)
    descriptors, slices = describe_gold(train_rows, gold_rows)
    gold = set(descriptors)
    selected = [entry for entry in public_manifest
                if entry["dataset"] == dataset and entry["group"] == "supervised"]
    if len(selected) != (55 if dataset == "CL-RuTerm3" else 50):
        raise ValueError(f"Unexpected supervised roster: {dataset}")
    profiles, scores, contexts = {}, [], []
    exported = []
    input_paths = [train_path, test_path]
    supports = {family + ":" + value: len(members)
                for (family, value), members in slices.items()}
    for entry in sorted(selected, key=lambda row: row["run"]):
        run = entry["run"]
        model, seed = run.rsplit("_seed", 1)
        typed = model not in UNTYPED
        from .io import load_prediction_rows
        # Hydration recovers only local surfaces; slices use gold/train norms.
        predicted_rows, path = load_prediction_rows(root, entry, index, prediction_root)
        predicted = {base_key(row) for row in real_rows(predicted_rows)}
        matched = gold & predicted
        scores.append({"dataset": dataset, "model": model, "group": "supervised",
                       "seed": int(seed), "run": run, "predicted": len(predicted),
                       "tp": len(matched), "gold": len(gold),
                       "precision": len(matched) / len(predicted) if predicted else 0.0,
                       "recall": len(matched) / len(gold) if gold else 0.0,
                       "f1": exact_f1(gold, predicted), "class_capable": typed})
        context = gap_context(gold_rows, predicted_rows, descriptors, typed)
        contexts.append({"dataset": dataset, "run": run, "model": model, **context})
        active = {}
        for (family, value), members in sorted(slices.items()):
            support = len(members)
            recall = len(matched & members) / support if support else None
            exported.append({"dataset": dataset, "run": run, "model": model,
                             "family": family, "slice": value, "support": support,
                             "active": support >= 30,
                             "core": family not in ("gold_class", "length_x_frequency"),
                             "value": recall})
            if support >= 30:
                active[family + ":" + value] = recall
        for gap in ("BoundaryGap", "NestedGap", "ClassGap"):
            if context[gap] is None:
                continue
            active["gap:" + gap] = context[gap]
            supports["gap:" + gap] = None
            exported.append({"dataset": dataset, "run": run, "model": model,
                             "family": "gap", "slice": gap, "support": None,
                             "active": True, "core": True, "value": context[gap]})
        profiles[run] = active
        input_paths.append(path)
    slice_rows = [{"dataset": dataset, "family": family, "slice": value,
                   "support": len(members), "active": len(members) >= 30,
                   "core": family not in ("gold_class", "length_x_frequency")}
                  for (family, value), members in sorted(slices.items())]
    return scores, profiles, supports, exported, contexts, slice_rows, input_paths


def calibrate(dataset: str, scores: list[dict], profiles: dict,
              supports: dict) -> tuple:
    model_runs = defaultdict(list)
    for row in scores:
        model_runs[row["model"]].append(row)
    if any(len(rows) != 5 or {row["seed"] for row in rows} != set(range(42, 47))
           for rows in model_runs.values()):
        raise ValueError("Expected training seeds 42 through 46 for every configuration")
    within = [pair_record(dataset, left, right, profiles, supports)
              for model in sorted(model_runs)
              for left, right in itertools.combinations(model_runs[model], 2)]
    cross, means = [], []
    for model_a, model_b in itertools.combinations(sorted(model_runs), 2):
        left, right = model_runs[model_a], model_runs[model_b]
        cross.extend(pair_record(dataset, a, b, profiles, supports)
                     for a, b in itertools.product(left, right))
        mean_profiles = []
        for runs in (left, right):
            shared = set.intersection(*(set(profiles[row["run"]]) for row in runs))
            mean_profiles.append({key: sum(profiles[row["run"]][key] for row in runs) / 5
                                  for key in shared})
        d = distance(*mean_profiles, supports)
        dp = abs(sum(row["precision"] for row in left) / 5
                 - sum(row["precision"] for row in right) / 5)
        dr = abs(sum(row["recall"] for row in left) / 5
                 - sum(row["recall"] for row in right) / 5)
        means.append({"dataset": dataset, "model_a": model_a, "model_b": model_b,
                      "delta_p": dp, "delta_r": dr, "pr_max": max(dp, dr),
                      "maximum": d["maximum"],
                      "weighted_family_maximum": d["weighted_family_maximum"],
                      "worst_coordinate": d["worst_coordinate"],
                      "coordinates": d["coordinates"]})
    p80 = nearest_rank([row["pr_max"] for row in within], .8)
    m80 = nearest_rank([row["maximum"] for row in within], .8)
    k80 = nearest_rank([row["joint_scale"] for row in within], .8)
    original = screens(within, cross, means, PR_ORIGINAL, PROFILE_ORIGINAL)
    componentwise = screens(within, cross, means, p80, m80)
    radial = screens(within, cross, means, PR_ORIGINAL * k80, PROFILE_ORIGINAL * k80)
    leave_one_out = []
    for model in sorted(model_runs):
        training = [row for row in within if row["model_a"] != model]
        held = [row for row in within if row["model_a"] == model]
        p = nearest_rank([row["pr_max"] for row in training], .8)
        m = nearest_rank([row["maximum"] for row in training], .8)
        k = nearest_rank([row["joint_scale"] for row in training], .8)
        leave_one_out.append({
            "radial": sum(row["joint_scale"] <= k + 1e-12 for row in held),
            "componentwise": sum(row["pr_max"] <= p + 1e-12
                                 and row["maximum"] <= m + 1e-12 for row in held)})
    summary = {
        "dataset": dataset, "configurations": len(model_runs), "within_pairs": len(within),
        "cross_configuration_pairs": len(means), "cross_run_pairs": len(cross),
        "pr_q50": nearest_rank([row["pr_max"] for row in within], .5),
        "pr_q80": p80, "pr_q90": nearest_rank([row["pr_max"] for row in within], .9),
        "profile_q50": nearest_rank([row["maximum"] for row in within], .5),
        "profile_q80": m80,
        "profile_q90": nearest_rank([row["maximum"] for row in within], .9),
        "weighted_q80": nearest_rank([row["weighted_family_maximum"] for row in within], .8),
        "joint_scale_q80": k80, "radial_pr_margin": PR_ORIGINAL * k80,
        "radial_profile_margin": PROFILE_ORIGINAL * k80,
        "loo_radial_pass_total": sum(row["radial"] for row in leave_one_out),
        "loo_radial_pass_min_model": min(row["radial"] for row in leave_one_out),
        "loo_radial_pass_max_model": max(row["radial"] for row in leave_one_out),
        "loo_componentwise_pass_total": sum(row["componentwise"] for row in leave_one_out),
        **{"original_" + key: value for key, value in original.items()},
        **{"componentwise_q80_" + key: value for key, value in componentwise.items()},
        **{"radial_q80_" + key: value for key, value in radial.items()},
    }
    return within, cross, means, summary


def check_saved(computed: list[dict], reference: list[dict], key_fields: tuple,
                fields: tuple | None = None) -> dict:
    def key(row):
        return tuple(str(row[field]) for field in key_fields)
    wanted = {key(row): row for row in computed}
    old = {key(row): row for row in reference if key(row) in wanted}
    if len(wanted) != len(computed) or set(old) != set(wanted):
        raise ValueError("Saved result roster differs from reconstructed output")
    checked = 0
    for identity, row in wanted.items():
        for field in fields or tuple(row):
            if field in key_fields or field not in old[identity]:
                continue
            expected, actual = old[identity][field], row[field]
            if actual is None:
                equal = expected in ("", None)
            elif isinstance(actual, bool):
                equal = expected == str(actual)
            elif isinstance(actual, (float, int)):
                equal = expected not in ("", None) and math.isclose(
                    actual, float(expected), rel_tol=0, abs_tol=1e-10)
            else:
                equal = str(actual) == expected
            if not equal:
                raise ValueError(f"Saved profile mismatch: {identity} {field}: {actual} != {expected}")
            checked += 1
    return {"rows": len(computed), "cells_checked": checked, "absolute_tolerance": 1e-10}


def reproduce_profiles(root: Path, gold_dir: Path, output_dir: Path,
                       prediction_root: Path | None = None) -> dict:
    from .io import compare_rows
    def compare(computed, reference, key_fields, fields=None):
        if prediction_root is None:
            return check_saved(computed, reference, key_fields, fields)
        return compare_rows(computed, reference, key_fields, fields)
    public = read_csv(root / "predictions/manifest.csv")
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = defaultdict(list)
    input_paths = {root / "predictions/manifest.csv"}
    checks = {}
    for dataset in DATASETS:
        scores, profiles, supports, rows, gaps, slices, paths = rebuild_dataset(
            root, gold_dir, dataset, public, prediction_root)
        print(f"{dataset}: reconstructed {len(scores)} supervised profiles", flush=True)
        within, cross, means, summary = calibrate(dataset, scores, profiles, supports)
        outputs["threshold_grid"].extend(
            {"dataset": dataset, "pr_margin": pm, "profile_margin": fm,
             **screens(within, cross, means, pm, fm)}
            for pm in (0.02, 0.03, 0.05, 0.10)
            for fm in (0.05, 0.10, 0.15, 0.20, 0.30))
        for name, data in (("run_scores", scores), ("profiles", rows), ("gap_context", gaps),
                           ("slice_supports", slices), ("within_pairs", within),
                           ("cross_pairs", cross), ("configuration_pairs", means),
                           ("calibration_summary", [summary])):
            outputs[name].extend(data)
        if dataset == "CL-RuTerm3":
            prefix = "cl_ruterm3_1_1_output_agreement_2026_10_13/"
        else:
            prefix = DATASETS[dataset] + "_v1_profile_recheck_"
        profile_reference = root / "results" / (prefix + "profiles.csv")
        score_reference = root / "results" / (prefix + "run_scores.csv")
        run_names = {row["run"] for row in scores}
        retained_profiles = [row for row in read_csv(profile_reference)
                             if row["dataset"] == dataset and row["run"] in run_names]
        if prediction_root is None and len(retained_profiles) != len(rows):
            raise ValueError(f"Applicable profile inventory differs from saved source: {dataset}")
        retained_scores = [row for row in read_csv(score_reference)
                           if row["dataset"] == dataset and row["run"] in run_names]
        if dataset == "CL-RuTerm3":
            retained_scores = [{**row, "class_capable": row["typed_applicable"]}
                               for row in retained_scores]
        configuration_reference = root / "results" / (prefix + "configuration_pairs.csv")
        retained_pairs = [row for row in read_csv(configuration_reference)
                          if row["dataset"] == dataset
                          and row["group_a"] == row["group_b"] == "supervised"]
        if dataset == "CL-RuTerm3":
            retained_pairs = [{**row, "maximum": row["maximum_profile_distance"],
                               "weighted_family_maximum": row["weighted_family_distance"],
                               "coordinates": row["active_profile_coordinates"]}
                              for row in retained_pairs]
        checks[dataset] = {
            "profiles": compare(rows, retained_profiles,
                                     ("dataset", "run", "family", "slice")),
            "scores": compare(scores, retained_scores, ("dataset", "run")),
            "configuration_means": compare(
                means, retained_pairs, ("dataset", "model_a", "model_b"),
                ("delta_p", "delta_r", "maximum", "weighted_family_maximum", "coordinates")),
        }
        input_paths.update(paths + [profile_reference, score_reference, configuration_reference])
    calibration_path = root / "results/profile_closeness_calibration_2026_10_13/calibration_summary.csv"
    checks["calibration"] = compare(outputs["calibration_summary"],
                                         read_csv(calibration_path), ("dataset",))
    input_paths.add(calibration_path)
    calibration_dir = root / "results/profile_closeness_calibration_2026_10_13"
    for name, key_fields in (("within_pairs", ("dataset", "run_a", "run_b")),
                             ("configuration_pairs", ("dataset", "model_a", "model_b"))):
        path = calibration_dir / (name + ".csv")
        if path.exists():
            checks[name] = compare(outputs[name], read_csv(path), key_fields)
            input_paths.add(path)
    grid_path = calibration_dir / "threshold_grid.csv"
    checks["threshold_grid"] = compare(outputs["threshold_grid"], read_csv(grid_path),
                                        ("dataset", "pr_margin", "profile_margin"))
    input_paths.add(grid_path)
    for name, rows in outputs.items():
        write_csv(output_dir / (name + ".csv"), rows)
    def receipt(path):
        if path.is_relative_to(root):
            name = path.relative_to(root).as_posix()
        elif path.is_relative_to(gold_dir):
            name = "LOCAL_PROCESSED/" + path.relative_to(gold_dir).as_posix()
        elif prediction_root is not None and path.resolve().is_relative_to(prediction_root.resolve()):
            name = "LOCAL_PREDICTIONS/" + path.resolve().relative_to(prediction_root.resolve()).as_posix()
        else:
            name = path.name
        return {"path": name, "sha256": sha256(path), "bytes": path.stat().st_size}
    verification = {
        "status": "PASS" if prediction_root is None else "COMPUTED",
        "input_mode": "published_predictions" if prediction_root is None else "new_predictions",
        "prediction_root": str(prediction_root.resolve()) if prediction_root is not None else None,
        "scope": "155 selected supervised outputs / 31 configurations",
        "profile_support_minimum": 30,
        "original_margins": {"precision_recall": PR_ORIGINAL, "maximum_profile": PROFILE_ORIGINAL},
        "interpretation": "Descriptive fixed-output checks; empirical q80 margins are post-hoc references, not calibrated equivalence boundaries.",
        "checks": checks, "inputs": [receipt(path) for path in sorted(input_paths)],
        "implementation": [receipt(Path(__file__)),
                           receipt(Path(__file__).parents[1] / "io.py"),
                           receipt(root / "scripts/reproduce_profiles.py")],
    }
    (output_dir / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    return verification
