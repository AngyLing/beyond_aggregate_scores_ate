"""Fixed-output IoU/distance, duplicate and CL assignment sensitivities.

Adapted from the study's two saved sensitivity verifiers. Nothing runs on import.
No model is trained and no new prediction or matching threshold is fitted here.
"""
from __future__ import annotations

import csv
import gzip
import itertools
import json
import math
import statistics
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from .io import DATASETS, SLUGS, nesting_categories, read_csv, span_set
from .matching import _components, _solve, match_unit, overlap_fraction
from .statistics import ratio
from ate_paper.predictions import sha256

GENIA_DUPLICATE = "MEDLINE:97218353#2"
TARGETS = ("exact", "relaxed_05", "relaxed_075")
MODEL_PAIRS = (("mBERT", "mbert_class_bio", "mbert_span"),
               ("XLM-R", "xlmr_bio", "xlmr_span_classifier"),
               ("RuBERT", "rubert_bio", "rubert_span"),
               ("ruRoBERTa", "ruroberta_bio", "ruroberta_span"))
# Explicitly fail rather than claim a complete sensitivity after truncation.
MAX_ENUMERATION_STATES = 100000


def finite(value):
    return float(value) if value is not None and math.isfinite(value) else None


def complete_mean(values):
    """An undefined run-pair value invalidates its fixed-population mean."""
    return statistics.mean(values) if values and all(x is not None for x in values) else None


def distance(row, target, metric):
    a, b, common = (int(row[f"{target}_{part}"]) for part in ("a", "b", "common"))
    if min(a, b, common) < 0 or common > min(a, b):
        raise ValueError("Inconsistent covered-gold counts")
    denominator = min(a, b) if metric == "smaller_set" else a + b - common
    return 1 - common / denominator if denominator > 0 else None


def aggregate(run_pairs, target, metric):
    result = []
    for dataset in sorted({r["dataset"] for r in run_pairs}):
        rows = [r for r in run_pairs if r["dataset"] == dataset]
        models = sorted({r["model_a"] for r in rows if str(r["within_configuration"]).lower() == "true"})
        within = {model: complete_mean([distance(r, target, metric) for r in rows
                                       if r["model_a"] == r["model_b"] == model]) for model in models}
        for a, b in itertools.combinations(models, 2):
            between = complete_mean([distance(r, target, metric) for r in rows
                                     if {r["model_a"], r["model_b"]} == {a, b}])
            reference = max(within[a], within[b]) if within[a] is not None and within[b] is not None else None
            excess = between - reference if between is not None and reference is not None else None
            result.append({"dataset": dataset, "model_a": a, "model_b": b,
                           "target": target, "distance": metric, "between": between,
                           "within_reference": reference, "excess": excess})
    return result


def summary(rows):
    values = [r["excess"] for r in rows if r["excess"] is not None]
    return {"median_excess": statistics.median(values) if values else None,
            "positive_pairs": sum(x > 0 for x in values), "pairs": len(rows),
            "defined_pairs": len(values), "undefined_pairs": len(rows) - len(values)}


def metric_summaries(run_pairs):
    all_rows, results, changes = {}, {}, {}
    for target in TARGETS:
        for metric in ("smaller_set", "jaccard"):
            key = target + "_" + metric
            all_rows[key] = aggregate(run_pairs, target, metric)
            results[key] = {dataset: summary([r for r in all_rows[key] if r["dataset"] == dataset])
                            for dataset in sorted({r["dataset"] for r in run_pairs})}
        d = {(r["dataset"], r["model_a"], r["model_b"]): r["excess"] for r in all_rows[target + "_smaller_set"]}
        j = {(r["dataset"], r["model_a"], r["model_b"]): r["excess"] for r in all_rows[target + "_jaccard"]}
        for dataset in sorted({r["dataset"] for r in run_pairs}):
            keys = [k for k in d if k[0] == dataset and d[k] is not None and j[k] is not None]
            changes[dataset + "/" + target] = {
                "smaller_set_negative_jaccard_positive": sum(d[k] < 0 < j[k] for k in keys),
                "smaller_set_positive_jaccard_negative": sum(j[k] < 0 < d[k] for k in keys),
                "defined_pairs": len(keys)}
    acl = changes.get("ACL RD-TEC 2.0/relaxed_05")
    if acl:
        results["acl_metric_sign_changes"] = {k: v for k, v in acl.items() if k != "defined_pairs"}
    results["metric_sign_changes"] = changes
    return results, [r for values in all_rows.values() for r in values]


def check_configuration_excess(detail, configurations):
    """Retain the original per-configuration producer consistency check."""
    saved = {(r["dataset"], frozenset((r["model_a"], r["model_b"]))): r for r in configurations}
    checked = 0
    for row in detail:
        if row["distance"] != "smaller_set":
            continue
        value = saved[row["dataset"], frozenset((row["model_a"], row["model_b"]))][row["target"] + "_excess"]
        expected = finite(float(value)) if value not in (None, "") else None
        actual = row["excess"]
        if (actual is None) != (expected is None) or (actual is not None and
                not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10)):
            raise ValueError("Smaller-set excess differs from gold-producer configuration: " + row["target"])
        checked += 1
    return checked


def load_assignments(path, datasets):
    rows = []
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["dataset"] in datasets:
                row["sent_id"] = row["sent_id"] or None
                rows.append(row)
    return rows


def gold_id(row):
    return (row["doc_id"], row["sent_id"], int(row["gold_start"]), int(row["gold_end"]))


def genia_without_duplicate(run_pairs, assignments, scores):
    covered = {r["run"]: set() for r in scores if r["dataset"] == "GENIA"}
    full = defaultdict(set)
    for row in assignments:
        if row["dataset"] != "GENIA":
            continue
        key = gold_id(row)
        if key in full[row["run"]]:
            raise ValueError("Duplicate GENIA gold-ID assignment")
        full[row["run"]].add(key)
        if row["doc_id"] != GENIA_DUPLICATE:
            covered[row["run"]].add(key)
    for row in scores:
        if row["dataset"] == "GENIA" and len(full[row["run"]]) != int(row["relaxed_05_tp"]):
            raise ValueError("GENIA assignment population differs from gold-producer scores")
    changed = []
    for row in run_pairs:
        if row["dataset"] == "GENIA":
            a, b = covered[row["run_a"]], covered[row["run_b"]]
            changed.append({**row, "relaxed_05_a": len(a), "relaxed_05_b": len(b),
                            "relaxed_05_common": len(a & b)})
    result = summary(aggregate(changed, "relaxed_05", "smaller_set"))
    result["excluded_test_doc_id"] = GENIA_DUPLICATE
    return result


def optimal_covered_sets(gold_offsets, prediction_offsets, max_states=MAX_ENUMERATION_STATES):
    """All covered-gold sets at the producer's cardinality/total-IoU optimum.

    Exact anchors stay fixed. Connected residual components are enumerated with
    solver-based pruning. Equal assignments with the same covered set collapse.
    An explicit resource limit raises an error; no alternatives are truncated.
    """
    exact = set(gold_offsets) & set(prediction_offsets)
    gs, ps = sorted(set(gold_offsets) - exact), sorted(set(prediction_offsets) - exact)
    edges = {(i, j): overlap_fraction(g, p) for i, g in enumerate(gs) for j, p in enumerate(ps)
             if overlap_fraction(g, p) >= Fraction(1, 2)}
    components = []
    states = 0
    for gi, pi in _components(edges):
        local = {e: v for e, v in edges.items() if e[0] in gi and e[1] in pi}
        solved = _solve(gi, pi, local)
        if solved is None:
            raise ValueError("Residual assignment has no feasible solution")
        optimum = solved[1]
        if optimum[0] == len(gi):
            components.append([frozenset(gs[i] for i in gi)])
            continue
        found = set()

        def visit(position, forced, forbidden):
            nonlocal states
            states += 1
            if states > max_states:
                raise ValueError("Unsupported CL tie size: complete optimal-set enumeration exceeds resource limit")
            candidate = _solve(gi, pi, local, forced, forbidden)
            if candidate is None or candidate[1] != optimum:
                return
            if position == len(gi):
                found.add(frozenset(gs[i] for i, _ in candidate[0]))
                return
            g = gi[position]
            used_predictions = {p for _, p in forced}
            for edge in sorted(local):
                if edge[0] == g and edge[1] not in used_predictions:
                    visit(position + 1, forced + (edge,), forbidden)
            incident = frozenset(e for e in local if e[0] == g)
            visit(position + 1, forced, forbidden | incident)
        visit(0, (), frozenset())
        components.append(sorted(found, key=lambda values: sorted(values)))
    result = {frozenset(exact)}
    for choices in components:
        if len(result) * len(choices) > max_states:
            raise ValueError("Unsupported CL tie size: too many optimal covered-gold sets")
        result = {left | right for left in result for right in choices}
    return sorted(result, key=lambda values: sorted(values))


def vector(keys, lookup):
    return np.bincount([lookup[item[0]] for item in keys], minlength=len(lookup)).astype(float)


def slice_stat(sets, bio_runs, span_runs, weights, lookup):
    pairs = (list(itertools.product(bio_runs, span_runs)),
             list(itertools.combinations(bio_runs, 2)), list(itertools.combinations(span_runs, 2)))
    runs = bio_runs + span_runs
    vectors = {run: vector(sets[run], lookup) for run in runs} if weights is not None else {}
    weighted = {run: weights @ vectors[run] for run in runs} if weights is not None else {}

    def mean_for(population):
        point, cloud = [], []
        for a, b in population:
            aa, bb, common = len(sets[a]), len(sets[b]), len(sets[a] & sets[b])
            denominator = min(aa, bb)
            point.append((denominator - common) / denominator if denominator else float("nan"))
            if weights is not None:
                denominator_draw = np.minimum(weighted[a], weighted[b])
                shared = weights @ vector(sets[a] & sets[b], lookup)
                cloud.append(ratio(denominator_draw - shared, denominator_draw))
        return (float(np.mean(point)) if point else float("nan"),
                np.mean(cloud, axis=0) if cloud else None)
    between, wa, wb = map(mean_for, pairs)
    point = (between[0], between[0] - float(np.maximum(wa[0], wb[0])))
    draws = (between[1], between[1] - np.maximum(wa[1], wb[1])) if weights is not None else (None, None)
    return point, draws


def mixed_choices(counts):
    """First local unit varies fastest, matching the original binary masks."""
    total = math.prod(counts)
    if total > MAX_ENUMERATION_STATES:
        raise ValueError("Unsupported CL tie size: too many per-encoder combinations")
    for index in range(total):
        value, choice = index, []
        for count in counts:
            choice.append(value % count)
            value //= count
        yield index, choice


def cl_ambiguity(specs, gold_rows, prediction_rows, assignment_rows, ambiguous_rows,
                 scores, docs=None, weights=None):
    dataset = "CL-RuTerm3"
    gold = span_set(gold_rows, dataset)
    categories = nesting_categories(gold)
    flat = {key for key, category in categories.items() if category == "flat"}
    selected_models = {m for _, a, b in MODEL_PAIRS for m in (a, b)}
    groups = {m: sorted(r["run"] for r in specs if r["model"] == m) for m in selected_models}
    if any(len(v) < 2 for v in groups.values()):
        raise ValueError("Selected CL contrasts require within-configuration run pairs")
    selected = set(itertools.chain.from_iterable(groups.values()))
    covered = {run: set() for run in selected}
    by_unit = defaultdict(list)
    for row in assignment_rows:
        if row["dataset"] == dataset and row["run"] in selected:
            key = gold_id(row)
            if key not in gold or key in covered[row["run"]]:
                raise ValueError("Invalid or duplicate CL assigned gold ID")
            covered[row["run"]].add(key)
            by_unit[row["run"], row["doc_id"], row["sent_id"]].append(row)
    for row in scores:
        if row["dataset"] == dataset and row["run"] in selected and len(covered[row["run"]]) != int(row["relaxed_05_tp"]):
            raise ValueError("CL assignment population differs from gold-producer scores")
    gold_units = defaultdict(set)
    for doc, sent, start, end in gold:
        gold_units[doc, sent].add((start, end))
    options = []
    wanted = {(r["run"], r["doc_id"], r["sent_id"] or None) for r in ambiguous_rows
              if r["dataset"] == dataset and r["run"] in selected}
    for run, doc, sent in sorted(wanted, key=lambda x: (x[0], x[1], x[2] or "")):
        preds = {(a, b) for d, s, a, b in span_set(prediction_rows[run], dataset) if (d, s) == (doc, sent)}
        choices = optimal_covered_sets(gold_units[doc, sent], preds)
        saved_pairs = {((int(r["gold_start"]), int(r["gold_end"])),
                        (int(r["pred_start"]), int(r["pred_end"]))) for r in by_unit[run, doc, sent]}
        anchored, _ = match_unit(gold_units[doc, sent], preds, Fraction(1, 2))
        if saved_pairs != {(g, p) for g, p, *_ in anchored}:
            raise ValueError("CL ambiguous-unit assignments differ from selected raw inputs")
        canonical = frozenset(g for g, _ in saved_pairs)
        if canonical not in choices:
            raise ValueError("Producer assignment is not an enumerated equal optimum")
        choices = [canonical, *[c for c in choices if c != canonical]]
        if len(choices) < 2:
            raise ValueError("Producer marks unstable gold coverage but no alternative optimum was found")
        options.append({"run": run, "doc_id": doc, "sent_id": sent,
                        "choices": [set((doc, sent, a, b) for a, b in c) for c in choices]})
    lookup = {doc: i for i, doc in enumerate(docs or [])}
    output, local_counts = [], {}
    for encoder, bio, span in MODEL_PAIRS:
        bio_runs, span_runs = groups[bio], groups[span]
        runs = bio_runs + span_runs
        local = [o for o in options if o["run"] in runs]
        local_counts[encoder] = math.prod(len(o["choices"]) for o in local)
        for index, choice in mixed_choices([len(o["choices"]) for o in local]):
            changed = {r: set(covered[r]) for r in runs}
            for option, selected_choice in zip(local, choice):
                original = option["choices"][0]
                changed[option["run"]].difference_update(original)
                changed[option["run"]].update(option["choices"][selected_choice])
            if any(len(changed[r]) != len(covered[r]) for r in runs):
                raise ValueError("Gold-ID alternative changed TP cardinality")
            # Recompute both slices: fresh alternatives may cross slice boundaries.
            flat_sets = {r: changed[r] & flat for r in runs}
            nonflat_sets = {r: changed[r] - flat for r in runs}
            f, fc = slice_stat(flat_sets, bio_runs, span_runs, weights, lookup)
            n, nc = slice_stat(nonflat_sets, bio_runs, span_runs, weights, lookup)
            shared = statistics.mean(len(nonflat_sets[a] & nonflat_sets[b]) for a, b in itertools.product(bio_runs, span_runs))
            for metric, position in (("between", 0), ("excess", 1)):
                draw = nc[position] - fc[position] if weights is not None else None
                valid = draw[np.isfinite(draw)] if draw is not None else []
                lo, hi = np.quantile(valid, [0.00625, 0.99375]) if len(valid) else (None, None)
                output.append({"dataset": "CL-RuTerm3 1.1", "encoder": encoder,
                               "contrast": f"nonflat-minus-flat {metric}", "switch_mask": index,
                               "switches_available": len(local), "nonflat_shared_tp_mean": shared,
                               "point": finite(n[position] - f[position]), "ci_low": finite(lo),
                               "ci_high": finite(hi), "valid_draws": len(valid),
                               "requested_draws": len(weights) if weights is not None else 0})
    info = {"selected_ambiguous_units": len(options),
            "global_equal_optimum_combinations": math.prod(len(o["choices"]) for o in options),
            "independent_local_combinations": local_counts,
            "enumeration": "All optimal covered-gold sets, exact anchors fixed; independent unit/run choices",
            "bootstrap": "Skipped" if weights is None else "Shared document draws; paired nonflat-minus-flat; 98.75% percentile intervals",
            "all_defined_lower_bounds_positive": (all(r["ci_low"] > 0 for r in output if r["ci_low"] is not None)
                                                   if weights is not None and all(r["ci_low"] is not None for r in output) else None)}
    return output, info


def receipt(path):
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_producer(root, source, required, prediction_root):
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] not in {"PASS", "COMPUTED"}:
        raise ValueError("Gold-agreement producer is incomplete")
    previous = manifest.get("prediction_root")
    previous = Path(previous).resolve() if previous is not None else None
    selected = Path(prediction_root).resolve() if prediction_root is not None else None
    if previous != selected:
        raise ValueError("Gold-output prediction root differs from --prediction-root")
    saved = {Path(r["path"]).name: r for r in manifest["outputs"]}
    paths = [manifest_path]
    for name in required:
        path = source / name
        if name not in saved or sha256(path) != saved[name]["sha256"]:
            raise ValueError("Gold-producer output identity differs: " + name)
        paths.append(path)
    recorded_inputs = {}
    for item in manifest["inputs"]:
        path = Path(item["path"])
        recorded_inputs[(path if path.is_absolute() else root / path).resolve()] = item["sha256"]
    return paths, recorded_inputs


def check_producer_inputs(paths, recorded):
    for path in paths:
        if path.resolve() not in recorded or sha256(path) != recorded[path.resolve()]:
            raise ValueError("Selected input differs from gold-producer inputs: " + str(path))


def check_reference(root, results, ambiguity, info, datasets, skip_bootstrap):
    reference = root / "results/paper_gold_agreement_sensitivity_2026_10_13.json"
    old = json.loads(reference.read_text(encoding="utf-8"))["results"]
    checked = 0
    for name in [target + "_" + metric for target in TARGETS for metric in ("smaller_set", "jaccard")]:
        for dataset in datasets:
            for field, expected in old[name][dataset].items():
                actual = results[name][dataset][field]
                if actual is None or not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10):
                    raise ValueError(f"Sensitivity reference differs: {name}/{dataset}/{field}")
                checked += 1
    for name in ("acl_metric_sign_changes", "genia_without_exact_train_test_text_duplicate"):
        if name in results:
            for field, expected in old[name].items():
                actual = results[name][field]
                if actual != expected and not (isinstance(expected, (int, float)) and actual is not None
                                                and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10)):
                    raise ValueError(f"Sensitivity reference differs: {name}/{field}")
                checked += 1
    paths = [reference]
    if "CL-RuTerm3" in datasets:
        path = root / "results/cl_flat_nonflat_tp_audit_2026_10_13/ambiguity_sensitivity.csv"
        expected = {(r["encoder"], r["contrast"], int(r["switch_mask"])): r for r in read_csv(path)}
        if len(ambiguity) != len(expected) or info["selected_ambiguous_units"] != 4 or info["global_equal_optimum_combinations"] != 16:
            raise ValueError("CL reference tie population differs")
        for row in ambiguity:
            saved = expected[row["encoder"], row["contrast"], row["switch_mask"]]
            fields = ["point", "nonflat_shared_tp_mean", "switches_available"]
            if not skip_bootstrap:
                fields += ["ci_low", "ci_high", "valid_draws"]
            for field in fields:
                if row[field] is None or not math.isclose(row[field], float(saved[field]), rel_tol=0, abs_tol=1e-10):
                    raise ValueError("CL ambiguity reference differs: " + field)
                checked += 1
        paths.append(path)
    return {"status": "PASS", "checked_values": checked}, paths


def run(root, gold_output, data_dir, output, datasets=DATASETS,
        prediction_root=None, skip_bootstrap=False, check_saved=False):
    from .io import load_analysis_rows, load_document_draws
    root, gold_output, data_dir, output = map(Path, (root, gold_output, data_dir, output))
    if output.exists():
        raise FileExistsError("Use a new sensitivity output directory: " + str(output))
    output.resolve().relative_to((root / "build").resolve())
    if check_saved and prediction_root is not None:
        raise ValueError("Reference checking is only available for the saved prediction release")
    required = ["run_pairs.csv", "run_scores.csv", "assignments_05.csv.gz", "configuration_pairs.csv"]
    if (gold_output / "ambiguous_units.csv").exists():
        required.append("ambiguous_units.csv")
    input_paths, recorded_inputs = verify_producer(root, gold_output, required, prediction_root)
    selected_gold = [data_dir / SLUGS[dataset] / "test.jsonl" for dataset in datasets]
    check_producer_inputs(selected_gold, recorded_inputs)
    input_paths.extend(selected_gold)
    pairs = [r for r in read_csv(gold_output / "run_pairs.csv") if r["dataset"] in datasets]
    scores = [r for r in read_csv(gold_output / "run_scores.csv") if r["dataset"] in datasets]
    if set(datasets) - {r["dataset"] for r in pairs}:
        raise ValueError("Selected corpus missing from gold-producer output")
    assignments = load_assignments(gold_output / "assignments_05.csv.gz", datasets)
    results, detail = metric_summaries(pairs)
    configurations = [r for r in read_csv(gold_output / "configuration_pairs.csv") if r["dataset"] in datasets]
    producer_checks = check_configuration_excess(detail, configurations)
    if "GENIA" in datasets:
        results["genia_without_exact_train_test_text_duplicate"] = genia_without_duplicate(pairs, assignments, scores)
    ambiguity, info = [], None
    if "CL-RuTerm3" in datasets:
        specs, gold_rows, prediction_rows, paths = load_analysis_rows(root, data_dir, "CL-RuTerm3", prediction_root, groups={"supervised"})
        check_producer_inputs(paths, recorded_inputs)
        input_paths.extend(paths)
        cl_scores = {r["run"]: r for r in scores if r["dataset"] == "CL-RuTerm3"}
        if set(cl_scores) != {r["run"] for r in specs}:
            raise ValueError("CL producer roster differs from selected prediction roster")
        ambiguous_rows = read_csv(gold_output / "ambiguous_units.csv") if "ambiguous_units.csv" in required else []
        docs, weights = None, None
        if not skip_bootstrap:
            docs, weights, path = load_document_draws(root, "CL-RuTerm3")
            check_producer_inputs([path], recorded_inputs)
            input_paths.append(path)
        ambiguity, info = cl_ambiguity(specs, gold_rows, prediction_rows, assignments,
                                        ambiguous_rows, scores, docs, weights)
        results["cl_equal_optimum_assignments"] = info
    checks, refs = ({"status": "NOT_REQUESTED"}, [])
    if check_saved:
        checks, refs = check_reference(root, results, ambiguity, info, datasets, skip_bootstrap)
    input_paths.extend(refs)
    output.mkdir(parents=True)
    paths = [output / "summary.json", output / "configuration_sensitivity.csv"]
    status = "COMPUTED" if prediction_root is not None else "PASS"
    paths[0].write_text(json.dumps({"status": status, "results": results}, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(paths[1], detail)
    if ambiguity:
        path = output / "ambiguity_sensitivity.csv"
        write_csv(path, ambiguity)
        paths.append(path)
    implementations = [Path(__file__), root / "src/ate_paper/analysis/matching.py",
                       root / "src/ate_paper/analysis/statistics.py", root / "src/ate_paper/analysis/io.py",
                       root / "scripts/reproduce_sensitivity.py"]
    manifest = {"status": status, "checks_against_saved_results": checks,
                "producer_consistency": {"configuration_excess_values_checked": producer_checks},
                "scope": "A12 IoU=.75; A13 Jaccard/signs; A15 CL equal-optimum coverage; A16 GENIA duplicate exclusion",
                "prediction_root": str(prediction_root) if prediction_root is not None else None,
                "protocol": {"denominators": "Undefined fixed run-pair constituents invalidate a configuration mean; summaries report defined/undefined configuration counts",
                             "A12_A13_A16": "Fixed-output point estimates; no new threshold matching, training or confidence interval",
                             "A15": "Exact-anchored maximum-cardinality/total-character-IoU=.5 assignments; all optimal covered sets in marked units; paired document draws; percentile .00625/.99375",
                             "resource_limit": "Enumeration exceeding 100000 states/combinations raises an explicit unsupported error; never truncates"},
                "inputs": [receipt(p) for p in sorted(set(input_paths))],
                "implementation": [receipt(p) for p in implementations],
                "outputs": [receipt(p) for p in paths]}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
