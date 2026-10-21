"""Full article panels: data counts, diagnostics, exact output sets and document CIs.

Only explicitly invoked reproduction reads corpora or predictions. The four
output-agreement estimands preserve the original fixed-run/document-only rules.
"""
from __future__ import annotations

import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from ate_paper.io import write_csv
from .matching import near_boundary_mask

import numpy as np

from ate_paper.evaluation import prf
from ate_paper.predictions import read_jsonl, sha256
from . import io
from .profiles import (UNTYPED, base_key, describe_gold, distance, gap_context,
                       nested_categories, real_rows)
from .statistics import complete_mean, interval, ratio, summarize_cloud

TARGETS = ("tp_turnover", "fp_turnover", "flat_tp_turnover", "fp_no_witness_share")
RUN_TARGET = {"tp_turnover": "tp", "fp_turnover": "fp", "flat_tp_turnover": "flat_tp"}
EXPECTED = {"ACL RD-TEC 2.0": (60, 20), "GENIA": (60, 20), "CL-RuTerm3": (65, 21)}


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def save_csv(path, records):
    write_csv(Path(path), [clean(row) for row in records])


def typed_status(spec):
    for key in ("typed_applicable", "class_capable"):
        value = spec.get(key)
        if value not in (None, ""):
            if isinstance(value, bool):
                return value
            if str(value).lower() in ("true", "1"):
                return True
            if str(value).lower() in ("false", "0"):
                return False
            raise ValueError("Invalid typed status: " + str(value))
    return spec["model"] not in UNTYPED


def data_statistics(dataset, split, rows):
    if any(row.get("dataset") != dataset or row.get("split") != split for row in rows):
        raise ValueError("Unexpected corpus/split while computing data statistics")
    units = {(row.get("doc_id"), row.get("sent_id")) for row in rows}
    spans = real_rows(rows)
    offsets = {base_key(row) for row in spans}
    mention_classes = {base_key(row) + (row.get("class_label"),) for row in spans}
    positive_units = {key[:2] for key in offsets}
    categories = Counter(nested_categories(offsets).values())
    classes = defaultdict(set)
    raw_counts = Counter()
    for row in spans:
        classes[row.get("class_label")].add(base_key(row))
        raw_counts[row.get("class_label")] += 1
    summary = {"dataset": dataset, "split": split, "documents": len({key[0] for key in units}),
               "text_units": len(units), "annotation_rows": len(spans),
               "unique_spans": len(offsets), "mention_class_pairs": len(mention_classes),
               "exact_duplicate_annotation_rows": len(spans) - len(mention_classes),
               "classes": len(classes), "term_free_units": len(units - positive_units),
               "inner": categories["inner-only"] + categories["both"],
               "outer": categories["outer-only"] + categories["both"],
               "both": categories["both"], "nonflat": len(offsets) - categories["flat"],
               "flat": categories["flat"]}
    class_rows = [{"dataset": dataset, "split": split, "class_label": label,
                   "unique_spans": len(offsets), "annotation_rows": raw_counts[label]}
                  for label, offsets in sorted(classes.items(), key=lambda pair: str(pair[0]))]
    return summary, class_rows


def run_diagnostics(dataset, specs, gold_rows, predictions, train_rows):
    descriptors, slices = describe_gold(train_rows, gold_rows)
    gold = set(descriptors)
    gold_typed = {base_key(row) + (row.get("class_label"),) for row in real_rows(gold_rows)}
    inner = {key for key, value in descriptors.items() if value["nesting"] in ("inner-only", "both")}
    outer = {key for key, value in descriptors.items() if value["nesting"] in ("outer-only", "both")}
    supports = {family + ":" + value: len(members) for (family, value), members in slices.items()}
    scores, exported, contexts, profiles, predicted_sets = [], [], [], {}, []
    for spec in specs:
        run, model = spec["run"], spec["model"]
        records = predictions[run]
        predicted = {base_key(row) for row in real_rows(records)}
        predicted_sets.append(predicted)
        typed = typed_status(spec)
        exact = prf(gold, predicted)
        pred_typed = {base_key(row) + (row.get("class_label"),) for row in real_rows(records)}
        typed_score = prf(gold_typed, pred_typed)
        context = gap_context(gold_rows, records, descriptors, typed)
        matched = gold & predicted
        extra = {"inner_recall": len(matched & inner) / len(inner) if inner else None,
                 "outer_recall": len(matched & outer) / len(outer) if outer else None,
                 "soft_precision": context["soft_matches"] / len(predicted) if predicted else 0.0,
                 "soft_recall": context["soft_matches"] / len(gold) if gold else 0.0,
                 "soft_f1": context["soft_f1"],
                 **{name: context[name] for name in ("ClassGap", "BoundaryGap", "NestedGap")}}
        scores.append({"dataset": dataset, "model": model, "group": spec["group"],
                       "seed": spec.get("seed", ""), "run": run,
                       "typed_applicable": typed, **exact,
                       **{"typed_" + name: typed_score[name] if typed else None
                          for name in ("predicted", "gold", "tp", "fp", "fn", "precision", "recall", "f1")},
                       **extra})
        contexts.append({"dataset": dataset, "run": run, "model": model, **context, **extra})
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
        for name in ("BoundaryGap", "NestedGap", "ClassGap"):
            if context[name] is None:
                continue
            active["gap:" + name] = context[name]
            supports["gap:" + name] = None
            exported.append({"dataset": dataset, "run": run, "model": model,
                             "family": "gap", "slice": name, "support": None,
                             "active": True, "core": True, "value": context[name]})
        profiles[run] = active
    slice_rows = [{"dataset": dataset, "family": family, "slice": value,
                   "support": len(members), "active": len(members) >= 30,
                   "core": family not in ("gold_class", "length_x_frequency")}
                  for (family, value), members in sorted(slices.items())]
    return scores, profiles, supports, exported, contexts, slice_rows, predicted_sets, gold


def configuration_scores(dataset, scores):
    grouped = defaultdict(list)
    for row in scores:
        grouped[row["model"]].append(row)
    fields = ("precision", "recall", "f1", "typed_precision", "typed_recall", "typed_f1",
              "soft_precision", "soft_recall", "soft_f1", "inner_recall", "outer_recall",
              "ClassGap", "BoundaryGap", "NestedGap")
    output = []
    for model, runs in sorted(grouped.items()):
        row = {"dataset": dataset, "model": model, "group": runs[0]["group"],
               "runs": len(runs), "typed_applicable": runs[0]["typed_applicable"]}
        for name in fields:
            values = [r[name] for r in runs]
            applicable = all(value is not None and np.isfinite(value) for value in values)
            row[name + "_mean"] = float(np.mean(values)) if applicable else None
            row[name + "_sample_sd"] = float(np.std(values, ddof=1)) if applicable and len(values) > 1 else None
        output.append(row)
    return output


def doc_vector(mask, key_docs, n_docs):
    return np.bincount(key_docs[mask], minlength=n_docs).astype(np.int32)


def bootstrap_values(weights, arrays, target, chunk=256):
    """Recompute corpus counts with document multiplicities before taking ratios."""
    i, j = arrays["pair_i"], arrays["pair_j"]
    base = "fp" if target == "fp_no_witness_share" else RUN_TARGET[target]
    run_totals = weights @ arrays["run_" + base].T.astype(float)
    output = np.empty((len(weights), len(i)), dtype=float)
    for start in range(0, len(i), chunk):
        sl = slice(start, start + chunk)
        a, b = run_totals[:, i[sl]], run_totals[:, j[sl]]
        if target == "fp_no_witness_share":
            na = weights @ arrays["pair_nw_fp_a"][sl].T.astype(float)
            nb = weights @ arrays["pair_nw_fp_b"][sl].T.astype(float)
            output[:, sl] = (ratio(na, a) + ratio(nb, b)) / 2
        else:
            common = weights @ arrays["pair_common_" + base][sl].T.astype(float)
            denominator = np.minimum(a, b)
            output[:, sl] = ratio(denominator - common, denominator)
    return output


def prepare_output_sets(dataset, run_rows, predictions, gold, docs):
    """Per-document counts for all actual fixed output pairs, in shared draw order."""
    if len(docs) != len(set(docs)) or not docs:
        raise ValueError("Document draw IDs must be unique and nonempty")
    doc_lookup = {str(doc): index for index, doc in enumerate(docs)}
    nesting = nested_categories(gold)
    flat = {key for key in gold if nesting[key] == "flat"}
    if len(predictions) != len(run_rows) or not predictions:
        raise ValueError("Prediction and run identities differ")
    if any(str(key[0]) not in doc_lookup for keys in [gold, *predictions] for key in keys):
        raise ValueError("Offsets include documents outside the saved draws")
    keys_tp = sorted(gold)
    keys_fp = sorted(set.union(*(prediction - gold for prediction in predictions)))
    membership = {
        "tp": np.array([[key in prediction for key in keys_tp] for prediction in predictions], dtype=bool),
        "fp": np.array([[key in prediction for key in keys_fp] for prediction in predictions], dtype=bool),
    }
    membership["flat_tp"] = membership["tp"] & np.array([key in flat for key in keys_tp], dtype=bool)
    # Explicit integer dtype also supports synthetic/perfect-output empty FP unions.
    key_docs = {"tp": np.array([doc_lookup[str(key[0])] for key in keys_tp], dtype=np.int64),
                "fp": np.array([doc_lookup[str(key[0])] for key in keys_fp], dtype=np.int64)}
    key_docs["flat_tp"] = key_docs["tp"]
    near = {"tp": np.array([near_boundary_mask(keys_tp, prediction) for prediction in predictions], dtype=bool),
            "fp": np.array([near_boundary_mask(keys_fp, prediction) for prediction in predictions], dtype=bool)}
    arrays = {"doc_ids": np.array(list(map(str, docs))),
              "run_names": np.array([row["run"] for row in run_rows])}
    for target, masks in membership.items():
        arrays["run_" + target] = np.stack([doc_vector(mask, key_docs[target], len(docs)) for mask in masks])
    gold_per_doc = np.bincount(key_docs["tp"], minlength=len(docs))
    document_rows = []
    for index, row in enumerate(run_rows):
        for position, doc in enumerate(docs):
            document_rows.append({"dataset": dataset, "run": row["run"], "model": row["model"],
                                  "doc_id": str(doc), "gold": gold_per_doc[position],
                                  **{target: arrays["run_" + target][index, position]
                                     for target in ("tp", "fp", "flat_tp")}})
    pairs = list(itertools.combinations(range(len(run_rows)), 2))
    arrays["pair_i"] = np.array([i for i, j in pairs], dtype=np.int64)
    arrays["pair_j"] = np.array([j for i, j in pairs], dtype=np.int64)
    for name in ("common_tp", "common_fp", "common_flat_tp", "nw_tp_a", "nw_tp_b", "nw_fp_a", "nw_fp_b"):
        arrays["pair_" + name] = np.empty((len(pairs), len(docs)), dtype=np.int32)
    pair_rows = []
    for number, (i, j) in enumerate(pairs):
        a, b = run_rows[i], run_rows[j]
        row = {"dataset": dataset, "run_a": a["run"], "run_b": b["run"],
               "model_a": a["model"], "model_b": b["model"],
               "within_configuration": a["model"] == b["model"]}
        for target in ("tp", "fp", "flat_tp"):
            ma, mb = membership[target][i], membership[target][j]
            common_mask = ma & mb
            arrays["pair_common_" + target][number] = doc_vector(common_mask, key_docs[target], len(docs))
            na, nb, common = int(ma.sum()), int(mb.sum()), int(common_mask.sum())
            row.update({target + "_a": na, target + "_b": nb, "common_" + target: common,
                        "unique_" + target + "_a": na - common,
                        "unique_" + target + "_b": nb - common,
                        target + "_turnover": float(ratio(min(na, nb) - common, min(na, nb)))})
        for target in ("tp", "fp"):
            for label, source, other in (("a", i, j), ("b", j, i)):
                mask = membership[target][source] & ~near[target][other]
                arrays["pair_nw_" + target + "_" + label][number] = doc_vector(mask, key_docs[target], len(docs))
                row["no_witness_" + target + "_" + label] = int(mask.sum())
        row["fp_no_witness_share"] = float((ratio(row["no_witness_fp_a"], row["fp_a"])
                                             + ratio(row["no_witness_fp_b"], row["fp_b"])) / 2)
        row["unique_fp_no_witness_share"] = float((ratio(row["no_witness_fp_a"], row["unique_fp_a"])
                                                    + ratio(row["no_witness_fp_b"], row["unique_fp_b"])) / 2)
        row["fn_a"], row["fn_b"] = len(gold) - row["tp_a"], len(gold) - row["tp_b"]
        row["common_fn"] = len(gold) - (row["tp_a"] + row["tp_b"] - row["common_tp"])
        pair_rows.append(row)
    groups = defaultdict(list)
    for index, row in enumerate(run_rows):
        groups[row["model"]].append(index)
    pair_lookup = {pair: index for index, pair in enumerate(pairs)}
    within = {name: [pair_lookup[i, j] for i, j in itertools.combinations(indices, 2)]
              for name, indices in groups.items()}
    comparisons = []
    for a, b in itertools.combinations(sorted(groups), 2):
        indices = [pair_lookup[tuple(sorted((i, j)))]
                   for i, j in itertools.product(groups[a], groups[b])]
        comparisons.append((a, b, indices, within[a], within[b]))
    return arrays, pair_rows, groups, within, comparisons, document_rows, flat


def configuration_profile_index(dataset, runs, groups, comparisons, profiles, supports):
    means, exported = {}, []
    for model, indices in sorted(groups.items()):
        names = [runs[index]["run"] for index in indices]
        shared = set.intersection(*(set(profiles[name]) for name in names))
        means[model] = {key: float(np.mean([profiles[name][key] for name in names])) for key in shared}
        for key in sorted(shared):
            family, value = key.split(":", 1)
            values = [profiles[name][key] for name in names]
            exported.append({"dataset": dataset, "model": model, "group": runs[indices[0]]["group"],
                             "runs": len(indices), "family": family, "slice": value,
                             "support": supports[key], "active": True,
                             "value": means[model][key],
                             "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None})
    result = {}
    for a, b, *_ in comparisons:
        d = distance(means[a], means[b], supports)
        pa, pb = [float(np.mean([runs[index]["precision"] for index in groups[model]])) for model in (a, b)]
        ra, rb = [float(np.mean([runs[index]["recall"] for index in groups[model]])) for model in (a, b)]
        result[dataset, frozenset((a, b))] = {
            "delta_p": abs(pa - pb), "delta_r": abs(ra - rb),
            "maximum": d["maximum"], "weighted_family_maximum": d["weighted_family_maximum"],
            "worst_coordinate": d["worst_coordinate"], "coordinates": d["coordinates"],
        }
    return result, exported

def point_summaries(dataset,run_rows,pair_rows,groups,within,comparisons,profile_index):
    values = np.array([[r[t] for r in pair_rows] for t in TARGETS]).T
    output = []
    within_rows = []
    for name,indices in within.items():
        if not indices:continue
        for t in TARGETS:
            vv = np.array([pair_rows[i][t] for i in indices])
            within_rows.append({"dataset":dataset,"model":name,"target":t,"run_pairs":len(indices),
                                "valid_pairs":int(np.isfinite(vv).sum()),"mean":float(np.mean(vv)),
                                "minimum":float(np.min(vv)),"maximum":float(np.max(vv))})
    for ma,mb,indices,wa,wb in comparisons:
        a,b = [run_rows[i] for i in groups[ma]],[run_rows[i] for i in groups[mb]]
        profile = profile_index[dataset,frozenset((ma,mb))]
        record = {"dataset":dataset,"model_a":ma,"model_b":mb,"group_a":a[0]["group"],"group_b":b[0]["group"],
                  "runs_a":len(a),"runs_b":len(b),"between_pairs":len(indices),
                  "within_pairs_a":len(wa),"within_pairs_b":len(wb),
                  "p_a":np.mean([float(r['precision']) for r in a]),"p_b":np.mean([float(r['precision']) for r in b]),
                  "r_a":np.mean([float(r['recall']) for r in a]),"r_b":np.mean([float(r['recall']) for r in b]),
                  "delta_p":float(profile['delta_p']),"delta_r":float(profile['delta_r']),
                  "maximum_profile_distance":float(profile['maximum']),
                  "weighted_family_distance":float(profile['weighted_family_maximum']),
                  "weighted_profiles_are_posthoc":True,"profile_from_current_inputs":True,
                  "worst_profile_coordinate":profile["worst_coordinate"],
                  "active_profile_coordinates":profile["coordinates"]}
        for col,t in enumerate(TARGETS):
            point = values[:,col][None,:]
            between,reference,excess = summarize_cloud(point,indices,wa,wb)
            record.update({t+"_mean":between[0],t+"_minimum":np.min(point[:,indices]),
                           t+"_maximum":np.max(point[:,indices]),t+"_valid_pairs":int(np.isfinite(point[:,indices]).sum()),
                           t+"_within_a":complete_mean(point,wa)[0],t+"_within_b":complete_mean(point,wb)[0],
                           t+"_within_reference":reference[0],t+"_excess":excess[0]})
        # Directional counts must follow the configuration labels, even if raw run indices reverse them.
        for field in ['tp_a','tp_b','common_tp','unique_tp_a','unique_tp_b','fp_a','fp_b','common_fp',
                      'unique_fp_a','unique_fp_b','fn_a','fn_b','common_fn','no_witness_tp_a','no_witness_tp_b',
                      'no_witness_fp_a','no_witness_fp_b','unique_fp_no_witness_share']:
            vv = []
            for i in indices:
                row=pair_rows[i];key=field
                if row['model_a']!=ma and field.endswith(('_a','_b')):
                    key=field[:-1]+('b' if field[-1]=='a' else 'a')
                vv.append(row[key])
            record[field+"_mean"] = np.mean(vv)
        output.append(record)
    return output,within_rows


def bootstrap_summaries(dataset, weights, arrays, pair_rows, comparisons):
    output = []
    for target in TARGETS:
        values = bootstrap_values(weights.astype(float), arrays, target)
        point = np.array([[row[target] for row in pair_rows]])
        for a, b, indices, wa, wb in comparisons:
            row = {"dataset": dataset, "model_a": a, "model_b": b, "target": target,
                   "requested": len(weights), "within_sides": int(bool(wa)) + int(bool(wb))}
            clouds = summarize_cloud(values, indices, wa, wb)
            points = summarize_cloud(point, indices, wa, wb)
            for name, cloud, estimate in zip(("between", "within_reference", "excess"), clouds, points):
                row[name + "_point"] = clean(estimate[0])
                row.update({name + "_" + key: value for key, value in interval(cloud).items()})
            output.append(clean(row))
        del values
    return output


def compare_saved_outputs(root, outputs, skip_bootstrap=False):
    """Compare only shared schemas, without making old numbers new measurements."""
    keys = {"run_scores": ("dataset", "run"),
            "run_pairs": ("dataset", "run_a", "run_b"),
            "configuration_pairs": ("dataset", "model_a", "model_b"),
            "bootstrap": ("dataset", "model_a", "model_b", "target"),
            "profiles": ("dataset", "run", "family", "slice"),
            "gap_context": ("dataset", "run")}
    cache, paths, comparisons = {}, set(), []
    for dataset in io.DATASETS:
        for name, identity in keys.items():
            if name == "bootstrap" and skip_bootstrap:
                continue
            if dataset == "CL-RuTerm3":
                file_name = "profiles" if name == "gap_context" else name
                path = root / "results/cl_ruterm3_1_1_output_agreement_2026_10_13" / (file_name + ".csv")
            elif name in ("profiles", "gap_context"):
                path = root / "results" / (io.SLUGS[dataset] + "_v1_profile_recheck_" + name + ".csv")
            else:
                path = root / "results" / ("v1_profile_verified_document_analysis_" + name + ".csv")
            entry = {"dataset": dataset, "output": name, "reference_path": path.relative_to(root).as_posix()}
            if not path.is_file():
                comparisons.append({**entry, "status": "REFERENCE_MISSING"})
                continue
            if path not in cache:
                cache[path] = io.read_csv(path)
            paths.add(path)
            reference = [row for row in cache[path] if row.get("dataset") == dataset]
            if dataset == "CL-RuTerm3" and name == "gap_context":
                # CL retains per-run gap values in the full profiles export.
                # Pivot only those values; no old support/count is invented.
                pivot = {}
                for row in reference:
                    if row.get("family") != "gap" or row.get("slice") not in ("BoundaryGap", "NestedGap", "ClassGap"):
                        continue
                    item = pivot.setdefault(row["run"], {"dataset": dataset, "run": row["run"],
                                                         "BoundaryGap": None, "NestedGap": None, "ClassGap": None})
                    item[row["slice"]] = row["value"]
                reference = list(pivot.values())
                entry["reference_projection"] = "family=gap cells pivoted by run; shared gap fields only"
            actual = [clean(row) for row in outputs[name] if row.get("dataset") == dataset]
            result = io.compare_rows(actual, reference, identity)
            comparisons.append({**entry, "status": "DIFFERENCES_REPORTED", **clean(result)})
    return {"status": "DIFFERENCES_REPORTED", "comparison": "Shared existing fields only; differences do not fail fresh recomputation",
            "interpretation": "A value match is not independent scientific validation or a claim that a new experiment reproduced the old execution",
            "difference_count": sum(row.get("difference_count", 0) for row in comparisons),
            "references_missing": sum(row["status"] == "REFERENCE_MISSING" for row in comparisons),
            "comparisons": comparisons}, paths


def _receipt(path, root, data_dir, prediction_root=None):
    path = Path(path).resolve()
    labels = [(root.resolve(), ""), (data_dir.resolve(), "LOCAL_PROCESSED/")]
    if prediction_root is not None:
        labels.append((prediction_root.resolve(), "FRESH_PREDICTIONS/"))
    name = str(path)
    for directory, prefix in labels:
        try:
            name = prefix + path.relative_to(directory).as_posix()
            break
        except ValueError:
            pass
    return {"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def reproduce_panels(root, data_dir, output_dir, prediction_root=None, skip_bootstrap=False):
    """Recompute from selected predictions into a new local build directory.

    Saved score files supply only the identity/type roster through the common
    loader. Every reported number here is computed from the loaded input rows.
    """
    root, data_dir, output_dir = Path(root).resolve(), Path(data_dir).resolve(), Path(output_dir).resolve()
    prediction_root = Path(prediction_root).resolve() if prediction_root is not None else None
    try:
        relative = output_dir.relative_to(root / "build")
    except ValueError:
        raise ValueError("Panel output must stay in a named directory under this repository's build/")
    if not relative.parts:
        raise ValueError("Give a named panel output directory under build/")
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = defaultdict(list)
    inputs = {}
    implementation = [Path(__file__), Path(__file__).parents[1] / "io.py",
                      Path(__file__).parent / "matching.py", root / "scripts/reproduce_panels.py",
                      Path(io.__file__), root / "src/ate_paper/analysis/profiles.py",
                      root / "src/ate_paper/analysis/statistics.py", root / "src/ate_paper/evaluation.py"]
    for path in implementation:
        if path.exists():
            inputs[path.resolve()] = sha256(path)
    scopes, array_paths = [], []
    for dataset in io.DATASETS:
        specs, gold_rows, predicted_rows, input_paths = io.load_analysis_rows(
            root, data_dir, dataset, prediction_root=prediction_root)
        expected_runs, expected_configurations = EXPECTED[dataset]
        if len(specs) != expected_runs or len({row["model"] for row in specs}) != expected_configurations:
            raise ValueError("Wrong base panel identity roster: " + dataset)
        if any(row["group"] not in ("supervised", "llm", "gazetteer") for row in specs):
            raise ValueError("Only the base panel belongs to full-panel analysis")
        for path in input_paths:
            path = Path(path).resolve()
            inputs[path] = sha256(path)
        splits = {}
        for split in ("train", "dev", "test"):
            path = data_dir / io.SLUGS[dataset] / (split + ".jsonl")
            rows = gold_rows if split == "test" else list(read_jsonl(path))
            splits[split] = rows
            statistic, classes = data_statistics(dataset, split, rows)
            outputs["data_statistics"].append(statistic)
            outputs["class_statistics"].extend(classes)
            inputs[path.resolve()] = sha256(path)
        result = run_diagnostics(dataset, specs, gold_rows, predicted_rows, splits["train"])
        scores, profiles, supports, profile_rows, gaps, slice_rows, predictions, gold = result
        unit_docs = {str(row["doc_id"]) for row in gold_rows}
        if skip_bootstrap:
            docs, weights, draw_path = sorted(unit_docs), None, None
        else:
            docs, weights, draw_path = io.load_document_draws(root, dataset, unit_docs)
            inputs[Path(draw_path).resolve()] = sha256(Path(draw_path))
        arrays, pairs, groups, within, comparisons, document_rows, flat = prepare_output_sets(
            dataset, scores, predictions, gold, docs)
        profile_index, mean_profiles = configuration_profile_index(
            dataset, scores, groups, comparisons, profiles, supports)
        configuration_rows, within_rows = point_summaries(
            dataset, scores, pairs, groups, within, comparisons, profile_index)
        for name, rows in (("run_scores", scores), ("configuration_score_summary", configuration_scores(dataset, scores)),
                           ("profiles", profile_rows), ("configuration_profiles", mean_profiles),
                           ("gap_context", gaps), ("slice_supports", slice_rows),
                           ("run_pairs", pairs), ("configuration_pairs", configuration_rows),
                           ("within_configuration", within_rows), ("document_counts", document_rows)):
            outputs[name].extend(rows)
        if not skip_bootstrap:
            outputs["bootstrap"].extend(bootstrap_summaries(dataset, weights, arrays, pairs, comparisons))
            arrays["weights"] = weights
        slug = io.SLUGS[dataset]
        array_path = output_dir / (slug + "_document_arrays.npz")
        np.savez_compressed(array_path, **arrays)
        array_paths.append(array_path)
        units = {(row.get("doc_id"), row.get("sent_id")) for row in gold_rows}
        scopes.append({"dataset": dataset, "documents": len(docs), "text_units": len(units),
                       "gold": len(gold), "flat_gold": len(flat), "runs": len(scores),
                       "configurations": len(groups), "run_pairs": len(pairs),
                       "configuration_pairs": len(comparisons), "draws": len(weights) if weights is not None else 0})
    comparison_report, comparison_paths = compare_saved_outputs(root, outputs, skip_bootstrap)
    for path in comparison_paths:
        inputs[path.resolve()] = sha256(path)
    for path, expected_sha in inputs.items():
        if sha256(path) != expected_sha:
            raise ValueError("Input changed during full-panel reproduction: " + str(path))
    for name, rows in outputs.items():
        save_csv(output_dir / (name + ".csv"), rows)
    protocol = {
        "analysis_blocks": ["A01", "A02", "A03", "A04", "A07"] + ([] if skip_bootstrap else ["A08"]),
        "skipped_blocks": ["A08"] if skip_bootstrap else [],
        "A05_A06": "existing supervised profiles/calibration command",
        "prediction_mode": "fresh" if prediction_root is not None else "saved_release",
        "profile_support_minimum": 30, "targets": list(TARGETS),
        "bootstrap": "skipped" if skip_bootstrap else "shared saved document multiplicities; fixed run pairs",
        "run_weighting": "equal actual between-run and distinct within-configuration pairs; no seed resampling",
        "within_reference": "larger available within-configuration mean, recomputed in each draw",
        "zero_denominators": "undefined constituent invalidates fixed-population mean",
        "intervals": "nominal exploratory percentile 95%; valid replicate counts reported separately",
        "score_sd": "sample SD across fixed seeds; single outputs have no SD; untyped typed metrics are N/A",
        "profiles": "all recall slices plus applicable gaps; support-weighted family distances are post-hoc",
        "comparison_with_saved_numbers": "not asserted; fresh output differences remain visible",
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    comparison_path = output_dir / "comparison_with_saved.json"
    comparison_path.write_text(json.dumps(clean(comparison_report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    files = [output_dir / (name + ".csv") for name in outputs] + array_paths + [output_dir / "protocol.json", comparison_path]
    manifest = {"status": "COMPUTED", "scope": scopes,
                "input_mode": "new_predictions" if prediction_root is not None else "published_predictions",
                "prediction_root": str(prediction_root) if prediction_root is not None else None, "protocol": protocol,
                "numpy_version": np.__version__, "inputs_unchanged": True,
                "comparison_with_saved": {"report": comparison_path.name,
                                          "difference_count": comparison_report["difference_count"],
                                          "references_missing": comparison_report["references_missing"]},
                "inputs": [_receipt(path, root, data_dir, prediction_root) for path in sorted(inputs)],
                "outputs": [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files],
                "limitations": ["Fixed outputs and exploratory document intervals; no new model randomness inferred",
                                "No semantic adjudication or linguistic validity claim for unmatched predictions"]}
    (output_dir / "manifest.json").write_text(json.dumps(clean(manifest), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return clean(manifest)
