"""Load selected fixed or newly generated predictions against prepared local gold."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from ate_paper.predictions import CorpusIndex, inside_root, read_jsonl, sha256, release_record, validate_record

DATASETS = ("ACL RD-TEC 2.0", "GENIA", "CL-RuTerm3")
SLUGS = {"ACL RD-TEC 2.0": "acl", "GENIA": "genia", "CL-RuTerm3": "cl"}
DRAW_NAMES = {"ACL RD-TEC 2.0": "acl_rd_tec_2_0", "GENIA": "genia_3_02", "CL-RuTerm3": "cl_ruterm3_1_1"}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def span_set(records, dataset):
    return {(str(row["doc_id"]), row.get("sent_id"), row["start"], row["end"])
            for row in records if row.get("dataset") == dataset
            and row.get("unit_type") == "span_mention"
            and row.get("start") is not None and row.get("end") is not None}


def load_fixed_dataset(root: Path, gold_dir: Path, dataset: str, prediction_root: Path | None = None):
    """Load public supervised predictions and validate them against local gold."""
    if prediction_root is not None:
        specs, gold_rows, records, inputs = load_analysis_rows(
            root, gold_dir, dataset, prediction_root, groups={"supervised"})
        gold = span_set(gold_rows, dataset)
        predictions = {run: span_set(rows, dataset) for run, rows in records.items()}
        index = CorpusIndex.build(gold_rows, dataset)
        docs, weights, draw_path = load_document_draws(
            root, dataset, {str(row["doc_id"]) for row in index.units.values()})
        return specs, gold, predictions, docs, weights, [*inputs, draw_path]

    source = root / ("results/v1_profile_verified_document_analysis_run_scores.csv"
                     if dataset != "CL-RuTerm3" else
                     "results/cl_ruterm3_1_1_output_agreement_2026_10_13/run_scores.csv")
    specs = [row for row in read_csv(source)
             if row["dataset"] == dataset and row["group"] == "supervised"]
    if len(specs) != (55 if dataset == "CL-RuTerm3" else 50):
        raise ValueError("Wrong supervised roster: " + dataset)
    gold_path = gold_dir / SLUGS[dataset] / "test.jsonl"
    if not gold_path.is_file():
        raise FileNotFoundError("Prepare the local corpus first: " + str(gold_path))
    gold_rows = list(read_jsonl(gold_path))
    validate_test_rows(gold_rows, dataset)
    index = CorpusIndex.build(gold_rows, dataset)
    gold = span_set(gold_rows, dataset)
    manifest_path = root / "predictions/manifest.csv"
    public = {(row["dataset"], row["run"]): row for row in read_csv(manifest_path)
              if row["group"] == "supervised"}
    predictions = {}
    input_paths = [gold_path, manifest_path]
    for row in specs:
        entry = public[dataset, row["run"]]
        path = inside_root(root, entry["release_path"])
        if sha256(path) != entry["release_sha256"]:
            raise ValueError("Prediction SHA-256 differs from manifest: " + str(path))
        records = list(read_jsonl(path))
        for prediction in records:
            index.match(prediction)
        predictions[row["run"]] = span_set(records, dataset)
        input_paths.append(path)
        pred = predictions[row["run"]]
        if (len(pred) != int(row["predicted"]) or len(pred & gold) != int(row["tp"])
                or len(gold) != int(row["gold"])):
            raise ValueError("Exact score differs from accepted source: " + row["run"])
    docs, weights, draw_path = load_document_draws(
        root, dataset, {str(row["doc_id"]) for row in gold_rows})
    config_path = root / (
        "results/v1_profile_verified_document_analysis_configuration_pairs.csv"
        if dataset != "CL-RuTerm3" else
        "results/cl_ruterm3_1_1_output_agreement_2026_10_13/configuration_pairs.csv"
    )
    return specs, gold, predictions, docs, weights, [source, draw_path, config_path, *input_paths]



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



def base_roster(root: Path, dataset: str, groups=None) -> tuple[list[dict], Path]:
    """Use saved results only for run identities/types, never their measurements."""
    path = root / ("results/v1_profile_verified_document_analysis_run_scores.csv"
                   if dataset != "CL-RuTerm3" else
                   "results/cl_ruterm3_1_1_output_agreement_2026_10_13/run_scores.csv")
    fields = ("dataset", "model", "group", "seed", "run", "class_capable", "typed_applicable")
    rows = [{key: row[key] for key in fields if key in row}
            for row in read_csv(path) if row["dataset"] == dataset
            and (groups is None or row["group"] in groups)]
    if len(rows) != len({row["run"] for row in rows}) or not rows:
        raise ValueError("Missing or duplicate selected run identities: " + dataset)
    return rows, path


def load_prediction_rows(root: Path, entry: dict, index: CorpusIndex,
                         prediction_root: Path | None = None) -> tuple[list[dict], Path]:
    """Validate/hydrate either a public export or one fresh launcher output.

    Fresh text-bearing rows are projected through the same release schema and
    exact unit hash. No old output counts or hashes are imposed on a new run.
    Missing files are errors; an existing empty prediction file means zero
    predicted spans on the full test, not an empty test population.
    """
    if prediction_root is None:
        path = inside_root(root, entry["release_path"])
        if sha256(path) != entry["release_sha256"]:
            raise ValueError("Prediction SHA-256 differs from manifest: " + str(path))
    else:
        source = prediction_root.resolve()
        path = source / SLUGS[entry["dataset"]] / entry["group"] / entry["run"] / "predictions.jsonl"
        path = path.resolve()
        if not path.is_relative_to(source):
            raise ValueError("Prediction path escapes its selected root")
    if not path.is_file():
        raise FileNotFoundError("Missing selected prediction file: " + str(path))
    result = []
    for row in read_jsonl(path):
        if prediction_root is not None and "text" in row:
            if "unit_hash" in row and row["unit_hash"] != release_record(row)["unit_hash"]:
                raise ValueError("Fresh prediction unit_hash disagrees with its text")
            safe = release_record(row)
        else:
            safe = row
        validate_record(safe, dataset=entry["dataset"])
        hydrated = index.hydrate(safe)
        if prediction_root is not None and "text" in row and row.get("surface") is not None:
            if row["surface"] != hydrated["surface"]:
                raise ValueError("Fresh prediction surface disagrees with its offsets")
        result.append(hydrated)
    return result, path


TEST_COUNTS = {
    "ACL RD-TEC 2.0": {"documents": 60, "units": 254, "spans": 799},
    "GENIA": {"documents": 400, "units": 3696, "spans": 18672},
    "CL-RuTerm3": {"documents": 317, "units": 317, "spans": 11547},
}


def validate_test_rows(rows: list[dict], dataset: str) -> None:
    """Require the retained full test, including units with no gold mentions."""
    if not rows or any(r.get("dataset") != dataset or r.get("split") != "test" for r in rows):
        raise ValueError("Prepared test dataset/split mismatch: " + dataset)
    units = {}
    for row in rows:
        key = (row.get("doc_id"), row.get("sent_id"))
        text = row.get("text")
        if not isinstance(text, str) or (key in units and units[key] != text):
            raise ValueError("Conflicting prepared test text-unit identity")
        units[key] = text
    actual = {"documents": len({key[0] for key in units}), "units": len(units),
              "spans": len(span_set(rows, dataset))}
    if actual != TEST_COUNTS[dataset]:
        raise ValueError(f"Prepared test is incomplete or differs from the selected corpus: {dataset}: {actual}")


def load_analysis_rows(root: Path, gold_dir: Path, dataset: str,
                       prediction_root: Path | None = None, groups=None):
    """Selected base-panel identities plus actual gold/prediction rows."""
    specs, roster_path = base_roster(root, dataset, groups)
    gold_path = gold_dir / SLUGS[dataset] / "test.jsonl"
    gold_rows = list(read_jsonl(gold_path))
    validate_test_rows(gold_rows, dataset)
    index = CorpusIndex.build(gold_rows, dataset)
    manifest_path = root / "predictions/manifest.csv"
    entries = {(row["dataset"], row["group"], row["run"]): row for row in read_csv(manifest_path)}
    predictions = {}
    inputs = [roster_path, manifest_path, gold_path]
    for spec in specs:
        entry = entries[dataset, spec["group"], spec["run"]]
        records, path = load_prediction_rows(root, entry, index, prediction_root)
        predictions[spec["run"]] = records
        inputs.append(path)
    return specs, gold_rows, predictions, inputs


def load_document_draws(root: Path, dataset: str, unit_docs=None):
    """Read the saved shared document multiplicities, including term-free units."""
    import numpy as np
    path = root / "results/document_draws" / (DRAW_NAMES[dataset] + ".npz")
    with np.load(path, allow_pickle=False) as loaded:
        docs = list(map(str, loaded["doc_ids"]))
        weights = loaded["weights"].astype(float)
    if (len(docs) != len(set(docs)) or weights.shape !=
            (10000 if dataset == "CL-RuTerm3" else 5000, len(docs))):
        raise ValueError("Wrong shared document-draw population or shape")
    if unit_docs is not None and set(map(str, unit_docs)) != set(docs):
        raise ValueError("Prepared full test document universe differs from saved draws")
    if not np.isfinite(weights).all() or (weights < 0).any() or (weights != np.floor(weights)).any():
        raise ValueError("Invalid document multiplicities")
    if not (weights.sum(axis=1) == len(docs)).all():
        raise ValueError("Shared draw has the wrong document population size")
    return docs, weights, path


def compare_rows(computed: list[dict], reference: list[dict], keys: tuple,
                 fields=None, tolerance=1e-10) -> dict:
    """Report actual differences; a fresh result is allowed to differ from paper."""
    import math
    def identity(row):
        return tuple(str(row.get(k)) for k in keys)
    def undefined(value):
        if value is None or value == "":
            return True
        try:
            return math.isnan(float(value))
        except (TypeError, ValueError):
            return False
    actual = {identity(row): row for row in computed}
    expected = {identity(row): row for row in reference}
    if len(actual) != len(computed) or len(expected) != len(reference):
        raise ValueError("Duplicate comparison identity")
    changes = []
    checked = 0
    for key in sorted(actual.keys() | expected.keys()):
        if key not in actual or key not in expected:
            changes.append({"key": list(key), "field": "__row__", "new": "present" if key in actual else "absent",
                            "reference": "present" if key in expected else "absent"})
            continue
        for field in fields or sorted(set(actual[key]) & set(expected[key]) - set(keys)):
            value, saved = actual[key].get(field), expected[key].get(field)
            checked += 1
            if (value == saved or str(value) == str(saved)
                    or (undefined(value) and undefined(saved))):
                continue
            delta = None
            try:
                a, b = float(value), float(saved)
                if (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=0, abs_tol=tolerance):
                    continue
                delta = a - b
            except (TypeError, ValueError):
                pass
            changes.append({"key": list(key), "field": field, "new": value,
                            "reference": saved, "delta": delta})
    return {"computed_rows": len(computed), "reference_rows": len(reference),
            "cells_checked": checked, "all_saved_values_match": not changes,
            "difference_count": len(changes), "differences": changes,
            "absolute_tolerance": tolerance}


def comparison_summary(value):
    """Keep detailed differences in output files, not in terminal summaries."""
    if isinstance(value, dict):
        return {key: comparison_summary(item) for key, item in value.items() if key != "differences"}
    if isinstance(value, list):
        return [comparison_summary(item) for item in value]
    return value
