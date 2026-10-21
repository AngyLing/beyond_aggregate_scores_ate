"""Exact set metrics for fixed text-free prediction outputs."""
from __future__ import annotations

from typing import Iterable

from .predictions import CorpusIndex, release_record


def prf(gold: set, predicted: set) -> dict:
    tp = len(gold & predicted)
    fp, fn = len(predicted - gold), len(gold - predicted)
    return {
        "predicted": len(predicted), "gold": len(gold), "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def score_predictions(gold_records: Iterable[dict], predictions: Iterable[dict],
                      dataset: str | None = None) -> dict:
    gold_records = [row for row in gold_records if dataset is None or row.get("dataset") == dataset]
    index = CorpusIndex.build(gold_records)
    gold_unlabeled, gold_typed = set(), set()
    for record in gold_records:
        if record.get("unit_type") != "span_mention":
            continue
        safe = release_record(record)
        index.match(safe)
        if safe["start"] is None:
            continue
        key = (safe["unit_hash"], safe["start"], safe["end"])
        gold_unlabeled.add(key)
        gold_typed.add((*key, safe.get("class_label")))
    predicted_unlabeled, predicted_typed = set(), set()
    prediction_rows, null_spans = 0, 0
    for prediction in predictions:
        if dataset is not None and prediction.get("dataset") != dataset:
            continue
        index.match(prediction)
        prediction_rows += 1
        if prediction["start"] is None:
            null_spans += 1
            continue
        key = (prediction["unit_hash"], prediction["start"], prediction["end"])
        predicted_unlabeled.add(key)
        predicted_typed.add((*key, prediction.get("class_label")))
    return {
        "dataset": dataset,
        "unit_identity": "sha256(dataset,doc_id,sent_id,text)",
        "units": len(index.units), "prediction_rows": prediction_rows,
        "null_spans": null_spans,
        "exact_unlabeled": prf(gold_unlabeled, predicted_unlabeled),
        "exact_typed": prf(gold_typed, predicted_typed),
    }
