"""Read, validate, and locally hydrate the text-free prediction release.

Character offsets refer to Python Unicode strings and use [start, end).
The unit hash binds offsets to an exact source string without shipping it.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .io import sha256

RELEASE_FIELDS = frozenset({
    "dataset", "version", "split", "doc_id", "sent_id", "unit_hash",
    "unit_type", "start", "end", "class_label", "is_nomen",
})
UNIT_METADATA = ("dataset", "version", "split", "doc_id", "sent_id", "text",
                 "language", "domain")
MANIFEST_FIELDS = {
    "dataset", "group", "run", "source_file", "source_sha256", "source_bytes",
    "release_path", "release_sha256", "release_bytes", "rows", "null_spans",
}
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")


def read_jsonl(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("record must be an object")
                except (json.JSONDecodeError, ValueError) as error:
                    raise ValueError(f"{path}:{number}: {error}") from error
                yield record
    except (EOFError, zlib.error) as error:
        raise ValueError(f"{path}: Invalid compressed JSONL stream: {error}") from error


def unit_hash(record: dict) -> str:
    if not isinstance(record.get("dataset"), str) or not record["dataset"]:
        raise ValueError("Unit needs a nonempty dataset")
    if not isinstance(record.get("text"), str):
        raise ValueError("Local corpus unit needs text")
    payload = json.dumps(
        [record["dataset"], record.get("doc_id"), record.get("sent_id"), record["text"]],
        ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_record(record: dict, *, dataset: str | None = None) -> bool:
    """Validate a release row; return True for a null-span placeholder."""
    extras = set(record) - RELEASE_FIELDS
    if extras:
        raise ValueError(f"Fields outside release whitelist: {sorted(extras)}")
    for key in ("dataset", "doc_id", "unit_hash", "unit_type", "start", "end"):
        if key not in record:
            raise ValueError(f"Missing release field: {key}")
    if not isinstance(record["dataset"], str) or not record["dataset"]:
        raise ValueError("Invalid dataset")
    if dataset is not None and record["dataset"] != dataset:
        raise ValueError(f"Dataset differs from manifest: {record['dataset']!r}")
    if not isinstance(record["doc_id"], (str, int)) or isinstance(record["doc_id"], bool):
        raise ValueError("Invalid doc_id")
    if record.get("sent_id") is not None and (not isinstance(record["sent_id"], (str, int)) or isinstance(record["sent_id"], bool)):
        raise ValueError("Invalid sent_id")
    for key in ("version", "split"):
        if key in record and record[key] is not None and not isinstance(record[key], str):
            raise ValueError(f"{key} must be a string or null")
    if not isinstance(record["unit_hash"], str) or not HASH_RE.fullmatch(record["unit_hash"]):
        raise ValueError("Invalid SHA-256 unit_hash")
    if record["unit_type"] != "span_mention":
        raise ValueError("Only span_mention units are supported")
    if record.get("class_label") is not None and not isinstance(record["class_label"], str):
        raise ValueError("class_label must be a string or null")
    if "is_nomen" in record and record["is_nomen"] is not None and not isinstance(record["is_nomen"], bool):
        raise ValueError("is_nomen must be a boolean or null")
    start, end = record["start"], record["end"]
    if start is None or end is None:
        if start is not None or end is not None:
            raise ValueError("Null-span placeholder needs both start and end null")
        return True
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ValueError("Span needs integer offsets 0 <= start < end")
    return False


def release_record(record: dict) -> dict:
    """Select the safe fields from a local source row, then validate them."""
    selected = {key: value for key, value in record.items() if key in RELEASE_FIELDS}
    selected["unit_hash"] = unit_hash(record)
    validate_record(selected)
    return selected


@dataclass
class CorpusIndex:
    units: dict[str, dict]

    @classmethod
    def build(cls, records: Iterable[dict], dataset: str | None = None) -> "CorpusIndex":
        units: dict[str, dict] = {}
        for record in records:
            if dataset is not None and record.get("dataset") != dataset:
                continue
            digest = unit_hash(record)
            if "unit_hash" in record and record["unit_hash"] != digest:
                raise ValueError("Local unit_hash disagrees with its text and identity")
            selected = {key: record[key] for key in UNIT_METADATA if key in record}
            # Corpus mention labels, gold variants and annotation flags deliberately
            # do not become prediction metadata.
            if digest in units:
                previous = units[digest]
                identity = ("dataset", "doc_id", "sent_id", "text")
                if any(previous.get(k) != selected.get(k) for k in identity):
                    raise ValueError("Unit hash collision: different text or identity")
                for key, value in selected.items():
                    if key in previous and previous[key] != value:
                        raise ValueError(f"Inconsistent local metadata for one unit: {key}")
                    previous[key] = value
            else:
                units[digest] = selected
        if not units:
            raise ValueError("No local corpus units in the requested dataset")
        return cls(units)

    def match(self, prediction: dict) -> dict:
        validate_record(prediction)
        digest = prediction["unit_hash"]
        if digest not in self.units:
            raise ValueError(f"Unknown unit_hash {digest}; check corpus version and split")
        unit = self.units[digest]
        for key in ("dataset", "doc_id", "sent_id"):
            if prediction.get(key) != unit.get(key):
                raise ValueError(f"Prediction {key} disagrees with matched local unit")
        for key in ("version", "split"):
            if key in prediction and key in unit and prediction[key] != unit[key]:
                raise ValueError(f"Prediction {key} disagrees with matched local unit")
        if prediction["end"] is not None and prediction["end"] > len(unit["text"]):
            raise ValueError("Prediction span extends beyond the local unit text")
        return unit

    def hydrate(self, prediction: dict) -> dict:
        unit = self.match(prediction)
        result = {**unit, **prediction}
        result["surface"] = (None if prediction["start"] is None
                             else unit["text"][prediction["start"]:prediction["end"]])
        # No `norm`: the source pipelines used different normalization rules.
        return result


def inside_root(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Unsafe manifest path: {relative}")
    target = (root / candidate).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path escapes release root: {relative}")
    return target


def verify_release(root: Path, manifest: Path | None = None, *,
                   require_summary: bool = False) -> dict:
    root = root.resolve()
    manifest = manifest or root / "predictions/manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not MANIFEST_FIELDS <= set(reader.fieldnames or []):
            raise ValueError("Missing required manifest columns")
        entries = list(reader)
    if not entries:
        raise ValueError("Empty prediction manifest")
    paths, runs = set(), set()
    totals = {"files": 0, "rows": 0, "null_spans": 0, "release_bytes": 0}
    for entry in entries:
        relative = entry["release_path"]
        run_key = (entry["dataset"], entry["group"], entry["run"])
        if relative in paths or run_key in runs:
            raise ValueError(f"Duplicate manifest path or run: {relative}")
        paths.add(relative)
        runs.add(run_key)
        path = inside_root(root, relative)
        if not path.is_file() or path.suffixes[-2:] != [".jsonl", ".gz"]:
            raise ValueError(f"Missing gzip JSONL prediction file: {relative}")
        for key in ("release_sha256", "source_sha256"):
            if not HASH_RE.fullmatch(entry[key]):
                raise ValueError(f"Invalid manifest {key}: {relative}")
        for key in ("source_bytes", "release_bytes", "rows", "null_spans"):
            if int(entry[key]) < 0:
                raise ValueError(f"Negative manifest count: {key}")
        if path.stat().st_size != int(entry["release_bytes"]):
            raise ValueError(f"Byte count differs: {relative}")
        if sha256(path) != entry["release_sha256"]:
            raise ValueError(f"SHA-256 differs: {relative}")
        rows = nulls = 0
        for record in read_jsonl(path):
            nulls += validate_record(record, dataset=entry["dataset"])
            rows += 1
        if rows != int(entry["rows"]) or nulls != int(entry["null_spans"]):
            raise ValueError(f"Row/null-span count differs: {relative}")
        totals["files"] += 1
        totals["rows"] += rows
        totals["null_spans"] += nulls
        totals["release_bytes"] += path.stat().st_size
    actual = {p.relative_to(root).as_posix() for p in (root / "predictions").rglob("*.jsonl.gz")}
    if actual != paths:
        raise ValueError(f"Manifest/file inventory mismatch; extra={sorted(actual-paths)}, missing={sorted(paths-actual)}")
    pairs_path = root / "predictions/llm_repeat_pairs.csv"
    if pairs_path.exists():
        pair_ids = set()
        with pairs_path.open(newline="", encoding="utf-8") as stream:
            for pair in csv.DictReader(stream):
                if pair["pair_id"] in pair_ids:
                    raise ValueError("Duplicate repeat pair id")
                pair_ids.add(pair["pair_id"])
                for key in ("base_path", "repeat_path"):
                    if pair[key] not in paths:
                        raise ValueError(f"Repeat pair references an unlisted prediction: {pair[key]}")
        totals["repeat_pairs"] = len(pair_ids)
    summary_path = root / "predictions/export_summary.json"
    if require_summary and not summary_path.is_file():
        raise ValueError("Missing prediction export summary: predictions/export_summary.json")
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        observed = {
            "released_files": totals["files"],
            "released_rows": totals["rows"],
            "released_bytes": totals["release_bytes"],
            "null_spans": totals["null_spans"],
            "base_outputs": sum(entry["group"] != "llm_repeats" for entry in entries),
            "published_repeat_pairs": totals.get("repeat_pairs", 0),
        }
        for key, actual in observed.items():
            expected = summary.get(key)
            if type(expected) is not int or expected < 0:
                raise ValueError(f"Invalid export summary count: {key}")
            if actual != expected:
                raise ValueError(f"Export summary differs for {key}: {actual} != {expected}")
    return totals
