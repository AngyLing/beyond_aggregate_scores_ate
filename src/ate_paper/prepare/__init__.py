"""Prepare the paper's local corpus splits from separately obtained sources."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from . import acl, acl_canonical, cl_ruterm3, genia, splits


DATASET_IDS = {"acl": "acl_rd_tec_2_0", "genia": "genia", "cl": "cl_ruterm3"}
EXPECTED_ROWS = {
    "acl": {"train": 3279, "dev": 364, "test": 809},
    "genia": {"train": 66820, "dev": 7429, "test": 18738},
    "cl": {"train": 17802, "dev": 2277, "test": 11547},
}
UNIT_FIELDS = ("dataset", "version", "split", "language", "domain",
               "doc_id", "sent_id", "text", "unit_type")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_source_files(source_root: Path, dataset: str, manifest: dict) -> None:
    entry = next(item for item in manifest["datasets"]
                 if item["id"] == DATASET_IDS[dataset])
    if dataset == "acl":
        root = source_root / "acl-rd-tec-2.0/distribution/annoitation_files"
        files = sorted(
            (path for layer in ("annotator1", "annotator2")
             for path in (root / layer).rglob("*.xml") if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
        checksum = entry["tree_checksum"]
        lines = "".join(f"{file_sha256(path)}  {path.relative_to(root).as_posix()}\n"
                        for path in files)
        actual = hashlib.sha256(lines.encode("utf-8")).hexdigest()
        size = sum(path.stat().st_size for path in files)
        if (len(files), size, actual) != (checksum["files"], checksum["size_bytes"],
                                          checksum["sha256"]):
            raise ValueError("ACL input XML tree differs from data/sources.json")
        return
    for source in entry["files"]:
        # The manifest uses repository-local acquisition paths. --source-root
        # replaces only data/local/sources while preserving the documented layout.
        relative = Path(source["local_path"]).relative_to("data/local/sources")
        path = source_root / relative
        actual = file_sha256(path)
        if path.stat().st_size != source["size_bytes"] or actual != source["sha256"]:
            raise ValueError(f"Source checksum mismatch: {relative}")


def convert_dataset(source_root: Path, dataset: str) -> list[dict]:
    if dataset == "acl":
        rows = acl.convert(
            source_root / "acl-rd-tec-2.0/distribution/annoitation_files", 0.8, 42)
        rows, _ = acl_canonical.make_canonical(rows, "ann1")
    elif dataset == "genia":
        rows = genia.convert(source_root / "GENIA_term_3.02/GENIAcorpus3.02.xml")
    elif dataset == "cl":
        root = source_root / "CL-RuTerm3"
        return cl_ruterm3.convert(
            root / "train.jsonl", root / "test_ref_v4.jsonl", root / "dev.jsonl",
            root / "dev_ref.jsonl", root / "test.jsonl")
    else:
        raise ValueError(f"Unknown corpus: {dataset}")

    source_train = [row for row in rows if row["split"] == "train"]
    dev_docs = splits.choose_dev_docs(source_train, dev_frac=0.1, seed=42)
    # Preserve the converter's row order; only split membership changes.
    return [{**row, "split": "dev"} if row["split"] == "train"
            and row["doc_id"] in dev_docs else row for row in rows]


def text_units(rows: list[dict]) -> list[dict]:
    units = {}
    for row in rows:
        key = (row["dataset"], row["doc_id"], row.get("sent_id"))
        unit = {field: row[field] for field in UNIT_FIELDS if field in row}
        if key in units and unit != units[key]:
            raise ValueError(f"Inconsistent metadata for text unit {key}")
        units[key] = unit
    return list(units.values())


def validate_splits(dataset: str, rows: list[dict], manifest: dict) -> None:
    observed = Counter(row["split"] for row in rows)
    if dict(observed) != EXPECTED_ROWS[dataset]:
        raise ValueError(f"Unexpected {dataset} split row counts: {dict(observed)}")
    documents = {split: {row["doc_id"] for row in rows if row["split"] == split}
                 for split in ("train", "dev", "test")}
    expected = next(item["split_documents"] for item in manifest["datasets"]
                    if item["id"] == DATASET_IDS[dataset])
    if {split: len(ids) for split, ids in documents.items()} != expected:
        raise ValueError(f"Unexpected {dataset} split document counts")
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if documents[left] & documents[right]:
            raise ValueError(f"Document IDs overlap {left}/{right}")
    for row in rows:
        start, end = row.get("start"), row.get("end")
        if start is None and end is None:
            continue
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(row["text"])
                or row.get("surface") != row["text"][start:end]):
            raise ValueError(f"Invalid converted span in {row['doc_id']!r}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_data(source_root: Path, output_root: Path, datasets: list[str],
                 source_manifest: Path) -> dict:
    """Verify source identities and write local train/dev/test plus text units.

    The generated files contain corpus text and annotations. Keep output_root
    inside ignored data/local/processed or another private local directory.
    """
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    for dataset in datasets:
        check_source_files(source_root, dataset, manifest)
    summary = {}
    for dataset in datasets:
        rows = convert_dataset(source_root, dataset)
        validate_splits(dataset, rows, manifest)
        units = text_units(rows)
        split_summary = {}
        for split in ("train", "dev", "test"):
            selected = [row for row in rows if row["split"] == split]
            write_jsonl(output_root / dataset / f"{split}.jsonl", selected)
            split_summary[split] = {
                "rows": len(selected),
                "documents": len({row["doc_id"] for row in selected}),
                "text_units": sum(unit["split"] == split for unit in units),
                "negative_placeholders": sum(row["start"] is None for row in selected),
            }
        write_jsonl(output_root / dataset / "units.jsonl", units)
        summary[dataset] = {"sources_verified": True, "splits": split_summary,
                            "total_text_units": len(units)}
    return summary
