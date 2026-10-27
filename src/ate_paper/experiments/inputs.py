"""Local adapters from prepared corpus splits; no work occurs on import."""
from __future__ import annotations

import json
from pathlib import Path

from ate_paper.prepare import text_units
from ate_paper.io import sha256

DATASETS = {"acl": "ACL RD-TEC 2.0", "genia": "GENIA", "cl": "CL-RuTerm3"}


def paths(processed_root: Path, output_root: Path, dataset: str) -> dict[str, Path]:
    source = processed_root / dataset
    target = output_root / dataset / "inputs"
    return {"train": source / "train.jsonl", "dev": source / "dev.jsonl",
            "test": source / "test.jsonl", "test_units": target / "test_units.jsonl",
            "train_untyped": target / "train_untyped.jsonl",
            "dev_untyped": target / "dev_untyped.jsonl",
            "gazetteer_train": target / "gazetteer_train.jsonl"}


def read_split(path: Path, dataset: str, split: str) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset") != DATASETS[dataset] or row.get("split") != split:
                raise ValueError(f"Wrong corpus or split in {path}")
            if row.get("unit_type") != "span_mention":
                raise ValueError(f"Expected span_mention rows in {path}")
            if not isinstance(row.get("text"), str) or row.get("doc_id") is None:
                raise ValueError(f"Missing text/ID in {path}")
            if "start" not in row or "end" not in row:
                raise ValueError(f"Missing annotation offsets in prepared split: {path}")
            start, end = row.get("start"), row.get("end")
            if start is not None or end is not None:
                if (type(start) is not int or type(end) is not int
                        or not 0 <= start < end <= len(row["text"])
                        or row.get("surface") != row["text"][start:end]
                        or not isinstance(row.get("norm"), str) or not row["norm"]):
                    raise ValueError(f"Malformed annotation in {path}")
            elif any(row.get(key) is not None for key in ("surface", "norm", "class_label")):
                raise ValueError(f"Malformed placeholder annotation in {path}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty prepared split: {path}")
    return rows


def prepare_inputs(processed_root: Path, output_root: Path, dataset: str) -> dict:
    """Write only private adapters; preserve mentions, order and the full test.

    Called only by an explicit execution, never by command planning. Dictionary
    inputs omit empty placeholders; model inputs retain their negative units.
    """
    p = paths(processed_root, output_root, dataset)
    manifest_path = p["test_units"].parent / "input_manifest.json"
    try:
        manifest_path.parent.resolve().relative_to(output_root.resolve())
    except ValueError:
        raise ValueError("Input adapter directory resolves outside --output-root")
    sources = {s: {"path": str(p[s]), "sha256": sha256(p[s])}
               for s in ("train", "dev", "test")}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["sources"] != sources:
            raise ValueError("Prepared inputs changed; use a fresh --output-root")
        for key, value in old["outputs"].items():
            if sha256(p[key]) != value["sha256"]:
                raise ValueError(f"Derived input changed: {p[key]}")
        return old
    if manifest_path.parent.exists() and any(manifest_path.parent.iterdir()):
        raise ValueError("Input folder exists without a manifest; use a fresh --output-root")
    rows = {s: read_split(p[s], dataset, s) for s in ("train", "dev", "test")}
    documents = {s: {row["doc_id"] for row in data} for s, data in rows.items()}
    if any(documents[a] & documents[b] for a, b in
           (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise ValueError("Prepared train/dev/test document IDs overlap")
    derived = {"test_units": text_units(rows["test"]),
               "train_untyped": [{**r, "class_label": None} for r in rows["train"]],
               "dev_untyped": [{**r, "class_label": None} for r in rows["dev"]],
               "gazetteer_train": [r for r in rows["train"] if r.get("start") is not None]}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for key, data in derived.items():
        with p[key].open("w", encoding="utf-8", newline="\n") as stream:
            for row in data:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {"dataset": dataset, "sources": sources,
              "outputs": {k: {"rows": len(data), "sha256": sha256(p[k])}
                          for k, data in derived.items()}}
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
