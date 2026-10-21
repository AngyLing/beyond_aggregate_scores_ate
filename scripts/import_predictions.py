#!/usr/bin/env python3
"""Migration only: create the fixed prediction release from a legacy workspace.

This migration tool is deliberately separate from evaluation. It copies the
selected saved outputs without rerunning models and records both source and
released SHA-256 hashes. Source corpora are never copied.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ate_paper.predictions import validate_record, verify_release

BASE_ROSTERS = (
    "results/acl_v1_profile_recheck_run_scores.csv",
    "results/genia_v1_profile_recheck_run_scores.csv",
    "results/cl_ruterm3_1_1_output_agreement_2026_10_13/run_scores.csv",
)
PAIR_ROSTER = "results/llm_repeat_output_agreement_2026_10_13/pair_roster.csv"
PUBLISHED_PAIRS = "results/llm_repeat_output_agreement_2026_10_13/limitations_full_repeat_f1.csv"
DATASETS = {
    "ACL RD-TEC 2.0": "acl_rd_tec_2_0",
    "GENIA": "genia_3_02",
    "CL-RuTerm3": "cl_ruterm3_1_1",
}
PAIR_DATASETS = {"acl": "ACL RD-TEC 2.0", "genia": "GENIA", "cl": "CL-RuTerm3"}
FIELDS = (
    "dataset", "version", "split", "doc_id", "sent_id", "unit_hash",
    "unit_type", "start", "end", "class_label", "is_nomen",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")


def unit_hash(record: dict, dataset: str) -> str:
    if not isinstance(record.get("text"), str):
        raise ValueError("Prediction record lacks source text needed for stable unit identity")
    payload = json.dumps(
        [dataset, record.get("doc_id"), record.get("sent_id"), record["text"]],
        ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def export_one(legacy: Path, release: Path, source_rel: str, dataset: str,
               group: str, run: str) -> dict[str, str | int]:
    source = legacy / source_rel
    if not source.is_file():
        raise FileNotFoundError(source)
    target_rel = Path("predictions") / DATASETS[dataset] / group / (safe_name(run) + ".jsonl.gz")
    target = release / target_rel
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    null_spans = 0
    with source.open(encoding="utf-8") as src, target.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as dst:
                for line in src:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("dataset") != dataset:
                        continue
                    selected = {key: record[key] for key in FIELDS if key in record and key != "unit_hash"}
                    selected["unit_hash"] = unit_hash(record, dataset)
                    validate_record(selected, dataset=dataset)
                    if selected.get("start") is None or selected.get("end") is None:
                        null_spans += 1
                    dst.write(json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                    written += 1
    if written == 0:
        target.unlink()
        raise ValueError(f"No rows for {dataset}: {source_rel}")
    return {
        "dataset": dataset, "group": group, "run": run,
        "source_path": source_rel, "source_sha256": sha256(source),
        "source_bytes": source.stat().st_size,
        "release_path": target_rel.as_posix(), "release_sha256": sha256(target),
        "release_bytes": target.stat().st_size, "rows": written,
        "null_spans": null_spans,
    }


def build_release(legacy: Path, release: Path) -> dict:
    """Write a complete export into a private temporary release root."""
    output_root = release / "predictions"
    if any(output_root.glob("*/*.jsonl.gz")) or any(output_root.glob("*/*/*.jsonl.gz")):
        raise FileExistsError("Refusing to overwrite existing prediction export")
    manifest: list[dict[str, str | int]] = []
    path_map: dict[tuple[str, str], str] = {}
    for roster in BASE_ROSTERS:
        for item in rows(legacy / roster):
            dataset = item["dataset"]
            source_rel = item["path"]
            group = item["group"]
            run = item["run"]
            entry = export_one(legacy, release, source_rel, dataset, group, run)
            manifest.append(entry)
            path_map[(dataset, str((legacy / source_rel).resolve()))] = str(entry["release_path"])
    published_ids = {item["pair_id"] for item in rows(legacy / PUBLISHED_PAIRS)}
    pair_rows = []
    for pair in rows(legacy / PAIR_ROSTER):
        if pair["pair_id"] not in published_ids:
            continue
        dataset = PAIR_DATASETS[pair["dataset"]]
        for key, artifact in (("base_prediction_path", "base_artifact"),
                              ("repeat_prediction_path", "repeat_artifact")):
            source_rel = pair[key]
            source_key = (dataset, str((legacy / source_rel).resolve()))
            if source_key not in path_map:
                entry = export_one(legacy, release, source_rel, dataset,
                                   "llm_repeats", pair[artifact])
                manifest.append(entry)
                path_map[source_key] = str(entry["release_path"])
        pair_rows.append({
            "pair_id": pair["pair_id"], "dataset": dataset,
            "decision": pair["decision"], "model_key": pair["model_key"],
            "base_path": path_map[(dataset, str((legacy / pair["base_prediction_path"]).resolve()))],
            "repeat_path": path_map[(dataset, str((legacy / pair["repeat_prediction_path"]).resolve()))],
        })
    if len(pair_rows) != len(published_ids):
        raise ValueError("Published pair roster mismatch")
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    with (output_root / "llm_repeat_pairs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    summary = {
        "source_rosters": list(BASE_ROSTERS) + [PAIR_ROSTER, PUBLISHED_PAIRS],
        "base_outputs": sum(len(rows(legacy / roster)) for roster in BASE_ROSTERS),
        "published_repeat_pairs": len(pair_rows),
        "released_files": len(manifest),
        "released_bytes": sum(int(item["release_bytes"]) for item in manifest),
        "released_rows": sum(int(item["rows"]) for item in manifest),
        "null_spans": sum(int(item["null_spans"]) for item in manifest),
        "excluded_fields": ["text", "surface", "norm", "language", "domain"],
        "unit_hash_recipe": "SHA256(compact UTF-8 JSON array [dataset,doc_id,sent_id,text])",
    }
    (output_root / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def migrate(legacy: Path, release: Path) -> dict:
    """Publish one complete validated directory; leave no partial release."""
    legacy, release = legacy.resolve(), release.resolve()
    if legacy == release:
        raise ValueError("Legacy source and destination must be different directories")
    output_root = release / "predictions"
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise FileExistsError("Refusing to replace a nonempty predictions directory")
    release.mkdir(parents=True, exist_ok=True)
    # Keep staging on the destination filesystem so publishing is a directory
    # rename. Errors remove only temporary output; existing releases are intact.
    with tempfile.TemporaryDirectory(prefix=".prediction-export-", dir=release) as temporary:
        staging = Path(temporary)
        summary = build_release(legacy, staging)
        verify_release(staging)
        if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
            raise FileExistsError("Prediction destination changed during export")
        os.replace(staging / "predictions", output_root)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", required=True, type=Path,
                        help="Legacy research workspace; needed only for migration")
    parser.add_argument("--release-root", default=Path(__file__).resolve().parents[1], type=Path)
    args = parser.parse_args()
    try:
        summary = migrate(args.legacy_root, args.release_root)
    except (ValueError, OSError, KeyError, IndexError) as error:
        parser.exit(1, f"Migration failed: {error}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
