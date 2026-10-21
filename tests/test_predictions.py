"""Synthetic examples only: no corpus text or copied corpus annotations."""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ate_paper.evaluation import score_predictions
from ate_paper.predictions import (CorpusIndex, RELEASE_FIELDS, release_record,
                                   sha256, unit_hash, validate_record, verify_release)


def mention(**changes):
    return {"dataset": "synthetic", "version": "1", "split": "test",
            "doc_id": "doc1", "sent_id": "sent1", "text": "blue cat naps",
            "unit_type": "span_mention", "start": 0, "end": 8,
            "class_label": "animal", **changes}


class PredictionTests(unittest.TestCase):
    def test_release_has_only_safe_fields(self):
        source = mention(surface="blue cat", norm="secret_norm", language="synthetic",
                         raw_response="secret", metadata={"text": "secret"})
        released = release_record(source)
        self.assertEqual(set(released) - RELEASE_FIELDS, set())
        for forbidden in ("text", "surface", "norm", "metadata", "raw_response", "language"):
            self.assertNotIn(forbidden, released)
        with self.assertRaisesRegex(ValueError, "whitelist"):
            validate_record({**released, "text": "secret"})

    def test_hash_handles_repeated_ids_and_unicode(self):
        original = mention(text="кот спит", end=3)
        other = mention(text="cat naps", end=3)
        self.assertNotEqual(unit_hash(original), unit_hash(other))
        index = CorpusIndex.build([original, original, other])
        self.assertEqual(len(index.units), 2)
        self.assertEqual(index.hydrate(release_record(original))["surface"], "кот")

    def test_hash_collision_is_rejected(self):
        with patch("ate_paper.predictions.unit_hash", return_value="a" * 64):
            with self.assertRaisesRegex(ValueError, "collision"):
                CorpusIndex.build([mention(), mention(text="another string")])

    def test_claimed_local_hash_is_checked(self):
        with self.assertRaisesRegex(ValueError, "disagrees"):
            CorpusIndex.build([mention(unit_hash="a" * 64)])

    def test_unknown_hash_does_not_fall_back_to_ids(self):
        index = CorpusIndex.build([mention()])
        released = release_record(mention(text="altered text"))
        with self.assertRaisesRegex(ValueError, "Unknown unit_hash"):
            index.hydrate(released)

    def test_hydration_does_not_copy_gold_class_or_normalization(self):
        index = CorpusIndex.build([mention(class_label="gold", is_nomen=True, norm="gold_norm")])
        released = release_record(mention(class_label="predicted", is_nomen=False))
        result = index.hydrate(released)
        self.assertEqual(result["class_label"], "predicted")
        self.assertFalse(result["is_nomen"])
        self.assertNotIn("norm", result)
        self.assertEqual(result["surface"], "blue cat")

    def test_offsets_and_half_null_are_rejected(self):
        for offsets in ({"start": True}, {"start": -1}, {"end": 0}, {"start": None}):
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                release_record(mention(**offsets))
        index = CorpusIndex.build([mention()])
        with self.assertRaisesRegex(ValueError, "beyond"):
            index.match(release_record(mention(end=100)))

    def test_zero_outputs_and_empty_gold_unit(self):
        gold = [mention(), mention(doc_id="doc2", start=None, end=None, class_label=None)]
        result = score_predictions(gold, [])
        self.assertEqual(result["units"], 2)
        self.assertEqual(result["exact_unlabeled"], {
            "predicted": 0, "gold": 1, "tp": 0, "fp": 0, "fn": 1,
            "precision": 0.0, "recall": 0.0, "f1": 0.0})
        false_positive = release_record(mention(doc_id="doc2"))
        result = score_predictions(gold, [false_positive])
        self.assertEqual(result["exact_unlabeled"]["fp"], 1)

    def test_duplicate_offsets_multiple_classes_use_sets(self):
        gold = [mention(class_label="A"), mention(class_label="A"), mention(class_label="B")]
        predicted = [release_record(mention(class_label="A"))] * 2
        result = score_predictions(gold, predicted)
        self.assertEqual(result["exact_unlabeled"]["gold"], 1)
        self.assertEqual(result["exact_unlabeled"]["f1"], 1)
        self.assertEqual(result["exact_typed"]["gold"], 2)
        self.assertEqual(result["exact_typed"]["recall"], 0.5)

    def test_null_prediction_is_not_a_term(self):
        null = release_record(mention(start=None, end=None, class_label=None))
        result = score_predictions([mention()], [null])
        self.assertEqual(result["prediction_rows"], 1)
        self.assertEqual(result["null_spans"], 1)
        self.assertEqual(result["exact_unlabeled"]["predicted"], 0)

    def test_corrupt_gzip_cli_reports_error_and_leaves_no_output(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold = root / "gold.jsonl"
            gold.write_text(json.dumps(mention()) + "\n", encoding="utf-8")
            predictions = root / "predictions.jsonl.gz"
            valid = gzip.compress((json.dumps(release_record(mention())) + "\n").encode(), mtime=0)
            bad_deflate = bytearray(valid)
            bad_deflate[10] = (bad_deflate[10] & ~7) | 7  # Invalid DEFLATE block type.
            damaged = {"truncated": valid[:-8], "invalid_deflate": valid + bad_deflate}
            for damage, payload in damaged.items():
                # The stream yields one valid row before detecting the missing
                # trailer or the invalid block in the second gzip member.
                predictions.write_bytes(payload)
                for tool, source_flag, suffix in (
                    ("hydrate_predictions.py", "--corpus", ".jsonl"),
                    ("score_predictions.py", "--gold", ".json"),
                ):
                    with self.subTest(damage=damage, tool=tool):
                        output = root / (damage + tool + suffix)
                        result = subprocess.run(
                            [sys.executable, str(scripts / tool), source_flag, str(gold),
                             "--predictions", str(predictions), "--output", str(output)],
                            capture_output=True, text=True, check=False,
                        )
                        self.assertEqual(result.returncode, 1)
                        self.assertIn("failed:", result.stderr)
                        self.assertNotIn("Traceback", result.stderr)
                        self.assertFalse(output.exists())

    def test_verifier_checks_inventory_counts_and_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "predictions/synthetic/test"
            folder.mkdir(parents=True)
            output = folder / "run.jsonl.gz"
            with gzip.open(output, "wt") as stream:
                stream.write(json.dumps(release_record(mention())) + "\n")
            entry = {"dataset": "synthetic", "group": "test", "run": "run",
                     "source_path": "private/source.jsonl", "source_sha256": "a" * 64,
                     "source_bytes": 1, "release_path": output.relative_to(root).as_posix(),
                     "release_sha256": sha256(output), "release_bytes": output.stat().st_size,
                     "rows": 1, "null_spans": 0}
            manifest = root / "predictions/manifest.csv"

            def write_manifest(entries):
                with manifest.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(entry))
                    writer.writeheader()
                    writer.writerows(entries)

            write_manifest([entry])
            self.assertEqual(verify_release(root)["files"], 1)
            write_manifest([entry, entry])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                verify_release(root)
            write_manifest([{**entry, "rows": 2}])
            with self.assertRaisesRegex(ValueError, "count differs"):
                verify_release(root)
            write_manifest([entry])
            (folder / "unlisted.jsonl.gz").write_bytes(output.read_bytes())
            with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                verify_release(root)

    def test_export_summary_detects_lost_run_and_repeat_roster(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "predictions/synthetic/test"
            folder.mkdir(parents=True)
            entries = []
            for run in ("first", "second"):
                output = folder / (run + ".jsonl.gz")
                with gzip.open(output, "wt", encoding="utf-8") as stream:
                    stream.write(json.dumps(release_record(mention())) + "\n")
                entries.append({
                    "dataset": "synthetic", "group": "test", "run": run,
                    "source_path": "private/" + run, "source_sha256": "a" * 64,
                    "source_bytes": 1, "release_path": output.relative_to(root).as_posix(),
                    "release_sha256": sha256(output), "release_bytes": output.stat().st_size,
                    "rows": 1, "null_spans": 0,
                })
            manifest = root / "predictions/manifest.csv"

            def write_manifest(rows):
                with manifest.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
                    writer.writeheader()
                    writer.writerows(rows)

            write_manifest(entries)
            pairs = root / "predictions/llm_repeat_pairs.csv"
            with pairs.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["pair_id", "base_path", "repeat_path"])
                writer.writeheader()
                writer.writerow({"pair_id": "pair", "base_path": entries[0]["release_path"],
                                 "repeat_path": entries[0]["release_path"]})
            summary = root / "predictions/export_summary.json"
            summary.write_text(json.dumps({
                "released_files": 2, "released_rows": 2,
                "released_bytes": sum(entry["release_bytes"] for entry in entries),
                "null_spans": 0, "base_outputs": 2, "published_repeat_pairs": 1,
            }))
            self.assertEqual(verify_release(root, require_summary=True)["files"], 2)
            # Removing an unreferenced run together with its manifest row leaves
            # an internally consistent inventory, but violates the saved release.
            lost = root / entries[1]["release_path"]
            saved_bytes = lost.read_bytes()
            lost.unlink()
            write_manifest(entries[:1])
            with self.assertRaisesRegex(ValueError, "released_files"):
                verify_release(root, require_summary=True)
            lost.write_bytes(saved_bytes)
            write_manifest(entries)
            pairs.unlink()
            with self.assertRaisesRegex(ValueError, "published_repeat_pairs"):
                verify_release(root, require_summary=True)
            summary.unlink()
            with self.assertRaisesRegex(ValueError, "Missing prediction export summary"):
                verify_release(root, require_summary=True)


if __name__ == "__main__":
    unittest.main()
