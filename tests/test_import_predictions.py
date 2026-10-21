"""The one-time migration must publish complete output or leave no output."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/import_predictions.py"
SPEC = importlib.util.spec_from_file_location("prediction_migration", SCRIPT)
migration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migration)


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class MigrationTests(unittest.TestCase):
    def setup_source(self, root, malformed=False):
        row = {"dataset": "ACL RD-TEC 2.0", "doc_id": "synthetic-document",
               "text": "tiny blue toy", "unit_type": "span_mention",
               "start": 0, "end": 4, "class_label": "Other"}
        (root / "source.jsonl").write_text(json.dumps(row) + "\n" + ("invalid JSON\n" if malformed else ""))
        roster = "roster.csv"
        write_csv(root / roster, ["dataset", "path", "group", "run"],
                  [{"dataset": "ACL RD-TEC 2.0", "path": "source.jsonl",
                    "group": "synthetic", "run": "first"}])
        pair = {"pair_id": "synthetic-pair", "dataset": "acl", "decision": "included",
                "model_key": "synthetic", "base_prediction_path": "source.jsonl",
                "repeat_prediction_path": "repeat.jsonl", "base_artifact": "first",
                "repeat_artifact": "second"}
        (root / "repeat.jsonl").write_text(json.dumps(row) + "\n")
        write_csv(root / "pairs.csv", list(pair), [pair])
        write_csv(root / "selected.csv", ["pair_id"], [{"pair_id": "synthetic-pair"}])
        return patch.multiple(migration, BASE_ROSTERS=(roster,),
                              PAIR_ROSTER="pairs.csv", PUBLISHED_PAIRS="selected.csv")

    def test_success_publishes_complete_validated_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy"
            legacy.mkdir()
            release = root / "release"
            with self.setup_source(legacy):
                summary = migration.migrate(legacy, release)
            self.assertEqual(summary["released_files"], 2)
            self.assertEqual(migration.verify_release(release)["files"], 2)
            self.assertEqual(list(release.glob(".prediction-export-*")), [])

    def test_malformed_source_leaves_no_partial_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy"
            legacy.mkdir()
            release = root / "release"
            with self.setup_source(legacy, malformed=True):
                with self.assertRaises(ValueError):
                    migration.migrate(legacy, release)
            self.assertFalse((release / "predictions").exists())
            self.assertEqual(list(release.glob(".prediction-export-*")), [])

    def test_existing_manifest_blocks_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "release/predictions").mkdir(parents=True)
            sentinel = root / "release/predictions/manifest.csv"
            sentinel.write_text("preserve me\n")
            with self.assertRaises(FileExistsError):
                migration.migrate(root / "missing-legacy", root / "release")
            self.assertEqual(sentinel.read_text(), "preserve me\n")


if __name__ == "__main__":
    unittest.main()
