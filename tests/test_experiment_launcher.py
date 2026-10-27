"""Command and adapter checks using metadata and synthetic text only."""
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from ate_paper.experiments.inputs import DATASETS, paths, prepare_inputs, sha256
from ate_paper.experiments.launcher import ROOT, checked_output_root, plan_record, records
from ate_paper.experiments.make_gazetteer_predictions import normalize_text


class ExperimentCommandsTest(unittest.TestCase):
    def test_all_selected_artifacts_and_explicit_scibert_choice(self):
        roster = records()
        out = ROOT / "build/experiments/check"
        source = ROOT / "data/local/processed"
        plans = [plan_record(r, source, out) for r in roster]
        self.assertEqual(len(plans), 214)
        self.assertEqual(sum(bool(p["requires"]) for p in plans), 20)
        for choice in ("false", "true"):
            full = [plan_record(r, source, out, choice) for r in roster]
            self.assertFalse(any(p["requires"] for p in full))
            self.assertEqual(sum(len(p["commands"]) for p in full), 320)
            self.assertEqual(len({p["run_dir"] for p in full}), 214)
            for p in full:
                Path(p["prediction"]).relative_to(out)
                if "scibert" in p["run"] and p["group"] == "supervised":
                    cmd = p["commands"][0]
                    self.assertEqual(cmd[cmd.index("--scibert-do-lower-case") + 1], choice)

    def test_transport_header_rule_and_untyped_inputs(self):
        plans = [plan_record(r, ROOT / "data/local/processed", ROOT / "build/experiments/check", "false")
                 for r in records()]
        modules = Counter(c[2].rsplit(".", 1)[-1] for p in plans for c in p["commands"])
        self.assertEqual(modules["run_llm_api"], 33)
        self.assertEqual(modules["run_together_batch"], 10)
        self.assertEqual(modules["run_mistral_batch"], 6)
        self.assertEqual(modules["run_anthropic_batch"], 4)
        self.assertEqual(sum("--legacy-exact-headers" in c for p in plans for c in p["commands"]), 2)
        untyped = [p for p in plans if "span_unlabeled" in p["run"]]
        self.assertEqual(len(untyped), 20)
        for p in untyped:
            c = p["commands"][0]
            self.assertTrue(c[c.index("--train")+1].endswith("train_untyped.jsonl"))
            self.assertTrue(c[c.index("--dev")+1].endswith("dev_untyped.jsonl"))

    def test_output_boundaries(self):
        with self.assertRaises(ValueError):
            checked_output_root(ROOT / "predictions")
        with self.assertRaises(ValueError):
            checked_output_root(ROOT / "build/experiments")
        self.assertEqual(checked_output_root(ROOT / "build/experiments/check"), ROOT / "build/experiments/check")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            (root / "build").mkdir(parents=True)
            (root / "build/experiments").symlink_to(Path(directory) / "elsewhere")
            with self.assertRaises(ValueError):
                checked_output_root(root / "build/experiments/check", root)


class ExperimentInputsTest(unittest.TestCase):
    @staticmethod
    def write_source(source, ds):
        for split in ("train", "dev", "test"):
            base = {"dataset": DATASETS[ds], "split": split, "language": "English",
                    "domain": "test", "doc_id": split + "-annotated", "sent_id": "1",
                    "text": "alpha beta", "unit_type": "span_mention"}
            mention = {**base, "start": 0, "end": 5, "surface": "alpha", "norm": "alpha", "class_label": "Specific"}
            # Preserve annotation duplicates, which contribute to dictionary class counts.
            empty = {**base, "doc_id": split + "-empty", "start": None, "end": None,
                     "surface": None, "norm": None, "class_label": None}
            path = source / ds / (split + ".jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(r) + "\n" for r in [mention, mention, empty]))

    def test_pool_untyped_and_full_label_free_test(self):
        with tempfile.TemporaryDirectory() as folder:
            source, out = Path(folder) / "processed", Path(folder) / "runs"
            for ds in DATASETS:
                self.write_source(source, ds)
                before = {s: sha256(source / ds / (s + ".jsonl")) for s in ("train", "dev", "test")}
                result = prepare_inputs(source, out, ds)
                p = paths(source, out, ds)
                pool = [json.loads(l) for l in p["gazetteer_train"].read_text().splitlines()]
                self.assertEqual(len(pool), 2)
                self.assertEqual({r["split"] for r in pool}, {"train"})
                units = [json.loads(l) for l in p["test_units"].read_text().splitlines()]
                self.assertEqual(len(units), 2)
                self.assertTrue(all(not ({"start", "end", "class_label", "norm", "surface"} & r.keys()) for r in units))
                untyped = [json.loads(l) for l in p["train_untyped"].read_text().splitlines()]
                self.assertEqual(len(untyped), 3)
                self.assertTrue(all(r["class_label"] is None for r in untyped))
                self.assertEqual(prepare_inputs(source, out, ds), result)
                self.assertEqual(before, {s: sha256(source / ds / (s + ".jsonl")) for s in before})

    def test_dictionary_normalization_ignores_null_and_preserves_lowercase_policy(self):
        self.assertEqual(normalize_text(None), "")
        self.assertEqual(normalize_text("  DNA \t Domain  "), "dna domain")

    def test_changed_source_and_malformed_annotation_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            source, out = Path(folder) / "processed", Path(folder) / "runs"
            self.write_source(source, "acl")
            prepare_inputs(source, out, "acl")
            train = source / "acl/train.jsonl"
            train.write_text(train.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                prepare_inputs(source, out, "acl")
            rows = [json.loads(l) for l in train.read_text().splitlines() if l]
            rows[0]["end"] = None
            train.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaisesRegex(ValueError, "Malformed annotation"):
                prepare_inputs(source, Path(folder) / "fresh", "acl")
            rows[0]["end"] = 5
            rows[0]["unit_type"] = "term_string"
            train.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaisesRegex(ValueError, "span_mention"):
                prepare_inputs(source, Path(folder) / "fresh", "acl")

    def test_empty_input_directory_is_accepted_but_unmanifested_content_is_not(self):
        with tempfile.TemporaryDirectory() as folder:
            source, out = Path(folder) / "processed", Path(folder) / "runs"
            self.write_source(source, "acl")
            target = paths(source, out, "acl")["test_units"].parent
            target.mkdir(parents=True)
            result = prepare_inputs(source, out, "acl")
            self.assertEqual(result["outputs"]["test_units"]["rows"], 2)
            self.assertTrue((target / "input_manifest.json").is_file())
            self.assertEqual(prepare_inputs(source, out, "acl"), result)
            blocked = Path(folder) / "unmanifested"
            target = paths(source, blocked, "acl")["test_units"].parent
            target.mkdir(parents=True)
            sentinel = target / "unfinished.txt"
            sentinel.write_text("preserve")
            with self.assertRaisesRegex(ValueError, "without a manifest"):
                prepare_inputs(source, blocked, "acl")
            self.assertEqual(sentinel.read_text(), "preserve")
            self.assertFalse((target / "test_units.jsonl").exists())

    def test_incomplete_annotation_is_not_a_negative_placeholder(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "processed"
            self.write_source(source, "acl")
            train = source / "acl/train.jsonl"
            rows = [json.loads(l) for l in train.read_text().splitlines()]
            rows[0]["start"] = rows[0]["end"] = None
            train.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaisesRegex(ValueError, "Malformed placeholder"):
                prepare_inputs(source, Path(folder) / "runs", "acl")
            rows[0]["surface"] = rows[0]["norm"] = rows[0]["class_label"] = None
            del rows[0]["start"]
            train.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaisesRegex(ValueError, "Missing annotation offsets"):
                prepare_inputs(source, Path(folder) / "runs", "acl")


if __name__ == "__main__":
    unittest.main()
