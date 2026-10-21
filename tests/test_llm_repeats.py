"""A17 contracts on artificial spans and metadata; no real predictions or corpus."""
import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ate_paper
from ate_paper.analysis import io, repeats
from ate_paper.predictions import read_jsonl, release_record, sha256


class LLMRepeatsTest(unittest.TestCase):
    def test_exact_detection_keeps_full_gold_and_deduplicates_classes(self):
        first = {"dataset": "GENIA", "doc_id": "toy1", "sent_id": 1, "unit_type": "span_mention",
                 "start": 0, "end": 5, "class_label": "protein_molecule"}
        second = {**first, "doc_id": "toy2"}
        empty = {**first, "doc_id": "empty", "start": None, "end": None}
        false_positive = {**first, "doc_id": "empty"}
        score = repeats.score_output([first, second, empty],
                                     [first, {**first, "class_label": "cell_type"}, false_positive, empty])
        self.assertEqual((score["gold"], score["predicted"], score["tp"], score["fp"], score["fn"]),
                         (2, 2, 1, 1, 1))
        self.assertEqual(score["f1"], 0.5)
        self.assertEqual(repeats.score_output([first, second], [first])["f1"], 2 / 3)

    def test_selected_roster_preserves_noncanonical_base_paths(self):
        root = Path(ate_paper.__file__).resolve().parents[2]
        pairs = io.read_csv(root / "predictions/llm_repeat_pairs.csv")
        manifest = io.read_csv(root / "predictions/manifest.csv")
        repeats.validate_roster(pairs, manifest)
        by_id = {row["pair_id"]: row for row in pairs}
        self.assertEqual(by_id["P07"]["decision"], "primary_documented")
        self.assertIn("/llm_repeats/pred_gemini_3_1_flash_lite.jsonl.gz", by_id["P07"]["base_path"])
        self.assertIn("deepinfra_repeat2", by_id["P23"]["base_path"])
        self.assertIn("deepinfra_repeat3", by_id["P23"]["repeat_path"])
        for pair in ("P24", "P25"):
            self.assertIn("gemini_3_1_flash_lite_repeat2", by_id[pair]["base_path"])
            self.assertIn("gemini_3_1_flash_lite_repeat3", by_id[pair]["repeat_path"])
        self.assertEqual({r["pair_id"] for r in pairs if r["decision"] == "transport_contrast"},
                         {"P03", "P17", "P21"})
        changed = [dict(row) for row in pairs]
        changed[2]["decision"] = "primary_raw"
        with self.assertRaises(ValueError):
            repeats.validate_roster(changed, manifest)

    def test_descriptive_sensitivity_changes_pair_population_only(self):
        root = Path(ate_paper.__file__).resolve().parents[2]
        pairs = io.read_csv(root / "predictions/llm_repeat_pairs.csv")
        rows = [{"pair_id": r["pair_id"], "dataset": io.SLUGS[r["dataset"]],
                 "decision": r["decision"], "absolute_delta_unlabeled_f1":
                 (0.4 if r["pair_id"] == "P19" else 0.1)} for r in pairs]
        summary = repeats.aggregate_pairs(rows)
        self.assertEqual(sum(v["n_pairs"] for v in summary["summary"].values()), 22)
        self.assertEqual(summary["sensitivity"]["cl_without_llama_format_failure_pair"]["n_pairs"], 5)
        self.assertEqual(summary["sensitivity"]["cl_without_llama_format_failure_pair"]["max_abs_delta_unlabeled_f1"], 0.1)
        self.assertEqual(summary["summary"]["cl"]["max_abs_delta_unlabeled_f1"], 0.4)
        self.assertIsNone(summary["transport_summary"]["genia"]["max_abs_delta_unlabeled_f1"])
        self.assertEqual(len(rows), 25)

    def test_complete_pipeline_with_toy_corpus_fixed_and_fresh_outputs(self):
        source = Path(ate_paper.__file__).resolve().parents[2]
        pairs = io.read_csv(source / "predictions/llm_repeat_pairs.csv")
        public = {r["release_path"]: r for r in io.read_csv(source / "predictions/manifest.csv")}
        bases = {r["base_path"] for r in pairs}
        repeated = {r["repeat_path"] for r in pairs}
        self.assertFalse(bases & repeated)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data, fresh = root / "data", root / "fresh"
            (root / "predictions").mkdir()
            with (root / "predictions/llm_repeat_pairs.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(pairs[0]))
                writer.writeheader()
                writer.writerows(pairs)
            golds = {}
            for dataset, slug in io.SLUGS.items():
                first = {"dataset": dataset, "version": "toy", "split": "test", "language": "en",
                         "domain": "toy", "doc_id": "toy1", "sent_id": 1, "text": "Alpha beta.",
                         "unit_type": "span_mention", "start": 0, "end": 5,
                         "surface": "Alpha", "norm": "alpha", "class_label": "toy"}
                second = {**first, "doc_id": "toy2", "sent_id": 2}
                empty = {**first, "doc_id": "empty", "sent_id": 0, "start": None,
                         "end": None, "surface": None, "norm": None, "class_label": None}
                golds[dataset] = [first, second, empty]
                path = data / slug / "test.jsonl"
                path.parent.mkdir(parents=True)
                path.write_text("".join(json.dumps(r) + "\n" for r in golds[dataset]))
            entries = []
            for relative in sorted(bases | repeated):
                entry = {k: public[relative][k] for k in ("dataset", "group", "run", "release_path")}
                first, second, empty = golds[entry["dataset"]]
                fp = {**empty, "start": 0, "end": 5, "surface": "Alpha", "norm": "alpha", "class_label": "toy"}
                saved = [first] if relative in bases else [first, fp]
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(gzip.compress("".join(json.dumps(release_record(r)) + "\n" for r in saved).encode(), mtime=0))
                entry["release_sha256"] = sha256(path)
                entries.append(entry)
                new_rows = [first, second] if relative in bases else [first]
                target = fresh / io.SLUGS[entry["dataset"]] / entry["group"] / entry["run"] / "predictions.jsonl"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("".join(json.dumps(r) + "\n" for r in new_rows))
            with (root / "predictions/manifest.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(entries[0]))
                writer.writeheader()
                writer.writerows(entries)
            delta = abs(2 / 3 - 0.5)
            reference = [{"pair_id": r["pair_id"], "dataset": io.SLUGS[r["dataset"]],
                          "decision": r["decision"], "model_key": r["model_key"],
                          "base_f1": 2 / 3, "repeat_f1": 0.5, "absolute_delta_unlabeled_f1": delta}
                         for r in pairs]
            def constant_summary(counts):
                return {ds: {"n_pairs": n, "median_abs_delta_unlabeled_f1": delta,
                             "max_abs_delta_unlabeled_f1": delta} for ds, n in counts.items()}
            retained = {"summary": constant_summary({"acl": 7, "genia": 9, "cl": 6}),
                        "all_saved_full_test_pairs": constant_summary({"acl": 8, "genia": 9, "cl": 8}),
                        "sensitivity": {
                            "genia_without_documented_raw_missing_pair": constant_summary({"genia": 8})["genia"],
                            "cl_without_llama_format_failure_pair": constant_summary({"cl": 5})["cl"]},
                        "transport_contrasts": [{k: r[k] for k in ("pair_id", "dataset", "absolute_delta_unlabeled_f1")}
                                                for r in reference if r["decision"] == "transport_contrast"]}
            where = root / repeats.REFERENCE_DIR
            where.mkdir(parents=True)
            with (where / "limitations_full_repeat_f1.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(reference[0]))
                writer.writeheader()
                writer.writerows(reference)
            (where / "limitations_full_repeat_f1_summary.json").write_text(json.dumps(retained))
            expected = {dataset: {"documents": 3, "units": 3, "spans": 2} for dataset in io.SLUGS}
            with patch.dict(io.TEST_COUNTS, expected, clear=True):
                fixed = repeats.reproduce_llm_repeats(root, data, root / "build/fixed")
                self.assertEqual(fixed["status"], "PASS")
                self.assertTrue(fixed["comparison_to_reference"]["all_saved_values_match"])
                table = io.read_csv(root / "build/fixed/limitations_full_repeat_f1.csv")
                self.assertTrue(all(r["gold"] == "2" and r["full_test_units"] == "3" for r in table))
                self.assertEqual(fixed["limitations"]["P19"]["historical_base_failed_batches"], 27)
                self.assertEqual(fixed["limitations"]["P19"]["historical_repeat_failed_batches"], 64)
                new = repeats.reproduce_llm_repeats(root, data, root / "build/new", fresh)
                self.assertEqual(new["status"], "COMPUTED")
                self.assertIsNone(new["checks"])
                self.assertFalse(new["comparison_to_reference"]["all_saved_values_match"])
                table = io.read_csv(root / "build/new/limitations_full_repeat_f1.csv")
                self.assertTrue(all(float(r["base_f1"]) == 1.0 for r in table))
                with self.assertRaises(FileExistsError):
                    repeats.reproduce_llm_repeats(root, data, root / "build/fixed")
                (root / entries[0]["release_path"]).write_bytes(b"corrupt synthetic release")
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    repeats.reproduce_llm_repeats(root, data, root / "build/corrupt")


if __name__ == "__main__":
    unittest.main()
