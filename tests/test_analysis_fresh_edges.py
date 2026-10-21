"""Fresh-output edge contracts with toy rows and toy producer artifacts only."""
import contextlib
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ate_paper
from ate_paper.analysis import profiles
try:
    import numpy as np
except ImportError:
    interval = None
else:
    from ate_paper.analysis.statistics import interval, ratio, summarize_cloud

ROOT = Path(ate_paper.__file__).resolve().parents[2]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src/ate_paper/analysis" / (name.removeprefix("reproduce_") + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@unittest.skipIf(interval is None, "Install the analysis profile from requirements.lock for numerical tests")
class AnalysisFreshEdgesTest(unittest.TestCase):
    def test_no_witness_empty_or_perfect_outputs_keep_undefined_ratios(self):
        module = script("reproduce_no_witness")
        module.np, module.interval, module.ratio, module.summarize_cloud = np, interval, ratio, summarize_cloud
        specs = [{"run": f"{model}_seed{seed}", "model": model, "seed": seed}
                 for model in ("a", "b") for seed in range(42, 47)]
        gold = {("toy", "s", 0, 5)}
        for condition in ("empty", "perfect", "mixed"):
            predictions = {}
            for spec in specs:
                value = set() if condition == "empty" else set(gold)
                if condition == "mixed" and spec["model"] == "b":
                    value.add(("toy", "s", 5, 10))
                predictions[spec["run"]] = value
            with self.subTest(condition=condition), contextlib.redirect_stdout(io.StringIO()):
                pairs, configs, boots = module.compute("GENIA", specs, gold, predictions,
                                                      ["toy"], np.ones((3, 1)), False)
                self.assertEqual(len(pairs), 45)
                self.assertEqual(len(configs), 1)
                self.assertEqual(configs[0][module.TARGET + "_valid_pairs"], 0)
                self.assertTrue(math.isnan(configs[0][module.TARGET + "_mean"]))
                self.assertEqual(boots[0]["between_valid"], 0)
                self.assertIsNone(boots[0]["between_ci_low"])

    def test_profiles_fresh_rejects_partial_gold_before_scoring(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split in ("train", "test"):
                row = {"dataset": "ACL RD-TEC 2.0", "version": "toy", "split": split,
                       "doc_id": split, "sent_id": "s", "text": "Alpha.",
                       "unit_type": "span_mention", "surface": "Alpha", "norm": "alpha",
                       "start": 0, "end": 5, "class_label": "Model"}
                path = root / "data/acl" / (split + ".jsonl")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "incomplete|full test"):
                profiles.rebuild_dataset(root, root / "data", "ACL RD-TEC 2.0", [], root / "fresh")

    def test_nesting_accepts_zero_matched_runs_and_binds_producer_inputs(self):
        module = script("reproduce_nesting_contrasts")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, data, output = root / "producer", root / "data", root / "build/nesting"
            source.mkdir()
            implementation = root / "src/ate_paper/analysis/io.py"
            implementation.parent.mkdir(parents=True)
            implementation.write_bytes((ROOT / "src/ate_paper/analysis/io.py").read_bytes())
            run_rows, config_rows, accepted = [], [], []
            gold_paths, arrays = {}, {}
            for dataset, pairs in module.PAIRS.items():
                slug = module.SLUGS[dataset]
                row = {"dataset": dataset, "version": "toy", "split": "test", "doc_id": "doc-" + slug,
                       "text": "Alpha beta.", "unit_type": "span_mention", "start": 0,
                       "end": 5, "surface": "Alpha", "norm": "alpha", "class_label": "toy"}
                rows = [{**row, "sent_id": "flat"}, {**row, "sent_id": "nested", "end": 10,
                                                          "surface": "Alpha beta", "norm": "alpha beta"},
                        {**row, "sent_id": "nested"}]
                gold_path = data / slug / "test.jsonl"
                gold_path.parent.mkdir(parents=True)
                gold_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
                gold_paths[dataset] = gold_path
                draw_path = root / "draws" / (slug + ".npz")
                draw_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(draw_path, doc_ids=np.array([row["doc_id"]]), weights=np.ones((3, 1), dtype=int))
                arrays[dataset] = draw_path
                for model in sorted({model for pair in pairs for model in pair}):
                    run_rows.extend({"dataset": dataset, "model": model, "run": f"{model}_seed{seed}"}
                                    for seed in range(42, 47))
                for a, b in pairs:
                    config_rows.append({"dataset": dataset, "model_a": a, "model_b": b,
                                        **{f"{scope}_05_{metric}": float("nan")
                                           for scope in ("flat", "nonflat") for metric in ("between", "excess")}})
                    accepted.append({"dataset": dataset, "model_a": a, "model_b": b,
                                     "between_pairs": 25, "within_pairs_per_side": 10,
                                     **{f"{scope}_{metric}": 0.0
                                        for scope in ("flat", "nonflat") for metric in ("between", "excess")},
                                     **{f"delta_{metric}": 0.0 for metric in ("between", "excess")}})
            write_csv(source / "run_scores.csv", run_rows)
            write_csv(source / "configuration_pairs.csv", config_rows)
            assignments = source / "assignments_05.csv.gz"
            with gzip.open(assignments, "wt", newline="") as f:
                csv.writer(f).writerow(("dataset", "run", "doc_id", "sent_id", "gold_start", "gold_end",
                                        "pred_start", "pred_end", "iou_num", "iou_den", "exact"))
            def receipt(path):
                return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            (source / "manifest.json").write_text(json.dumps({
                "status": "COMPUTED", "input_mode": "new_predictions", "prediction_root": str(root / "fresh"),
                "outputs": [receipt(source / name) for name in
                            ("run_scores.csv", "configuration_pairs.csv", "assignments_05.csv.gz")],
                "inputs": [receipt(p) for p in [*gold_paths.values(), *arrays.values()]]}))
            write_csv(root / "results/relaxed_gold_id_agreement_2026_10_13/nested_contrasts.csv", accepted)
            args = ["nesting", "--data-dir", str(data), "--gold-output", str(source), "--output", str(output)]
            with patch.object(module, "ROOT", root), patch.object(module, "ARRAYS", arrays), \
                    patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                module.main()
            result = json.loads((output / "nested_contrasts_manifest.json").read_text())
            self.assertEqual(result["status"], "COMPUTED")
            computed = module.rows(output / "nested_contrasts.csv")
            self.assertEqual(len(computed), 8)
            self.assertTrue(all(row["delta_between_valid"] == "0" for row in computed))
            self.assertTrue(all(math.isnan(float(row["delta_between"])) for row in computed))
            # A different gold file cannot be combined with the producer's assignments/draws.
            first = next(iter(gold_paths.values()))
            first.write_text(first.read_text() + "\n")
            args[-1] = str(root / "build/wrong-gold")
            with patch.object(module, "ROOT", root), patch.object(module, "ARRAYS", arrays), \
                    patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(ValueError, "Gold/draw input differs"):
                module.main()


if __name__ == "__main__":
    unittest.main()
