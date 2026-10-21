"""Small invented sets only; no article data or numerical result is recomputed."""
import itertools
import unittest
import hashlib
import json
import tempfile
from pathlib import Path
from collections import defaultdict

try:
    import numpy as np
    import scipy
except ImportError:
    aggregate = None
else:
    from ate_paper.analysis.sensitivity import (MODEL_PAIRS, aggregate, cl_ambiguity,
        distance, metric_summaries, mixed_choices, optimal_covered_sets, slice_stat,
        verify_producer, check_producer_inputs)


@unittest.skipIf(aggregate is None, "Install requirements-analysis.lock for numerical tests")
class SensitivityTest(unittest.TestCase):
    def test_distance_and_undefined(self):
        row = {"exact_a": 2, "exact_b": 4, "exact_common": 1}
        self.assertEqual(distance(row, "exact", "smaller_set"), .5)
        self.assertEqual(distance(row, "exact", "jaccard"), .8)
        row.update(exact_a=0, exact_common=0)
        self.assertIsNone(distance(row, "exact", "smaller_set"))
        self.assertEqual(distance(row, "exact", "jaccard"), 1)

    def test_undefined_within_constituent_is_not_dropped(self):
        runs = [("a1", "a"), ("a2", "a"), ("b1", "b"), ("b2", "b")]
        rows = []
        for (a, ma), (b, mb) in itertools.combinations(runs, 2):
            rows.append({"dataset": "toy", "run_a": a, "run_b": b,
                         "model_a": ma, "model_b": mb, "within_configuration": ma == mb,
                         "exact_a": 0 if a == "b1" else 2, "exact_b": 2, "exact_common": 0})
        self.assertIsNone(aggregate(rows, "exact", "smaller_set")[0]["excess"])

    def test_exact_anchor_and_two_way_coverage_tie(self):
        gold = {(0, 4), (2, 6), (10, 12)}
        predictions = {(1, 5), (10, 12)}
        choices = optimal_covered_sets(gold, predictions)
        self.assertEqual(set(choices), {frozenset({(0, 4), (10, 12)}),
                                       frozenset({(2, 6), (10, 12)})})

    def test_complex_component_has_three_optimal_covered_sets(self):
        gold = {(0, 4), (1, 5), (2, 6)}
        choices = optimal_covered_sets(gold, {(0, 5), (1, 6)})
        self.assertEqual(set(choices), {frozenset(x) for x in itertools.combinations(gold, 2)})

    def test_resource_limit_never_truncates_with_success(self):
        with self.assertRaisesRegex(ValueError, "Unsupported CL tie size"):
            optimal_covered_sets({(0, 4), (2, 6)}, {(1, 5)}, max_states=1)

    def test_mixed_choice_order_preserves_old_binary_masks(self):
        self.assertEqual(list(mixed_choices([2, 2])),
                         [(0, [0, 0]), (1, [1, 0]), (2, [0, 1]), (3, [1, 1])])
        self.assertEqual(list(mixed_choices([])), [(0, [])])

    def test_undefined_second_within_side_propagates(self):
        sets = {"a1": {("d", None, 0, 1)}, "a2": {("d", None, 0, 1)},
                "b1": set(), "b2": set()}
        point, _ = slice_stat(sets, ["a1", "a2"], ["b1", "b2"], None, {})
        self.assertTrue(np.isnan(point[1]))

    def test_wrong_prediction_root_is_rejected_before_reading_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps(
                {"status": "COMPUTED", "prediction_root": str(root / "first")}))
            with self.assertRaisesRegex(ValueError, "prediction root differs"):
                verify_producer(root, root, [], root / "second")

    def test_changed_input_is_rejected_against_producer_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.jsonl"
            path.write_text("toy input")
            recorded = {path.resolve(): hashlib.sha256(path.read_bytes()).hexdigest()}
            check_producer_inputs([path], recorded)
            path.write_text("changed toy input")
            with self.assertRaisesRegex(ValueError, "differs from gold-producer inputs"):
                check_producer_inputs([path], recorded)

    def test_zero_ambiguities_are_a_valid_dynamic_population(self):
        dataset = "CL-RuTerm3"
        offsets = [(0, 8), (1, 4), (10, 14)]
        gold = [{"dataset": dataset, "unit_type": "span_mention", "doc_id": "d",
                 "sent_id": None, "start": a, "end": b} for a, b in offsets]
        specs, predictions, assignments, scores = [], {}, [], []
        for model in {m for _, a, b in MODEL_PAIRS for m in (a, b)}:
            for seed in (1, 2):
                run = model + str(seed)
                specs.append({"run": run, "model": model})
                predictions[run] = gold
                scores.append({"dataset": dataset, "run": run, "relaxed_05_tp": 3})
                assignments.extend({"dataset": dataset, "run": run, "doc_id": "d", "sent_id": None,
                                    "gold_start": a, "gold_end": b, "pred_start": a, "pred_end": b}
                                   for a, b in offsets)
        rows, info = cl_ambiguity(specs, gold, predictions, assignments, [], scores,
                                  ["d"], np.array([[1], [1]], dtype=float))
        self.assertEqual(info["global_equal_optimum_combinations"], 1)
        self.assertEqual(info["selected_ambiguous_units"], 0)
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(r["point"] == r["ci_low"] == r["ci_high"] == 0 for r in rows))


if __name__ == "__main__":
    unittest.main()
