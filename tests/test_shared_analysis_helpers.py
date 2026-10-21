"""Serialization and overlap contracts on invented data only."""
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

from ate_paper.io import read_csv, write_csv

try:
    from ate_paper.analysis.matching import near_boundary_mask
except ImportError:
    near_boundary_mask = None


class SharedFileContracts(unittest.TestCase):
    def test_csv_preserves_nulls_missing_fields_unicode_and_lf(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.csv"
            write_csv(path, [{"a": None, "b": "x,y"}, {"a": 0.5, "c": "слово"}])
            self.assertEqual(path.read_bytes(), 'a,b,c\n,"x,y",\n0.5,,слово\n'.encode())
            self.assertEqual(read_csv(path), [{"a": "", "b": "x,y", "c": ""},
                                              {"a": "0.5", "b": "", "c": "слово"}])
            with self.assertRaises(FileExistsError):
                write_csv(path, [{"a": "replacement"}])
            self.assertEqual(read_csv(path)[1]["c"], "слово")

    def test_empty_populations_keep_each_producer_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_csv(root / "empty.csv", [])
            self.assertEqual((root / "empty.csv").read_bytes(), b"\n")
            write_csv(root / "skipped.csv", [], empty="skip")
            self.assertFalse((root / "skipped.csv").exists())
            with self.assertRaisesRegex(ValueError, "Empty result"):
                write_csv(root / "rejected.csv", [], empty="raise")
            self.assertFalse((root / "rejected.csv").exists())


@unittest.skipIf(near_boundary_mask is None, "Install the analysis profile from requirements.lock")
class WitnessContracts(unittest.TestCase):
    def test_same_unit_many_to_one_screen_matches_rational_oracle(self):
        keys = [(doc, unit, start, end)
                for doc, unit in [("d1", "s1"), ("d1", "s2"), ("d2", "s1")]
                for start, end in [(0, 2), (0, 4), (1, 3), (2, 4), (4, 6), (6, 9)]]
        witnesses = {("d1", "s1", 0, 4), ("d1", "s1", 5, 8), ("d2", "s1", 1, 3)}
        for threshold in [Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1)]:
            expected = []
            for doc, unit, start, end in keys:
                expected.append(any(
                    other[:2] == (doc, unit)
                    and Fraction(max(0, min(end, other[3]) - max(start, other[2])),
                                 max(end, other[3]) - min(start, other[2])) >= threshold
                    for other in witnesses))
            with self.subTest(threshold=threshold):
                self.assertEqual(near_boundary_mask(keys, witnesses, float(threshold)).tolist(), expected)
        self.assertEqual(near_boundary_mask(keys, set()).tolist(), [False] * len(keys))


if __name__ == "__main__":
    unittest.main()
