"""Synthetic exhaustive checks of exact-first relaxed gold matching."""

from fractions import Fraction
import itertools
import unittest

try:
    import numpy
    import scipy
except ImportError:
    match_unit = overlap_fraction = None
else:
    from ate_paper.analysis.matching import match_unit, overlap_fraction


@unittest.skipIf(match_unit is None, "Install the analysis extra to test scipy matching")
class MatchingTests(unittest.TestCase):
    def test_exact_identity_and_half_threshold(self):
        gold = {(0, 4), (8, 12)}
        pred = {(0, 4), (8, 10)}
        exact, _ = match_unit(gold, pred, Fraction(1))
        relaxed, _ = match_unit(gold, pred)
        assert [(g, p) for g, p, *_ in exact] == [((0, 4), (0, 4))]
        assert {(g, p) for g, p, *_ in relaxed} == {
            ((0, 4), (0, 4)), ((8, 12), (8, 10))
        }
        assert overlap_fraction((0, 4), (3, 7)) == Fraction(1, 7)


    def test_exact_priority_does_not_reassign_gold_to_raise_cardinality(self):
        # p=(0,4) is exact for g=(0,4), but also overlaps g=(0,8).
        # The other prediction overlaps only the exact gold.  Full-graph maximum
        # cardinality can match both, whereas exact-priority intentionally cannot.
        gold = {(0, 4), (0, 8)}
        pred = {(0, 4), (1, 4)}
        matched, _ = match_unit(gold, pred)
        assert matched[0][:2] == ((0, 4), (0, 4))
        assert len(matched) == 1


    def test_maximum_cardinality_beats_greedy_and_is_one_to_one(self):
        gold = {(0, 4), (0, 2)}
        pred = {(0, 3), (2, 4)}
        matched, _ = match_unit(gold, pred)
        assert {(g, p) for g, p, *_ in matched} == {
            ((0, 2), (0, 3)), ((0, 4), (2, 4))
        }


    def test_ambiguous_coverage_and_stable_tie_break(self):
        gold = {(0, 4), (1, 5)}
        pred = {(1, 4)}
        matched, ambiguous = match_unit(gold, pred)
        assert len(matched) == 1
        assert ambiguous == 2
        assert matched == match_unit(set(reversed(sorted(gold))), set(pred))[0]


    def test_nested_offsets_are_distinct_and_empty_is_safe(self):
        gold = {(0, 10), (2, 8), (3, 6)}
        pred = {(0, 10), (2, 8), (3, 6)}
        matched, ambiguous = match_unit(gold, pred)
        assert len(matched) == 3 and ambiguous == 0
        assert all(exact for *_, exact in matched)
        assert match_unit(gold, set()) == ([], 0)


    def test_invalid_span_rejected(self):
        try:
            match_unit({(3, 3)}, set())
        except ValueError:
            pass
        else:
            raise AssertionError("Zero-length span was accepted")


    def test_all_small_offset_subsets_against_exhaustive_lex_optimum(self):
        gold_options = ((0, 2), (0, 4), (1, 3), (2, 5))
        pred_options = ((0, 3), (0, 4), (1, 4), (2, 5))

        def all_subsets(options):
            return [set(group) for n in range(len(options) + 1)
                    for group in itertools.combinations(options, n)]

        def brute(gold, pred, threshold):
            anchored = gold & pred
            gs, ps = sorted(gold - anchored), sorted(pred - anchored)
            candidates = []

            def recurse(index, used, current):
                if index == len(gs):
                    chosen = tuple(sorted(current))
                    score = (len(chosen), sum((overlap_fraction(g, p) for g, p in chosen),
                                              Fraction(0)))
                    candidates.append((score, chosen))
                    return
                recurse(index + 1, used, current)
                for p in ps:
                    if p not in used and overlap_fraction(gs[index], p) >= threshold:
                        recurse(index + 1, used | {p}, current + [(gs[index], p)])

            recurse(0, set(), [])
            best_score = max(score for score, _ in candidates)
            best_edges = min(edges for score, edges in candidates if score == best_score)
            return {(g, p) for g, p in best_edges} | {(x, x) for x in anchored}

        for gold in all_subsets(gold_options):
            for pred in all_subsets(pred_options):
                for threshold in (Fraction(1, 2), Fraction(3, 4)):
                    actual, _ = match_unit(gold, pred, threshold)
                    assert {(g, p) for g, p, *_ in actual} == brute(gold, pred, threshold)

if __name__ == "__main__":
    unittest.main()
