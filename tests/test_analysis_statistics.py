"""Hand-calculated fixed-population examples; no saved scientific inputs."""
import unittest
import tempfile
from pathlib import Path

try:
    import numpy as np
except ImportError:
    interval = None
else:
    from ate_paper.analysis.statistics import interval, ratio, summarize_cloud


@unittest.skipIf(interval is None, 'Install the analysis profile from requirements.lock for numerical tests')
class AnalysisStatisticsTest(unittest.TestCase):
    def test_fixed_pair_means_and_within_max_are_recomputed_per_draw(self):
        values = np.array([[.8, .2, .1, .3], [.2, .4, .8, .1], [np.nan, .4, .1, .1]])
        between, within, excess = summarize_cloud(values, [0, 1], [2], [3])
        np.testing.assert_allclose(between, [.5, .3, np.nan], equal_nan=True)
        np.testing.assert_allclose(within, [.3, .8, .1])
        np.testing.assert_allclose(excess, [.2, -.5, np.nan], equal_nan=True)
        bounds = interval(excess)
        self.assertEqual(bounds['valid'], 2)
        self.assertAlmostEqual(bounds['ci_low'], -.4825)
        self.assertAlmostEqual(bounds['ci_high'], .1825)

    def test_single_output_reference_rules_do_not_create_fake_repeats(self):
        values = np.array([[.2, .6], [.4, .1]])
        between, reference, excess = summarize_cloud(values, [0], [1], [])
        np.testing.assert_allclose(between, [.2, .4])
        np.testing.assert_allclose(reference, [.6, .1])
        np.testing.assert_allclose(excess, [-.4, .3])
        _, reference, excess = summarize_cloud(values, [0], [], [])
        self.assertTrue(np.isnan(reference).all())
        self.assertTrue(np.isnan(excess).all())
        self.assertEqual(interval(excess), {'valid': 0, 'ci_low': None, 'ci_high': None})

    def test_document_counts_are_aggregated_before_corpus_ratios(self):
        # Two documents: rates 1/1 and 1/9. A draw taking both is 2/10, not 5/9.
        counts, totals = np.array([1, 1]), np.array([1, 9])
        draws = np.array([[1, 1], [2, 0], [0, 2], [0, 0]])
        actual = ratio(draws @ counts, draws @ totals)
        np.testing.assert_allclose(actual, [.2, 1., 1/9, np.nan], equal_nan=True)

    def test_shared_draw_loader_checks_full_document_population_and_weights(self):
        from ate_paper.analysis.io import load_document_draws
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'results/document_draws/acl_rd_tec_2_0.npz'
            path.parent.mkdir(parents=True)
            weights = np.ones((5000, 2), dtype=int)
            np.savez(path, doc_ids=np.array(['term', 'negative']), weights=weights)
            docs, _, _ = load_document_draws(root, 'ACL RD-TEC 2.0', {'term', 'negative'})
            self.assertEqual(docs, ['term', 'negative'])
            with self.assertRaisesRegex(ValueError, 'universe'):
                load_document_draws(root, 'ACL RD-TEC 2.0', {'term'})
            for values, message in [([3, -1], 'Invalid'), ([3, 0], 'population size')]:
                weights[0] = values
                np.savez(path, doc_ids=np.array(['term', 'negative']), weights=weights)
                with self.assertRaisesRegex(ValueError, message):
                    load_document_draws(root, 'ACL RD-TEC 2.0', {'term', 'negative'})


if __name__ == '__main__':
    unittest.main()
