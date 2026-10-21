"""Corner cases use invented sets/CSV fixtures only, never article data."""
import csv
import gzip
import hashlib
import importlib.util
import io
import itertools
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
try:
    import numpy as np
    import scipy
except ImportError:
    sensitivity = None
else:
    from ate_paper.analysis.statistics import ratio, interval, summarize_cloud
    from ate_paper.analysis.matching import match_unit
    from ate_paper.analysis import sensitivity, io as analysis_io


def script(name, candidate=False):
    path = ROOT / 'scripts' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name + ('_candidate' if candidate else ''), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.np, module.ratio, module.interval, module.summarize_cloud = np, ratio, interval, summarize_cloud
    return module


def write_csv(path, rows, fields=None):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@unittest.skipIf(sensitivity is None, "Install requirements-analysis.lock for numerical tests")
class GoldReviewCorners(unittest.TestCase):
    def test_common_six_is_fixed_when_only_acl_is_selected(self):
        module = script('reproduce_no_witness', True)
        models = sorted(module.COMMON_MODELS) + ['extra1', 'extra2', 'extra3', 'extra4']
        rows = [{'dataset': 'ACL RD-TEC 2.0', 'model_a': a, 'model_b': b,
                 module.TARGET + '_mean': .5, module.TARGET + '_excess': .1}
                for a, b in itertools.combinations(models, 2)]
        result = module.configuration_summary(rows, [], ['ACL RD-TEC 2.0'])['ACL RD-TEC 2.0']
        self.assertEqual(result['configuration_pairs'], 45)
        self.assertEqual(result['shared_model_count'], 6)
        self.assertEqual(result['shared_configuration_pairs'], 15)
        rows[0][module.TARGET + '_excess'] = float('nan')
        first = module.configuration_summary(rows, [], ['ACL RD-TEC 2.0'])
        last = module.configuration_summary(list(reversed(rows)), [], ['ACL RD-TEC 2.0'])
        self.assertEqual(first, last)
        self.assertIsNone(first['ACL RD-TEC 2.0']['median_excess'])
        self.assertEqual(first['ACL RD-TEC 2.0']['undefined_excess_pairs'], 1)
        json.dumps(first, allow_nan=False)

    def test_no_false_positives_keep_undefined_ratios_and_zero_valid_draws(self):
        module = script('reproduce_no_witness', True)
        specs = [{'run': model + str(seed), 'model': model}
                 for model in sorted(module.COMMON_MODELS) for seed in range(5)]
        predictions = {row['run']: set() for row in specs}
        with redirect_stdout(io.StringIO()):
            pairs, configs, boots = module.compute('toy', specs, {('d', None, 0, 2)},
                                                  predictions, ['d'], np.array([[1], [2]]), False)
        self.assertTrue(all(math.isnan(row[module.TARGET]) for row in pairs))
        self.assertTrue(all(row['excess_valid'] == 0 and row['excess_ci_low'] is None for row in boots))
        summary = module.configuration_summary(configs, boots, ['toy'])['toy']
        self.assertIsNone(summary['median_share'])
        self.assertEqual(summary['defined_share_pairs'], 0)
        self.assertEqual(summary['shared_configuration_pairs'], 15)

    def test_zero_matched_gold_runs_and_bootstrap_do_not_crash(self):
        module = script('reproduce_gold_agreement')
        module.match_unit = match_unit
        specs = [{'run': model + str(seed), 'model': model} for model in ['a', 'b'] for seed in range(5)]
        predictions = {row['run']: set() for row in specs}
        gold = {('d', None, 0, 2)}
        stream = io.StringIO()
        reference = [{'dataset': 'toy', 'group_a': 'supervised', 'group_b': 'supervised',
                      'model_a': 'a', 'model_b': 'b', 'tp_turnover_mean': 0, 'tp_turnover_excess': 0}]
        with redirect_stdout(io.StringIO()), patch.object(module, 'read_csv', side_effect=lambda path: [] if 'run_pairs' in path.name else reference):
            matched, ambiguity, scores = module.match_runs('toy', specs, gold, predictions, gold, csv.writer(stream))
            pairs, pair_indices, run_counts, common, resolved, unique, _ = module.pair_counts(
                'toy', specs, matched, gold, gold, ['d'], check_reference=False)
            module.CONFIG_PATHS['toy'] = Path('toy-reference.csv')
            configs, indices = module.configuration_summaries('toy', specs, pairs, pair_indices, check_reference=False)
            boots = module.bootstrap('toy', np.array([[1], [2]]), pair_indices,
                                     run_counts, common, resolved, unique, configs, indices)
        self.assertFalse(ambiguity)
        self.assertEqual(len(scores), 10)
        self.assertTrue(all(r['relaxed_05_tp'] == 0 for r in scores))
        self.assertTrue(all(r['valid_draws'] == 0 and r['ci_low'] is None for r in boots))
        self.assertTrue(all(math.isnan(r['relaxed_05_between']) for r in configs))

    def test_nesting_inherits_available_partial_upstream_corpora(self):
        module = script('reproduce_nesting_contrasts', True)
        self.assertEqual(module.selected_datasets([{'dataset': 'CL-RuTerm3'}]), ('CL-RuTerm3',))
        self.assertEqual(module.selected_datasets([{'dataset': 'GENIA'}, {'dataset': 'CL-RuTerm3'}], ['cl']),
                         ('CL-RuTerm3',))
        with self.assertRaisesRegex(ValueError, 'absent'):
            module.selected_datasets([{'dataset': 'CL-RuTerm3'}], ['genia'])

    def test_sensitivity_partial_cl_and_zero_assignments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, data, output = root / 'upstream', root / 'data', root / 'build/out'
            source.mkdir(); (data / 'cl').mkdir(parents=True)
            gold_path = data / 'cl/test.jsonl'; gold_path.write_text('toy fixture marker')
            for name in ['src/ate_paper/analysis/matching.py', 'src/ate_paper/analysis/statistics.py',
                         'src/ate_paper/analysis/io.py', 'scripts/reproduce_sensitivity.py']:
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# fixture\n')
            models = sorted({m for _, a, b in sensitivity.MODEL_PAIRS for m in (a, b)})
            specs = [{'run': model + str(seed), 'model': model} for model in models for seed in (1, 2)]
            scores = [{'dataset': 'CL-RuTerm3', **s, 'relaxed_05_tp': 0} for s in specs]
            pairs = []
            for a, b in itertools.combinations(specs, 2):
                row = {'dataset': 'CL-RuTerm3', 'run_a': a['run'], 'run_b': b['run'],
                       'model_a': a['model'], 'model_b': b['model'],
                       'within_configuration': a['model'] == b['model']}
                row.update({target + '_' + part: 0 for target in sensitivity.TARGETS for part in ('a', 'b', 'common')})
                pairs.append(row)
            configs = [{'dataset': 'CL-RuTerm3', 'model_a': a, 'model_b': b,
                        **{target + '_excess': float('nan') for target in sensitivity.TARGETS}}
                       for a, b in itertools.combinations(models, 2)]
            write_csv(source / 'run_scores.csv', scores)
            write_csv(source / 'run_pairs.csv', pairs)
            write_csv(source / 'configuration_pairs.csv', configs)
            with gzip.open(source / 'assignments_05.csv.gz', 'wt', newline='') as stream:
                csv.writer(stream).writerow(['dataset', 'run', 'doc_id', 'sent_id', 'gold_start', 'gold_end', 'pred_start', 'pred_end'])
            fresh = root / 'fresh'
            def receipt(path):return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            (source / 'manifest.json').write_text(json.dumps(
                {'status': 'COMPUTED', 'prediction_root': str(fresh), 'inputs': [receipt(gold_path)],
                 'outputs': [receipt(path) for path in source.iterdir()]}))
            gold = [{'dataset': 'CL-RuTerm3', 'unit_type': 'span_mention', 'doc_id': 'd',
                     'sent_id': None, 'start': a, 'end': b} for a, b in [(0, 8), (1, 4), (10, 14)]]
            with patch.object(analysis_io, 'load_analysis_rows', return_value=(specs, gold, {s['run']: [] for s in specs}, [])), \
                    patch.object(analysis_io, 'load_document_draws', side_effect=AssertionError('bootstrap was skipped')):
                result = sensitivity.run(root, source, data, output, ['CL-RuTerm3'], fresh, True)
            self.assertEqual(result['status'], 'COMPUTED')
            summary = json.loads((output / 'summary.json').read_text(),
                                 parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            self.assertEqual(summary['results']['cl_equal_optimum_assignments']['global_equal_optimum_combinations'], 1)
            self.assertEqual(summary['results']['exact_smaller_set']['CL-RuTerm3']['defined_pairs'], 0)


    def test_full_sensitivity_cli_all_zero_with_bootstrap_and_a16(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, data, output = root / 'upstream', root / 'data', root / 'build/out'
            source.mkdir()
            gold_paths = []
            for slug in ['acl', 'genia', 'cl']:
                path = data / slug / 'test.jsonl'; path.parent.mkdir(parents=True)
                path.write_text('toy gold marker'); gold_paths.append(path)
            for name in ['src/ate_paper/analysis/matching.py', 'src/ate_paper/analysis/statistics.py',
                         'src/ate_paper/analysis/io.py', 'scripts/reproduce_sensitivity.py']:
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# fixture\n')
            all_specs, scores, pairs, configs = {}, [], [], []
            for dataset in ['ACL RD-TEC 2.0', 'GENIA', 'CL-RuTerm3']:
                models = (sorted({m for _, a, b in sensitivity.MODEL_PAIRS for m in (a, b)})
                          if dataset == 'CL-RuTerm3' else ['toy-a', 'toy-b'])
                specs = [{'run': model + str(seed), 'model': model} for model in models for seed in (1, 2)]
                all_specs[dataset] = specs
                scores.extend({'dataset': dataset, **r, 'relaxed_05_tp': 0} for r in specs)
                for a, b in itertools.combinations(specs, 2):
                    row = {'dataset': dataset, 'run_a': a['run'], 'run_b': b['run'],
                           'model_a': a['model'], 'model_b': b['model'],
                           'within_configuration': a['model'] == b['model']}
                    row.update({target + '_' + part: 0 for target in sensitivity.TARGETS for part in ('a', 'b', 'common')})
                    pairs.append(row)
                configs.extend({'dataset': dataset, 'model_a': a, 'model_b': b,
                                **{target + '_excess': float('nan') for target in sensitivity.TARGETS}}
                               for a, b in itertools.combinations(models, 2))
            write_csv(source / 'run_scores.csv', scores)
            write_csv(source / 'run_pairs.csv', pairs)
            write_csv(source / 'configuration_pairs.csv', configs)
            with gzip.open(source / 'assignments_05.csv.gz', 'wt', newline='') as stream:
                csv.writer(stream).writerow(['dataset', 'run', 'doc_id', 'sent_id', 'gold_start', 'gold_end', 'pred_start', 'pred_end'])
            draw_path = root / 'toy_draw_receipt'; draw_path.write_text('toy draw marker, not an array file')
            fresh = root / 'fresh'
            def receipt(path): return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            (source / 'manifest.json').write_text(json.dumps(
                {'status': 'COMPUTED', 'prediction_root': str(fresh),
                 'inputs': [receipt(path) for path in [*gold_paths, draw_path]],
                 'outputs': [receipt(path) for path in source.iterdir()]}))
            cl_specs = all_specs['CL-RuTerm3']
            gold = [{'dataset': 'CL-RuTerm3', 'unit_type': 'span_mention', 'doc_id': 'd',
                     'sent_id': None, 'start': a, 'end': b} for a, b in [(0, 8), (1, 4), (10, 14)]]
            cli = script('reproduce_sensitivity')
            args = ['reproduce_sensitivity.py', '--gold-output', str(source), '--data-dir', str(data),
                    '--output', str(output), '--prediction-root', str(fresh)]
            with patch.object(cli, 'ROOT', root), patch.object(sys, 'argv', args), \
                    patch.object(analysis_io, 'load_analysis_rows', return_value=(cl_specs, gold, {r['run']: [] for r in cl_specs}, [])), \
                    patch.object(analysis_io, 'load_document_draws', return_value=(['d'], np.array([[1], [2]], dtype=float), draw_path)), \
                    redirect_stdout(io.StringIO()):
                cli.main()
            summary = json.loads((output / 'summary.json').read_text(),
                                 parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))['results']
            self.assertIsNone(summary['genia_without_exact_train_test_text_duplicate']['median_excess'])
            self.assertEqual(summary['genia_without_exact_train_test_text_duplicate']['defined_pairs'], 0)
            self.assertEqual(summary['cl_equal_optimum_assignments']['global_equal_optimum_combinations'], 1)
            with (output / 'ambiguity_sensitivity.csv').open() as stream:
                tie_rows = list(csv.DictReader(stream))
            self.assertTrue(all(r['valid_draws'] == '0' and r['point'] == '' and r['ci_low'] == '' for r in tie_rows))


if __name__ == '__main__':
    unittest.main()
