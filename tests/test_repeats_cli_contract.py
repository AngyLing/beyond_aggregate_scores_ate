"""Independent artificial A17 CLI integration; no corpus/prediction artifacts read."""
import contextlib
import csv
import gzip
import importlib.util
import io as textio
import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ate_paper
NEW = Path(ate_paper.__file__).resolve().parents[2]
from ate_paper.analysis import io, repeats
from ate_paper.predictions import release_record, sha256


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_script():
    spec = importlib.util.spec_from_file_location('review_repeats_cli', NEW / 'scripts/reproduce_llm_repeats.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RepeatsCLIContract(unittest.TestCase):
    def test_custom_data_dir_fresh_empty_and_changed_outputs(self):
        pairs = io.read_csv(NEW / 'predictions/llm_repeat_pairs.csv')
        originals = {r['release_path']: r for r in io.read_csv(NEW / 'predictions/manifest.csv')}
        with tempfile.TemporaryDirectory(prefix='toy_a17_') as temporary:
            root = Path(temporary)
            data = root / 'unusual_location'
            fresh = root / 'launcher_outputs'
            csv_write(root / 'predictions/llm_repeat_pairs.csv', pairs)
            gold, entries, wanted = {}, {}, []
            for dataset, slug in io.SLUGS.items():
                first = {'dataset': dataset, 'version': 'toy', 'split': 'test', 'doc_id': 'one',
                         'sent_id': 's', 'text': 'Alpha beta.', 'unit_type': 'span_mention',
                         'start': 0, 'end': 5, 'surface': 'Alpha', 'norm': 'alpha', 'class_label': 'toy'}
                other = {**first, 'doc_id': 'two'}
                empty = {**first, 'doc_id': 'negative', 'start': None, 'end': None,
                         'surface': None, 'norm': None, 'class_label': None}
                gold[dataset] = [first, other, empty]
                p = data / slug / 'test.jsonl'
                p.parent.mkdir(parents=True)
                p.write_text(''.join(json.dumps(r) + '\n' for r in gold[dataset]))
            for pair in pairs:
                first, other, empty = gold[pair['dataset']]
                # Reference scores are deliberately different from the new outputs.
                wanted.append({'pair_id': pair['pair_id'], 'dataset': io.SLUGS[pair['dataset']],
                               'model_key': pair['model_key'], 'decision': pair['decision'],
                               'base_f1': 0.25, 'repeat_f1': 0.25, 'absolute_delta_unlabeled_f1': 0.0})
                for side in ('base', 'repeat'):
                    relative = pair[side + '_path']
                    entry = {k: originals[relative][k] for k in ('dataset', 'group', 'run', 'release_path')}
                    entries[relative] = entry
                    # All bases recover one of two gold spans; P19 repeat is empty.
                    rows = ([first] if side == 'base' else
                            ([] if pair['pair_id'] == 'P19' else [first, other]))
                    path = fresh / io.SLUGS[entry['dataset']] / entry['group'] / entry['run'] / 'predictions.jsonl'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                    # No real release is needed in fresh mode; a deliberately bad old hash proves this.
                    entry['release_sha256'] = '0' * 64
            csv_write(root / 'predictions/manifest.csv', list(entries.values()))
            reference = root / repeats.REFERENCE_DIR
            csv_write(reference / 'limitations_full_repeat_f1.csv', wanted)
            (reference / 'limitations_full_repeat_f1_summary.json').write_text('{}')
            module = load_script()
            output = root / 'build/selected'
            args = ['reproduce_llm_repeats', '--data-dir', str(data), '--prediction-root',
                    str(fresh), '--output', str(output)]
            toy_counts = {d: {'documents': 3, 'units': 3, 'spans': 2} for d in io.SLUGS}
            with patch.object(module, 'ROOT', root), patch.dict(io.TEST_COUNTS, toy_counts, clear=True), \
                    patch.object(sys, 'argv', args), contextlib.redirect_stdout(textio.StringIO()):
                module.main()
            table = io.read_csv(output / 'limitations_full_repeat_f1.csv')
            summary = json.loads((output / 'limitations_full_repeat_f1_summary.json').read_text())
            self.assertEqual(summary['status'], 'COMPUTED')
            self.assertIsNone(summary['checks'])
            self.assertFalse(summary['comparison_to_reference']['all_saved_values_match'])
            self.assertEqual(summary['data_dir'], str(data.resolve()))
            self.assertEqual(summary['prediction_root'], str(fresh.resolve()))
            self.assertEqual(len(table), 25)
            self.assertTrue(all(int(r['gold']) == 2 and int(r['full_test_units']) == 3 for r in table))
            by_id = {r['pair_id']: r for r in table}
            self.assertAlmostEqual(float(by_id['P19']['base_f1']), 2 / 3)
            self.assertEqual(float(by_id['P19']['repeat_f1']), 0)
            self.assertEqual(int(by_id['P19']['repeat_predicted']), 0)
            self.assertEqual(summary['limitations']['P19']['historical_base_failed_batches'], 27)
            self.assertEqual(summary['limitations']['P19']['historical_repeat_failed_batches'], 64)
            self.assertEqual({r['pair_id'] for r in summary['transport_contrasts']}, {'P03', 'P17', 'P21'})
            for slug, count in {'acl': 7, 'genia': 9, 'cl': 6}.items():
                subset = [float(r['absolute_delta_unlabeled_f1']) for r in table
                          if r['dataset'] == slug and r['decision'] != 'transport_contrast']
                self.assertEqual(len(subset), count)
                self.assertEqual(summary['summary'][slug]['n_pairs'], count)
                self.assertEqual(summary['summary'][slug]['median_abs_delta_unlabeled_f1'], statistics.median(subset))
                self.assertEqual(summary['summary'][slug]['max_abs_delta_unlabeled_f1'], max(subset))
            self.assertEqual(summary['sensitivity']['genia_without_documented_raw_missing_pair']['n_pairs'], 8)
            self.assertEqual(summary['sensitivity']['cl_without_llama_format_failure_pair']['n_pairs'], 5)
            self.assertEqual(summary['summary']['cl']['max_abs_delta_unlabeled_f1'], 2 / 3)
            self.assertAlmostEqual(summary['sensitivity']['cl_without_llama_format_failure_pair']['max_abs_delta_unlabeled_f1'], 1 / 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
