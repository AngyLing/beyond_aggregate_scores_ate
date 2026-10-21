"""Parse all analysis CLIs from another cwd, stopping before any producer runs."""
import argparse
import contextlib
import tempfile
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import ate_paper
ROOT = Path(ate_paper.__file__).resolve().parents[2]
NAMES = ('panels', 'profiles', 'gold_agreement', 'no_witness', 'nesting_contrasts', 'sensitivity', 'llm_repeats')
ALT = Path(__file__).parent


class Parsed(Exception):
    pass


def parse_only(name, arguments):
    spec = importlib.util.spec_from_file_location('review_cli_' + name, ROOT / 'scripts' / f'reproduce_{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = argparse.ArgumentParser.parse_args
    captured = []
    def stop(parser, *args, **kwargs):
        captured.append(actual(parser, *args, **kwargs))
        raise Parsed()
    with patch.object(argparse.ArgumentParser, 'parse_args', stop), \
            patch.object(sys, 'argv', [name, *map(str, arguments)]):
        try:
            module.main()
        except Parsed:
            pass
        else:
            raise AssertionError('CLI did not stop after parsing: ' + name)
    return module, captured[0]


class CLIParsing(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.enterContext(contextlib.chdir(folder.name))

    def test_all_seven_defaults_are_bound_to_repo(self):
        self.assertNotEqual(Path.cwd().resolve(), ROOT.resolve())
        for name in NAMES:
            with self.subTest(cli=name):
                _, args = parse_only(name, [])
                gold = args.gold_dir if name == 'profiles' else args.data_dir
                self.assertEqual(gold, ROOT / 'data/local/processed')
                self.assertEqual(args.prediction_root if hasattr(args, 'prediction_root') else None, None)
                output = args.out if hasattr(args, 'out') else args.output
                self.assertTrue(output.is_relative_to(ROOT / 'build'))

    def test_alternate_input_output_and_fresh_modes_are_parsed(self):
        for name in NAMES:
            with self.subTest(cli=name):
                data_flag = '--data-dir'
                output_flag = '--out' if name in ('panels', 'profiles') else '--output'
                arguments = [data_flag, ALT / 'fake_data', output_flag, ALT / 'build/fake_output']
                if name != 'nesting_contrasts':
                    arguments += ['--prediction-root', ALT / 'fake_predictions']
                if name in ('nesting_contrasts', 'sensitivity'):
                    arguments += ['--gold-output', ALT / 'fake_gold_producer']
                if name in ('panels', 'gold_agreement', 'no_witness', 'sensitivity'):
                    arguments += ['--skip-bootstrap']
                if name in ('gold_agreement', 'no_witness', 'sensitivity', 'nesting_contrasts'):
                    arguments += ['--dataset', 'cl']
                _, args = parse_only(name, arguments)
                self.assertEqual(args.gold_dir if name == 'profiles' else args.data_dir, ALT / 'fake_data')
                self.assertEqual(args.out if name in ('panels', 'profiles') else args.output, ALT / 'build/fake_output')
                if hasattr(args, 'prediction_root'):
                    self.assertEqual(args.prediction_root, ALT / 'fake_predictions')
                if hasattr(args, 'skip_bootstrap'):
                    self.assertTrue(args.skip_bootstrap)
                if hasattr(args, 'dataset'):
                    self.assertEqual(args.dataset, ['cl'])

    def test_legacy_profile_gold_dir_flag_is_retained(self):
        _, args = parse_only('profiles', ['--gold-dir', ALT / 'legacy_data'])
        self.assertEqual(args.gold_dir, ALT / 'legacy_data')

    def test_sensitivity_reference_is_explicit_and_rejects_fresh(self):
        _, args = parse_only('sensitivity', ['--check-reference', '--skip-bootstrap'])
        self.assertTrue(args.check_reference)
        self.assertTrue(args.skip_bootstrap)
        self.assertIsNone(args.prediction_root)
        try:
            from ate_paper.analysis.sensitivity import run
        except ImportError:
            self.skipTest('Install the analysis profile from requirements.lock for numerical runtime import')
        with self.assertRaisesRegex(ValueError, 'only available for the saved prediction release'):
            run(ALT, ALT / 'no_producer', ALT / 'no_data', ALT / 'build/no_writes',
                ['ACL RD-TEC 2.0'], ALT / 'fresh', skip_bootstrap=True, check_saved=True)
        self.assertFalse((ALT / 'build/no_writes').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
