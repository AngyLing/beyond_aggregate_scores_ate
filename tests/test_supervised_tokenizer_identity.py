"""Tokenizer-identity contracts; no transformers, tokenizer or model is loaded."""
import json
import tempfile
import unittest
from pathlib import Path

from ate_paper.experiments.supervised_tokenizers import tokenizer_kwargs


class SupervisedTokenizerIdentityTest(unittest.TestCase):
    def test_run_tag_does_not_change_non_scibert_bundle(self):
        path = '/tmp/scibert-false/genia/supervised/xlmr_span_classifier_seed42/model'
        self.assertEqual(tokenizer_kwargs(path, model_name='xlm-roberta-base'), {'use_fast': True})
        self.assertEqual(tokenizer_kwargs(path, model_name='bert-base-multilingual-cased'), {'use_fast': True})
        self.assertEqual(tokenizer_kwargs(path, model_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'), {'use_fast': True})

    def test_saved_encoder_name_has_priority(self):
        self.assertEqual(tokenizer_kwargs('/tmp/scibert/model', model_name='xlm-roberta-base'), {'use_fast': True})
        self.assertEqual(tokenizer_kwargs('/tmp/control/model', 'false', model_name='allenai/scibert_scivocab_uncased'), {'use_fast': True, 'do_lower_case': False})

    def test_local_path_parent_does_not_imply_scibert(self):
        self.assertEqual(tokenizer_kwargs('/tmp/scibert-control/bert-base-multilingual-cased'), {'use_fast': True})

    def test_saved_local_tokenizer_metadata_identifies_scibert(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'tokenizer_config.json').write_text(json.dumps({'name_or_path':'allenai/scibert_scivocab_uncased','do_lower_case':False}))
            with self.assertRaisesRegex(ValueError,'requires --scibert-do-lower-case'):
                tokenizer_kwargs(root)
            self.assertEqual(tokenizer_kwargs(root,'false'), {'use_fast': True, 'do_lower_case': False})

    def test_real_scibert_remains_explicit(self):
        with self.assertRaisesRegex(ValueError,'requires --scibert-do-lower-case'):
            tokenizer_kwargs('allenai/scibert_scivocab_uncased')
        self.assertEqual(tokenizer_kwargs('allenai/scibert_scivocab_uncased','true',revision='revision123'),
                         {'use_fast':True,'revision':'revision123','do_lower_case':True})


if __name__=='__main__':
    unittest.main()
