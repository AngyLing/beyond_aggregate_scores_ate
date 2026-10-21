"""Pure BIO loader-argument contract: only its pre-import AST prefix is executed."""
import ast
import json
import tempfile
import types
import unittest
from pathlib import Path

from ate_paper.experiments import train_mbert_sequence_labeler as bio
from ate_paper.experiments.supervised_tokenizers import tokenizer_kwargs


def loader_kwargs_only(args):
    # Stop before the try/import transformers block. No train/predict function,
    # tokenizer or model code is called by this contract.
    source=Path(bio.__file__).read_text(encoding='utf-8')
    function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='train_and_predict')
    prefix=[]
    for node in function.body:
        if isinstance(node,ast.Try):break
        prefix.append(node)
    block=ast.Module(body=prefix,type_ignores=[])
    namespace={'args':args,'tokenizer_kwargs':tokenizer_kwargs}
    exec(compile(ast.fix_missing_locations(block),bio.__file__,'exec'),namespace)
    return namespace['tokenizer_load_kwargs']


class BioPredictOnlyTokenizerIdentityTest(unittest.TestCase):
    def test_saved_scibert_ignores_parser_mbert_default(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle=Path(directory)/'model';bundle.mkdir()
            (bundle/'tokenizer_config.json').write_text(json.dumps({'name_or_path':'allenai/scibert_scivocab_uncased','do_lower_case':False}))
            args=types.SimpleNamespace(predict_only=True,model_dir=str(bundle),model_name='bert-base-multilingual-cased',model_revision='unused-during-local-inference',scibert_do_lower_case='false')
            self.assertEqual(loader_kwargs_only(args),{'use_fast':True,'do_lower_case':False})
            args.scibert_do_lower_case=None
            with self.assertRaisesRegex(ValueError,'requires --scibert-do-lower-case'):
                loader_kwargs_only(args)

    def test_non_scibert_checkpoint_under_scibert_run_tag(self):
        with tempfile.TemporaryDirectory(prefix='scibert-control-') as directory:
            bundle=Path(directory)/'model';bundle.mkdir()
            (bundle/'tokenizer_config.json').write_text(json.dumps({'name_or_path':'bert-base-multilingual-cased','do_lower_case':False}))
            args=types.SimpleNamespace(predict_only=True,model_dir=str(bundle),model_name='bert-base-multilingual-cased',model_revision=None,scibert_do_lower_case=None)
            self.assertEqual(loader_kwargs_only(args),{'use_fast':True})

    def test_training_uses_explicit_encoder_and_revision(self):
        args=types.SimpleNamespace(predict_only=False,model_dir='/tmp/model',model_name='allenai/scibert_scivocab_uncased',model_revision='revision123',scibert_do_lower_case='true')
        self.assertEqual(loader_kwargs_only(args),{'use_fast':True,'do_lower_case':True,'revision':'revision123'})


if __name__=='__main__':
    unittest.main()
