"""Explicit tokenizer casing and optional encoder revision for supervised runs.

Only standard-library imports occur here. Importing or requesting --help never
loads torch, transformers, model weights or data.
"""
import json
from pathlib import Path


def pretrained_revision_kwargs(revision):
    return {"revision": revision} if revision else {}


def _scibert_model_name(value):
    # Only the model name itself identifies the encoder. A run directory such
    # as build/experiments/scibert-false must not turn another encoder into SciBERT.
    return bool(value) and "scibert" in Path(str(value)).name.lower()


def _is_scibert(source, model_name=None):
    if model_name:
        # Span bundles retain the actual training encoder in their config.
        return _scibert_model_name(model_name)
    config_path = Path(str(source)) / "tokenizer_config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        names = [config.get(key) for key in ("name_or_path", "_name_or_path") if config.get(key)]
        if names:
            return any(_scibert_model_name(name) for name in names)
    return _scibert_model_name(source)


def tokenizer_kwargs(source, scibert_do_lower_case=None, revision=None, model_name=None):
    """Preserve the old fast loader, requiring an explicit SciBERT case choice.

    For SciBERT the selected historical outputs and later prepared recipes
    disagree about do_lower_case. Callers must choose false or true instead of
    inheriting the installed transformers version's implicit casing behavior.
    """
    kwargs = {"use_fast": True}
    kwargs.update(pretrained_revision_kwargs(revision))
    is_scibert = _is_scibert(source, model_name=model_name)
    if is_scibert and scibert_do_lower_case is None:
        raise ValueError("SciBERT requires --scibert-do-lower-case false|true; "
                         "choose explicitly; implicit defaults do not identify the saved-run settings.")
    if scibert_do_lower_case is not None:
        if not is_scibert:
            raise ValueError("--scibert-do-lower-case is only valid for a SciBERT encoder.")
        if scibert_do_lower_case not in ("false", "true"):
            raise ValueError("--scibert-do-lower-case must be false or true.")
        kwargs["do_lower_case"] = scibert_do_lower_case == "true"
    return kwargs


def add_pretrained_arguments(parser):
    parser.add_argument("--model-revision", default=None,
                        help="Optional immutable Hugging Face revision for the base model and tokenizer.")
    parser.add_argument("--scibert-do-lower-case", choices=["false", "true"], default=None,
                        help="Required for SciBERT training: choose casing explicitly; "
                             "the historical outputs and later recipe disagree. "
                             "Other encoders keep their own original tokenizer settings.")
