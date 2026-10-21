"""Epoch budgets and five training seeds for the article's three corpora.

All runs train the full budget and choose a checkpoint on development data;
there is no early stopping. The values preserve the 2026-09-18 base-panel recipe.
"""
import os

EPOCHS_BY_DATASET = {"CL-RuTerm3": 15, "ACL RD-TEC 2.0": 15, "GENIA": 5}
SEEDS = ("42", "43", "44", "45", "46")


def epochs_for(dataset, env_var=None):
    """Return the budget as a CLI string; explicit nonempty env override wins."""
    override = os.environ.get(env_var) if env_var else None
    if override:
        return override
    try:
        return str(EPOCHS_BY_DATASET[dataset])
    except KeyError:
        raise KeyError(f"no epoch budget for dataset {dataset!r}; known: {sorted(EPOCHS_BY_DATASET)}") from None
