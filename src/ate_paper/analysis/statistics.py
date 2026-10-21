"""Original fixed-population ratio, interval, and within-reference rules."""

import numpy as np

def ratio(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(numerator, denominator,
                     out=np.full(numerator.shape, np.nan, dtype=float), where=denominator > 0)


def complete_mean(values, indices):
    """Keep the fixed pair population: an undefined constituent invalidates a mean."""
    if not indices:
        return np.full(values.shape[0], np.nan)
    return np.mean(values[:, indices], axis=1)


def interval(values):
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"valid": 0, "ci_low": None, "ci_high": None}
    lo, hi = np.quantile(finite, [0.025, 0.975])
    return {"valid": len(finite), "ci_low": float(lo), "ci_high": float(hi)}


def summarize_cloud(values, between_indices, within_a, within_b):
    between = complete_mean(values, between_indices)
    references = [complete_mean(values, indices) for indices in (within_a, within_b) if indices]
    reference = np.maximum.reduce(references) if references else np.full(len(between), np.nan)
    excess = between - reference
    return between, reference, excess


