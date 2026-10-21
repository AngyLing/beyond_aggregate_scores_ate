"""Exact-preserving, one-to-one character-IoU matching to gold mentions.

The output is a set of (gold offset, prediction offset) assignments in one
original text unit.  Gold and prediction offsets are half-open character spans.
Exact edges are fixed before optimizing residual cardinality and total IoU.
"""

from __future__ import annotations

from collections import defaultdict, deque
from fractions import Fraction

import numpy as np
from scipy.optimize import linear_sum_assignment

Span = tuple[int, int]
Edge = tuple[int, int]


def overlap_fraction(gold: Span, prediction: Span) -> Fraction:
    overlap = max(0, min(gold[1], prediction[1]) - max(gold[0], prediction[0]))
    if not overlap:
        return Fraction(0)
    union = max(gold[1], prediction[1]) - min(gold[0], prediction[0])
    return Fraction(overlap, union)


def _components(edges: dict[Edge, Fraction]) -> list[tuple[list[int], list[int]]]:
    by_gold: dict[int, set[int]] = defaultdict(set)
    by_prediction: dict[int, set[int]] = defaultdict(set)
    for gi, pi in edges:
        by_gold[gi].add(pi)
        by_prediction[pi].add(gi)
    components = []
    unseen = set(by_gold)
    while unseen:
        start = min(unseen)
        gs, ps = {start}, set()
        queue = deque([("g", start)])
        unseen.remove(start)
        while queue:
            side, vertex = queue.popleft()
            if side == "g":
                for pi in by_gold[vertex] - ps:
                    ps.add(pi)
                    queue.append(("p", pi))
            else:
                for gi in by_prediction[vertex] - gs:
                    gs.add(gi)
                    unseen.discard(gi)
                    queue.append(("g", gi))
        components.append((sorted(gs), sorted(ps)))
    return components


def _solve(
    gold_indices: list[int],
    prediction_indices: list[int],
    edges: dict[Edge, Fraction],
    forced: tuple[Edge, ...] = (),
    forbidden: frozenset[Edge] = frozenset(),
) -> tuple[tuple[Edge, ...], tuple[int, Fraction]] | None:
    used_g = {g for g, _ in forced}
    used_p = {p for _, p in forced}
    if len(used_g) != len(forced) or len(used_p) != len(forced):
        return None
    if any(edge not in edges or edge in forbidden for edge in forced):
        return None
    remaining_g = [g for g in gold_indices if g not in used_g]
    remaining_p = [p for p in prediction_indices if p not in used_p]
    assignments = list(forced)
    if remaining_g:
        n, m = len(remaining_g), len(remaining_p)
        # A single extra match is worth more than all possible IoU gains.
        bonus = len(gold_indices) + 1
        cost = np.full((n, m + n), np.inf, dtype=float)
        cost[:, m:] = 0.0  # Dedicated dummy columns allow unmatched gold.
        for ri, gi in enumerate(remaining_g):
            for ci, pi in enumerate(remaining_p):
                edge = gi, pi
                if edge in edges and edge not in forbidden:
                    cost[ri, ci] = -bonus - float(edges[edge])
        rows, cols = linear_sum_assignment(cost)
        assignments.extend((remaining_g[ri], remaining_p[ci])
                           for ri, ci in zip(rows, cols) if ci < m)
    result = tuple(sorted(assignments))
    return result, (len(result), sum((edges[edge] for edge in result), Fraction(0)))


def _optimal_component(
    gs: list[int], ps: list[int], edges: dict[Edge, Fraction]
) -> tuple[tuple[Edge, ...], int]:
    best = _solve(gs, ps, edges)
    assert best is not None
    assignment, optimum = best
    # Most components have only their unique matching's edges.  Otherwise,
    # check whether any chosen edge can be replaced without changing the exact
    # cardinality/IoU objective before doing the lexicographic refinement.
    has_alternative = len(edges) > len(assignment) and any(
        (candidate := _solve(gs, ps, edges, forbidden=frozenset({edge}))) is not None
        and candidate[1] == optimum for edge in assignment
    )
    if has_alternative:
        chosen: tuple[Edge, ...] = ()
        for edge in sorted(edges):
            if len(chosen) == optimum[0]:
                break
            if any(edge[0] == g or edge[1] == p for g, p in chosen):
                continue
            candidate = _solve(gs, ps, edges, chosen + (edge,))
            if candidate is not None and candidate[1] == optimum:
                chosen += (edge,)
        if len(chosen) != optimum[0]:
            raise AssertionError("Lexicographic matching did not reach optimum")
        assignment = tuple(sorted(chosen))

    ambiguous_gold = 0
    if has_alternative and optimum[0] < len(gs):
        covered = {g for g, _ in assignment}
        for gi in gs:
            incident = tuple(edge for edge in edges if edge[0] == gi)
            if gi in covered:
                other = _solve(gs, ps, edges, forbidden=frozenset(incident))
                ambiguous_gold += other is not None and other[1] == optimum
            else:
                ambiguous_gold += any(
                    (other := _solve(gs, ps, edges, (edge,))) is not None
                    and other[1] == optimum for edge in incident
                )
    return assignment, ambiguous_gold


def match_unit(
    gold_offsets: set[Span], prediction_offsets: set[Span], threshold: Fraction = Fraction(1, 2)
) -> tuple[list[tuple[Span, Span, Fraction, bool]], int]:
    """Return assignments and count of gold vertices with unstable coverage."""
    if not Fraction(0) < threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    for start, end in gold_offsets | prediction_offsets:
        if not isinstance(start, int) or not isinstance(end, int) or start >= end:
            raise ValueError("Invalid half-open character span")
    exact = sorted(gold_offsets & prediction_offsets)
    out = [(span, span, Fraction(1), True) for span in exact]
    remaining_g = sorted(gold_offsets - set(exact))
    remaining_p = sorted(prediction_offsets - set(exact))
    edges: dict[Edge, Fraction] = {}
    for gi, gold in enumerate(remaining_g):
        for pi, prediction in enumerate(remaining_p):
            value = overlap_fraction(gold, prediction)
            if value >= threshold:
                edges[gi, pi] = value
    ambiguous = 0
    for gs, ps in _components(edges):
        local = {edge: value for edge, value in edges.items()
                 if edge[0] in gs and edge[1] in ps}
        assignment, count = _optimal_component(gs, ps, local)
        ambiguous += count
        out.extend((remaining_g[gi], remaining_p[pi], local[gi, pi], False)
                   for gi, pi in assignment)
    assert len({g for g, *_ in out}) == len(out)
    assert len({p for _, p, *_ in out}) == len(out)
    return sorted(out), ambiguous


def near_boundary_mask(keys, prediction_keys, threshold=0.5):
    """Original many-to-one presence screen; exact membership is a fast path."""
    by_sentence = defaultdict(list)
    for key in prediction_keys:
        by_sentence[key[:2]].append(key[2:])
    result = np.zeros(len(keys), dtype=bool)
    for i, key in enumerate(keys):
        if key in prediction_keys:
            result[i] = True
            continue
        start, end = key[2:]
        for other_start, other_end in by_sentence[key[:2]]:
            if other_end <= start or end <= other_start:
                continue
            overlap = max(0, min(end, other_end) - max(start, other_start))
            union = max(end, other_end) - min(start, other_start)
            if union and overlap / union >= threshold:
                result[i] = True
                break
    return result
