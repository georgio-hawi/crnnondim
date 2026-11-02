"""Batched integer-coordinate searches for unimodular basis transformations.

Candidates follow target row, source row, then sign (+1, -1) order. Strict L1
search is the default; a sparsity-first score supplies an alternative starting
basis for polynomial search. Integer arithmetic falls back to Python integers
when candidate calculations cannot be bounded safely within int64.
"""
from __future__ import annotations
import numpy as np
import sympy as sp

_LIMIT = np.iinfo(np.int64).max


def _scores(rows):
    negative = np.minimum(rows, 0)
    return np.column_stack((np.count_nonzero(rows, axis=1),
                            -negative.sum(axis=1),
                            np.count_nonzero(negative, axis=1),
                            np.abs(rows).sum(axis=1)))


def _lex_less(scores, reference):
    less = np.zeros(scores.shape[0], dtype=bool)
    equal = np.ones(scores.shape[0], dtype=bool)
    for k, value in enumerate(reference):
        less |= equal & (scores[:, k] < value)
        equal &= scores[:, k] == value
    return less


def fast_transform(vecs, max_passes=None, return_inverse=False, *, legacy=False):
    """Find a unimodular transform by first-improvement descent.

    Args:
        vecs: Equal-length integer exponent vectors, including multiplicities.
        max_passes: Maximum search passes, or None to stop at a local minimum.
        return_inverse: Whether to return the exact inverse with the transform.
        legacy: Use the lexicographic score (nonzeros, negative magnitude,
            negative count, L1) instead of strict L1 descent. Polynomial search
            uses this mode to construct one of its starting bases.

    Returns:
        The integer matrix T, or (T, T_inverse) when return_inverse is True.
        Returns None for empty or zero-dimensional input. An unimproved
        nonempty input retains an identity transform, up to sign choices.
    """
    if not vecs or not vecs[0]:
        return None
    d = len(vecs[0])
    if any(len(v) != d for v in vecs):
        raise ValueError("All exponent vectors must have the same length.")
    exact = [[int(v[i]) for v in vecs] for i in range(d)]
    total_abs = sum(abs(x) for row in exact for x in row)
    rows = np.asarray(exact, dtype=np.int64 if 2*total_abs <= _LIMIT else object)
    T = [[int(i == j) for j in range(d)] for i in range(d)]
    A = [[int(i == j) for j in range(d)] for i in range(d)] if return_inverse else None

    def accept(i, j=None, sign=-1):
        if j is None:
            rows[i] *= -1
            T[i] = [-v for v in T[i]]
            if A is not None:
                for row in A:
                    row[i] = -row[i]
        else:
            rows[i] += sign*rows[j]
            T[i] = [a+sign*b for a, b in zip(T[i], T[j])]
            if A is not None:
                for row in A:
                    row[j] -= sign*row[i]

    # Cache pairwise candidate row scores. After changing row k, only pairs
    # with target k or source k change. Updating O(d*n) entries per accepted
    # move replaces an O(d*d*n) rescan while preserving the exact move order.
    scores = _scores(rows)
    pair_scores = np.empty((d,d,2,4),dtype=rows.dtype)

    def refresh(k):
        plus = _scores(rows+rows[k])
        minus_rows = rows[k]-rows
        minus = _scores(minus_rows)
        opposite = _scores(-minus_rows)
        pair_scores[k,:,0,:] = plus
        pair_scores[:,k,0,:] = plus
        pair_scores[k,:,1,:] = minus
        pair_scores[:,k,1,:] = opposite

    for k in range(d):
        refresh(k)
    passes = 0
    while max_passes is None or passes < max_passes:
        passes += 1
        if legacy:
            flipped = _scores(-rows)
            indices = [i for i in range(d) if tuple(flipped[i]) < tuple(scores[i])]
            if indices:
                i = indices[0]
                accept(i)
                scores[i] = flipped[i]
                refresh(i)
                continue
        if legacy:
            better = np.zeros((d,d,2),dtype=bool)
            equal = np.ones((d,d,2),dtype=bool)
            for field in range(4):
                ref = scores[:,field,None,None]
                better |= equal & (pair_scores[:,:,:,field] < ref)
                equal &= pair_scores[:,:,:,field] == ref
        else:
            better = pair_scores[:,:,:,3] < scores[:,3,None,None]
        for i in range(d):
            better[i,i,:] = False
        hits = np.flatnonzero(better)
        if not hits.size:
            break
        position = int(hits[0])
        i, rem = divmod(position,2*d)
        j, sign_index = divmod(rem,2)
        sign = 1 if sign_index == 0 else -1
        next_score = pair_scores[i,j,sign_index].copy()
        total_abs += int(next_score[3])-int(scores[i,3])
        accept(i,j,sign)
        # Sparsity-first descent can increase L1, unlike strict L1 descent.
        # Promote before computing any new pair whose sum may overflow int64.
        if rows.dtype != object and 2*total_abs > _LIMIT:
            rows = rows.astype(object)
            scores = scores.astype(object)
            pair_scores = pair_scores.astype(object)
        scores[i] = next_score
        refresh(i)
    if not legacy:
        # Orient signs to reduce negative exponents without changing L1.
        scores, flipped = _scores(rows), _scores(-rows)
        for i in range(d):
            if tuple(flipped[i]) < tuple(scores[i]):
                accept(i)
    return (T, sp.Matrix(A)) if return_inverse else T
