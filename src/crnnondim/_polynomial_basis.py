"""Local minimisation of polynomial coefficient, generator and time-scale costs.

Each candidate is scored after the common shift to nonnegative exponents.
The best of three starting bases is refined by deterministic unimodular descent.
The returned score is no worse than any starting score; global optimality over
all unimodular bases is not guaranteed.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
import sympy as sp
from ._integer_search import fast_transform, _lex_less

_LIMIT = np.iinfo(np.int64).max


def _valuation(expr, parameters):
    expression = sp.cancel(expr)
    powers = expression.as_powers_dict()
    result = [powers.get(s, sp.Integer(0)) for s in parameters]
    if not all(v.is_Integer for v in result):
        raise ValueError('Polynomial basis scores require integer Laurent valuations.')
    if sp.cancel(expression/sp.prod(s**v for s,v in zip(parameters,result))) != 1:
        raise ValueError('Polynomial basis scores require unit Laurent monomials.')
    return [int(v) for v in result]


@lru_cache(maxsize=512)
def _seed(vecs, legacy):
    T, A = fast_transform(vecs, return_inverse=True, legacy=legacy)
    return tuple(map(tuple,T)), tuple(tuple(int(v) for v in A.row(i)) for i in range(A.rows))


def _object_matrix(rows):
    return np.asarray(rows, dtype=object)


def _candidate_state(T, A, vectors, B0, time0, weights):
    # Python-integer arrays at seed construction avoid overflow from even a
    # very large unimodular inverse. Narrowing is allowed only after a bound.
    rows = _object_matrix(T) @ vectors.T
    generators = B0 @ _object_matrix(A)
    minimum = rows.min(axis=1)
    time = time0-generators@minimum
    row_cost = rows@weights-int(sum(weights))*minimum
    score = (int(sum(row_cost)), int(np.abs(generators).sum()), int(np.abs(time).sum()))
    return score, rows, generators, time


def _safe_dtype(rows, generators, time, weights):
    bound = max((abs(int(v)) for a in (rows,generators,time) for v in a.flat),default=0)
    # Covers candidate row sums, G_i*(m'_i-m_i-sign*m_j), weighted
    # reductions and global scores. If this sufficient bound fails, object
    # arrays continue with Python arbitrary-precision integers.
    n = rows.size+generators.size+time.size+1
    return np.int64 if 16*n*(sum(map(int,weights))+1)*(bound+1)**2 <= _LIMIT else object


def _polynomial_descent(T, A, rows, generators, time0, weights):
    T, A = [list(row) for row in T], [list(row) for row in A]
    rows, generators, time0 = (a.astype(object) for a in (rows,generators,time0))
    weights = np.asarray(weights,dtype=object)
    d = rows.shape[0]
    W = int(sum(weights))
    while True:
        minimum = rows.min(axis=1)
        time = time0-generators@minimum
        dtype = _safe_dtype(rows,generators,time,weights)
        rows, generators, time0, weights, minimum, time = (
            a.astype(dtype,copy=False) for a in (rows,generators,time0,weights,minimum,time))
        row_cost = rows@weights-W*minimum
        gen_cost = np.abs(generators).sum(axis=0)
        best = (int(sum(map(int,row_cost))),int(sum(map(int,gen_cost))),int(np.abs(time).sum()))
        accepted = False
        # Final polynomial costs can change under sign flips, unlike raw L1.
        for i in range(d):
            maximum = rows[i].max()
            candidate_F = best[0]-int(row_cost[i])+int(-rows[i]@weights+W*maximum)
            if candidate_F > best[0]:
                continue
            candidate_time = time+generators[:,i]*(minimum[i]-maximum)
            score = (candidate_F,best[1],int(np.abs(candidate_time).sum()))
            if score < best:
                rows[i] *= -1
                generators[:,i] *= -1
                T[i] = [-v for v in T[i]]
                for row in A:
                    row[i] = -row[i]
                accepted = True
                break
        if accepted:
            continue
        signs = np.tile(np.asarray([1,-1],dtype=dtype),d)
        source_minima = np.repeat(minimum,2)
        source_gen_cost = np.repeat(gen_cost,2)
        for i in range(d):
            candidate_rows = np.stack((rows+rows[i],rows[i]-rows),axis=1).reshape(2*d,-1)
            candidate_minima = candidate_rows.min(axis=1)
            F_mono = best[0]-row_cost[i]+candidate_rows@weights-W*candidate_minima
            potentially_better = F_mono <= best[0]
            potentially_better[2*i:2*i+2] = False
            if not potentially_better.any():
                continue
            # Row i += sign*row j induces generator column j -= sign*column i.
            candidate_generators = np.repeat(generators.T,2,axis=0)-signs[:,None]*generators[:,i]
            F_gen = best[1]-source_gen_cost+np.abs(candidate_generators).sum(axis=1)
            offsets = candidate_minima-minimum[i]-signs*source_minima
            candidate_times = time[None,:]-offsets[:,None]*generators[:,i]
            F_time = np.abs(candidate_times).sum(axis=1)
            scores = np.column_stack((F_mono,F_gen,F_time))
            better = _lex_less(scores,best) & potentially_better
            hits = np.flatnonzero(better)
            if not hits.size:
                continue
            position = int(hits[0]);j=position//2;sign=1 if position%2==0 else -1
            rows[i] += sign*rows[j]
            generators[:,j] -= sign*generators[:,i]
            T[i] = [a+sign*b for a,b in zip(T[i],T[j])]
            for row in A:
                row[j] -= sign*row[i]
            accepted = True
            break
        if not accepted:
            return T, [tuple(map(int,v)) for v in rows.T], sp.Matrix(A)


def polynomial_basis(vecs, weights, generator_expressions, K_T_expr):
    """Select a starting basis and descend on the final polynomial cost tuple.

    Args:
        vecs: Distinct integer exponent vectors, one per coefficient class.
        weights: Positive integer multiplicities of the coefficient classes.
        generator_expressions: Unit Laurent monomials defining the input basis.
        K_T_expr: Unit Laurent monomial defining the initial time scale.

    Returns:
        A tuple (T, transformed_vecs, T_inverse). The transformed vectors are
        unshifted; the caller applies their componentwise minimum shift and
        updates the time scale. Empty or zero-dimensional input returns
        (None, vecs, None).

    The starting bases are sparsity-first, coefficient-weighted L1 and identity,
    in that tie-breaking order. One descent follows the best starting basis.
    """
    if not vecs or not vecs[0]:
        return None, list(vecs), None
    if len(weights)!=len(vecs) or any(int(w)!=w or w<=0 for w in weights):
        raise ValueError('One positive integer multiplicity is required per class.')
    d=len(vecs[0])
    if any(len(v)!=d for v in vecs) or len(generator_expressions)!=d:
        raise ValueError('Generator and exponent-vector dimensions do not match.')
    params=sorted(set(K_T_expr.free_symbols).union(*(g.free_symbols for g in generator_expressions)),key=str)
    B0=_object_matrix([_valuation(g,params) for g in generator_expressions]).T
    time0=np.asarray(_valuation(K_T_expr,params),dtype=object)
    vectors=_object_matrix(vecs)
    weights=np.asarray(list(map(int,weights)),dtype=object)
    unique=tuple(tuple(map(int,v)) for v in vecs)
    expanded=tuple(v for v,w in zip(unique,weights) for _ in range(int(w)))
    identity=tuple(tuple(int(i==j) for j in range(d)) for i in range(d))
    seeds=[_seed(unique,True),_seed(expanded,False),(identity,identity)]
    best_state=None
    for T,A in seeds:
        score,rows,generators,time=_candidate_state(T,A,vectors,B0,time0,weights)
        if best_state is None or score<best_state[0]:
            best_state=(score,T,A,rows,generators)
    _,T,A,rows,generators=best_state
    return _polynomial_descent(T,A,rows,generators,time0,weights)
