from typing import List, Tuple, Dict, Any, Callable, Optional, Union
from pathlib import Path
import pickle
import warnings
import re
import sympy as sp
import numpy as np
from functools import lru_cache
from collections.abc import Mapping
# Bound caches so repeated model analyses do not retain every past model.
_BASIS_CACHE_SIZE = 256
# Cached integer-basis transformations.
_basis_cache: Dict[Tuple[Tuple[int, ...], ...], Tuple[List[List[int]], List[Tuple[int, ...]], 'sp.Matrix']] = {}

def _get_basis(vecs: List[Tuple[int, ...]]):
    """
    Compute a simpler integer basis for a set of exponent vectors.

    Args:
        vecs (List[Tuple[int, ...]]): Integer exponent vectors, with repeated
            coefficients included according to their multiplicities.

    Returns:
        Tuple[Optional[List[List[int]]], List[Tuple[int, ...]], Optional[sp.Matrix]]:
            A triple (T, vecs_T, Am) where:
            - T: Unimodular transformation matrix (as list-of-lists), or None
              for empty or zero-dimensional input.
            - vecs_T: Transformed exponent vectors under T.
            - Am: Exact inverse transform as a SymPy matrix, or None when T
              is None. An identity transform has an identity inverse.
    """
    # Optionally find a unimodular transform that simplifies the exponent
    # vectors, then return both the transform and transformed vectors.
    if not vecs: 
        return None, [], None
    k = tuple(vecs)
    entry = _basis_cache.get(k)
    if entry is not None: 
        T, vecs_T, Am = entry
        return ([row[:] for row in T] if T is not None else None,
                list(vecs_T), Am.copy() if Am is not None else None)
    basis = choose_unimodular_transform_for_simplicity(vecs, return_inverse=True)
    if basis:
        T, Am = basis
        vecs_T = _transform_vectors(T, vecs)
        entry = (T, vecs_T, Am)
    else:
        entry = (None, list(vecs), None)
    if len(_basis_cache) >= _BASIS_CACHE_SIZE:
        _basis_cache.pop(next(iter(_basis_cache)))
    _basis_cache[k] = entry
    T, vecs_T, Am = entry
    return ([row[:] for row in T] if T is not None else None,
            list(vecs_T), Am.copy() if Am is not None else None)

def _unique_ordered(seq):
    """
    Remove duplicates from a sequence while preserving first occurrence order.

    Args:
        seq (Iterable[Any]): Input sequence of hashable items.

    Returns:
        List[Any]: List of unique elements in the order of first appearance.
    """
    return list(dict.fromkeys(seq))

def _normalise_symbol_mapping(values, symbols, name, *, require_all=False):
    """Resolve unambiguous symbol/string keys by name without changing inputs."""
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping keyed by symbols or their names.")
    symbols = _unique_ordered(symbols)
    by_name = {}
    for symbol in symbols:
        if symbol.name in by_name and by_name[symbol.name] != symbol:
            raise ValueError(f"Distinct model symbols share the name {symbol.name!r}.")
        by_name[symbol.name] = symbol
    result = {}
    for key, value in values.items():
        canonical = by_name.get(str(key), key)
        if canonical in result and result[canonical] != value:
            raise ValueError(f"Conflicting values for {canonical} in {name}.")
        result[canonical] = value
    if require_all:
        missing = [str(symbol) for symbol in symbols if symbol not in result]
        if missing:
            raise ValueError(f"{name} is missing values for: {', '.join(missing)}.")
    return result


def _validate_scale_symbols(scales, X, V):
    """Keep scale names independent of species and rate-parameter names."""
    scales = list(scales)
    if not all(isinstance(scale, sp.Symbol) for scale in scales):
        raise ValueError("All nondimensionalisation scales must be SymPy symbols.")
    if len(set(map(str, scales))) != len(set(scales)):
        raise ValueError("Distinct scale symbols must have distinct names.")
    model_names = {str(x) for x in X}
    model_names.update(str(symbol) for v in V for symbol in sp.sympify(v).free_symbols)
    collisions = sorted({str(scale) for scale in scales} & model_names)
    if collisions:
        raise ValueError(
            "Scale symbols must have names distinct from species and rate constants: "
            + ", ".join(collisions) + "."
        )


def _normalise_dimensionally_consistent_sets(
    X: List[sp.Symbol],
    dimensionally_consistent_sets: Optional[List[set[sp.Symbol]]],
):
    """Return a validated partition of the species into compatible-unit sets."""
    groups = (
        [set(X)]
        if dimensionally_consistent_sets is None
        else [set(group) for group in dimensionally_consistent_sets]
    )
    if len(X) != len(set(X)):
        raise ValueError("Species in X must be distinct.")
    if any(not group for group in groups):
        raise ValueError("dimensionally_consistent_sets cannot contain an empty set.")

    X_set = set(X)
    flattened = [species for group in groups for species in group]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Each species must occur in exactly one dimensionally consistent set.")
    if set(flattened) != X_set:
        raise ValueError("dimensionally_consistent_sets must contain every species in X exactly once.")

    normalised = tuple(frozenset(group) for group in groups)
    group_of = {
        species: group_index
        for group_index, group in enumerate(normalised)
        for species in group
    }
    return normalised, group_of

def _scale_dimension_groups(species_to_scale, species_group):
    """Infer each scale symbol's unit group and reject inconsistent reuse."""
    scale_group = {}
    scale_species = {}
    for species, scale in species_to_scale.items():
        group = species_group[species]
        if scale in scale_group and scale_group[scale] != group:
            raise ValueError(
                f"Scale symbol {scale} is assigned to species {scale_species[scale]} "
                f"and {species}, which are not in the same dimensionally consistent set."
            )
        scale_group[scale] = group
        scale_species[scale] = species
    return scale_group

def _apply_row_op(T, op):
    """
    Apply a unimodular row operation to an integer matrix in-place.

    Args:
        T (List[List[int]]): Integer matrix represented as a list of row lists.
        op (Tuple): Row operation of one of the forms:
            ("swap", i, j), ("neg", i), ("add", i, j, s) where s in {+1, -1}.

    Returns: None.
    """
    # Decode the operation and update the relevant row(s) accordingly.
    if op[0] == "swap":
        _, i, j = op
        T[i], T[j] = T[j], T[i]
    elif op[0] == "neg":
        _, i = op
        T[i] = [-t for t in T[i]]
    elif op[0] == "add":
        _, i, j, s = op
        Ti = T[i]
        T[i] = [Ti[k] + s*T[j][k] for k in range(len(Ti))]
    else:
        raise ValueError("unknown unimodular op")

def _transform_vectors(T, vecs):
    """
    Apply an integer row-transform matrix to a list of exponent vectors.

    Args:
        T (List[List[int]]): Unimodular integer matrix.
        vecs (List[Tuple[int, ...]]): Exponent vectors to transform.

    Returns:
        List[Tuple[int, ...]]: Transformed vectors T * v for each v in vecs.
    """
    # Multiply T by each vector v and return the transformed list.
    r = len(T)
    out = []
    for v in vecs:
        w = [0]*r
        for i in range(r):
            acc = 0
            Ti = T[i]
            for k in range(r):
                acc += Ti[k]*v[k]
            w[i] = acc
        out.append(tuple(int(x) for x in w))
    return out

def choose_unimodular_transform_for_simplicity(
    vecs: List[Tuple[int, ...]],
    max_passes: int | None = None,
    return_inverse: bool = False,
):
    """Find a unimodular transform by strict L1 descent on the input vectors.

    Args:
        vecs: Equal-length integer exponent vectors. Repeated vectors retain
            their multiplicities in the objective.
        max_passes: Maximum search passes, or None to stop at a local minimum.
        return_inverse: Whether to return the inverse as a SymPy Matrix.

    Returns:
        An integer matrix T, or (T, T_inverse). Empty or zero-dimensional input
        returns None. Candidate arithmetic uses int64 when safe and Python
        integers otherwise; both the transform and inverse remain exact.
    """
    from ._integer_search import fast_transform
    return fast_transform(vecs, max_passes, return_inverse)

def _reference_unimodular_transform_for_simplicity(
    vecs: List[Tuple[int, ...]],
    max_passes: int | None = None,
    return_inverse: bool = False,
):
    """
    Scalar implementation of strict L1 descent for checking batched results.

    Args:
        vecs (List[Tuple[int, ...]]): Integer exponent vectors spanning a
            lattice.
        max_passes (Optional[int]): Optional limit on the number of greedy
            local-improvement passes. None means "until no improvement".
        return_inverse (bool): If True, also return the inverse transform
            maintained during the row-operation search.

    Returns:
        Optional[List[List[int]]]: A unimodular integer matrix T whose rows
        transform the exponent coordinates, or None for empty or
        zero-dimensional input. If return_inverse is True, returns
        Tuple[T, T^{-1}] with the inverse as a SymPy Matrix.
    """
    # Additive row scores avoid copying the full matrix for each candidate.
    if not vecs:
        return None
    r = len(vecs[0])
    if r == 0:
        return None

    rows = [[int(v[i]) for v in vecs] for i in range(r)]

    def _row_score(row):
        """
        Compute the contribution of one row to the global complexity score.
        """
        nnz = 0
        neg_abs = 0
        neg_count = 0
        l1 = 0
        for x in row:
            if x:
                nnz += 1
                if x < 0:
                    neg_count += 1
                    neg_abs -= x
                    l1 -= x
                else:
                    l1 += x
        return nnz, neg_abs, neg_count, l1

    def _neg_row_score(row):
        """
        Score the row that would result from multiplying row by -1.
        """
        nnz = 0
        neg_abs = 0
        neg_count = 0
        l1 = 0
        for a in row:
            x = -a
            if x:
                nnz += 1
                if x < 0:
                    neg_count += 1
                    neg_abs -= x
                    l1 -= x
                else:
                    l1 += x
        return nnz, neg_abs, neg_count, l1

    def _add_row_score(row_i, row_j, s):
        """
        Score the row that would result from row_i += s * row_j.
        """
        nnz = 0
        neg_abs = 0
        neg_count = 0
        l1 = 0
        if s == 1:
            for a, b in zip(row_i, row_j):
                x = a + b
                if x:
                    nnz += 1
                    if x < 0:
                        neg_count += 1
                        neg_abs -= x
                        l1 -= x
                    else:
                        l1 += x
        else:
            for a, b in zip(row_i, row_j):
                x = a - b
                if x:
                    nnz += 1
                    if x < 0:
                        neg_count += 1
                        neg_abs -= x
                        l1 -= x
                    else:
                        l1 += x
        return nnz, neg_abs, neg_count, l1

    def _added_row(row_i, row_j, s):
        """
        Materialise row_i + s * row_j after an improving operation is accepted.
        """
        if s == 1:
            return [a + b for a, b in zip(row_i, row_j)]
        return [a - b for a, b in zip(row_i, row_j)]

    def _replace_score(total, old, new):
        """
        Return the global score after replacing one row contribution.
        """
        return (
            total[0] - old[0] + new[0],
            total[1] - old[1] + new[1],
            total[2] - old[2] + new[2],
            total[3] - old[3] + new[3],
        )

    def _apply_inverse_update(A, op):
        """
        Update A = T^{-1} after accepting the corresponding row operation on T.
        """
        if op[0] == "neg":
            _, i = op
            for row in A:
                row[i] = -row[i]
        elif op[0] == "add":
            _, i, j, s = op
            for row in A:
                row[j] -= s * row[i]
        elif op[0] == "swap":
            _, i, j = op
            for row in A:
                row[i], row[j] = row[j], row[i]
        else:
            raise ValueError("unknown unimodular op")

    T = [[1 if i == j else 0 for j in range(r)] for i in range(r)]
    T_inverse = [[1 if i == j else 0 for j in range(r)] for i in range(r)] if return_inverse else None
    row_scores = [_row_score(row) for row in rows]
    best_score = tuple(sum(score[k] for score in row_scores) for k in range(4))

    passes = 0
    while True:
        if max_passes is not None and passes >= max_passes:
            break
        passes += 1
        improved = False

        # Swaps preserve L1 and cannot be accepted by strict descent.
        # Sign flips also preserve L1; sign orientation is applied below.
        for i in range(r):
            old_score = row_scores[i]
            new_score = _neg_row_score(rows[i])
            sc = _replace_score(best_score, old_score, new_score)
            if sc[3] < best_score[3]:
                op = ("neg", i)
                _apply_row_op(T, op)
                if T_inverse is not None:
                    _apply_inverse_update(T_inverse, op)
                rows[i] = [-a for a in rows[i]]
                row_scores[i] = new_score
                best_score = sc
                improved = True
                break
        if improved:
            continue

        # Visit target i, source j, then signs +1 and -1.
        for i in range(r):
            old_score = row_scores[i]
            row_i = rows[i]
            for j in range(r):
                if i == j:
                    continue
                row_j = rows[j]
                for s in (1, -1):
                    new_score = _add_row_score(row_i, row_j, s)
                    sc = _replace_score(best_score, old_score, new_score)
                    if sc[3] < best_score[3]:
                        op = ("add", i, j, s)
                        _apply_row_op(T, op)
                        if T_inverse is not None:
                            _apply_inverse_update(T_inverse, op)
                        rows[i] = _added_row(row_i, row_j, s)
                        row_scores[i] = new_score
                        best_score = sc
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            break
    # Orient signs after strict L1 descent.
    # Sign flips preserve L1 cost and local optimality under signed shears.
    for i, row in enumerate(rows):
        if _neg_row_score(row) < row_scores[i]:
            op = ("neg", i)
            _apply_row_op(T, op)
            if T_inverse is not None:
                _apply_inverse_update(T_inverse, op)
    if return_inverse:
        return T, sp.Matrix(T_inverse)
    return T

def _lambdify_V(V, X, k_numeric: dict):
    """
    Build a fast numerical callable for the rate monomials V.

    Args:
        V (Sequence[sp.Expr]): Symbolic rate monomials V_j(X).
        X (Sequence[sp.Symbol]): Species symbols; order of arguments.
        k_numeric (Dict[sp.Symbol, float]): Numeric value for each k_j.

    Returns:
        Callable[[np.ndarray], np.ndarray]: Function fV(x) that evaluates all
        monomials V_j at once for a given vector x of species values.
    """
    # Substitute numeric rate constants and lambdify V_j with respect to X.
    V_num = sp.Matrix([v.subs(k_numeric) for v in V])
    fV = sp.lambdify(X, V_num, modules="numpy")
    return fV

def _rhs_factory(S: sp.Matrix, fV):
    """
    Create a right-hand side function for the ODE system dx/dt = S * V(x).

    Args:
        S (sp.Matrix): Stoichiometric matrix (n_species x n_reactions).
        fV (Callable): Function mapping x -> V(x) for all reactions.

    Returns:
        Callable[[float, np.ndarray], np.ndarray]: Function rhs(t, x) suitable
        for scipy.integrate.solve_ivp.
    """
    # Construct a closure that computes S @ fV(x) for each call.
    Snp = np.array(S.tolist(), dtype=float)  # shape (n, r)
    def rhs(t, x):
        """
        Right-hand side wrapper used by the ODE solver.

        Args:
            t (float): Current time (ignored if the system is autonomous).
            x (np.ndarray): Current state vector.

        Returns:
            np.ndarray: Time derivative dx/dt at (t, x).
        """
        # Evaluate the reaction rates at this state.
        vals = np.asarray(fV(*x), dtype=float).reshape(-1)  # shape (r,)
        return Snp @ vals
    return rhs

def _simulate(
    S,
    V,
    X,
    ks,
    k_numeric: dict,
    x0_numeric: dict,
    T_min: float,
    T_max: float,
    solver_opts: Mapping[str, Any] | None = None,
):
    """
    Numerically simulate the CRN and compute per-species ranges.

    The interval from 0 to T_min is treated as burn-in.  If T_min is strictly
    between 0 and T_max, a second integration is started at T_min so that both
    endpoints of the observation window are included in the range calculation.
    By default, dense output samples each adaptive step and refines bracketed
    stationary points. These are numerical extrema, not certified bounds; use
    max_step to resolve fast transients. dense_output=False explicitly requests
    the cheaper adaptive-node-only estimate.

    Args:
        S (sp.Matrix): Stoichiometric matrix of the CRN.
        V (Sequence[sp.Expr]): Rate monomials V_j(X).
        X (Sequence[sp.Symbol]): Species symbols (state ordering).
        ks (Sequence[sp.Symbol]): Rate-constant symbols.
        k_numeric (Dict[sp.Symbol, float]): Numeric rate constants.
        x0_numeric (Dict[sp.Symbol, float]): Numeric initial conditions.
        T_min (float): Time from which ranges are measured.
        T_max (float): Final simulation time.
        solver_opts (Mapping[str, Any]): Optional options for the solver.

    Returns:
        Dict[str, Any]: Dictionary with keys 'Xmin', 'Xmax',
        describing the simulated trajectories and ranges.
    """
    from scipy.integrate import solve_ivp
    opts: dict[str, Any] = dict(
        method="Radau",
        atol=1e-12,
        rtol=1e-9,
    )
    if solver_opts is not None:
        opts.update(solver_opts)
    opts.setdefault("dense_output", True)
    # Build a numerical rhs and integrate the burn-in and observation windows.
    x0_numeric = _normalise_symbol_mapping(x0_numeric, X, "x0_numeric", require_all=True)
    y0 = np.array([float(x0_numeric[xi]) for xi in X], dtype=float)
    if not np.all(np.isfinite(y0)):
        raise ValueError("Initial concentrations must have finite numerical values.")
    rhs, jac, jac_sparsity = build_rhs_and_jac(S, V, X, ks, k_numeric)
    method = opts.get("method", "Radau")
    # Explicit RK solvers do not use a Jacobian. LSODA uses a dense Jacobian
    # unless the caller requests its packed banded representation.
    if method in ("Radau", "BDF", "LSODA") and not (
        method == "LSODA" and ("lband" in opts or "uband" in opts)
    ):
        opts.setdefault("jac", jac)
    if method in ("Radau", "BDF") and opts.get("jac") is None:
        opts.setdefault("jac_sparsity", jac_sparsity)
    else:
        opts.pop("jac_sparsity", None)

    T_min = float(T_min)
    T_max = float(T_max)
    if not np.isfinite(T_min) or not np.isfinite(T_max) or not 0 <= T_min <= T_max:
        raise ValueError("The observation window must satisfy 0 <= T_min <= T_max with finite times.")
    if opts.get("t_eval") is not None:
        raise ValueError(
            "solver_opts['t_eval'] is not supported because extrema are "
            "computed from adaptive steps and their dense interpolant."
        )

    if 0.0 < T_min < T_max:
        # Burn-in integration: only its final state is required.
        sol_burn = solve_ivp(
            rhs,
            (0.0, T_min),
            y0,
            **opts,
        )
        if sol_burn.status != 0:
            raise RuntimeError(
                f"ODE integration did not reach T_min: {sol_burn.message}"
            )

        # The first and last columns correspond to T_min and T_max.
        sol_window = solve_ivp(
            rhs,
            (T_min, T_max),
            sol_burn.y[:, -1],
            **opts,
        )
        if sol_window.status != 0:
            raise RuntimeError(
                f"ODE integration did not reach T_max: {sol_window.message}"
            )
        Yw = sol_window.y
    else:
        sol_window = solve_ivp(
            rhs,
            (0.0, T_max),
            y0,
            **opts,
        )
        if sol_window.status != 0:
            raise RuntimeError(
                f"ODE integration did not reach T_max: {sol_window.message}"
            )
        if T_min == T_max:
            Yw = sol_window.y[:, -1:]
        else:
            Yw = sol_window.y

    Xmins = np.min(Yw, axis=1)
    Xmaxs = np.max(Yw, axis=1)
    if T_min < T_max and sol_window.sol is not None:
        # Include adaptive endpoints and three interior points per step.
        times = sol_window.t
        grid = np.unique(np.concatenate([
            times,
            *(times[:-1] + fraction * np.diff(times) for fraction in (0.25, 0.5, 0.75)),
        ]))
        dense_y = sol_window.sol(grid)
        Xmins = np.minimum(Xmins, np.min(dense_y, axis=1))
        Xmaxs = np.maximum(Xmaxs, np.max(dense_y, axis=1))
        slopes = rhs(0.0, dense_y)  # autonomous vector field, evaluated in a batch
        from scipy.optimize import brentq
        for i in range(len(X)):
            crossings = np.flatnonzero(
                ((slopes[i, :-1] < 0) & (slopes[i, 1:] > 0))
                | ((slopes[i, :-1] > 0) & (slopes[i, 1:] < 0))
            )
            for j in crossings:
                root = brentq(lambda t: rhs(t, sol_window.sol(t))[i], grid[j], grid[j + 1])
                value = float(sol_window.sol(root)[i])
                Xmins[i] = min(Xmins[i], value)
                Xmaxs[i] = max(Xmaxs[i], value)
    if not np.all(np.isfinite(Xmins)) or not np.all(np.isfinite(Xmaxs)):
        raise ValueError("The simulated concentration ranges must be finite.")
    return {"Xmin": Xmins, "Xmax": Xmaxs}

def as_matrix(obj) -> sp.Matrix:
    """
    Convert an object into a SymPy matrix.

    Args:
        obj: Any object acceptable to sp.Matrix (list-of-lists, ndarray, etc.).

    Returns:
        sp.Matrix: The corresponding SymPy matrix.
    """
    # Simple wrapper to normalise different input types.
    return sp.Matrix(obj)

def ensure_constant_stoichiometry(S) -> sp.Matrix:
    """
    Convert S to a SymPy matrix and verify that every entry is an
    integer or rational constant (no symbolic dependence).

    Parameters
    ----------
    S : array_like or sp.Matrix
        Stoichiometry matrix.

    Returns
    -------
    sp.Matrix
        The same matrix, converted to a SymPy Matrix, if the check passes.

    Raises
    ------
    ValueError
        If any entry of S is not an integer or rational constant.
    """
    S_mat = as_matrix(S)
    for entry in S_mat:
        # Any symbolic dependence (parameters, time, etc.) is disallowed
        if getattr(entry, "free_symbols", None):
            if entry.free_symbols:
                raise ValueError("The stoichiometry matrix must be constant")
        # Require SymPy's Rational (includes integers)
        if not isinstance(entry, sp.Rational):
            raise ValueError("The stoichiometry matrix must be constant")
    return S_mat

def has_all_zero_row_or_col(M: sp.Matrix) -> bool:
    """
    Check if a SymPy matrix has any all-zero rows or all-zero columns.

    Args:
        M (sp.Matrix): The SymPy matrix to check.

    Returns:
        True if M has at least one all-zero row or all-zero column, False otherwise
    """

    # Since M will be a matrix of integers, converting to numpy is faster.

    A = np.array(M.tolist(), dtype=object)
    zero_mask = (A == 0)

    # Any row with all zeros?
    if np.any(np.all(zero_mask, axis=1)):
        return True

    # Any column with all zeros?
    if np.any(np.all(zero_mask, axis=0)):
        return True

    return False

def build_P(S: sp.Matrix) -> sp.Matrix:
    """
    Build the 0/1 nonzero-pattern matrix P from a stoichiometric matrix S.

    Args:
        S (sp.Matrix): Stoichiometric matrix of the CRN.

    Returns:
        sp.Matrix: Matrix P of the same shape as S, with P[i, j] = 1 if
        S[i, j] != 0, else 0.
    """
    # Apply a nonzero test entrywise and cast to an integer matrix.
    S = as_matrix(S)
    n, r = S.shape
    data = [[1 if S[i, j] != 0 else 0 for j in range(r)] for i in range(n)]
    return sp.Matrix(data)

def _infer_ks_from_V(V, X):
    """
    Infer rate-constant symbols k_j from a list of monomials V_j(X).

    Args:
        V (Sequence[sp.Expr]): Rate monomials.
        X (Sequence[sp.Symbol]): Species symbols (state variables).

    Returns:
        List[sp.Symbol]: Inferred rate-constant symbols, one per V_j.
    """
    # Factor each V_j into a scalar times a product of X_i**nu_ij and treat
    # the scalar as the rate constant.
    Xset = set(X)
    ks = []
    for j, Vj in enumerate(V):
        # Symbols that are not species (i.e., parameters)
        params = [s for s in Vj.free_symbols if s not in Xset]
        if len(params) != 1:
            raise ValueError(
                f"V[{j}] = {Vj} must contain exactly one independent symbolic "
                "rate parameter for nondimensionalisation. Parameter-free terms "
                "and products of parameters are not supported by this analysis; "
                "poly_system_to_CRN preserves them but does not make them independent."
            )
        ks.append(params[0])
    return ks

def parse_mass_action(
    V: List[sp.Expr],
    X: List[sp.Symbol],
    ks: List[sp.Symbol],
) -> Tuple[sp.Matrix, List[sp.Expr]]:
    """
    Factor mass-action monomials into exponent matrix and rate constants.

    Args:
        V (List[sp.Expr]): Reaction-rate monomials V_j(X).
        X (List[sp.Symbol]): Species symbols (rows of nu).
        ks (List[sp.Symbol]): Rate-constant symbols, one per reaction column.

    Returns:
        Tuple[sp.Matrix, List[sp.Expr]]:
            nu (sp.Matrix): Integer exponent matrix nu_ij for X_i in V_j.
            ks_out (List[sp.Symbol]): Possibly adjusted list of k_j symbols.
    """
    # Divide each V_j by its associated k_j to recover exponents on X_i.
    n = len(X); r = len(V)
    nu = sp.zeros(n, r)
    Xset = set(X)
    for j, Vj in enumerate(V):
        # Factor and divide by k_j; avoid heavy Poly when possible
        Vj = sp.factor(Vj); kj = ks[j]
        rest = sp.together(Vj/kj)
        # Only X symbols should remain
        if rest != 1:
            if any(sym for sym in rest.free_symbols if sym not in Xset):
                raise ValueError(f"V[{j}] has symbols other than X and k_j: {rest}")
            pd = rest.as_powers_dict()
            # Build exponent vector for each species
            for i in range(n):
                e = sp.sympify(pd.get(X[i], 0))
                if not e.is_Integer or e < 0:
                    raise ValueError("Mass-action exponents must be nonnegative integers.")
                nu[i, j] = int(e)
            monomial = sp.prod(X[i]**nu[i, j] for i in range(n))
            if sp.cancel(rest - monomial) != 0:
                raise ValueError(f"V[{j}] must be k_j times a unit species monomial.")
        else:
            # all zeros for this reaction column
            pass
    return sp.Matrix(nu), ks

@lru_cache(maxsize=256)
def _snf_with_transforms_int_cached(rows_tuple):
    """
    Cached integer Smith Normal Form with transform matrices.

    Args:
        rows_tuple (Tuple[Tuple[int, ...], ...]): Row representation of the
            integer matrix whose SNF is required.

    Returns:
        Tuple[Optional[sp.Matrix], sp.Matrix, Optional[sp.Matrix]]:
            (U, D, V) where D is the SNF diagonal, and U, V are unimodular
            integer transforms (or None if not available).
    """
    # Check the cache for an existing decomposition before recomputing.
    if not rows_tuple:
        return None, sp.zeros(0, 0), None
    M = sp.Matrix(rows_tuple)
    return tuple(matrix.as_immutable() if matrix is not None else None
                 for matrix in _snf_with_transforms_int(M))

def _snf_diagonal_matrix_int(M: sp.Matrix) -> sp.Matrix:
    """Return the exact integer Smith diagonal (not transformation matrices)."""
    from sympy.matrices.normalforms import smith_normal_form
    return smith_normal_form(M, domain=sp.ZZ)


def _snf_with_transforms_int(M: sp.Matrix):
    """Return (None, D, None); SymPy's matrix SNF API returns only D.

    The caller obtains exact coordinates from the integer row lattice instead
    of relying on unavailable Smith transformation matrices.
    """
    return None, _snf_diagonal_matrix_int(M), None

def monomial_str(exp_tuple: Tuple[int, ...], gens: List[str]) -> str:
    """
    Convert an exponent vector into a human-readable monomial string.

    Args:
        exp_tuple (Tuple[int, ...]): Integer exponents per generator.
        gens (List[str]): Generator names corresponding to exp_tuple.

    Returns:
        str: Monomial string such as "g1^2*g2" or "1" when all exponents are 0.
    """
    # Build a product over gens with nonzero exponents, or "1" for the zero vector.
    if not exp_tuple or all(e == 0 for e in exp_tuple): return "1"
    parts = []
    for g, e in zip(gens, exp_tuple):
        if e == 0: continue
        parts.append(g if e == 1 else f"{g}^{int(e)}")
    return "*".join(parts) if parts else "1"

def _gen_symbols(gens: List[str]):
    """
    Create SymPy symbols from generator names.

    Args:
        gens (List[str]): Generator names, e.g. ["alpha", "beta"].

    Returns:
        Dict[str, sp.Symbol]: Mapping from name to positive, real Symbol.
    """
    # Construct sympy.Symbol objects for each generator name.
    return {g: sp.symbols(g, positive=True, real=True) for g in gens}

def _key_to_vec(key: str, gens: List[str]) -> Tuple[int, ...]:
    """
    Parse a monomial key string into an exponent vector.

    Args:
        key (str): String such as "alpha^2*beta^-1" or "1".
        gens (List[str]): Ordered list of generator names.

    Returns:
        Tuple[int, ...]: Integer exponents for each generator in gens.
    """
    # Parse the key and extract integer exponents in the gens order.
    # Empty/1 denotes the zero exponent vector.
    if not key or key.strip() == "1":
        return tuple(0 for _ in gens)
    exp = {g: 0 for g in gens}
    for part in key.split("*"):
        part = part.strip()
        if not part: continue
        if "^" in part:
            g, e = part.split("^", 1)
            exp[g.strip()] += int(e.strip())
        else:
            exp[part] += 1
    return tuple(exp[g] for g in gens)

def _vec_to_monom(v: Tuple[int, ...], gens: List[str], sym_g=None) -> sp.Expr:
    """
    Convert an exponent vector into a SymPy monomial in generator symbols.

    Args:
        v (Tuple[int, ...]): Integer exponents.
        gens (List[str]): Generator names corresponding to v.
        sym_g (Optional[Dict[str, sp.Symbol]]): Optional name->Symbol map;
            if None, new symbols are created by _gen_symbols.

    Returns:
        sp.Expr: SymPy expression prod_i g_i^v_i.
    """
    # Use provided or newly-created generator symbols to build the monomial.
    if sym_g is None: sym_g = _gen_symbols(gens)
    expr = sp.Integer(1)
    for g, e in zip(gens, v):
        if e:
            expr *= sym_g[g]**int(e)
    return expr

def _build_key_from_vec_global(v: Tuple[int, ...], gens: List[str]) -> str:
    """
    Build a global key string from an exponent vector.

    Args:
        v (Tuple[int, ...]): Integer exponent vector.
        gens (List[str]): Ordered generator names.

    Returns:
        str: Compact monomial key, e.g. "alpha^2*beta^-1" or "1".
    """
    # Inverse of _key_to_vec: convert vector back to a monomial key string.
    if not v or all(x == 0 for x in v):
        return "1"
    out = []
    for g, e in zip(gens, v):
        if e == 0:
            continue
        out.append(g if e == 1 else f"{g}^{int(e)}")
    return "*".join(out) if out else "1"

def format_sympy_matrix(M, name=None, joining_sym="=", elem_formatter=None):
    """
    Format a SymPy matrix or vector as a Python-style list string, optionally
    labelled with a name.

    Args:
        M (sympy.Matrix): The SymPy matrix or column vector to format.
        name (Optional[str]): Optional variable name label to place before the
            equals sign, e.g. "Stilde" or "Vtilde". If not provided or None,
            only the list representation is returned without a leading label.
        joining_sym (Optional[str]): Joining symbol. If not provided or None,
            use '='.
    Returns:
        str: A multi-line string of the form

            "Name =
            [
                [a11, a12, ...],
                [a21, a22, ...],
            ]"

            when name is provided for general matrices, or

            "Name =
            [
              v1,
              v2,
              ...
            ]"

            for column vectors.

            If name is None, the leading "Name =\n" line is omitted and only

            "[
                [a11, a12, ...],
                ...
            ]"

            (or the analogous vector form) is returned.
    """
    if joining_sym is None:
        joining_sym = '='
    prefix = f"{name}{joining_sym} \n" if name is not None else ""

    # Column vector case
    if M.cols == 1:
        lines = [f"  {elem_formatter(M[i, 0]) if elem_formatter else str(M[i, 0])}," for i in range(M.rows)]
        return prefix + "[\n" + "\n".join(lines) + "\n]"
    # General matrix case
    else:
        row_lines = []
        for i in range(M.rows):
            row_elems = ", ".join((elem_formatter(M[i, j]) if elem_formatter else str(M[i, j])) for j in range(M.cols))
            row_lines.append(f"    [{row_elems}],")
        return prefix + "[\n" + "\n".join(row_lines) + "\n]"
    
@lru_cache(maxsize=4096)
def _parse_grouped_monomial_key(key: str, gens: Tuple[str, ...]) -> Tuple[int, ...]:
    symbols = {name: sp.Symbol(name) for name in gens}
    # Treat complete generator names as atoms, including non-identifier labels.
    prefix = '_monomial_'
    while prefix in key:
        prefix = '_' + prefix
    aliases = {name: f'{prefix}{i}' for i, name in enumerate(gens)}
    if aliases:
        pattern = r'(?<!\w)(?:' + '|'.join(
            re.escape(name) for name in sorted(aliases, key=len, reverse=True)
        ) + r')(?!\w)'
        key = re.sub(pattern, lambda match: aliases[match.group()], key)
    expr = sp.sympify(key.replace('^', '**'),
                      locals={alias: symbols[name] for name, alias in aliases.items()})
    powers = expr.as_powers_dict()
    exponents = tuple(powers.get(symbols[name], sp.S.Zero) for name in gens)
    if (not all(value.is_Integer for value in exponents)
            or sp.cancel(expr / sp.prod(symbols[name]**value
                         for name, value in zip(gens, exponents))) != 1):
        raise ValueError("A unit monomial with integer generator exponents is required.")
    return tuple(map(int, exponents))


def _parse_monomial_key_to_exponents(key: str, gens: List[str]) -> List[int]:
    """Parse monomial keys into integer exponent vectors using arbitrary generator names.

    Args:
        key (str): Monomial key formatted as a product or quotient of generator tokens,
            e.g., "x*y^-2" or "alpha*beta^3" or "alpha/beta". Tokens must match names in gens,
            may include an optional "^p" integer exponent, and are separated by "*".
            The special keys "1" or "" denote the identity.
        gens (Sequence[str]): Ordered list of generator names.
            (len(gens) = d; token matching gens[i] maps to position i).

    Returns:
        List[int]: Integer exponent vector z in Z^d where z[k] is the exponent
        of generator gens[k], with absent generators contributing 0.
    """
    L = len(gens)
    gen_map = {name: i for i, name in enumerate(gens)}
    exps = [0] * L
    
    if key == '1' or key.strip() == '':
        return exps
    if '/' in key or '(' in key or ')' in key:
        return list(_parse_grouped_monomial_key(key, tuple(gens)))

    # SymPy prints powers as "**"; normalise to this parser's "^" syntax.
    key = key.replace('**', '^')
    
    for token in key.split('*'):
        token = token.strip()
        if not token:
            continue
        if '^' in token:
            base, exp_str = token.rsplit('^', 1)
            try:
                val = int(exp_str)
            except ValueError:
                raise ValueError("Monomial exponents must be integers.") from None
        else:
            base, val = token, 1
        if base in gen_map:
            exps[gen_map[base]] += val
        elif base != '1':
            raise ValueError(f"Unknown monomial generator: {base}")

    return exps

@lru_cache(maxsize=128)
def _build_symbolic_rhs_and_jacobian(
    S_key: Tuple[Tuple[sp.Expr, ...], ...],
    V_key: Tuple[sp.Expr, ...],
    X_key: Tuple[sp.Symbol, ...],
) -> Tuple[sp.Matrix, sp.Matrix]:
    """
    Build symbolic RHS f(x) = S * V(x) and its Jacobian df/dx.

    S_key:    hashable representation of S (tuple-of-tuples of exact constants)
    V_key:    tuple of SymPy expressions (the entries of V)
    X_key:    tuple of SymPy symbols (species)
    """
    S = sp.Matrix(S_key)
    V = sp.Matrix(V_key)
    X = sp.Matrix(X_key)

    f = S * V                 # vector field
    J = f.jacobian(X)         # df/dx

    return f.as_immutable(), J.as_immutable()


def build_rhs_and_jac(
    S: sp.Matrix,
    V: sp.Matrix,
    X: sp.Matrix,
    ks: sp.Matrix | list[sp.Symbol],
    k_numeric: Mapping[sp.Symbol, float],
) -> tuple[Callable, Callable, np.ndarray]:
    """
    Given symbolic S, V(X, ks), X, ks and numeric values for ks,
    return (rhs, jac, jac_sparsity) suitable for scipy.integrate.solve_ivp.
    """

    # Make hashable keys for caching
    S_key = tuple(tuple(sp.sympify(s_ij) for s_ij in row) for row in S.tolist())
    V_key = tuple(sp.sympify(v_ij) for v_ij in list(V))
    X_key = tuple(X)

    f_sym, J_sym = _build_symbolic_rhs_and_jacobian(S_key, V_key, X_key)

    # Resolve numeric keys to the model's symbols, including their assumptions.
    ks = _unique_ordered(ks)
    k_numeric = _normalise_symbol_mapping(k_numeric, ks, "k_numeric", require_all=True)
    k_vec = [float(k_numeric[k]) for k in ks]
    if not np.all(np.isfinite(k_vec)):
        raise ValueError("Rate constants must have finite numerical values.")

    # Lambdify f and J; they take (X, ks) -> f / J
    f_lam = sp.lambdify((X, ks), f_sym, modules="numpy")
    J_lam = sp.lambdify((X, ks), J_sym, modules="numpy")

    def rhs(t, y):
        # y is a 1D array of species values, same order as X
        if getattr(y, "ndim", 1) == 2:
            return np.column_stack([
                np.asarray(f_lam(column, k_vec), dtype=float).reshape(-1)
                for column in y.T
            ])
        return np.asarray(f_lam(y, k_vec), dtype=float).reshape(-1)

    def jac(t, y):
        return np.asarray(J_lam(y, k_vec), dtype=float)

    # Structural sparsity pattern (Jacobian nonzero structure)
    rows = [[(entry != 0) for entry in row] for row in J_sym.tolist()]
    jac_sparsity = np.array(rows, dtype=bool)

    return rhs, jac, jac_sparsity

def _build_rate_and_species_orders(V: sp.Matrix, X: list[sp.Symbol]):
    """
    Build consistent global orders for rate constants and species.

    Rate constants are discovered by scanning the reaction vector V
    top-to-bottom (row-wise) and, for each entry, iterating over the
    free symbols in a stable (name-sorted) order. Any symbol that is
    not a species in X is treated as a rate constant and assigned its
    first-seen index. Species are ordered by their index in X.

    Args:
        V (sp.Matrix): Column vector of reaction-rate monomials (n_reactions x 1).
        X (list[sp.Symbol]): Species symbols (ordered); defines the species index.

    Returns:
        tuple[dict[sp.Symbol, int], dict[sp.Symbol, int]]:
            A pair (rate_order, x_index) where:
              - rate_order maps each rate-constant symbol to its first-appearance index.
              - x_index maps each species symbol in X to its positional index.
    """
    # Species order by index (used for consistent variable ordering).
    x_index = {sym: i for i, sym in enumerate(X)}

    # Rate-constant order discovered from first appearance in V.
    rate_order = {}
    next_idx = 0
    for i in range(V.rows):
        expr = V[i, 0]
        # Name-sorted to keep discovery deterministic across Python runs.
        for sym in sorted(expr.free_symbols, key=lambda s: s.name):
            if sym not in x_index and sym not in rate_order:
                rate_order[sym] = next_idx
                next_idx += 1
    return rate_order, x_index

def _format_factor_ordered(expr: sp.Expr, rate_order: dict, x_index: dict) -> str:
    """
    Format an expression as a string with factors ordered within each monomial.

    Ordering within each multiplicative term (monomial):
      1) rate constants ascending by rate_order
      2) state variables ascending by x_index
      3) any remaining factors left in SymPy's default order

    Additive term order (i.e., the sequence of monomials in a sum) is preserved.

    Args:
        expr (sp.Expr): SymPy expression to render.
        rate_order (dict): Mapping from rate-constant symbols to integer order.
        x_index (dict): Mapping from species symbols to integer order.

    Returns:
        str: String representation with requested intra-monomial factor ordering.
    """
    def fmt_mul(term: sp.Expr) -> str:
        # Extract a numeric coefficient and the remaining product.
        coeff, rest = term.as_coeff_Mul()
        sign = "-" if coeff < 0 else ""
        # Take the multiplicative arguments (factors) of the non-coefficient part.
        factors = sp.Mul.make_args(rest if coeff != 0 else term)

        # Bucket factors for sorting by category.
        rate_factors, state_factors, other = [], [], []
        for f in factors:
            base = f.base if isinstance(f, sp.Pow) else f
            if isinstance(base, sp.Symbol) and base in rate_order:
                rate_factors.append(f)
            elif isinstance(base, sp.Symbol) and base in x_index:
                state_factors.append(f)
            else:
                other.append(f)

        # Sort categories by their declared orders.
        rate_factors.sort(key=lambda f: rate_order[f.base] if isinstance(f, sp.Pow) else rate_order[f])
        state_factors.sort(key=lambda f: x_index[f.base] if isinstance(f, sp.Pow) else x_index[f])

        # Reassemble: rates + states + other (keep other stable).
        ordered = rate_factors + state_factors + other

        # Rebuild textual product (omit unit coefficient 1).
        pieces = [str(abs(coeff))] if abs(coeff) not in (1, 1.0) else []
        pieces += [f"({p})" if isinstance(p, sp.Add) else str(p) for p in ordered]
        return sign + ("*".join(pieces) if pieces else "0")

    # Preserve additive term order; only reorder factors inside each term.
    if isinstance(expr, sp.Add):
        terms = list(sp.Add.make_args(expr))
        out = []
        for i, t in enumerate(terms):
            s = fmt_mul(t)
            out.append(s if i == 0 else ((" + " + s) if not s.startswith("-") else (" - " + s[1:])))
        return "".join(out)
    elif isinstance(expr, sp.Mul):
        return fmt_mul(expr)
    else:
        return str(expr)

def relabel_generators_by_first_use(Stilde, Vtilde, family):
    """
    Relabel the generators so that their subscripts increase first in the
    order they appear in Vtilde, and then in the order they appear in
    Stilde (left-to-right, top-to-bottom).

    Parameters
    ----------
    Stilde : sympy.Matrix
        Nondimensional stoichiometry matrix.
    Vtilde : sympy.Matrix
        Nondimensional reaction vector (column).
    family : dict
        One element of processed_CRN_dict["families"], containing
        "generators" and "generator_expressions".

    Returns
    -------
    Stilde_new, Vtilde_new, family_new
    """
    gens_old = list(family.get("generators", []))  # ["g_1", ..., "g_r"]
    if not gens_old:
        return Stilde, Vtilde, family

    # Find the actual SymPy symbols used for the generators in Vtilde/Stilde.
    sym_by_name = {}
    for s in Vtilde.free_symbols.union(Stilde.free_symbols):
        n = str(s)
        if n in gens_old:
            sym_by_name[n] = s

    # First row at which each generator appears in Vtilde.
    first_use_V = {name: None for name in gens_old}
    for row in range(Vtilde.rows):
        expr = Vtilde[row, 0]
        for name, sym in sym_by_name.items():
            if first_use_V[name] is None and expr.has(sym):
                first_use_V[name] = row

    # For generators not used in Vtilde, continue the ordering by scanning
    # Stilde in row-major order (left-to-right, top-to-bottom).
    first_use_S = {name: None for name in gens_old}
    for row in range(Stilde.rows):
        for col in range(Stilde.cols):
            expr = Stilde[row, col]
            for name, sym in sym_by_name.items():
                if first_use_S[name] is None and expr.has(sym):
                    first_use_S[name] = (row, col)

    # Order generators by first-use in Vtilde; if absent there, use their
    # first appearance in Stilde; finally fall back to original index.
    order = sorted(
        range(len(gens_old)),
        key=lambda idx: (
            0 if first_use_V[gens_old[idx]] is not None else 1,
            first_use_V[gens_old[idx]] if first_use_V[gens_old[idx]] is not None else float('inf'),
            first_use_S[gens_old[idx]] if first_use_S[gens_old[idx]] is not None else (float('inf'), float('inf')),
            idx,
        ),
    )
    gens_ordered_old = [gens_old[i] for i in order]
    gens_new = [f"g_{i+1}" for i in range(len(gens_ordered_old))]

    # Build substitution map old_symbol -> new_symbol
    subs = {}
    for old_name, new_name in zip(gens_ordered_old, gens_new):
        old_sym = sym_by_name.get(old_name)
        if old_sym is None:
            continue  # this generator never actually appears; ignore
        new_sym = sp.symbols(new_name, positive=True, real=True)
        subs[old_sym] = new_sym

    Stilde_new = Stilde.xreplace(subs)
    Vtilde_new = Vtilde.xreplace(subs)

    # Relabel generator expressions so they match the new names/order
    gen_exprs_old = list(family.get("generator_expressions", []))
    expr_map = {g: e for g, e in zip(gens_old, gen_exprs_old)}
    gen_exprs_new = [expr_map[g] for g in gens_ordered_old]

    # Relabel coefficient-class monomials using the same permutation.  This
    # keeps the family internally consistent and makes relabelling idempotent.
    classes_old = family.get("coeff_classes_by_generators", {})
    classes_new = {}
    for key, edges in classes_old.items():
        vec_old = _key_to_vec(key, gens_old)
        vec_new = tuple(vec_old[i] for i in order)
        key_new = _build_key_from_vec_global(vec_new, gens_new)
        classes_new.setdefault(key_new, []).extend(edges)

    family_new = dict(family)  # shallow copy
    family_new["generators"] = gens_new
    family_new["generator_expressions"] = gen_exprs_new
    family_new["coeff_classes_by_generators"] = classes_new

    return Stilde_new, Vtilde_new, family_new


def export_crnnondim_results(
    crnnondim_result: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = "out/",
    *,
    S: Optional[sp.Matrix] = None,
    V: Optional[List[Any]] = None,
    X: Optional[Union[List[Any], sp.Matrix]] = None,
) -> Path:
    """Export CRNNondim results to a pickle.

    The format of the pickle is as below:

        {
          "res_nd": {"opt": <dict>, "nd": <dict>, "crnnondim_result": <dict>},
          "nondim_S_sym": <sympy.Matrix>,
          "nondim_V_sym": <sympy.Matrix>,
          "crn_family": <dict|None>,
          "generators_syms": <list>,
          "generators_exprs": <list>,
        }

    Args:
        crnnondim_result: The dictionary returned by one of CRNNondim's
            nondimensionalisation routines.
        output_path: File path (or directory) to write. If a directory is
            supplied, {model_name}_{nondim_metric}_nondim.pkl is created within
            it. Metadata is sanitised to a safe filename. Missing metadata
            defaults to "crnnondim" and "deviation". If None, the file is
            created in the current working directory. Explicit file paths
            are trusted caller input and are used as supplied.

    Returns:
        Path: The path written.
    """
    # Determine a sensible default output filename from the model name / id
    model_name = None
    if isinstance(crnnondim_result, dict):
        for key in ("model_name", "model_id", "name", "crn_name"):
            val = crnnondim_result.get(key)
            if val:
                model_name = str(val)
                break
    if not model_name:
        model_name = "crnnondim"

    model_name = re.sub(r"[^A-Za-z0-9_]", "_", model_name.strip())
    while "__" in model_name:
        model_name = model_name.replace("__", "_")
    model_name = model_name.strip("_") or "crnnondim"

    nondim_metric = str((crnnondim_result or {}).get("nondim_metric", "deviation")).strip().lower()
    nondim_metric = re.sub(r"[^a-z0-9_]", "_", nondim_metric).strip("_") or "deviation"
    default_fname = f"{model_name}_{nondim_metric}_nondim.pkl"

    out = Path(output_path) if output_path is not None else Path.cwd()
    # Use the metadata-derived filename when the target is a directory.
    if out.suffix.lower() != ".pkl":
        out.mkdir(parents=True, exist_ok=True)
        candidate = out / default_fname
        if candidate.resolve().parent != out.resolve():
            raise ValueError("Metadata-derived output path must stay inside the output directory.")
        out = candidate
    else:
        out.parent.mkdir(parents=True, exist_ok=True)

    # Defaults (match preprocessor's empty fallbacks)
    nondim_S_sym = sp.Matrix([])
    nondim_V_sym = sp.Matrix([])
    crn_family = None
    generators_syms: List[Any] = []
    generators_exprs: List[Any] = []

    # Compute the symbolic nondimensional (S~, V~) pair when possible
    try:
        # Local import avoids import cycles during module import.
        from .analyse_results import analyse_results as _analyse_results

        pairs = _analyse_results(crnnondim_result, verbose=False)
        if pairs:
            nondim_S_sym, nondim_V_sym = pairs[0]
    except Exception as exc:
        warnings.warn(f"Could not export nondimensional matrices: {exc}", RuntimeWarning, stacklevel=2)

    # Extract the first family + generator metadata
    try:
        fams = crnnondim_result.get("families", []) if isinstance(crnnondim_result, dict) else []
        if fams:
            crn_family = fams[0]
            generators_syms = list(crn_family.get("generators", [])) if crn_family else []
            generators_exprs = list(crn_family.get("generator_expressions", [])) if crn_family else []
    except Exception as exc:
        warnings.warn(f"Could not export family metadata: {exc}", RuntimeWarning, stacklevel=2)

    # Optionally build the full res_nd payload
    opt: Dict[str, Any] = {}
    nd: Dict[str, Any] = {}
    if S is not None and V is not None and X is not None:
        try:
            S_mat = sp.Matrix(S)
            V_list = [sp.sympify(v) for v in list(V)]
            X_vec = sp.Matrix(X)

            # Dimensional RHS: dX/dT = S * V
            rhs_dim = S_mat * sp.Matrix(V_list)

            # Use symbol names as species names
            species_names = [str(X_vec[i]) for i in range(X_vec.rows)]
            x_syms = [X_vec[i] for i in range(X_vec.rows)]

            fam = crn_family
            if fam is not None and fam.get("K_family") is not None:
                # Species and time scalings from the family
                K_species = fam["K_family"].get("K_species")
                if K_species is None:
                    raise ValueError("Chosen family missing K_species.")
                if isinstance(K_species, sp.MatrixBase):
                    K_species_list = [sp.sympify(K_species[i]) for i in range(K_species.rows)]
                else:
                    K_species_list = [sp.sympify(v) for v in list(K_species)]

                if len(K_species_list) != len(species_names):
                    raise ValueError("Length mismatch between X and K_species.")

                K_map = {name: K_species_list[i] for i, name in enumerate(species_names)}
                kT_expr = sp.sympify(fam["K_family"].get("K_T", sp.Symbol("K_T", positive=True)))

                x_nd_syms = [sp.Symbol(f"x_{i+1}", nonnegative=True) for i in range(len(species_names))]
                subs_dim_to_nd = {Xsym: K_map[name] * x_nd
                                  for name, Xsym, x_nd in zip(species_names, x_syms, x_nd_syms)}
                rhs_sub = rhs_dim.xreplace(subs_dim_to_nd)
                rhs_nd_sym = sp.Matrix([(kT_expr / K_map[name]) * rhs_i
                                        for name, rhs_i in zip(species_names, rhs_sub)])

                nd = {
                    "rhs_nd_sym": rhs_nd_sym,
                    "k_T": kT_expr,
                    "k_species_map": K_map,
                    "subs_dim_to_nd": subs_dim_to_nd,
                    "x_nd_syms": x_nd_syms,
                }

                Nu = crnnondim_result.get("nu", None) if isinstance(crnnondim_result, dict) else None
                rate_constants = crnnondim_result.get("rate_constants", []) if isinstance(crnnondim_result, dict) else []
                params_set = set(rate_constants)

                # Extract coefficient term per reaction (product of rate constants)
                coeffs = []
                for expr in V_list:
                    c, _ = expr.as_coeff_mul(*x_syms)
                    coeff_term = sp.S(1)
                    for factor in c.as_ordered_factors():
                        if factor in params_set:
                            coeff_term *= factor
                    coeffs.append(coeff_term)

                # If Nu missing, reconstruct from monomial structure (shape nxm)
                if Nu is None:
                    Nu_rows = []
                    for expr, coeff in zip(V_list, coeffs):
                        mono = sp.simplify(expr / coeff)
                        pdict = mono.as_powers_dict()
                        Nu_rows.append([int(pdict.get(s, 0)) for s in x_syms])
                    Nu = sp.Matrix(Nu_rows).T

                alpha = []
                n = len(species_names)
                m = len(V_list)
                for j in range(m):
                    prodK = sp.Integer(1)
                    for i in range(n):
                        exp = int(Nu[i, j])
                        if exp != 0:
                            prodK *= K_map[species_names[i]] ** exp
                    alpha.append(sp.simplify(kT_expr * coeffs[j] * prodK))

                normalised_block = fam.get("normalised_block_index", None)
                opt = {
                    "K_map": K_map,
                    "kT": kT_expr,
                    "alpha": alpha,
                    "normalised_idx": ([normalised_block] if normalised_block is not None else []),
                }
        except Exception as exc:
            warnings.warn(f"Could not export opt/nd payload: {exc}", RuntimeWarning, stacklevel=2)
            # Keep opt/nd empty if anything goes wrong (still loads as a cache container)
            opt = {}
            nd = {}

    data = {
        "res_nd": {"opt": opt, "nd": nd, "crnnondim_result": crnnondim_result},
        "nondim_S_sym": nondim_S_sym,
        "nondim_V_sym": nondim_V_sym,
        "crn_family": crn_family,
        "generators_syms": generators_syms,
        "generators_exprs": generators_exprs,
    }

    with open(out, "wb") as f:
        pickle.dump(data, f)
    return out
