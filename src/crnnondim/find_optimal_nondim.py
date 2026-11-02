from itertools import chain, product
from contextlib import closing
from ._parallel_assignments import evaluate_assignments, resolve_workers
from typing import List, Tuple, Dict, Any, Optional, Union, Iterable
from pathlib import Path
import sympy as sp
import numpy as np
from .CRN_analyser import (
    _add_rate_rows,
    _assignment_rank_lower_bound,
    _exact_row_basis,
    _exact_row_rank,
)
from .objective_function_utils import compute_families_sup_lexi_constrained
from .utils import (
    _infer_ks_from_V,
    _simulate,
    has_all_zero_row_or_col,
    ensure_constant_stoichiometry,
    _parse_monomial_key_to_exponents,
    export_crnnondim_results,
    _normalise_dimensionally_consistent_sets,
)
from ._LEXIWrapper import _LEXIWrapper
from .process_CRN import process_CRN
from .process_CRN_given_nondimensionalisation import process_CRN_given_nondimensionalisation
# ------------- Assignment-dependent numerical scale selection ---------------

def _compare_sorted_costs(
    first: Tuple[float, ...],
    second: Tuple[float, ...],
    tol: float = 1e-12,
) -> int:
    """
    Compare descending cost vectors lexicographically.

    Args:
        first (Tuple[float, ...]): First descending cost vector.
        second (Tuple[float, ...]): Second descending cost vector.
        tol (float): Absolute comparison tolerance.

    Returns:
        int: -1 if first is preferred, 1 if second is preferred, and 0 for a
        tie within the supplied tolerance.
    """
    for first_value, second_value in zip(first, second):
        if first_value < second_value - tol:
            return -1
        if first_value > second_value + tol:
            return 1
    if len(first) < len(second):
        return -1
    if len(first) > len(second):
        return 1
    return 0

def _merge_sorted_costs(
    first: Tuple[float, ...],
    second: Tuple[float, ...],
) -> Tuple[float, ...]:
    """
    Merge two descending cost vectors.

    Args:
        first (Tuple[float, ...]): First descending cost vector.
        second (Tuple[float, ...]): Second descending cost vector.

    Returns:
        Tuple[float, ...]: Descending merge of both vectors.
    """
    return tuple(sorted((*first, *second), reverse=True))

def _maximum_ranks(maximum: float, values: List[float], tol: float = 1e-12) -> List[int]:
    """Rank positive numerical scales without breaking equal-value ties.

    Feasible values satisfy maximum/value <= 1 + tol and are ranked from
    smallest to largest. Infeasible values follow, largest first. Labels do
    not enter the ranks: distinct symbols with the same value remain tied.
    """
    preference = [
        (0, float(value)) if maximum / max(float(value), 1e-300) <= 1.0 + tol
        else (1, -float(value))
        for value in values
    ]
    rank = {key: index for index, key in enumerate(sorted(set(preference)))}
    return [rank[key] for key in preference]


def _dependent_optimal_choices(
    pre: Dict[str, Any],
    X: List[sp.Symbol],
    Xmin: np.ndarray,
    Xmax: np.ndarray,
    K_candidates: List[float],
    base_syms: List[sp.Symbol],
    fixed_K_idx: Dict[int, sp.Symbol],
    x0_numeric: Dict[sp.Symbol, float],
    nondim_metric: str,
) -> Tuple[Optional[float], Iterable[List[str]]]:
    """
    Select numerically optimal assignments among minimum-rank assignments.

    Block assignments are searched by exact branch-and-bound.  Numerical ties
    are streamed lazily so that downstream symbolic scoring remains exact
    without materialising all tied assignments in memory.

    Args:
        pre (Dict[str, Any]): Structural preprocessing output from process_CRN.
        X (List[sp.Symbol]): Species symbols corresponding to rows of S.
        Xmin (np.ndarray): Minimum simulated species values.
        Xmax (np.ndarray): Maximum simulated species values.
        K_candidates (List[float]): Numeric values of the base scale symbols.
        base_syms (List[sp.Symbol]): Base scale symbols.
        fixed_K_idx (Dict[int, sp.Symbol]): Pre-fixed species scale symbols.
        x0_numeric (Dict[sp.Symbol, float]): Numeric initial conditions.
        nondim_metric (str): Numerical scale metric, either deviation or maximum.

    Returns:
        Tuple[Optional[float], Iterable[List[str]]]: Level-1 objective value and
        a lazy stream of optimal full species-scale assignments.
    """
    analysis = pre["_assignment_analysis"]
    structure = analysis["structure"]
    minimum_r_rate = int(analysis["minimum_r_rate"])
    minimum_plan = analysis["minimum_plan"]
    component_data = structure["component_data"]
    mandatory_blocks = structure["mandatory_blocks"]
    structure_symbols = structure["scale_symbols"]
    rate_rows = structure["rate_rows"]
    block_count = structure["num_mandatory_blocks"]
    n = len(X)
    tol = 1e-12

    symbol_to_label = {
        symbol: index for index, symbol in enumerate(base_syms)
    }
    block_labels = {
        block: tuple(
            symbol_to_label[structure_symbols[index]] for index in indices
        )
        for block, indices in enumerate(mandatory_blocks)
    }
    fixed_labels = {
        i: symbol_to_label[symbol] for i, symbol in fixed_K_idx.items()
    }

    if nondim_metric == "deviation":
        alpha = np.log(np.minimum(Xmin, Xmax))
        beta = np.log(np.maximum(Xmin, Xmax))
        midpoint = 0.5 * (alpha + beta)
        half_range = 0.5 * (beta - alpha)
        log_scales = np.log(np.asarray(K_candidates, dtype=float))
        costs = np.empty((n, len(base_syms)), dtype=float)
        for i in range(n):
            costs[i, :] = (
                half_range[i] + np.abs(log_scales - midpoint[i])
            )
    else:
        costs = np.empty((n, len(base_syms)), dtype=float)
        for i in range(n):
            if i in fixed_labels:
                costs[i, :] = 0.0
                continue
            try:
                x0_i = float(x0_numeric.get(X[i], 0.0))
            except Exception:
                x0_i = 0.0
            if x0_i != 0.0:
                costs[i, :] = np.arange(len(base_syms), dtype=float)
            else:
                costs[i, :] = _maximum_ranks(float(Xmax[i]), K_candidates, tol)

    if nondim_metric == "deviation":
        def compare_costs(first, second):
            return _compare_sorted_costs(first, second, tol)

        def merge_costs(first, second):
            return _merge_sorted_costs(first, second)

        def one_cost(i, label):
            return (float(costs[i, label]),)
    else:
        def compare_costs(first, second):
            for first_item, second_item in zip(first, second):
                first_index, first_value = first_item
                second_index, second_value = second_item
                if first_index != second_index:
                    raise RuntimeError(
                        "Order-1 costs have inconsistent species indices."
                    )
                if first_value < second_value:
                    return -1
                if first_value > second_value:
                    return 1
            if len(first) < len(second):
                return -1
            if len(first) > len(second):
                return 1
            return 0

        def merge_costs(first, second):
            return tuple(sorted((*first, *second), key=lambda item: item[0]))

        def one_cost(i, label):
            return ((i, float(costs[i, label])),)

    local_cache: Dict[Tuple[int, int], Tuple[Any, Any]] = {}

    def local_option(component_index: int, block: int):
        key = (component_index, block)
        if key in local_cache:
            return local_cache[key]

        data = component_data[component_index]
        labels = block_labels[block]
        local_cost = tuple()
        label_options = []
        for i in data["species_indices"]:
            if i in fixed_labels:
                best_labels = (fixed_labels[i],)
            else:
                best_value = min(float(costs[i, label]) for label in labels)
                best_labels = tuple(
                    label for label in labels
                    if abs(float(costs[i, label]) - best_value) <= tol
                )
            local_cost = merge_costs(
                local_cost, one_cost(i, best_labels[0])
            )
            label_options.append((i, best_labels))

        result = (local_cost, tuple(label_options))
        local_cache[key] = result
        return result

    width = len(rate_rows[0]) if rate_rows else 0
    zero_row = tuple(sp.Integer(0) for _ in range(width))
    block_sums = [zero_row for _ in range(block_count)]
    base_cost = tuple()
    floating_nonzero: List[int] = []
    floating_zero: List[int] = []

    for component_index, data in enumerate(component_data):
        fixed_block = data["fixed_block"]
        if fixed_block is None:
            if any(rate_rows[component_index]):
                floating_nonzero.append(component_index)
            else:
                floating_zero.append(component_index)
            continue
        block_sums[fixed_block] = _add_rate_rows(
            block_sums[fixed_block], rate_rows[component_index]
        )
        local_cost, _ = local_option(component_index, fixed_block)
        base_cost = merge_costs(base_cost, local_cost)

    floating_nonzero.sort(key=lambda component_index: (
        -sum(abs(value) for value in rate_rows[component_index]),
        rate_rows[component_index],
        component_index,
    ))

    zero_block_options: Dict[int, Tuple[int, ...]] = {}
    for component_index in floating_zero:
        best_cost = None
        best_blocks: List[int] = []
        for block in range(block_count):
            local_cost, _ = local_option(component_index, block)
            if best_cost is None or compare_costs(local_cost, best_cost) < 0:
                best_cost = local_cost
                best_blocks = [block]
            elif compare_costs(local_cost, best_cost) == 0:
                best_blocks.append(block)
        zero_block_options[component_index] = tuple(best_blocks)
        base_cost = merge_costs(base_cost, best_cost)

    suffix_bases: List[Tuple[Tuple[sp.Expr, ...], ...]] = [
        tuple() for _ in range(len(floating_nonzero) + 1)
    ]
    basis: Tuple[Tuple[sp.Expr, ...], ...] = tuple()
    for position in range(len(floating_nonzero) - 1, -1, -1):
        row = rate_rows[floating_nonzero[position]]
        basis = _exact_row_basis((row,) + basis)
        suffix_bases[position] = basis

    total_row = zero_row
    for row in block_sums:
        total_row = _add_rate_rows(total_row, row)
    for component_index in floating_nonzero:
        total_row = _add_rate_rows(
            total_row, rate_rows[component_index]
        )
    total_nonzero = any(total_row)

    rank_cache: Dict[Tuple[Tuple[sp.Expr, ...], ...], int] = {}

    def cached_rank(rows):
        rows = tuple(rows)
        if rows not in rank_cache:
            rank_cache[rows] = _exact_row_rank(rows)
        return rank_cache[rows]

    def lower_bound(position, current_sums):
        return _assignment_rank_lower_bound(
            tuple(current_sums),
            suffix_bases[position],
            total_nonzero,
            minimum_r_rate,
        )

    best_suffix_cost = [tuple() for _ in range(len(floating_nonzero) + 1)]
    for position in range(len(floating_nonzero) - 1, -1, -1):
        component_index = floating_nonzero[position]
        best_local = None
        for block in range(block_count):
            local_cost, _ = local_option(component_index, block)
            if best_local is None or compare_costs(local_cost, best_local) < 0:
                best_local = local_cost
        best_suffix_cost[position] = merge_costs(
            best_local, best_suffix_cost[position + 1]
        )

    initial_plan = [selected[0] for selected in minimum_plan]
    initial_sums = list(block_sums)
    initial_cost = base_cost
    for component_index in floating_nonzero:
        block = initial_plan[component_index]
        initial_sums[block] = _add_rate_rows(
            initial_sums[block], rate_rows[component_index]
        )
        local_cost, _ = local_option(component_index, block)
        initial_cost = merge_costs(initial_cost, local_cost)
    if cached_rank(initial_sums) != minimum_r_rate:
        raise RuntimeError("The minimum-rank assignment is inconsistent.")

    best_cost = initial_cost
    seen_costs: Dict[Tuple[int, Tuple[Any, ...]], Any] = {}
    stack = [(0, tuple(block_sums), base_cost)]
    while stack:
        position, current_sums, partial_cost = stack.pop()
        if lower_bound(position, current_sums) > minimum_r_rate:
            continue
        optimistic = merge_costs(
            partial_cost, best_suffix_cost[position]
        )
        if compare_costs(optimistic, best_cost) > 0:
            continue

        state = (position, current_sums)
        previous = seen_costs.get(state)
        if previous is not None and compare_costs(partial_cost, previous) >= 0:
            continue
        seen_costs[state] = partial_cost

        if position == len(floating_nonzero):
            if cached_rank(current_sums) == minimum_r_rate:
                if compare_costs(partial_cost, best_cost) < 0:
                    best_cost = partial_cost
            continue

        component_index = floating_nonzero[position]
        row = rate_rows[component_index]
        branches = []
        for block in range(block_count):
            next_sums = list(current_sums)
            next_sums[block] = _add_rate_rows(next_sums[block], row)
            next_tuple = tuple(next_sums)
            if lower_bound(position + 1, next_tuple) > minimum_r_rate:
                continue
            local_cost, _ = local_option(component_index, block)
            next_cost = merge_costs(partial_cost, local_cost)
            branches.append((local_cost, block, next_tuple, next_cost))
        branches.sort(key=lambda item: (item[0], item[1]))
        for _, _, next_sums, next_cost in reversed(branches):
            stack.append((position + 1, next_sums, next_cost))

    def stream_choices():
        zero_components = tuple(sorted(floating_zero))
        zero_options = tuple(
            zero_block_options[component_index]
            for component_index in zero_components
        )
        search_stack = [(0, tuple(block_sums), base_cost, tuple())]
        while search_stack:
            position, current_sums, partial_cost, blocks = search_stack.pop()
            if lower_bound(position, current_sums) > minimum_r_rate:
                continue
            optimistic = merge_costs(
                partial_cost, best_suffix_cost[position]
            )
            if compare_costs(optimistic, best_cost) > 0:
                continue

            if position == len(floating_nonzero):
                if (
                    cached_rank(current_sums) != minimum_r_rate
                    or compare_costs(partial_cost, best_cost) != 0
                ):
                    continue

                zero_assignments = (
                    (tuple(),) if not zero_options else product(*zero_options)
                )
                for zero_blocks in zero_assignments:
                    component_blocks = [
                        data["fixed_block"] for data in component_data
                    ]
                    for component_index, block in zip(
                        floating_nonzero, blocks
                    ):
                        component_blocks[component_index] = block
                    for component_index, block in zip(
                        zero_components, zero_blocks
                    ):
                        component_blocks[component_index] = block

                    species_options: List[Tuple[int, ...]] = [tuple()] * n
                    for component_index, block in enumerate(component_blocks):
                        _, local_labels = local_option(component_index, block)
                        for i, labels_for_species in local_labels:
                            species_options[i] = labels_for_species
                    for assignment in product(*species_options):
                        yield [str(base_syms[label]) for label in assignment]
                continue

            component_index = floating_nonzero[position]
            row = rate_rows[component_index]
            branches = []
            for block in range(block_count):
                next_sums = list(current_sums)
                next_sums[block] = _add_rate_rows(next_sums[block], row)
                next_tuple = tuple(next_sums)
                if lower_bound(position + 1, next_tuple) > minimum_r_rate:
                    continue
                local_cost, _ = local_option(component_index, block)
                next_cost = merge_costs(partial_cost, local_cost)
                optimistic = merge_costs(
                    next_cost, best_suffix_cost[position + 1]
                )
                if compare_costs(optimistic, best_cost) <= 0:
                    branches.append((block, next_tuple, next_cost))
            for block, next_sums, next_cost in reversed(branches):
                search_stack.append((
                    position + 1,
                    next_sums,
                    next_cost,
                    blocks + (block,),
                ))

    if nondim_metric == "deviation" and best_cost:
        Rstar = float(best_cost[0])
    else:
        Rstar = None
    return Rstar, stream_choices()

def find_optimal_nondim(
    S: sp.Matrix,
    V: List[sp.Expr],
    X: List[sp.Symbol],
    fixed_K: Optional[Dict[sp.Symbol, sp.Symbol]] = None,
    k_numeric: Optional[Dict[sp.Symbol, float]] = None,
    x0_numeric: Optional[Dict[sp.Symbol, float]] = None,
    T_max: Optional[float] = None,
    T_min: Optional[float] = None,
    basis_type: str = "Laurent",
    nondim_metric: str = "deviation",
    solver_opts: dict | None = None,
    pre: Optional[Dict[str, Any]] = None,
    export_results: bool = False,
    output_path: Optional[Union[str, Path]] = "out",
    dimensionally_consistent_sets: Optional[List[set[sp.Symbol]]] = None,
    *,
    n_jobs: Optional[int] = None,
    parallel_start_method: str = "spawn",
    parallel_diagnostics: bool = False,
) -> Dict[str, Any]:
    """
    Search over families of nondimensionalisations and pick an optimal one.

    If r_rate depends on the scale assignment, the structural objective
    s - c_scale + r_rate is minimised before the numerical scale objective.
    Cases in which component merges can tie the structural minimum use an
    exhaustive assignment fallback. Tied assignments are compared by
    (F_mono, F_gen, F_time). Basis searches
    use deterministic local descent, not a global unimodular optimum.

    Args:
        S (sp.Matrix): Stoichiometric matrix of the CRN.
        V (List[sp.Expr]): Reaction-rate monomials V_j(X).
        X (List[sp.Symbol]): Species symbols corresponding to S's rows.
        fixed_K (Optional[Dict[sp.Symbol, sp.Symbol]]): Optional mapping
            species_symbol -> fixed scale symbol, imposing partial choices.
        k_numeric (Optional[Dict[sp.Symbol, float]]): Numeric values for rate
            constants; required for simulation-based objectives.
        x0_numeric (Optional[Dict[sp.Symbol, float]]): Numeric initial
            conditions for species.
        T_min (Optional[float]): Time from which the objective is evaluated. If
            omitted (or passed as None/empty), it defaults to T_max, meaning the
            O(1) deviation penalty is evaluated at the final time only.
        T_max (Optional[float]): Final integration time; must be provided.
            Explicitly passing 0 evaluates only the initial state.
        basis_type (str): Type of exponent basis ("Laurent" or "Polynomial") controlling
            whether negative exponents are allowed.
        nondim_metric (str): Optimisation metric used to choose the per-species
            scale assignment. "deviation" minimises O(1) penalty lexicographically;
            "maximum" chooses, for zero-initial-condition species, the smallest
            admissible scale whose simulated window maximum keeps non-dimensional
            variables <= 1 over the interval [T_min, T_max]; if none does, the
            largest allowed value is preferred. Equal numerical values retain
            all allowed symbolic labels for subsequent coefficient scoring.
            For Cartesian restrictions, linear scans select each coordinate's
            preferred values before the tied assignments are streamed lazily.
        solver_opts (Optional[dict]): Optional options for Scipy's solve_ivp solver.
        pre (Optional[Dict[str, Any]]): Optional precomputed symbolic bundle
            returned by process_CRN, to avoid recomputing structural data.
        export_results (bool): If True, export the returned dictionary to a pickle
            file.
        output_path (Optional[str | Path]): File path (or directory) for the pickle.
            Directory targets use {model_name}_{nondim_metric}_nondim.pkl;
            missing metadata defaults to "crnnondim" and "deviation".
        dimensionally_consistent_sets (Optional[List[set[sp.Symbol]]]):
            Partition of X into sets whose species scales have compatible units.
            Each species must occur exactly once. If None, all species are
            assumed to have compatible units. When pre is supplied, its stored
            partition is used unless this argument is also supplied and matches it.

        n_jobs (Optional[int]): Maximum assignment processes. The actual count
            is min(available CPUs - 1, n_jobs, number of tied assignments),
            respecting CPU affinity/quota and platform process-pool limits.
            None uses all available CPUs except one. Single-CPU environments
            and single-assignment inputs run serially; 1 also forces serial.
        parallel_start_method (str): Multiprocessing context; default "spawn".
            A package-owned coordinator creates the process pool; workers do
            not import the calling script or repeat its setup and output.
            This also supports ordinary interactive calls. Frozen executables
            require n_jobs=1. Application scripts should still use a main guard
            for their own multiprocessing or import-time side effects.
        parallel_diagnostics (bool): Include per-task process IDs and timings
            in result["assignment_parallelism"].

    Returns:
        Dict[str, Any]: A dictionary describing the selected optimal family,
        including generator rank, uniqueness counts, selected objective
        values, assignment diagnostics and the chosen nondimensionalisation.
    """

    # Validate before preprocessing/integration. A singleton stays serial.
    resolve_workers(n_jobs)
    assignment_stats: Dict[str, Any] = {}

    def _finish(result: Dict[str, Any]) -> Dict[str, Any]:
        result.setdefault("nondim_metric", nondim_metric)
        if export_results:
            # If available, attach a model name/id so the exporter can choose
            # {model_name}_{nondim_metric}_nondim.pkl for a directory target.
            if isinstance(result, dict) and (not result.get("model_name")) and pre is not None:
                name = None
                if isinstance(pre, dict):
                    name = pre.get("model_name") or pre.get("model_id") or pre.get("name") or pre.get("id")
                else:
                    name = getattr(pre, "model_name", None) or getattr(pre, "model_id", None) or getattr(pre, "name", None) or getattr(pre, "id", None)
                if name:
                    result["model_name"] = name

            export_crnnondim_results(result, output_path=output_path, S=S, V=V, X=X)
        return result

    # --- Input validation: basis_type ---------------------------------------------
    allowed_basis_types = ("Laurent", "Polynomial")
    if basis_type not in allowed_basis_types:
        raise ValueError("basis_type must be either 'Laurent' or 'Polynomial'.")
    if fixed_K is not None and (
            not hasattr(fixed_K, "items") or not set(fixed_K).issubset(set(X))
            or not all(isinstance(scale, sp.Symbol) for scale in fixed_K.values())):
        raise ValueError("fixed_K must map species in X to SymPy scale symbols.")
    allowed_nondim_metrics = ("deviation", "maximum")
    nondim_metric = str(nondim_metric).strip().lower()
    if nondim_metric not in allowed_nondim_metrics:
        raise ValueError("nondim_metric must be either 'deviation' or 'maximum'.")
    if solver_opts is None:
        solver_opts = {}
    # --- Basic numeric requirement -------------------------------------------------
    run_numeric = (k_numeric is not None and x0_numeric is not None and T_max is not None)
    if not run_numeric:
        raise ValueError("Numerical values for all initial conditions, rate constants and a final time T_max need to be provided to find the optimal nondim.")
    # If T_min is not provided, default to T_max (evaluate O(1) penalty at the final time).
    if T_min is None or (isinstance(T_min, str) and T_min.strip() == ""):
        T_min = T_max
    T_min = float(T_min)
    T_max = float(T_max)
    if T_max < T_min:
        raise ValueError("T_max must be >= T_min.")
    # --- Basic structural checks on S ---------------------------------------------
    S = ensure_constant_stoichiometry(S)
    if has_all_zero_row_or_col(S):
        raise ValueError("All rows and all columns of the stoichiometry matrix, S, must be non-zero.")
    if S.rows != len(X):
        raise ValueError(f"Dimension mismatch: S has {S.rows} rows but X has {len(X)} species.")
    if S.cols != len(V):
        raise ValueError(f"Dimension mismatch: S has {S.cols} columns but there are {len(V)} reactions in V.")
    # Per-reaction rate symbols (returned at the end for compatibility)
    ks = _infer_ks_from_V(V, X)
    # --- Structural preprocessing & component analysis (reuse process_CRN) ---------
    # Optionally reuse precomputed structural data (components, nu, etc.).
    # This bundles the mass-action parsing, P/nu construction, and component-level
    # restrictions that are also needed by process_CRN and process_CRN_given_nondimensionalisation.
    if pre is None:
        pre = process_CRN(
            S, V, X, fixed_K=fixed_K, find_example_nondim=False,
            basis_type=basis_type,
            dimensionally_consistent_sets=dimensionally_consistent_sets,
        )
    else:
        if (pre.get("_S") != S or tuple(pre.get("_X", ())) != tuple(X)
                or sp.Matrix(pre.get("_V", ())) != sp.Matrix(V)):
            raise ValueError("pre must be constructed from the same S, V and X.")
        if fixed_K is not None:
            requested_fixed = {i: fixed_K[x] for i, x in enumerate(X) if x in fixed_K}
            if requested_fixed != pre.get("fixed_K_indexed", {}):
                raise ValueError("fixed_K must match the scales used to construct pre.")
    pre_dimensional_sets, _ = _normalise_dimensionally_consistent_sets(
        X, pre.get("dimensionally_consistent_sets")
    )
    if dimensionally_consistent_sets is not None:
        requested_sets, _ = _normalise_dimensionally_consistent_sets(
            X, dimensionally_consistent_sets
        )
        if frozenset(requested_sets) != frozenset(pre_dimensional_sets):
            raise ValueError(
                "dimensionally_consistent_sets must match the partition used "
                "to construct pre."
            )
    dimensional_sets = pre_dimensional_sets
    has_dimensional_restrictions = len(dimensional_sets) > 1
    # Reuse the exact rank-first enumeration for cases where component merges
    # may preserve the minimum parameter count. The non-merging search alone
    # can omit the best numerical assignment in shared-rate networks.
    search_all_assignments = has_dimensional_restrictions or pre.get(
        "_assignment_analysis", {}
    ).get("requires_full_assignment_search", False)
    P = pre["P"]
    nu = pre["nu"]
    n, r = P.rows, P.cols
    # Active coefficient slots
    E = sorted([(i, j) for i in range(n) for j in range(r) if P[i, j] != 0])
    if not E:
        return _finish({
            "num_unique_coefficients": 0,
            "num_generators": 0,
            "families": [],
            "P": P,
            "nu": nu,
            "rate_constants": ks,
            "_S": S, "_X": X
        })
    # Fixed-scale bookkeeping (per-species assignment of base symbols)
    fixed_K_idx = pre.get("fixed_K_indexed", {}) or {}
    base_syms = pre.get("base_symbols", [])
    free_indices = [i for i in range(n) if i not in fixed_K_idx]
    # If no base symbols exist and there are free species, nothing to assign => no families
    if not base_syms and free_indices:
        return _finish({
            "num_unique_coefficients": None,
            "num_generators": None,
            "families": [],
            "P": P,
            "nu": nu,
            "rate_constants": ks,
            "_S": S, "_X": X
        })
    # Restrictions and label mapping
    restrictions = pre.get("restrictions", [None] * n)
    labels = [str(s) for s in base_syms]
    label_to_sym = {str(s): s for s in base_syms}
    # --- Simulate to obtain Xmin/Xmax over [T_min, T_max] --------------------------
    sim = _simulate(S, V, X, ks, k_numeric, x0_numeric, T_min, T_max, solver_opts)
    Xmin = np.array(sim["Xmin"], dtype=float)
    Xmax = np.array(sim["Xmax"], dtype=float)
    # A logarithmic deviation is defined only for strictly positive trajectories.
    if nondim_metric == "deviation" and (
        not np.all(np.isfinite(Xmin)) or not np.all(np.isfinite(Xmax))
        or np.any(Xmin <= 0.0)
    ):
        raise ValueError(
            "The deviation objective requires finite, strictly positive "
            "concentrations throughout the observation window."
        )
    eps = 1e-300
    # --- Prepare numeric candidates for base scales K_l ----------------------------
    if not base_syms:
        raise ValueError("No base (fixed) K symbols available; cannot enumerate choices.")
    extra_rhs_numeric: Dict[sp.Symbol, float] = {}
    for i, s in fixed_K_idx.items():
        if x0_numeric is None or X[i] not in x0_numeric or x0_numeric[X[i]] is None:
            raise ValueError("All initial conditions must be known to find the optimal nondim.")
        value = float(x0_numeric[X[i]])
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Fixed species scales require finite positive initial values.")
        if s in extra_rhs_numeric and extra_rhs_numeric[s] != value:
            raise ValueError("Species sharing a fixed scale must have equal initial values.")
        extra_rhs_numeric[s] = value
    K_candidates = [max(float(extra_rhs_numeric[s]), eps) for s in base_syms]
    # --- Compute constrained lexicographic family ranges / O(1) choice ------------
    Rstar = None
    dimensional_choice_cost = None
    dimensional_cost_compare = None
    if search_all_assignments:
        label_index = {label: index for index, label in enumerate(labels)}
        if nondim_metric == "deviation":
            alpha = np.log(np.minimum(Xmin, Xmax))
            beta = np.log(np.maximum(Xmin, Xmax))
            midpoint = 0.5 * (alpha + beta)
            half_range = 0.5 * (beta - alpha)
            log_scales = np.log(np.asarray(K_candidates, dtype=float))
            dimensional_costs = (
                half_range[:, None]
                + np.abs(log_scales[None, :] - midpoint[:, None])
            )

            def dimensional_choice_cost(choice):
                return tuple(sorted(
                    (
                        float(dimensional_costs[i, label_index[str(choice[i])]])
                        for i in range(n)
                    ),
                    reverse=True,
                ))

            dimensional_cost_compare = _compare_sorted_costs
        else:
            dimensional_costs = np.zeros((n, len(base_syms)), dtype=float)
            for i in range(n):
                if i in fixed_K_idx:
                    continue
                try:
                    x0_i = float(x0_numeric.get(X[i], 0.0))
                except Exception:
                    x0_i = 0.0
                if x0_i != 0.0:
                    dimensional_costs[i, :] = np.arange(len(base_syms), dtype=float)
                else:
                    dimensional_costs[i, :] = _maximum_ranks(float(Xmax[i]), K_candidates)

            def dimensional_choice_cost(choice):
                return tuple(
                    (i, float(dimensional_costs[i, label_index[str(choice[i])]]))
                    for i in range(n)
                )

            def dimensional_cost_compare(first, second):
                return -1 if first < second else (1 if first > second else 0)

        choices = _LEXIWrapper(labels, restrictions).stream()
    elif pre.get("_assignment_analysis") is not None:
        Rstar, choices = _dependent_optimal_choices(
            pre,
            X,
            Xmin,
            Xmax,
            K_candidates,
            base_syms,
            fixed_K_idx,
            x0_numeric,
            nondim_metric,
        )
    elif nondim_metric == "deviation":
        L1, LEXI = compute_families_sup_lexi_constrained(
            Xmin, Xmax, K_candidates, restrictions, labels, tol=1e-12
        )
        Rstar = float(L1.Rstar)   # Level-1 max O(1) penalty
        choices = LEXI.stream(as_labels=True)
    else:
        def _allowed_labels_for_index(i: int) -> List[str]:
            ritem = restrictions[i] if i < len(restrictions) else None
            if ritem is None:
                return list(labels)
            if isinstance(ritem, str):
                return [ritem]
            if hasattr(ritem, 'dtype') and hasattr(ritem, '__len__') and getattr(ritem, 'dtype', None) is not None and ritem.dtype == bool:
                return [labels[j] for j, flag in enumerate(ritem) if flag]
            return [str(s) for s in list(ritem)]

        label_to_value = {lab: float(K_candidates[idx]) for idx, lab in enumerate(labels)}
        # Resolve each coordinate once, without starting a Cartesian product.
        allowed = [_allowed_labels_for_index(i) for i in range(n)]
        first_choice = [options[0] for options in allowed] if all(allowed) else None
        if first_choice is None:
            choices = []
        else:
            choice_maximum = [[label] for label in first_choice]
            for i in range(n):
                if i in fixed_K_idx:
                    choice_maximum[i] = [str(fixed_K_idx[i])]
                    continue
                try:
                    x0_i = float(x0_numeric.get(X[i], 0.0))
                except Exception:
                    x0_i = 0.0
                if x0_i != 0.0:
                    continue
                allowed_labels = allowed[i]
                if not allowed_labels:
                    raise ValueError(f"No admissible scale choices are available for species {X[i]}.")
                # Only the preferred value is needed; do not rank every scale.
                values = [label_to_value[lab] for lab in allowed_labels]
                maximum = float(Xmax[i])
                admissible = [
                    value for value in values
                    if maximum / max(value, eps) <= 1.0 + 1e-12
                ]
                best_value = min(admissible) if admissible else max(values)
                choice_maximum[i] = [
                    lab for lab, value in zip(allowed_labels, values) if value == best_value
                ]
            choices = _LEXIWrapper(labels, choice_maximum).stream()
    # --- One-time structural invariants (avoid recomputing per choice) -------------
    E = [(i, j) for i in range(n) for j in range(r) if P[i, j] != 0]
    if not E:
        return _finish({"num_unique_coefficients": 0, "num_generators": 0, "families": [], "P": P, "nu": nu, "rate_constants": ks, "_S": S, "_X": X})
    E_map = {pair: idx for idx, pair in enumerate(E)}
    mE = len(E)

    # Build the C matrix s.t. log c_ij = log k_j + C_row * log(K)
    C = sp.zeros(mE, n)
    for idx, (i, j) in enumerate(E):
        C[idx, i] -= 1
        for l in range(n):
            C[idx, l] += nu[l, j]

    choice_iterator = iter(choices)
    first_choice = next(choice_iterator, None)
    if first_choice is None:
        return _finish({"num_unique_coefficients": 0, "num_generators": 0, "families": [], "P": P, "nu": nu, "rate_constants": ks, "_S": S, "_X": X})

    # --- Enumerate nondimensionalisations and score families -----------------------
    best_min_rank = None
    best_dimensional_cost = None
    best_symbolic_score = None
    best_families: List[Dict[str, Any]] = []
    def _assignment_jobs():
        for choice in chain((first_choice,), choice_iterator):
            fixed_full: Dict[int, sp.Symbol] = dict(fixed_K_idx)
            if isinstance(choice, (list, tuple)) and len(choice) == n:
                for i, sym in enumerate(choice):
                    fixed_full[i] = label_to_sym[sym] if isinstance(sym, str) else sym
            else:
                for i, sym in zip(free_indices, choice):
                    fixed_full[i] = label_to_sym[sym] if isinstance(sym, str) else sym
            yield choice, tuple(fixed_full[i] for i in range(n))

    assignment_results = evaluate_assignments(
        _assignment_jobs(), (S, V, X, basis_type, dimensional_sets),
        n_jobs, parallel_start_method, parallel_diagnostics, assignment_stats,
    )
    with closing(assignment_results):
        for choice, fam_res in assignment_results:
            families = fam_res.get("families", [])
            if search_all_assignments and families:
                choice_rank = min(int(fam.get("free_group_rank", 0)) for fam in families)
                choice_cost = dimensional_choice_cost(choice)
                cost_comparison = (
                    0 if best_dimensional_cost is None
                    else dimensional_cost_compare(choice_cost, best_dimensional_cost)
                )
                if (
                    best_min_rank is None
                    or choice_rank < best_min_rank
                    or (choice_rank == best_min_rank and cost_comparison < 0)
                ):
                    best_min_rank = choice_rank
                    best_dimensional_cost = choice_cost
                    best_families = []
                    best_symbolic_score = None
                elif choice_rank > best_min_rank or cost_comparison > 0:
                    continue
                families = [
                    fam for fam in families
                    if int(fam.get("free_group_rank", 0)) == best_min_rank
                ]
            # De-duplicate anchors within this nondim choice by K_T expression
            seen_KT_expr = set()
            for fam in families:
                K_T_expr = fam.get("K_family", {}).get("K_T", None)
                if K_T_expr is None:
                    continue
                if K_T_expr in seen_KT_expr:
                    continue
                seen_KT_expr.add(K_T_expr)

                # Exact lexicographic reduction of worker-scored final families.
                # Only current minima are retained; completion order cannot
                # change this minimum or the established deterministic tie order.
                symbolic_score = tuple(int(fam[k]) for k in
                                       ("L1_cost", "gen_L1_cost", "time_L1_cost"))
                if best_symbolic_score is None or symbolic_score < best_symbolic_score:
                    best_symbolic_score = symbolic_score
                    best_families = []
                elif symbolic_score > best_symbolic_score:
                    continue

                # Attach the level-1 objective bound (noting that the objective is Rstar/log(10))
                if search_all_assignments and nondim_metric == "deviation":
                    fam["max_objective"] = best_dimensional_cost[0] / float(np.log(10))
                else:
                    fam["max_objective"] = Rstar/float(np.log(10)) if Rstar is not None else None

                # Numeric score based on |log10(T_max / K_T)| whenever possible
                if (k_numeric is not None) and (T_max is not None):
                    try:
                        subs_numeric: Dict[sp.Symbol, float] = {}
                        subs_numeric.update({k: float(v) for k, v in k_numeric.items()})
                        if extra_rhs_numeric:
                            subs_numeric.update({s: float(v) for s, v in extra_rhs_numeric.items()})
                        K_T_val = float(sp.N(K_T_expr.subs(subs_numeric)))
                        if (K_T_val is None) or not (K_T_val > 0.0):
                            score_T = float("inf")
                        else:
                            score_T = abs(np.log10(float(T_max) / float(K_T_val)))
                    except Exception:
                        K_T_val = None
                        score_T = float("inf")

                    fam["K_T_numeric"] = K_T_val
                    fam["objective_T_ratio"] = score_T
                best_families.append(fam)
    if best_families:
        try:
            min_F = min(f.get("L1_cost", float("inf")) for f in best_families)
            best_families = [f for f in best_families if f.get("L1_cost", float("inf")) == min_F]
        except Exception:
            pass
        try:
            min_gen_F = min(f.get("gen_L1_cost", float("inf")) for f in best_families)
            best_families = [f for f in best_families if f.get("gen_L1_cost", float("inf")) == min_gen_F]
        except Exception:
            pass
        try:
            min_time_F = min(f.get("time_L1_cost", float("inf")) for f in best_families)
            best_families = [f for f in best_families if f.get("time_L1_cost", float("inf")) == min_time_F]
        except Exception:
            pass
    # Compute numeric bests based on |log10(T_max / K_T)|
    if best_families:
        finite = [
            f for f in best_families
            if np.isfinite(f.get("objective_T_ratio", float("inf")))
        ]
        if finite:
            min_score = min(f["objective_T_ratio"] for f in finite)
            tol = 1e-12  # tie tolerance
            all_minimisers = [
                f for f in finite
                if abs(f["objective_T_ratio"] - min_score) <= tol
            ]

            # Deterministic pick among ties
            def _key_absbest(f: Dict[Any, Any]):
                """
                Helper to pick the 'best' key based on absolute objective value.

                Args:
                    f (Dict[Any, Any]): Mapping from candidate key to a result
                        object containing an objective value (e.g. error magnitude).

                Returns:
                    Any: The key corresponding to the best (usually minimum) objective
                    value according to the selection rule used in this helper.
                """
                # Inspect result entries and return a key for the best objective value.
                return (
                    f.get("objective_T_ratio", float("inf")),
                    f.get("normalised_block_index", 0),
                    str(f.get("partition", "")),
                )

            absolute_best = sorted(all_minimisers, key=_key_absbest)[0]
            # Place the absolute best first so the example banner prints it
            best_families = [absolute_best]
    # Report the selected family's coefficient and generator counts.
    final_families = best_families
    # Report BOTH minima
    if final_families:
        min_unique = final_families[0]["unique_monomial_count"]
        # Ensure best_min_rank is an integer
        best_min_rank = final_families[0]["free_group_rank"]
    else:  # Should not happen if best_families was not empty, but handle defensively
        min_unique = 0
        best_min_rank = 0
    return _finish({
        "num_unique_coefficients": min_unique,
        "num_generators": best_min_rank,
        "families": final_families,
        "P": P,
        "nu": nu,
        "rate_constants": ks,
        "nondim_metric": nondim_metric,
        "assignment_parallelism": assignment_stats,
        "_S": S, "_X": X
    })
