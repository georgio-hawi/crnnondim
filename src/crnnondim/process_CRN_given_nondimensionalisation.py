from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
import sympy as sp
import numpy as np
from .factor_generators_columnwise import factor_generators_columnwise
from .utils import (
    _build_key_from_vec_global,
    _get_basis,
    _infer_ks_from_V,
    _snf_with_transforms_int_cached,
    _unique_ordered,
    build_P,
    monomial_str,
    parse_mass_action,
    has_all_zero_row_or_col,
    ensure_constant_stoichiometry,
    _parse_monomial_key_to_exponents,
    export_crnnondim_results,
    _normalise_dimensionally_consistent_sets,
    _scale_dimension_groups,
    _validate_scale_symbols,
    relabel_generators_by_first_use,
)

def process_CRN_given_nondimensionalisation(
    S: sp.Matrix,
    V,
    X: List[sp.Symbol],
    nondim: Dict[sp.Symbol, sp.Symbol],
    basis_type: str = "Laurent",
    return_all_families: bool = True,
    export_results: bool = False,
    output_path: Optional[Union[str, Path]] = "out",
    dimensionally_consistent_sets: Optional[List[set[sp.Symbol]]] = None,
) -> Dict[str, Any]:
    """
    Reprocess a CRN for a specific nondimensionalisation choice.

    Args:
        S (sp.Matrix): Stoichiometric matrix of the original CRN.
        V: Reaction monomials; sequence of sympy expressions or sp.Matrix.
        X (List[sp.Symbol]): Species symbols corresponding to rows of S.
        nondim (Dict[sp.Symbol, sp.Symbol]): Mapping species_symbol -> chosen
            scale symbol (one particular nondimensionalisation).
        basis_type (str): Type of exponent basis ("Laurent" or "Polynomial") controlling
            whether negative exponents are allowed.
        return_all_families (bool): Return all families tied at the best
            evaluated score, or one representative when False.
        export_results (bool): If True, export the returned dictionary to a pickle
            file.
        output_path (Optional[str | Path]): File path (or directory) for the pickle.
            Directory targets use {model_name}_{nondim_metric}_nondim.pkl;
            missing metadata defaults to "crnnondim" and "deviation".
        dimensionally_consistent_sets (Optional[List[set[sp.Symbol]]]):
            Partition of X into sets whose species scales have compatible units.
            Each species must occur exactly once. If None, all species are
            assumed to have compatible units.

    Returns:
        Dict[str, Any]: Same general structure as the output of process_CRN,
        but restricted to the provided nondimensionalisation (families,
        generators, exponent matrices, and rate-constant factoring results).
    """

    # --- Input validation: basis_type and nondim ----------------------------------
    allowed_basis_types = ("Laurent", "Polynomial")
    if basis_type not in allowed_basis_types:
        raise ValueError("basis_type must be either 'Laurent' or 'Polynomial'.")
    
    # Nondim should be a mapping from species in X to scaling symbols.
    if not hasattr(nondim, "keys") or not hasattr(nondim, "values"):
        raise ValueError("nondim must be a mapping from species in X to scaling symbols.")

    # All keys in nondim must be species in X.
    if not set(nondim.keys()).issubset(set(X)):
        raise ValueError("All keys in 'nondim' must be species in X.")

    missing = [str(xi) for xi in X if xi not in nondim]
    if missing:
        raise ValueError("Species missing from 'nondim' mapping: " + ", ".join(missing) + ".")
    _validate_scale_symbols(nondim.values(), X, V)
    
    # Plug the chosen nondimensionalisation into the symbolic structures,
    # rebuild transformed exponent matrices and families, and return the
    # resulting CRN representation.
    # --- Preprocessing ---
    S = ensure_constant_stoichiometry(S)
    if has_all_zero_row_or_col(S):
        raise ValueError("All rows and all columns of the stoichiometry matrix, S, must be non-zero.")
    if S.rows != len(X):
        raise ValueError(f"Dimension mismatch: S has {S.rows} rows but X has {len(X)} species.")
    if S.cols != len(V):
        raise ValueError(f"Dimension mismatch: S has {S.cols} columns but there are {len(V)} reactions in V.")
    ks = _infer_ks_from_V(V, X)
    nu, _ = parse_mass_action(V, X, ks)
    nu_list = nu.tolist()
    P = build_P(S)
    n, r = P.rows, P.cols

    # Active coefficient slots
    E = sorted([(i, j) for i in range(n) for j in range(r) if P[i, j] != 0])
    if not E:
        return {"num_unique_coefficients": 0, "num_generators": 0, "families": [], "P": P, "nu": nu, "rate_constants": ks, "_S": S, "_X": X}
    E_map = {pair: idx for idx, pair in enumerate(E)}
    mE = len(E)

    allow_negatives = (basis_type == "Laurent")

    # Build coeffs in log-space: log c_ij = log kappa_j + C_row * x, where x = log(K)
    C = sp.zeros(mE, n)
    for idx, (i, j) in enumerate(E):
        C[idx, i] -= 1
        for l in range(n):
            C[idx, l] += nu_list[l][j]

    # Vectorised precompute: integer exponents for species K_l in each active (i,j)
    _E_exp = np.zeros((mE, n), dtype=object)
    for _eidx, (_i, _j) in enumerate(E):
        jj = int(_j)
        _E_exp[_eidx, :] = [int(nu_list[l][jj]) for l in range(n)]
        _E_exp[_eidx, _i] -= 1

    # Integer basis bookkeeping
    k_unique = _unique_ordered(ks)
    rks = len(k_unique)
    k_slot = {s: i for i, s in enumerate(k_unique)}        # symbol -> unique slot
    kpos_of_reaction = [k_slot[ks[j]] for j in range(r)]

    # Precompute k-column index for each active edge eidx (one-hot position in the 'k' block)
    _kcols = np.fromiter((kpos_of_reaction[E[_e][1]] for _e in range(mE)), dtype=int, count=mE)

    # base_syms come from nondim's values, in stable order
    base_syms = _unique_ordered([nondim[xi] for xi in X])
    pbs = len(base_syms)
    sym_index = {s: idx for idx, s in enumerate(base_syms)}  # s -> [0..p-1]

    # --- Single "choice": the provided mapping ---
    fixed_full = {}
    for i, xi in enumerate(X):
        fixed_full[i] = nondim[xi]

    _, species_group = _normalise_dimensionally_consistent_sets(
        X, dimensionally_consistent_sets
    )
    _scale_dimension_groups(
        {X[i]: scale for i, scale in fixed_full.items()}, species_group
    )

    # --- Calculate v_e, partition, and Rank ---
    families = []
    _sp_pos = np.fromiter((sym_index[fixed_full[i]] for i in range(n)), dtype=int, count=n)
    Vabs = np.zeros((mE, rks + pbs), dtype=object)
    Vabs[np.arange(mE), _kcols] = 1

    #Calculate valuations (we consider transpose(A), noting all results are identical to considering A)
    H = np.zeros((n, pbs), dtype=int)
    H[np.arange(n), _sp_pos] = 1
    Vabs[:, rks:] = _E_exp @ H

    v_e = [tuple(int(x) for x in Vabs[row].tolist()) for row in range(mE)]

    # TRUE partition
    part_map = {}
    for eidx, vec in enumerate(v_e):
        part_map.setdefault(vec, []).append(E[eidx])
    true_partition = list(part_map.values())
    true_unique_count = len(true_partition)

    # --- Calculate free rank via integer SNF of DeltaG ---
    _anchor0_vec = Vabs[0]
    DeltaG = Vabs - _anchor0_vec
    diff_rows = [tuple(int(x) for x in DeltaG[row].tolist()) for row in range(mE)]
    Mq = sp.Matrix(diff_rows)

    if diff_rows:
        rows_tuple = tuple(tuple(row) for row in diff_rows)
        _, D, _ = _snf_with_transforms_int_cached(rows_tuple)
        diag_len = min(D.rows, D.cols)
        free_rank = sum(1 for t in range(diag_len) if D[t, t] != 0)
    else:
        D = None
        free_rank = 0

    # Exact coordinates in a basis of the INTEGER row lattice. A rational
    # rowspace basis alone need not span this lattice.
    coords = None
    B = sp.zeros(0, Mq.cols)
    if free_rank > 0:
        basis_int = []
        for vec in Mq.rowspace()[:free_rank]:
            lcm = 1
            for value in vec:
                lcm = sp.ilcm(lcm, int(sp.denom(value)))
            basis_int.append([int(value * lcm) for value in vec])
        B = sp.Matrix(basis_int)
        from sympy.matrices.normalforms import hermite_normal_form
        lattice_basis = hermite_normal_form(Mq.T)
        if hermite_normal_form(B.T) != lattice_basis:
            B = lattice_basis.T
        if B.rows != free_rank:
            raise RuntimeError("Integer row-lattice rank does not match the Smith rank.")
        Z_num, Z_den = B.T.to_DM().solve_den(Mq.T.to_DM())
        Z_matrix = Z_num.to_Matrix() / Z_num.domain.to_sympy(Z_den)
        coords = []
        for j in range(mE):
            z = list(Z_matrix.col(j))
            if not all(value.is_Integer for value in z):
                raise ValueError("Non-integral coordinates in the integer lattice basis.")
            coords.append(tuple(map(int, z)))

    # Solve K-species symbols & x_sol (logs)
    K_species_expr = sp.Matrix([fixed_full[i] for i in range(n)])
    x_sol = sp.Matrix([sp.log(K_species_expr[i]) for i in range(n)])

    # Generator definitions are independent of the normalised coefficient block.
    # Build them once and keep them by index, avoiding repeated string-key work.
    all_parameters = _unique_ordered(list(ks) + list(base_syms))
    generator_exprs = [
        sp.prod(symbol**int(exponent) for symbol, exponent in zip(all_parameters, row))
        for row in B.tolist()
    ]

    # Build families normalised on each block
    for block_idx, block in enumerate(true_partition):
        anchor_ij = block[0]
        i_anchor, j_anchor = anchor_ij

        # tau / K_T
        C_row_anchor_sym = C[E_map[(i_anchor, j_anchor)], :]
        k_j_sym = ks[j_anchor]
        log_kj = sp.log(k_j_sym)
        tau_expr = -log_kj - C_row_anchor_sym.dot(x_sol)
        K_T_expr = sp.exp(tau_expr)

        # Generator labels remain fresh per family, as in the original output.
        gens = [f"g_{i+1}" for i in range(free_rank)]

        # Coefficient classes relative to anchor
        classes_by_gens = {}
        key_order = []
        vecs = []
        if free_rank > 0 and coords:
            shift = coords[E_map[(i_anchor, j_anchor)]]
            for eidx in range(mE):
                exp_vec = tuple(ci - si for ci, si in zip(coords[eidx], shift))
                key = monomial_str(exp_vec, gens)
                if key not in classes_by_gens:
                    key_order.append(key)
                    vecs.append(exp_vec)
                classes_by_gens.setdefault(key, []).append(E[eidx])
        else:
            classes_by_gens = {'1': [E[eidx] for eidx in range(mE)]}
            key_order = ['1']
            vecs = [tuple(0 for _ in gens)] if gens else []

        fam_partition = true_partition

        def _shift_to_nonnegative(vecs_in: List[Tuple[int, ...]]) -> Tuple[List[Tuple[int, ...]], Tuple[int, ...]]:
            """
            Shift a list of exponent vectors so all entries become nonnegative.

            Args:
                vecs_in (List[Tuple[int, ...]]): Integer exponent vectors for generators.

            Returns:
                Tuple[List[Tuple[int, ...]], Tuple[int, ...]]:
                    A pair (shifted_vecs, shift) where
                    shifted_vecs are the transformed exponent vectors and
                    shift is the componentwise offset applied to each original vector.
            """
            # Compute the componentwise minimum and subtract it so everything is >= 0.
            if not vecs_in:
                return vecs_in, tuple()
            d = len(vecs_in[0])
            mins = tuple(int(min(v[i] for v in vecs_in)) for i in range(d))
            shifted = [tuple(int(v[i] - mins[i]) for i in range(d)) for v in vecs_in]
            return shifted, mins
        
        # Transform to printing basis; ensure non-negative exponents by per-coordinate shift
        T, vecs_T, Am = (None, [], None)
        if vecs:
            if not allow_negatives:
                from ._polynomial_basis import polynomial_basis
                T, vecs_T, Am = polynomial_basis(
                    vecs, [len(classes_by_gens[key]) for key in key_order],
                    generator_exprs, K_T_expr,
                )
            else:
                # F_mono counts active edges, including repeated classes.
                T, vecs_T, Am = _get_basis([
                    v for key, v in zip(key_order, vecs)
                    for _ in classes_by_gens[key]
                ])
                vecs_T = list(dict.fromkeys(vecs_T))

        shift_vec = tuple(0 for _ in gens)
        if vecs_T:
            if allow_negatives == False:
                if any(e < 0 for v in vecs_T for e in v):
                    vecs_T, shift_vec = _shift_to_nonnegative(vecs_T)
        else:
            vecs_T, shift_vec = _shift_to_nonnegative(vecs)

        classes_print = {}
        if T:
            for key, vT in zip(key_order, vecs_T):
                new_key = _build_key_from_vec_global(vT, gens)
                where = classes_by_gens[key]
                for edge in where:
                    classes_print.setdefault(new_key, []).append(edge)
        else:
            classes_print = dict(classes_by_gens)

        # A strictly worse primary score cannot survive the final filtering.
        if families:
            best_L1 = min(f["L1_cost"] for f in families)
            candidate_L1 = sum(
                len(edges) * sum(abs(e) for e in _parse_monomial_key_to_exponents(key, gens))
                for key, edges in classes_print.items()
            )
            if candidate_L1 > best_L1:
                continue

        # Generator expressions in printing basis
        generator_expressions = []
        if Am is not None:
            gen_exprs_print = {}
            for i_gen in range(len(gens)):
                expr = sp.Integer(1)
                for j in range(len(gens)):
                    e = int(Am[j, i_gen])
                    if e:
                        base = generator_exprs[j]
                        expr *= base**e
                gen_exprs_print[gens[i_gen]] = sp.simplify(expr)
            generator_expressions = [gen_exprs_print[g] for g in gens]
        else:
            generator_expressions = list(generator_exprs)
        # Absorb global shift of generator exponents into K_T, if any
        if any(shift_vec):
            shift_factor = sp.Integer(1)
            for g_expr, m in zip(generator_expressions, shift_vec):
                if m != 0:
                    shift_factor *= g_expr**(-m)
            K_T_expr = sp.simplify(K_T_expr * shift_factor)
        classes_by_gens = classes_print

        # Build final dict (omit any numeric fields)
        fam_dict = {
            'partition': fam_partition, 'normalised_block_index': block_idx,
            'K_family': {'K_species': K_species_expr, 'K_T': K_T_expr},
            'unique_monomial_count': true_unique_count, 'free_group_rank': free_rank,
            'generators': gens, 'generator_expressions': generator_expressions,
            'coeff_classes_by_generators': classes_by_gens,
        }

        #compute overall monomial sparsity F_mono and generator sparsity F_gen on the representation
        try:
            classes = fam_dict.get("coeff_classes_by_generators", {}) or {}
            gens_list = fam_dict.get("generators", []) or []
            generator_expressions = fam_dict.get("generator_expressions", []) or []
            K_T_expr = fam_dict.get("K_family", []).get("K_T",[]) or []
            L1_cost = 0
            for mkey, edges in classes.items():
                z = _parse_monomial_key_to_exponents(mkey, gens_list)
                # number of edges in the class x L1 norm of exponent vector
                L1_cost += (len(edges) * sum(abs(x) for x in z))
            fam_dict["L1_cost"] = int(L1_cost)
            def exponent_L1(expr):
                powers = sp.expand_power_base(expr, force=True).as_powers_dict()
                return sum(abs(int(powers.get(sym, 0))) for sym in all_parameters)

            fam_dict["gen_L1_cost"] = sum(exponent_L1(gen) for gen in generator_expressions)
            fam_dict["time_L1_cost"] = exponent_L1(K_T_expr)
        except Exception as exc:
            raise ValueError("Cannot compute exact monomial sparsity scores for a nondimensionalisation family.") from exc
        families.append(fam_dict)
        
    # Exact lexicographic reduction; malformed scores must not be ignored.
    if families:
        def score(family):
            return tuple(family[key] for key in ("L1_cost", "gen_L1_cost", "time_L1_cost"))
        best_score = min(map(score, families))
        families = [family for family in families if score(family) == best_score]
    if families and not return_all_families:
        families = [families[0]]
    # Establish first-use labels when creating the result, not while printing it.
    for index, family in enumerate(families):
        Stilde, Vtilde = factor_generators_columnwise(S, family, k_unique, nu)
        _, _, families[index] = relabel_generators_by_first_use(Stilde, Vtilde, family)
    if families:
        min_unique = min(f.get("unique_monomial_count", 0) for f in families)
        min_free_rank = min(f.get("free_group_rank", 0) for f in families)
    else:
        min_unique = 0
        min_free_rank = 0

    result = {
        "num_unique_coefficients": int(min_unique),
        "num_generators": int(min_free_rank),
        "families": families,
        "P": P,
        "nu": nu,
        "rate_constants": k_unique,
        "_S": S,
        "_X": X,
    }
    if export_results:
        export_crnnondim_results(result, output_path=output_path, S=S, V=V, X=X)
    return result
