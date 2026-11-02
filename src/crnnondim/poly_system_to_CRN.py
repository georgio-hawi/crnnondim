import sympy as sp
from sympy.polys.polyerrors import PolynomialError
from typing import Any, Dict, Sequence
from .utils import format_sympy_matrix, _build_rate_and_species_orders, _format_factor_ordered
def poly_system_to_CRN(
    _X: Sequence[sp.Symbol],
    ode_rhs: Sequence[sp.Expr],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Infer a mass-action-style CRN representation (S, V, X, parameters)
    from a polynomial ODE system x' = f(x).

    Args:
        _X (Sequence[sp.Symbol]):
            Species symbols, in the same order as the ODEs ode_rhs.
            That is, ode_rhs[i] is d(_X[i])/dt.

        ode_rhs (Sequence[sp.Expr]):
            Right-hand sides [f1, ..., fn] for x' = f(x),
            with fi corresponding to d(_X[i])/dt.

        verbose (bool): If True, print out information about the corresponding CRN.
    Returns:
        Dict[str, Any]: Dictionary with keys matching build_mass_action_system:

            - 'species' (List[str]): Species names in the supplied order.
            - '_S' (sp.Matrix): Stoichiometric matrix (n_species x n_reactions),
                                 with rational entries. Floating-point numerical
                                 coefficients are converted to exact rationals.
            - '_X' (List[sp.Symbol]): Species symbols in the supplied order.
            - '_V' (sp.Matrix): Column vector of reaction-rate expressions V(x).
            - 'rate_constants' (Tuple[sp.Symbol, ...]): Tuple of distinct
              rate symbols that appear in the entries of V(x) and are not
              species.

        The ordering of reactions (columns) is not guaranteed to match any
        external list; it is determined by the order in which distinct rate
        monomials are encountered while scanning the ODEs.
    """
    if len(_X) != len(ode_rhs):
        raise ValueError("Length of _X and ode_rhs must match (one ODE per species).")

    # Sanity check: all right-hand sides must be polynomial in the species
    for ode in ode_rhs:
        try:
            sp.Poly(ode, *_X)
        except PolynomialError:
            raise ValueError("All ODEs must be polynomial")

    # ---------------------------------------------------------------------
    # 1. Preserve the supplied species order and aligned ODEs.
    # ---------------------------------------------------------------------

    species_syms = list(_X)
    rhs_sorted   = list(ode_rhs)  # already aligned with _X

    species_names = [s.name for s in species_syms]
    n = len(species_syms)

    # ---------------------------------------------------------------------
    # 2. Decompose each fi into rational-coefficient * rate_expr(x)
    #    and group terms with the same rate_expr across all equations.
    #    This gives a rational stoichiometric matrix S and a vector V(x)
    #    of rate expressions.
    # ---------------------------------------------------------------------
    rate_to_index = {}      # rate_expr -> column index
    rate_exprs = []         # list of distinct rate_expr (V entries)
    stoich_cols = []        # list of columns, each is length n (species)

    for row, expr in enumerate(rhs_sorted):
        expr = sp.expand(expr)

        # SymPy represents a zero RHS as [0]; it contributes no reaction.
        for term in expr.as_ordered_terms():
            if term.is_zero:
                continue
            coeff_num, rest = term.as_coeff_Mul()
            coeff_num = sp.simplify(coeff_num)

            # Keep all numerical factors in S, as required by parse_mass_action.
            # This preserves shared rate symbols instead of introducing new
            # independent parameters. Rational(Float) preserves the stored value.
            stoich_coeff = sp.Rational(coeff_num)
            base_expr = sp.simplify(rest)

            # Register reaction / rate expression
            col = rate_to_index.get(base_expr)
            if col is None:
                col = len(rate_exprs)
                rate_to_index[base_expr] = col
                rate_exprs.append(base_expr)
                stoich_cols.append([0] * n)  # one row per species

            stoich_cols[col][row] += stoich_coeff

    # Number of reactions = number of distinct rate expressions
    m = len(rate_exprs)

    # Build S as an n x m matrix (rows: species, columns: reactions)
    S = sp.Matrix(n, m, lambda i, j: stoich_cols[j][i])

    # Column vector V(x) of rate expressions
    V = sp.Matrix(m, 1, rate_exprs)

    # ---------------------------------------------------------------------
    # 3. Infer rate constants as all non-species symbols appearing in V(x),
    #    optionally augmented by the user-provided "rate_constants" hint.
    # ---------------------------------------------------------------------
    species_set = set(species_syms)

    # Symbols in V(x) that are not species are parameters
    params_in_V = set()
    for vj in V:
        for sym in vj.free_symbols:
            if sym not in species_set:
                params_in_V.add(sym)

    final_params = tuple(sorted(params_in_V, key=lambda s: s.name))

    # ---------------------------------------------------------------------
    # 4. Assemble the result dict with the same structure as
    #    build_mass_action_system.
    # ---------------------------------------------------------------------
    
    # Print a small summary of the parsed CRN.
    if verbose:
        print(f"Species names: {species_names}")
        print(format_sympy_matrix(S, "Stoichiometry matrix",':'))
        rate_order, x_index = _build_rate_and_species_orders(V, _X)
        print(format_sympy_matrix(V, "Reaction vector", ':', elem_formatter=lambda e: _format_factor_ordered(e, rate_order, x_index)))
        print(f"Unique rate constants: {final_params}")

    return {
        "species": species_names,
        "_S": S,
        "_X": species_syms,
        "_V": V,
        "rate_constants": final_params,
    }