from .factor_generators_columnwise import factor_generators_columnwise
from .utils import format_sympy_matrix, relabel_generators_by_first_use
import sympy as sp

def _format_expr_generators(expr: sp.Expr,
                            rate_symbols: list[sp.Symbol],
                            species_scale_symbols: list[sp.Symbol]) -> str:
    """
    Format a generator expression as a string with ordered factors.

    Within each multiplicative term, factors are ordered as:
      1) rate constants, in the order given by rate_symbols
      2) species-scale symbols, in the order given by species_scale_symbols
      3) any remaining factors (kept in SymPy's default order)

    The overall rational expression is printed as "num/den".
    If the denominator simplifies to 1, only the numerator string is returned.
    If the denominator contains multiple factors, it is wrapped in parentheses.

    Args:
        expr (sp.Expr): SymPy expression for a generator coefficient.
        rate_symbols (list[sp.Symbol]): Ordered list of rate-constant symbols
            (defines the sort order for all rate factors).
        species_scale_symbols (list[sp.Symbol]): Ordered list of species-scale
            symbols (defines the sort order for all scale factors).

    Returns:
        str: A string representation of expr with factors ordered as specified.
    """
    # Normalise expression and split into numerator/denominator.
    expr = sp.simplify(expr)
    num, den = sp.fraction(expr)

    # Pull out multiplicative factors from numerator/denominator.
    def factors(e): return list(sp.Mul.make_args(e))
    num_factors = factors(num); den_factors = factors(den)

    # Categorize factors into (rates, scales, other).
    def categorize(factors):
        rates, scales, other = [], [], []
        for f in factors:
            base = f.base if isinstance(f, sp.Pow) else f
            if isinstance(base, sp.Symbol) and base in rate_symbols:
                rates.append(f)
            elif isinstance(base, sp.Symbol) and base in species_scale_symbols:
                scales.append(f)
            else:
                other.append(f)
        return rates, scales, other

    nr, ns, no = categorize(num_factors)
    dr, ds, do = categorize(den_factors)

    # Build indices for stable ordering within each category.
    rate_index = {s: i for i, s in enumerate(rate_symbols)}
    scale_index = {s: i for i, s in enumerate(species_scale_symbols)}

    # Sort numerator/denominator categories by their declared orders.
    keyr = lambda f: rate_index[f.base] if isinstance(f, sp.Pow) else rate_index[f]
    keys = lambda f: scale_index[f.base] if isinstance(f, sp.Pow) else scale_index[f]
    nr.sort(key=keyr); ns.sort(key=keys)
    dr.sort(key=lambda f: rate_index.get(f.base, 10**9) if isinstance(f, sp.Pow) else rate_index.get(f, 10**9))
    ds.sort(key=lambda f: scale_index.get(f.base, 10**9) if isinstance(f, sp.Pow) else scale_index.get(f, 10**9))

    # Join factors back into strings; omit empty parts.
    def join(fs): return "*".join(f"({x})" if isinstance(x, sp.Add) else str(x) for x in fs) if fs else ""
    nums = [join(nr), join(ns), join(no)]
    dens = [join(dr), join(ds), join(do)]
    num_str = "*".join([s for s in nums if s])
    den_str = "*".join([s for s in dens if s])

    # Render as "num/den", applying small formatting conventions.
    if not den_str or den_str == "1":
        return num_str if num_str else "1"
    if "*" in den_str:
        return f"{num_str}/(" + den_str + ")"
    else:
        return f"{num_str}/" + den_str

def analyse_results(all_results, verbose=True):
    """
    Summarise and pretty-print nondimensionalisation results.

    Args:
        all_results (dict | list[dict]): Either a single result
            dictionary or a list/iterable of such dictionaries,
            typically as returned by the CRN processing pipeline.
        verbose (bool): Print out nondimensionalisation results
            (defaults to True).

    Returns:
        all_transformed_results: A list of tuples (Stilde, Vtilde)
            for each family in all_results, where
            Stilde is the transformed stoichiometry matrix, and
            Vtilde is transformed reaction vector.
    """
    # Normalise to a list so the following loop works for both single / multiple results.
    if isinstance(all_results, dict):
        all_results = [all_results]

    all_transformed_results = []
    # Pretty-print a summary of the optimal non-dimensionalisati...amilies found and the corresponding non-dimensionalised systems.
    for processed_CRN_dict in all_results:
        for i,f in enumerate(processed_CRN_dict['families'],1):
            if verbose:
                if "max_objective" in f and f["max_objective"] is not None:
                    print("Minimum max O(1) penalty:", f.get("max_objective"))
                print("Number of unique monomial coefficients:", processed_CRN_dict["num_unique_coefficients"])
                print("Number of generators:", processed_CRN_dict["num_generators"])

                # Species and time scaling factors.
                print("K_i:", f["K_family"]["K_species"].T)
                # grab the symbolic K_T
                KT_expr = f["K_family"]["K_T"]
                # use the same ordered symbol lists as for generators
                rate_syms = processed_CRN_dict["rate_constants"]
                scale_syms = list(f.get("K_species", {}).values())
                # print with consistent factor ordering
                print("K_T:", _format_expr_generators(KT_expr, rate_syms, scale_syms))
                if "K_T_numeric" in f and f["K_T_numeric"] is not None:
                    print("K_T (numeric):", f.get("K_T_numeric"))

            # Compute the transformed (nondimensional) stoichiometry matrix and reaction vector.
            Stilde, Vtilde = factor_generators_columnwise(processed_CRN_dict["_S"], f, processed_CRN_dict["rate_constants"], processed_CRN_dict["nu"])
            Stilde, Vtilde, f = relabel_generators_by_first_use(Stilde, Vtilde, f)
            processed_CRN_dict["families"][i-1] = f
            if verbose:
                # Generator (free parameter) definitions.
                gens = f.get("generators", [])
                gen_exprs_print = {g: e for g, e in zip(gens, f.get("generator_expressions", []))}
                print("Generator definitions (in terms of rate constants and species scales):")
                if gens:
                    for g in gens:
                        if g in gen_exprs_print:
                            print(f"  {g} = " + _format_expr_generators(gen_exprs_print[g], processed_CRN_dict["rate_constants"], list(f.get("K_species", {}).values())))
                        else:
                            print(f"  {g} = (unavailable)")
                else:
                    print("  (none)")
                    
                # Explain the substitution from dimensional species to nondimensional x_i.
                print("Denoting [" + ", ".join(f"x_{i+1}" for i in range(len(processed_CRN_dict["_X"]))) + "] = [" + ", ".join(f"{processed_CRN_dict['_X'][i]}/K_{i+1}" for i in range(len(processed_CRN_dict["_X"])))+"], then the non-dimensionalised system is given by dx/dt = Stilde*Vtilde with")

                # Pretty-print the transformed stoichiometry matrix and reaction vector (the full non-dimensionalised 
                # system has equations dx/dt = Stilde*Vtilde).
                print(format_sympy_matrix(Stilde, "Stilde"))
                print()
                print(format_sympy_matrix(Vtilde, "Vtilde"))
            all_transformed_results.append((Stilde, Vtilde))
            if verbose:
                try:
                    print(f'F_mono: {f.get("L1_cost")}')
                except:
                    pass
                try:
                    print(f'F_gen: {f.get("gen_L1_cost")}')
                except:
                    pass
                try:
                    print(f'F_time: {f.get("time_L1_cost")}')
                except:
                    pass
    return all_transformed_results