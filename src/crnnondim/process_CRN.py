from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
import sympy as sp
from .CRN_analyser import analyse_crn, _assignment_from_structural_plan
from .utils import (
    _infer_ks_from_V,
    _unique_ordered,
    as_matrix,
    build_P,
    parse_mass_action,
    has_all_zero_row_or_col,
    ensure_constant_stoichiometry,
    export_crnnondim_results,
    _normalise_dimensionally_consistent_sets,
    _scale_dimension_groups,
)
from ._LEXIWrapper import _LEXIWrapper
from .process_CRN_given_nondimensionalisation import process_CRN_given_nondimensionalisation

def process_CRN(
    S: sp.Matrix,
    V,
    X: List[sp.Symbol],
    fixed_K: Optional[Dict[sp.Symbol, sp.Symbol]] = None,
    find_example_nondim: bool = True,
    verbose: bool = False,
    basis_type: str = "Laurent",
    export_results: bool = False,
    output_path: Optional[Union[str, Path]] = "out/",
    dimensionally_consistent_sets: Optional[List[set[sp.Symbol]]] = None,
) -> Dict[str, Any]:
    """
    Structural preprocessing of a CRN before searching for nondimensionalisation.

    Args:
        S (sp.Matrix): Stoichiometric matrix of the CRN.
        V: Reaction monomials; can be a sequence of sympy expressions or an
            sp.Matrix column vector.
        X (List[sp.Symbol]): Species symbols corresponding to rows of S.
        fixed_K (Optional[Dict[sp.Symbol, sp.Symbol]]): Optional mapping
            species_symbol -> fixed scale symbol to encode prior choices.
        find_example_nondim (bool): If True, also construct one example
            nondimensionalisation family (symbolically).
        verbose (bool): If True, print diagnostic banners and summaries.
        basis_type (str): Type of exponent basis ("Laurent" or "Polynomial") controlling
            whether negative exponents are allowed.
        export_results (bool): If True, export the returned dictionary to a pickle file.
        output_path (Optional[str | Path]): File path (or directory) for the pickle.
        dimensionally_consistent_sets (Optional[List[set[sp.Symbol]]]):
            Partition of X into sets whose species scales have compatible units.
            Each species must occur exactly once. If None, all species are
            assumed to have compatible units.

    Returns:
        Dict[str, Any]: Bundle including S, V, X, inferred rate constants,
        nonzero pattern P, exponent matrix nu, connected-component info,
        scale-constraint data, internal assignment analysis when required, and
        optionally an example family structure. With dimensional constraints,
        min_theor_gen is a theoretical lower bound, not necessarily attainable.
    """

    def _finish(result: Dict[str, Any]) -> Dict[str, Any]:
        if export_results:
            export_crnnondim_results(result, output_path=output_path, S=S, V=V, X=X)
        return result
    
    # --- Input validation: basis_type ---------------------------------------------
    allowed_basis_types = ("Laurent", "Polynomial")
    if basis_type not in allowed_basis_types:
        raise ValueError("basis_type must be either 'Laurent' or 'Polynomial'.")
    
    # Infer rate constants and exponents from V, then call analyse_crn to
    # obtain connected-component and generator information for this CRN.
    # --- Basic preprocessing ---
    S = ensure_constant_stoichiometry(S)
    if has_all_zero_row_or_col(S):
        raise ValueError("All rows and all columns of the stoichiometry matrix, S, must be non-zero.")
    if S.rows != len(X):
        raise ValueError(f"Dimension mismatch: S has {S.rows} rows but X has {len(X)} species.")
    if S.cols != len(V):
        raise ValueError(f"Dimension mismatch: S has {S.cols} columns but there are {len(V)} reactions in V.")
    ks = _infer_ks_from_V(V, X)
    nu, _ = parse_mass_action(V, X, ks)
    P = build_P(S)
    n = P.rows

    # --- Build fixed index mapping from species->scale_symbol ---
    fixed_K = fixed_K or {}
    if (not hasattr(fixed_K, "items") or not set(fixed_K).issubset(set(X))
            or not all(isinstance(scale, sp.Symbol) for scale in fixed_K.values())):
        raise ValueError("fixed_K must map species in X to SymPy scale symbols.")
    if len(set(map(str, fixed_K.values()))) != len(set(fixed_K.values())):
        raise ValueError("Distinct scale symbols must have distinct names.")
    # species index -> symbol
    fixed_K_idx = {i: fixed_K[X[i]] for i in range(n) if X[i] in fixed_K}
    base_syms = _unique_ordered([fixed_K_idx[i] for i in sorted(fixed_K_idx.keys())])
    dimensional_sets, species_group = _normalise_dimensionally_consistent_sets(
        X, dimensionally_consistent_sets
    )
    scale_group = _scale_dimension_groups(
        {X[i]: scale for i, scale in fixed_K_idx.items()}, species_group
    )
    has_dimensional_restrictions = len(dimensional_sets) > 1

    # --- Analyse components & constraints ---
    out = analyse_crn(S, V, X, K_fixed=fixed_K_idx)
    # Build restrictions array exactly like find_optimal_nondim
    restrictions = [None]*n
    labels = [str(s) for s in base_syms]
    label_to_sym = {str(s): s for s in base_syms}
    # fixed indices
    for i, s in fixed_K_idx.items():
        restrictions[i] = str(s)
    if has_dimensional_restrictions:
        for i in range(n):
            if i not in fixed_K_idx:
                restrictions[i] = [
                    str(scale) for scale in base_syms
                    if scale_group[scale] == species_group[X[i]]
                ]
        unavailable = [
            str(X[i]) for i in range(n)
            if i not in fixed_K_idx and not restrictions[i]
        ]
        if unavailable:
            raise ValueError(
                "No dimensionally compatible fixed scale is available for: "
                + ", ".join(unavailable)
                + "."
            )
    elif out.get("_assignment_analysis", {}).get("requires_full_assignment_search", False):
        # A component merge may be offset by a fall in r_rate. The optimiser
        # must compare the full structural objective before numerical costs.
        for i in range(n):
            if i not in fixed_K_idx:
                restrictions[i] = list(labels)
    else:
        # allowed sets from components
        for ci, info in out["comp_info"].items():
            if "unfixed_indices" in info and "allowed_set" in info:
                allowed = [str(s) for s in info["allowed_set"]]
                for i in info["unfixed_indices"]:
                    restrictions[i] = list(allowed)
    # Prepare default return
    ret = {
        "_S": S, "_V": V.copy() if hasattr(V, "copy") else list(V), "_X": list(X),
        "rate_constants": _unique_ordered(ks),
        "P": P, "nu": nu,
        # Everything from analyse_crn, surfaced at top-level for convenience
        "connected": out.get("connected"),
        "components": out.get("components"),
        "components_species": out.get("components_species"),
        "comp_info": out.get("comp_info"),
        "min_theor_gen": out.get("min_theor_gen"),
        "fixed_K_indexed": fixed_K_idx,
        "base_symbols": base_syms,
        "restrictions": restrictions,
        "num_unique_coefficients": None,
        "num_generators": None,
        "families": [],
    }
    if dimensionally_consistent_sets is not None:
        ret["dimensionally_consistent_sets"] = dimensional_sets
    if "_assignment_analysis" in out:
        ret["_assignment_analysis"] = out["_assignment_analysis"]

    if verbose:
        print("Is the species co-participation graph connected?:", ["No", "Yes"][ret["connected"]])
        print("Components (species):", [[str(s) for s in comp] for comp in ret["components_species"]])
        qualifier = " under dimensional consistency" if has_dimensional_restrictions else ""
        print("Theoretical minimum number of free parameters" + qualifier + ":", ret["min_theor_gen"])
        if out.get("_assignment_analysis", {}).get("requires_full_assignment_search", False):
            print("Non-merging component constraints (additional tied merges are also searched):")
        else:
            print("Component scale constraints to minimise free parameters:")
        # Iterate over each connected component and show which scales can be chosen freely.
        for ci, info in ret["comp_info"].items():
            # Convert unfixed indices -> symbolic K_1, K_2, ...
            unfixed_idxs = info.get("unfixed_indices", ())
            unfixed_scales = tuple(sp.Symbol(f"K_{i+1}") for i in unfixed_idxs)
            idx2name = dict(zip(info.get("species_indices", ()), info.get("species_names", ())))
            display = {
                "species_names_within_compartment": info.get("species_names"),
                "unfixed_species": tuple(str(idx2name[i]) for i in unfixed_idxs),
                "unfixed_scales": unfixed_scales,
            }
            if unfixed_scales==() or "allowed_set" not in info:continue
            # Keep only the relevant constraint key.
            display["allowed_set"] = info["allowed_set"]
            print(f"  Component {ci}: {display}")
    # If we are not asked to find an example nondimensionalisation, return the pre-analysis bundle.
    if not find_example_nondim:
        return _finish(ret)

    # If there are free species but we have no base symbols, we cannot assign choices (same behaviour as enumerator).
    free_indices = [i for i in range(n) if i not in fixed_K_idx]
    if not base_syms and free_indices:
        return _finish(ret)  # no families possible

    # Non-numeric constrained streamer
    if base_syms:
        assignment_analysis = ret.get("_assignment_analysis")
        if assignment_analysis is not None and not has_dimensional_restrictions:
            fixed_full = _assignment_from_structural_plan(
                assignment_analysis["structure"],
                fixed_K_idx,
                assignment_analysis["minimum_plan"],
            )
            choices = [[str(fixed_full[i]) for i in range(n)]]
        else:
            choices = _LEXIWrapper(labels, restrictions).stream()

        best_rank = None
        for choice in choices:
            fixed_full = dict(fixed_K_idx)
            if isinstance(choice, (list, tuple)) and len(choice) == n:
                # Full per-species assignment
                for i in range(n):
                    symi = choice[i]
                    fixed_full[i] = symi if not isinstance(symi, str) else label_to_sym[symi]
            else:
                # Only free indices provided in order
                for idx, sym in zip(free_indices, choice):
                    fixed_full[idx] = sym if not isinstance(sym, str) else label_to_sym[sym]
            # species -> scale symbol
            nondim = {X[i]: fixed_full[i] for i in range(n)}
            # Restrict to a single family
            fam_res = process_CRN_given_nondimensionalisation(
                S, V, X, nondim=nondim, basis_type=basis_type,
                return_all_families=False,
                dimensionally_consistent_sets=dimensional_sets,
            )
            families = fam_res.get("families", [])
            if families:
                rank = int(families[0].get("free_group_rank", 0))
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    ret["families"] = families  # ensure one family only
                    ret["num_unique_coefficients"] = int(families[0].get("unique_monomial_count", 0))
                    ret["num_generators"] = rank
                if not has_dimensional_restrictions or rank == ret["min_theor_gen"]:
                    break
    return _finish(ret)
