from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence, Union
from .utils import format_sympy_matrix, _build_rate_and_species_orders, _format_factor_ordered
import sympy as sp
import re
Number = Union[int, float]
Rate = Union[str, Number, sp.Symbol]

@dataclass
class Reaction:
    """
    A mass-action reaction of the form:

        sum_i nu_ir * X_i  ->  sum_i nu'_ir * X_i  with rate k_r

    Attributes:
        reactants (Dict[str, int]): Mapping species name -> stoichiometric
            coefficient on the left-hand side.
        products (Dict[str, int]): Mapping species name -> stoichiometric
            coefficient on the right-hand side.
        k (Rate): Mass-action rate parameter, which can be a string name,
            a number, or a sympy.Symbol.
    """
    reactants: Dict[str, int]
    products: Dict[str, int]
    k: Rate


_species_token = re.compile(r"^\s*(\d+)?\s*([A-Za-z_][\w\-\+\.]*)\s*$")


def _parse_side(side: str) -> Dict[str, int]:
    """
    Parse one side of a reaction equation (e.g. '2H2 + O2').

    Args:
        side (str): String representing one side of a reaction equation.
            It may be empty or use '0' to denote no species.

    Returns:
        Dict[str, int]: Mapping from species name to stoichiometric
        coefficient (non-negative integers). Returns an empty dict if the
        side is empty/zero.
    """
    side = side.strip()
    # Handle empty or explicit zero/empty-symbol cases
    if side in ["", "0"]:
        return {}
    out: Dict[str, int] = {}
    # Split by '+' to get tokens like '2H2', 'O2', 'A'
    for tok in side.split("+"):
        tok = tok.strip()
        if not tok:
            continue
        # Match optional integer coefficient plus species name
        m = _species_token.match(tok)
        if not m:
            raise ValueError(f"Cannot parse species token: '{tok}'. Use valid identifiers.")
        coeff = int(m.group(1)) if m.group(1) else 1
        spname = m.group(2)
        # Accumulate coefficients if a species appears multiple times
        out[spname] = out.get(spname, 0) + coeff
    return out


def reaction(eq: str, k: Rate) -> Reaction:
    """
    Create an irreversible Reaction object from a reaction string.

    Args:
        eq (str): Reaction equation using '->' for an irreversible arrow,
            e.g. '2A + B -> C'. Must not contain '<->'.
        k (Rate): Rate parameter for this reaction. May be a string name,
            a numeric constant, or a sympy.Symbol.

    Returns:
        Reaction: Dataclass instance with parsed reactants, products,
        and the given rate parameter.
    """
    # For reversible arrows, the dedicated function must be used
    if "<->" in eq:
        raise ValueError("Use reversible() for reversible arrows '<->'")
    # Split into left- and right-hand sides and parse each side
    lhs, rhs = eq.split("->")
    return Reaction(_parse_side(lhs), _parse_side(rhs), k)


def reversible(eq: str, kf: Rate, kr: Rate) -> List[Reaction]:
    """
    Create a pair of irreversible Reaction objects for a reversible reaction.

    Args:
        eq (str): Reaction equation using '<->' between left and right,
            e.g. 'A + B <-> C'.
        kf (Rate): Forward reaction rate parameter (left -> right).
        kr (Rate): Reverse reaction rate parameter (right -> left).

    Returns:
        List[Reaction]: Two Reaction instances, [forward, reverse],
        representing the reversible reaction as a pair of irreversible ones.
    """
    arrow = "<->"
    # Split at the reversible arrow and build both directions
    lhs, rhs = eq.split(arrow)
    r1 = Reaction(_parse_side(lhs), _parse_side(rhs), kf)
    r2 = Reaction(_parse_side(rhs), _parse_side(lhs), kr)
    return [r1, r2]


def build_mass_action_system(reactions: Sequence[Reaction], verbose:bool = True):
    """
    Construct the symbolic mass-action system (S, V, X, parameters) for a CRN.

    Args:
        reactions (Sequence[Reaction]): Iterable of Reaction objects
            describing the network.
        verbose (bool): If True, print out information about the CRN.

    Returns:
        Dict[str, Any]: Dictionary with keys:
            - 'species' (List[str]): Sorted list of species names.
            - '_S' (sp.Matrix): Stoichiometric matrix (n_species x n_reactions).
            - '_X' (List[sp.Symbol]): Sympy symbols for species concentrations.
            - '_V' (sp.Matrix): Column vector of reaction-rate monomials.
            - 'rate_constants' (Tuple[sp.Symbol, ...]): Tuple of distinct
              rate symbols appearing in the monomials.
            - 'expanded_ODEs' (sp.Matrix): Vector field S * V for dX/dt.
    """
    # Build the global set of species appearing in any reaction
    species_set = set()
    for r in reactions:
        species_set.update(r.reactants.keys())
        species_set.update(r.products.keys())
    species = sorted(list(species_set))

    # Map species names to row indices
    idx = {s: i for i, s in enumerate(species)}
    n, m = len(species), len(reactions)

    # Build stoichiometric matrix S (products minus reactants)
    S = sp.zeros(n, m)
    for j, r in enumerate(reactions):
        # Add product stoichiometry
        for s, c in r.products.items():
            S[idx[s], j] += c
        # Subtract reactant stoichiometry
        for s, c in r.reactants.items():
            S[idx[s], j] -= c

    # Create one nonnegative sympy symbol per species
    x_syms = [sp.symbols(s, nonnegative=True) for s in species]

    def _as_rate_symbol(k: Rate, j: int) -> sp.Expr:
        """
        Convert a rate specifier into a symbolic parameter or numeric constant.

        Args:
            k (Rate): Rate specifier (Symbol, string name, or numeric).
            j (int): Index of the reaction, used for default naming.

        Returns:
            sp.Expr: The symbolic or numeric rate constant.
        """
        if isinstance(k, sp.Symbol):
            return k
        if isinstance(k, str):
            return sp.Symbol(k, positive=True)
        return sp.sympify(k)

    # Build the V vector of mass-action monomials
    V_entries: List[sp.Expr] = []
    params_set = set()
    for j, r in enumerate(reactions):
        # Ensure each reaction gets a rate symbol (re-used if needed)
        kj = _as_rate_symbol(r.k, j)
        if isinstance(kj, sp.Symbol):
            params_set.add(kj)

        # Multiply species symbols according to reactant stoichiometry
        monomial = sp.Integer(1)
        for s, coeff in r.reactants.items():
            monomial *= x_syms[idx[s]] ** coeff
        V_entries.append(kj * monomial)

    V = sp.Matrix(V_entries)

    # Sort parameters by name for reproducibility
    params = tuple(sorted(list(params_set), key=lambda s: s.name))

    # Symbolic right-hand side of the ODE system
    rhs_sym = S * V

    # Print a small summary of the parsed CRN.
    if verbose:
        print(f"Species names: {species}")
        print(format_sympy_matrix(S, "Stoichiometry matrix",':'))
        rate_order, x_index = _build_rate_and_species_orders(V, x_syms)
        print(format_sympy_matrix(V, "Reaction vector", ':', elem_formatter=lambda e: _format_factor_ordered(e, rate_order, x_index)))
        print(f"Unique rate constants: {params}")
        print(f"Expanded ODEs (where ' denotes the derivative with respect to dimensional time):")
        # Show each expanded ODE in human-readable form.
        for i in range(S.rows):
            pieces = []
            first = True
            for j in range(V.rows):               # iterate columns of V in order
                if S[i, j] == 0:
                    continue
                term = S[i, j] * V[j, 0]          # one reaction's contribution to species i
                s = _format_factor_ordered(term, rate_order, x_index)  # order factors inside monomial
                if first:
                    pieces.append(s)
                    first = False
                else:
                    pieces.append((" + " + s) if not s.startswith("-") else (" - " + s[1:]))
            print(f"{species[i]}' = {''.join(pieces)}")
    return {
        "species": species,
        "_S": S,
        "_X": x_syms,
        "_V": V,
        "rate_constants": params,
        "expanded_ODEs": rhs_sym,
    }
