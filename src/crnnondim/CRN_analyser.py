# -----------------------------------------------------------------------------
# Core analysis for CRN nondimensionalisation diagnostics.
# This module provides:
#   - compute_connected_components(S): co-participation graph (species x species)
#     and per-reaction participating species lists, all computed purely from
#     the stoichiometric support (S != 0).
#   - analyse_crn(S, V, X, K_fixed): main analyser; returns:
#       * connected (bool)  -- from S-based connected-components
#       * per_reaction_species_1b, components_1b (1-based lists)
#       * components, components_species (0-based + names)
#       * comp_info (per-component scale constraints)
#       * distinct_rate_parameters, min_theor_gen
#   - exact assignment-dependence analysis for r_rate when required.
#     All component logic below is driven by the S-based components.
# -----------------------------------------------------------------------------
from __future__ import annotations
from collections import Counter
from typing import Any, Dict, List, Tuple, Set
import numpy as np
import sympy as sp
# ------------------------------ Utilities -----------------------------------

def _distinct_rate_parameters(V: List[sp.Expr], X: List[sp.Symbol]) -> Set[sp.Symbol]:
    """
    Extract the set of distinct rate parameters appearing in V, excluding species.

    Args:
        V (List[sp.Expr]): Reaction-rate monomials V_j(X) for each reaction.
        X (List[sp.Symbol]): Species symbols; any free symbol in this list
            is treated as a state variable rather than a parameter.

    Returns:
        Set[sp.Symbol]: All free symbols appearing in V that are not in X,
        interpreted as rate parameters.
    """
    # Treat species as "non-parameters"
    xset = set(X)
    # Collect all free symbols across all rate monomials, excluding species
    params: Set[sp.Symbol] = set()
    for vj in V:
        for s in vj.free_symbols:
            if s not in xset:
                params.add(s)
    return params
# -------------------- Connected components from S != 0 -----------------------
def compute_connected_components(S: sp.Matrix) -> Dict[str, Any]:
    """
    Compute per-reaction participating species and S-based connected components.

    Args:
        S (sp.Matrix): Stoichiometric matrix (n_species x n_reactions). Only the
            support pattern S[i, j] != 0 is used; magnitudes are ignored.

    Returns:
        Dict[str, Any]: Dictionary with keys:
            per_reaction_species_1b (List[List[int]]): For each reaction j,
                the sorted 1-based indices i of species with S[i, j] != 0.
            components_1b (List[List[int]]): Connected components (1-based) of
                the undirected species graph where i ~ i' if they co-occur in
                some reaction column.
            connected (bool): True iff there is exactly one connected component.
    """
    # Extract matrix shape and build a boolean nonzero mask
    n, r = S.shape
    S_nz = (np.array(S, dtype=object) != 0)

    # Per-reaction participating species (1-based indices)
    per_reac: List[List[int]] = []
    for j in range(r):
        idx = np.flatnonzero(S_nz[:, j]).tolist()
        per_reac.append([i + 1 for i in idx])

    # Species co-participation adjacency: edge if appear together in some reaction
    S8 = S_nz.astype(np.int64)
    counts = S8 @ S8.T                       # (n x n) co-appearance counts
    G = counts > 0
    np.fill_diagonal(G, False)

    # BFS to get components
    seen = np.zeros(n, dtype=bool)
    components: List[List[int]] = []
    for s in range(n):
        if not seen[s]:
            q = [s]
            seen[s] = True
            comp = [s]
            head = 0
            while head < len(q):
                u = q[head]
                head += 1
                for v in np.flatnonzero(G[u]):
                    if not seen[v]:
                        seen[v] = True
                        q.append(v)
                        comp.append(v)
            comp.sort()
            components.append([i + 1 for i in comp])   # 1-based

    components.sort(key=lambda c: c[0] if c else -1)
    connected = (len(components) == 1)

    return dict(
        per_reaction_species_1b=per_reac,
        components_1b=components,
        connected=connected,
    )
# -------------------- Per-component scale-constraint assembly ----------------
def _scale_constraints_for_cscale_max(
    comps_0b: List[List[int]],
    K_fixed: Dict[int, sp.Symbol],
    X: List[sp.Symbol],
) -> Tuple[Dict[int, Dict[str, Any]], Set[sp.Symbol], List[Tuple[sp.Symbol, sp.Symbol]], List[List[sp.Symbol]]]:
    """
    Build per-component scale constraints for maximising c_scale across the CRN.

    Args:
        comps_0b (List[List[int]]): Connected components of the species graph,
            as 0-based species index lists.
        K_fixed (Dict[int, sp.Symbol]): Map from 0-based species index i to a
            pre-fixed scale symbol K_i (imposed nondimensionalisation choices).
        X (List[sp.Symbol]): Species symbols corresponding to indices.

    Returns:
        Tuple[...]: A tuple
            (comp_info, s_used, scale_edges, comps_as_names) where
            comp_info (Dict[int, Dict[str, Any]]): Per-component scale data
                such as required equalities between scales.
            s_used (Set[sp.Symbol]): Set of all scale symbols that appear.
            scale_edges (List[Tuple[sp.Symbol, sp.Symbol]]): Undirected edges
                between scale symbols indicating they must be linked.
            comps_as_names (List[List[sp.Symbol]]): Components as lists
                of species symbols (names instead of indices).
    """
    # Convert component indices into species symbols
    # and enforce fixed-scale constraints per component.
    comp_info: Dict[int, Dict[str, Any]] = {}
    used_symbols: Set[sp.Symbol] = set()
    scale_edges: List[Tuple[sp.Symbol, sp.Symbol]] = []
    comps_as_names: List[List[sp.Symbol]] = []

    for ci, comp in enumerate(comps_0b):
        fixed_syms = {K_fixed[i] for i in comp if i in K_fixed}
        unfixed_idx = [i for i in comp if i not in K_fixed]
        species_names = [X[i] for i in comp]
        comps_as_names.append(species_names)

        info: Dict[str, Any] = dict(
            species_indices=tuple(comp),
            species_names=tuple(species_names),
            fixed_symbols=tuple(sorted(fixed_syms, key=lambda s: str(s))),
            unfixed_indices=tuple(unfixed_idx),
        )

        if fixed_syms:
            # Unfixed may choose any fixed symbol present in their component
            info["allowed_set"] = tuple(sorted(fixed_syms, key=lambda s: str(s)))
            used_symbols |= fixed_syms
            # Add edges among fixed symbols within the component to connect them
            fl = sorted(list(fixed_syms), key=lambda s: str(s))
            for a in range(len(fl)):
                for b in range(a + 1, len(fl)):
                    scale_edges.append((fl[a], fl[b]))
        else:
            # No fixed scales in component: introduce a per-component anchor
            anc = sp.Symbol(f"A_comp{ci+1}", positive=True)
            info["anchor"] = anc

        comp_info[ci] = info

    if used_symbols:
        symbol_to_component, symbol_components = _symbol_components_map(
            used_symbols, scale_edges
        )
        for info in comp_info.values():
            fixed_symbols = info["fixed_symbols"]
            if fixed_symbols:
                info["allowed_set"] = tuple(sorted(
                    symbol_components[symbol_to_component[fixed_symbols[0]]],
                    key=str,
                ))

    return comp_info, used_symbols, scale_edges, comps_as_names

def _n_components_in_symbol_graph(nodes: Set[sp.Symbol], edges: List[Tuple[sp.Symbol, sp.Symbol]]) -> int:
    """
    Count connected components in an undirected graph of scale symbols.

    Args:
        nodes (Set[sp.Symbol]): All scale symbols that appear as vertices.
        edges (List[Tuple[sp.Symbol, sp.Symbol]]): Undirected edges between
            scale symbols; each pair indicates that the scales must be linked.

    Returns:
        int: Number of connected components in the symbol graph.
    """
    # Initialise disjoint-set structure for union-find
    # and join nodes that appear together in an edge.
    if not nodes:
        return 0
    # Union-Find
    parent = {u: u for u in nodes}
    rank = {u: 0 for u in nodes}

    def find(x):
        """
        Find the representative of a node in a union-find data structure.

        Args:
            x: Node identifier (hashable) whose representative is requested.

        Returns:
            Same type as x: The root/representative element of x's set.
        """
        # Path-compressed find operation for disjoint-set union.
        px = parent[x]
        if px != x:
            parent[x] = find(px)
        return parent[x]

    def union(a, b):
        """
        Union two sets in the union-find structure by rank.

        Args:
            a: Representative or element of the first set.
            b: Representative or element of the second set.

        Returns: None.
        """
        # Merge sets containing a and b, using union by rank.
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for u, v in edges:
        union(u, v)

    reps = {find(u) for u in nodes}
    return len(reps)

def _compute_r_rate(
    V: List[sp.Expr],
    X: List[sp.Symbol],
    per_reaction_species_0b: List[List[int]],
    comp_info: Dict[int, Dict[str, Any]],
    used_symbols: Set[sp.Symbol],
    scale_edges: List[Tuple[sp.Symbol, sp.Symbol]],
) -> int:
    """Compute r_rate = rank_Z(Lambda_rate)

    Args:
        V (List[sp.Expr]): Reaction-rate monomials V_j(X) for each reaction.
        X (List[sp.Symbol]): Species symbols (ordered), used to read exponents nu_ij.
        per_reaction_species_0b (List[List[int]]): For each reaction j, the 0-based species indices
            participating in that reaction (nonzero column support in S).
        comp_info (Dict[int, Dict[str, Any]]): Per-species-component scale info produced upstream;
            must include species_indices, and either fixed_symbols or an anchor symbol.
        used_symbols (Set[sp.Symbol]): The set of scale symbols actually present across components/anchors.
        scale_edges (List[Tuple[sp.Symbol, sp.Symbol]]): Undirected edges between scale symbols, defining
            the scale graph H_scale.

    Returns:
        int: The integer rank of the stoichiometric rate lattice Lambda_rate.
    """
    if not used_symbols:
        return 0

    # Component map of the scale graph H_scale
    sym_to_comp, comps = _symbol_components_map(used_symbols, scale_edges)
    c_scale = len(comps)

    # Map species to psi(i) (component id in H_scale) using comp_info
    species_to_comp = {}
    for ci, info in comp_info.items():
        fixed_syms = info.get('fixed_symbols', ())
        if not fixed_syms:
            continue
        any_sym = next(iter(fixed_syms))
        comp_id = sym_to_comp[any_sym]
        for i in info['species_indices']:
            species_to_comp[i] = comp_id

    # Helper: extract nu_{ij} exponents from V_j wrt each species
    xset = set(X)

    def reactant_exponents(vj: sp.Expr) -> List[int]:
        pows = vj.as_powers_dict()
        return [int(pows.get(xi, 0)) for xi in X]

    # Build z_j vectors and group reactions by rate-parameter key
    z_cols: List[List[int]] = []
    rate_groups: Dict[Tuple[sp.Symbol, ...], List[int]] = {}

    for j, vj in enumerate(V):
        nu_j = reactant_exponents(vj)

        # psi_V(j): component of any participating species from S support
        psiVj = None
        for i in per_reaction_species_0b[j]:
            if i in species_to_comp:
                psiVj = species_to_comp[i]
                break

        # z_j in Z^{c_scale}
        z = [0] * c_scale
        for i, exp in enumerate(nu_j):
            if exp and i in species_to_comp:
                z[species_to_comp[i]] += exp
        if psiVj is not None:
            z[psiVj] -= 1

        z_cols.append(z)

        # group by non-species parameters
        params = tuple(sorted([s for s in vj.free_symbols if s not in xset], key=lambda s: str(s)))
        rate_groups.setdefault(params, []).append(j)

    # Generator set: differences within each rate group
    gens: List[List[int]] = []
    for _, idxs in rate_groups.items():
        if len(idxs) <= 1:
            continue
        anchor = idxs[0]
        z0 = z_cols[anchor]
        for j in idxs[1:]:
            diff = [z_cols[j][k] - z0[k] for k in range(c_scale)]
            gens.append(diff)

    if not gens:
        return 0

    Lambda_rate = sp.Matrix(gens).T  # c_scale x num_gens
    return int(Lambda_rate.rank())

def _symbol_components_map(nodes: Set[sp.Symbol], edges: List[Tuple[sp.Symbol, sp.Symbol]]) -> Tuple[Dict[sp.Symbol, int], List[List[sp.Symbol]]]:
    """Compute connected components of the scale-symbol graph and map symbols to component IDs.

    Args:
        nodes (Set[sp.Symbol]): All scale symbols (vertices) present.
        edges (List[Tuple[sp.Symbol, sp.Symbol]]): Undirected edges between symbols indicating coupling.

    Returns:
        Tuple[Dict[sp.Symbol, int], List[List[sp.Symbol]]]:
            - A dictionary mapping each symbol to its component index.
            - A list of components, each a list of symbols belonging to that component.
    """
    nodes = list(nodes)
    # Union-find on symbol indices
    parent = {i: i for i in range(len(nodes))}
    idx_of = {nodes[i]: i for i in range(len(nodes))}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for a, b in edges:
        ia, ib = idx_of[a], idx_of[b]
        union(ia, ib)

    # collect components
    root_to_nodes: Dict[int, List[sp.Symbol]] = {}
    for i, sym in enumerate(nodes):
        r = find(i)
        root_to_nodes.setdefault(r, []).append(sym)

    comps = list(root_to_nodes.values())
    sym_to_comp = {}
    for cid, comp in enumerate(comps):
        for sym in comp:
            sym_to_comp[sym] = cid
    return sym_to_comp, comps

# -------------- Assignment-dependent rate-lattice analysis -------------------

def _build_bar_lambda_rate(
    V: List[sp.Expr],
    X: List[sp.Symbol],
    comps_0b: List[List[int]],
    per_reaction_species_0b: List[List[int]],
) -> sp.Matrix:
    """
    Build the assignment-independent rate lattice on species components.

    Args:
        V (List[sp.Expr]): Reaction-rate monomials V_j(X) for each reaction.
        X (List[sp.Symbol]): Species symbols corresponding to rows of S.
        comps_0b (List[List[int]]): S-based species components as 0-based
            species-index lists.
        per_reaction_species_0b (List[List[int]]): For each reaction, the
            0-based species indices with nonzero stoichiometric entries.

    Returns:
        sp.Matrix: Matrix whose columns generate the component-level rate
        lattice before a scale assignment is applied.
    """
    q = len(comps_0b)
    species_to_component = [0] * len(X)
    for component_index, component in enumerate(comps_0b):
        for i in component:
            species_to_component[i] = component_index

    xset = set(X)
    bar_z: List[List[int]] = []
    rate_groups: Dict[Tuple[sp.Symbol, ...], List[int]] = {}

    for j, vj in enumerate(V):
        z = [0] * q
        powers = vj.as_powers_dict()
        for i, xi in enumerate(X):
            exponent = int(powers.get(xi, 0))
            if exponent:
                z[species_to_component[i]] += exponent

        participating_species = per_reaction_species_0b[j]
        if participating_species:
            component_index = species_to_component[participating_species[0]]
            z[component_index] -= 1
        bar_z.append(z)

        rate_key = tuple(sorted(
            [symbol for symbol in vj.free_symbols if symbol not in xset],
            key=lambda symbol: str(symbol),
        ))
        rate_groups.setdefault(rate_key, []).append(j)

    generators: List[List[int]] = []
    for indices in rate_groups.values():
        if len(indices) <= 1:
            continue
        anchor = indices[0]
        for j in indices[1:]:
            generators.append([
                bar_z[j][component_index] - bar_z[anchor][component_index]
                for component_index in range(q)
            ])

    if not generators:
        return sp.zeros(q, 0)
    return sp.Matrix(generators).T

def _exact_row_rank(rows: Tuple[Tuple[sp.Expr, ...], ...]) -> int:
    """
    Compute the exact rank of a tuple of row vectors.

    Args:
        rows (Tuple[Tuple[sp.Expr, ...], ...]): Row vectors over the rationals.

    Returns:
        int: Exact row rank over the rationals.
    """
    if not rows or not rows[0]:
        return 0
    matrix = sp.Matrix(rows)
    return int(matrix.rank())


def _exact_row_basis(
    rows: Tuple[Tuple[sp.Expr, ...], ...],
) -> Tuple[Tuple[sp.Expr, ...], ...]:
    """
    Return an exact reduced row-space basis.

    Args:
        rows (Tuple[Tuple[sp.Expr, ...], ...]): Row vectors over the rationals.

    Returns:
        Tuple[Tuple[sp.Expr, ...], ...]: Nonzero rows of a reduced row-echelon
        basis for the supplied row space.
    """
    nonzero_rows = tuple(row for row in rows if any(value != 0 for value in row))
    if not nonzero_rows:
        return tuple()
    reduced, _ = sp.Matrix(nonzero_rows).rref()
    return tuple(
        tuple(reduced[i, j] for j in range(reduced.cols))
        for i in range(reduced.rows)
        if any(reduced[i, j] != 0 for j in range(reduced.cols))
    )


def _add_rate_rows(
    first: Tuple[sp.Expr, ...],
    second: Tuple[sp.Expr, ...],
) -> Tuple[sp.Expr, ...]:
    """
    Add two component-level rate rows.

    Args:
        first (Tuple[sp.Expr, ...]): First row vector.
        second (Tuple[sp.Expr, ...]): Second row vector.

    Returns:
        Tuple[sp.Expr, ...]: Entrywise sum of the two rows.
    """
    return tuple(a + b for a, b in zip(first, second))


def _compress_rate_rows(bar_lambda_rate: sp.Matrix) -> Tuple[Tuple[int, ...], ...]:
    """
    Compress the rate lattice to a column basis without changing left kernels.

    Args:
        bar_lambda_rate (sp.Matrix): Component-level rate-lattice matrix.

    Returns:
        Tuple[Tuple[int, ...], ...]: One integer row for each species component,
        expressed in a minimum-width set of independent lattice columns.
    """
    if bar_lambda_rate.rows == 0:
        return tuple()
    if bar_lambda_rate.cols == 0:
        return tuple(tuple() for _ in range(bar_lambda_rate.rows))
    _, pivot_columns = bar_lambda_rate.rref()
    if not pivot_columns:
        return tuple(tuple() for _ in range(bar_lambda_rate.rows))
    compressed = bar_lambda_rate[:, list(pivot_columns)]
    return tuple(
        tuple(int(compressed[i, j]) for j in range(compressed.cols))
        for i in range(compressed.rows)
    )


def _build_assignment_structure(
    V: List[sp.Expr],
    X: List[sp.Symbol],
    comps_0b: List[List[int]],
    per_reaction_species_0b: List[List[int]],
    K_fixed: Dict[int, sp.Symbol],
    bar_lambda_rate: sp.Matrix | None = None,
) -> Dict[str, Any]:
    """
    Build the fixed scale blocks and component-level rate rows.

    Args:
        V (List[sp.Expr]): Reaction-rate monomials V_j(X).
        X (List[sp.Symbol]): Species symbols corresponding to rows of S.
        comps_0b (List[List[int]]): S-based species components as 0-based
            species-index lists.
        per_reaction_species_0b (List[List[int]]): Per-reaction active species
            as 0-based index lists.
        K_fixed (Dict[int, sp.Symbol]): Pre-fixed species scale symbols.
        bar_lambda_rate (sp.Matrix | None): Optional precomputed component-level
            rate-lattice matrix.

    Returns:
        Dict[str, Any]: Fixed scale blocks, species-component data and compressed
        rate rows used by the exact assignment search.
    """
    scale_symbols: List[sp.Symbol] = []
    for i in sorted(K_fixed):
        symbol = K_fixed[i]
        if symbol not in scale_symbols:
            scale_symbols.append(symbol)

    if bar_lambda_rate is None:
        bar_lambda_rate = _build_bar_lambda_rate(
            V, X, comps_0b, per_reaction_species_0b
        )

    s = len(scale_symbols)
    symbol_index = {symbol: index for index, symbol in enumerate(scale_symbols)}
    parent = list(range(s))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)

    for component in comps_0b:
        fixed_indices = [
            symbol_index[K_fixed[i]] for i in component if i in K_fixed
        ]
        for index in fixed_indices[1:]:
            union(fixed_indices[0], index)

    root_to_symbols: Dict[int, List[int]] = {}
    for index in range(s):
        root_to_symbols.setdefault(find(index), []).append(index)
    mandatory_blocks = tuple(
        tuple(indices)
        for _, indices in sorted(
            root_to_symbols.items(), key=lambda item: min(item[1])
        )
    )

    symbol_to_block: Dict[sp.Symbol, int] = {}
    for block, indices in enumerate(mandatory_blocks):
        for index in indices:
            symbol_to_block[scale_symbols[index]] = block

    component_data: List[Dict[str, Any]] = []
    anchored_blocks: List[int | None] = []
    for component in comps_0b:
        fixed_species = tuple(i for i in component if i in K_fixed)
        free_species = tuple(i for i in component if i not in K_fixed)
        fixed_blocks = {
            symbol_to_block[K_fixed[i]] for i in fixed_species
        }
        if len(fixed_blocks) > 1:
            raise RuntimeError(
                "Fixed scale symbols in one species component were not merged."
            )
        fixed_block = next(iter(fixed_blocks), None)
        anchored_blocks.append(fixed_block)
        component_data.append(dict(
            species_indices=tuple(component),
            fixed_indices=fixed_species,
            free_indices=free_species,
            fixed_block=fixed_block,
        ))

    rate_rows = _compress_rate_rows(bar_lambda_rate)
    return dict(
        scale_symbols=tuple(scale_symbols),
        mandatory_blocks=mandatory_blocks,
        symbol_to_block=symbol_to_block,
        component_data=tuple(component_data),
        anchored_blocks=tuple(anchored_blocks),
        rate_rows=rate_rows,
        bar_lambda_rate=bar_lambda_rate,
        bar_lambda_rate_rank=(len(rate_rows[0]) if rate_rows else 0),
        num_species_components=len(comps_0b),
        num_scale_symbols=s,
        num_mandatory_blocks=len(mandatory_blocks),
    )


class _ModularSearchLimit(RuntimeError):
    """Signal that an optional modular certificate exceeded its state budget."""


def _encode_modular_row(row: Tuple[int, ...], prime: int) -> int:
    """Encode a row over a prime field as a base-prime integer."""
    code = 0
    multiplier = 1
    for value in row:
        code += (int(value) % prime) * multiplier
        multiplier *= prime
    return code


def _decode_modular_row(code: int, prime: int, width: int) -> List[int]:
    """Decode a base-prime integer into a row over a prime field."""
    row: List[int] = []
    for _ in range(width):
        row.append(code % prime)
        code //= prime
    return row


def _modular_support_rank(
    support: Tuple[int, ...],
    prime: int,
    width: int,
) -> int:
    """Compute the exact rank of encoded nonzero rows over a prime field."""
    rows = [_decode_modular_row(code, prime, width) for code in support]
    rank = 0
    column = 0
    while rank < len(rows) and column < width:
        pivot = next(
            (
                i for i in range(rank, len(rows))
                if rows[i][column] % prime != 0
            ),
            None,
        )
        if pivot is None:
            column += 1
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        inverse = pow(rows[rank][column] % prime, -1, prime)
        rows[rank] = [(value * inverse) % prime for value in rows[rank]]
        for i in range(rank + 1, len(rows)):
            factor = rows[i][column] % prime
            if factor:
                rows[i] = [
                    (value - factor * pivot_value) % prime
                    for value, pivot_value in zip(rows[i], rows[rank])
                ]
        rank += 1
        column += 1
    return rank


def _minimum_modular_rate_rank(
    rate_rows: Tuple[Tuple[int, ...], ...],
    anchored_blocks: Tuple[int | None, ...],
    block_count: int,
    prime: int,
    state_limit: int,
) -> Tuple[int | None, int]:
    """
    Compute an exact finite-field lower bound for the assignment problem.

    Args:
        rate_rows (Tuple[Tuple[int, ...], ...]): Compressed component rate rows.
        anchored_blocks (Tuple[int | None, ...]): Fixed block for each component,
            with None for components having no fixed species scale.
        block_count (int): Number of mandatory scale blocks.
        prime (int): Prime defining the finite field.
        state_limit (int): Maximum number of modular dynamic-programming states.

    Returns:
        Tuple[int | None, int]: Exact minimum rank over the finite field, or
        None if the optional state budget is exceeded, and the state count.
    """
    width = len(rate_rows[0]) if rate_rows else 0
    alphabet_size = prime ** width
    decoded = [
        _decode_modular_row(code, prime, width)
        for code in range(alphabet_size)
    ]
    addition_tables: Dict[int, Tuple[int, ...]] = {}

    def add_code(first: int, second: int) -> int:
        table = addition_tables.get(second)
        if table is None:
            second_row = decoded[second]
            table = tuple(
                _encode_modular_row(
                    tuple(
                        (a + b) % prime
                        for a, b in zip(decoded[code], second_row)
                    ),
                    prime,
                )
                for code in range(alphabet_size)
            )
            addition_tables[second] = table
        return table[first]

    block_sums = [0] * block_count
    free_rows: List[int] = []
    for row, block in zip(rate_rows, anchored_blocks):
        code = _encode_modular_row(row, prime)
        if block is None:
            if code != 0:
                free_rows.append(code)
        else:
            block_sums[block] = add_code(block_sums[block], code)

    multiplicities = Counter(free_rows)
    free_rows.sort(key=lambda code: (-multiplicities[code], code))
    initial_counts = Counter(block_sums)
    initial_state = tuple(sorted(initial_counts.items()))
    rank_cache: Dict[Tuple[int, ...], int] = {}

    def rank_from_counts(
        state: Tuple[Tuple[int, int], ...],
    ) -> int:
        support = tuple(
            code for code, count in state
            if code != 0 and count > 0
        )
        if support not in rank_cache:
            rank_cache[support] = _modular_support_rank(
                support, prime, width
            )
        return rank_cache[support]

    states = {initial_state}
    state_count = 1
    for code in free_rows:
        next_states = set()
        for state in states:
            for current, count in state:
                new_code = add_code(current, code)
                next_counts = dict(state)
                if count == 1:
                    del next_counts[current]
                else:
                    next_counts[current] = count - 1
                next_counts[new_code] = next_counts.get(new_code, 0) + 1
                next_states.add(tuple(sorted(next_counts.items())))
                if state_count + len(next_states) > state_limit:
                    return None, state_count + len(next_states)
        states = next_states
        state_count += len(states)

    minimum_rank = min(rank_from_counts(state) for state in states)
    return minimum_rank, state_count


def _assignment_rank_lower_bound(
    block_sums: Tuple[Tuple[sp.Expr, ...], ...],
    remaining_basis: Tuple[Tuple[sp.Expr, ...], ...],
    total_nonzero: bool,
    global_lower: int = 0,
) -> int:
    """
    Bound the rank attainable after assigning the remaining components.

    Args:
        block_sums (Tuple[Tuple[sp.Expr, ...], ...]): Current block-sum rows.
        remaining_basis (Tuple[Tuple[sp.Expr, ...], ...]): Exact row-space basis
            of all unassigned component rows.
        total_nonzero (bool): Whether the invariant sum of all final rows is
            nonzero.
        global_lower (int): Any additional certified global lower bound.

    Returns:
        int: Exact lower bound on every completion of the current state.
    """
    quotient_rank = (
        _exact_row_rank(block_sums + remaining_basis) - len(remaining_basis)
    )
    total_rank = 1 if total_nonzero else 0
    return max(global_lower, total_rank, quotient_rank)


def _disjoint_row_basis_count(
    rows: Tuple[Tuple[sp.Expr, ...], ...],
    target_rank: int,
    stop_after: int,
) -> int:
    """
    Find row-disjoint exact bases as a rank-robustness certificate.

    Args:
        rows (Tuple[Tuple[sp.Expr, ...], ...]): Candidate rows.
        target_rank (int): Rank required of every extracted basis.
        stop_after (int): Stop once this many disjoint bases are found.

    Returns:
        int: Number of row-disjoint bases found by repeated exact greedy scans.
    """
    if target_rank == 0:
        return stop_after
    remaining = list(rows)
    count = 0
    while remaining and count < stop_after:
        basis: List[Tuple[sp.Expr, ...]] = []
        selected: List[int] = []
        for index, row in enumerate(remaining):
            if _exact_row_rank(tuple(basis + [row])) > len(basis):
                basis.append(row)
                selected.append(index)
                if len(basis) == target_rank:
                    break
        if len(basis) != target_rank:
            break
        selected_set = set(selected)
        remaining = [
            row for index, row in enumerate(remaining)
            if index not in selected_set
        ]
        count += 1
    return count


def _minimum_assignment_rate_rank(
    structure: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Minimise projected rate-lattice rank over non-merging assignments.

    The search first applies exact quotient, basis-packing and finite-field
    certificates.  If these do not certify the rational optimum, it uses an
    exact memoised branch-and-bound search with a quotient-space lower bound.
    States are identified up to permutations of blocks having identical
    current rate sums.

    Args:
        structure (Dict[str, Any]): Assignment structure produced by
            _build_assignment_structure.

    Returns:
        Dict[str, Any]: Minimum rate rank, one minimising block plan, state count
        and the method that certified optimality.
    """
    rate_rows = structure["rate_rows"]
    anchored_blocks = structure["anchored_blocks"]
    block_count = structure["num_mandatory_blocks"]
    width = len(rate_rows[0]) if rate_rows else 0
    zero_row = tuple(sp.Integer(0) for _ in range(width))

    block_sums = [zero_row for _ in range(block_count)]
    floating: List[Tuple[int, Tuple[int, ...]]] = []
    for component_index, (row, block) in enumerate(
        zip(rate_rows, anchored_blocks)
    ):
        if block is None:
            floating.append((component_index, row))
        else:
            block_sums[block] = _add_rate_rows(block_sums[block], row)

    nonzero_floating = [item for item in floating if any(item[1])]
    multiplicities = Counter(row for _, row in nonzero_floating)
    nonzero_floating.sort(key=lambda item: (
        -multiplicities[item[1]],
        -sum(abs(value) for value in item[1]),
        item[1],
        item[0],
    ))
    component_order = tuple(index for index, _ in nonzero_floating)
    free_rows = tuple(row for _, row in nonzero_floating)

    rank_cache: Dict[Tuple[Tuple[sp.Expr, ...], ...], int] = {}

    def cached_rank(rows: Tuple[Tuple[sp.Expr, ...], ...]) -> int:
        if rows not in rank_cache:
            rank_cache[rows] = _exact_row_rank(rows)
        return rank_cache[rows]

    total_free = zero_row
    for _, row in floating:
        total_free = _add_rate_rows(total_free, row)

    candidate_rank = None
    candidate_block = 0
    for block in range(block_count):
        candidate_sums = list(block_sums)
        candidate_sums[block] = _add_rate_rows(
            candidate_sums[block], total_free
        )
        value = cached_rank(tuple(candidate_sums))
        if candidate_rank is None or value < candidate_rank:
            candidate_rank = value
            candidate_block = block

    total_row = zero_row
    for row in block_sums:
        total_row = _add_rate_rows(total_row, row)
    total_row = _add_rate_rows(total_row, total_free)
    total_nonzero = any(total_row)

    basis_cache: Dict[
        Tuple[
            Tuple[int, ...],
            Tuple[Tuple[sp.Expr, ...], ...],
        ],
        Tuple[Tuple[sp.Expr, ...], ...],
    ] = {}
    suffix_bases: List[Tuple[Tuple[sp.Expr, ...], ...]] = [
        tuple() for _ in range(len(free_rows) + 1)
    ]
    basis: Tuple[Tuple[sp.Expr, ...], ...] = tuple()
    for position in range(len(free_rows) - 1, -1, -1):
        key = (free_rows[position], basis)
        if key not in basis_cache:
            basis_cache[key] = _exact_row_basis((free_rows[position],) + basis)
        basis = basis_cache[key]
        suffix_bases[position] = basis

    root_lower = _assignment_rank_lower_bound(
        tuple(block_sums),
        suffix_bases[0],
        total_nonzero,
    )
    if candidate_rank == root_lower:
        plan = [block for block in anchored_blocks]
        for component_index, _ in floating:
            plan[component_index] = candidate_block
        return dict(
            minimum_r_rate=candidate_rank,
            minimum_plan=tuple((int(block),) for block in plan),
            state_count=1,
            search_method="quotient",
            modular_primes=tuple(),
        )

    nonzero_count = len(nonzero_floating)
    anchor_rank = cached_rank(tuple(block_sums))
    if candidate_rank == anchor_rank:
        basis_count = _disjoint_row_basis_count(
            tuple(block_sums),
            anchor_rank,
            nonzero_count + 1,
        )
        if basis_count > nonzero_count:
            plan = [block for block in anchored_blocks]
            for component_index, _ in floating:
                plan[component_index] = candidate_block
            return dict(
                minimum_r_rate=candidate_rank,
                minimum_plan=tuple((int(block),) for block in plan),
                state_count=basis_count,
                search_method="basis_packing",
                modular_primes=tuple(),
            )

    modular_lower = root_lower
    modular_states = 0
    modular_primes: List[int] = []
    for prime in (2, 3, 5, 7):
        if prime ** width > 256:
            continue
        value, state_count = _minimum_modular_rate_rank(
            rate_rows,
            anchored_blocks,
            block_count,
            prime,
            250000,
        )
        modular_states += state_count
        if value is not None:
            modular_lower = max(modular_lower, value)
            modular_primes.append(prime)
        if modular_lower == candidate_rank:
            plan = [block for block in anchored_blocks]
            for component_index, _ in floating:
                plan[component_index] = candidate_block
            return dict(
                minimum_r_rate=candidate_rank,
                minimum_plan=tuple((int(block),) for block in plan),
                state_count=modular_states,
                search_method="modular",
                modular_primes=tuple(modular_primes),
            )

    def lower_bound(
        position: int,
        current_sums: Tuple[Tuple[sp.Expr, ...], ...],
    ) -> int:
        return _assignment_rank_lower_bound(
            current_sums,
            suffix_bases[position],
            total_nonzero,
            modular_lower,
        )

    incumbent = candidate_rank
    initial_state = tuple(sorted(block_sums))
    seen_states = set()
    stack = [(0, initial_state)]
    while stack:
        position, current_sums = stack.pop()
        state = (position, current_sums)
        if state in seen_states:
            continue
        seen_states.add(state)
        if incumbent == modular_lower:
            break
        if lower_bound(position, current_sums) >= incumbent:
            continue
        if position == len(free_rows):
            incumbent = min(incumbent, cached_rank(current_sums))
            continue

        row = free_rows[position]
        branches = []
        seen_rows = set()
        for block_row in current_sums:
            if block_row in seen_rows:
                continue
            seen_rows.add(block_row)
            next_sums = list(current_sums)
            block = next_sums.index(block_row)
            next_sums[block] = _add_rate_rows(block_row, row)
            canonical = tuple(sorted(next_sums))
            branch_lower = lower_bound(position + 1, canonical)
            if branch_lower < incumbent:
                branches.append((
                    branch_lower,
                    cached_rank(canonical),
                    canonical,
                ))
        for _, _, next_sums in reversed(sorted(branches)):
            stack.append((position + 1, next_sums))

    minimum_r_rate = incumbent
    plan = [block for block in anchored_blocks]
    for component_index, row in floating:
        if not any(row):
            plan[component_index] = 0

    initial_recovery = (0, tuple(block_sums))
    recovery_stack = [initial_recovery]
    recovery_seen = {initial_recovery}
    parents: Dict[
        Tuple[int, Tuple[Tuple[sp.Expr, ...], ...]],
        Tuple[
            Tuple[int, Tuple[Tuple[sp.Expr, ...], ...]],
            int,
        ],
    ] = {}
    final_state = None

    while recovery_stack:
        position, current_sums = recovery_stack.pop()
        if lower_bound(position, current_sums) > minimum_r_rate:
            continue
        if position == len(free_rows):
            if cached_rank(current_sums) == minimum_r_rate:
                final_state = (position, current_sums)
                break
            continue

        row = free_rows[position]
        seen_rows = set()
        branches = []
        for block, block_row in enumerate(current_sums):
            if block_row in seen_rows:
                continue
            seen_rows.add(block_row)
            next_sums = list(current_sums)
            next_sums[block] = _add_rate_rows(block_row, row)
            next_tuple = tuple(next_sums)
            if lower_bound(position + 1, next_tuple) <= minimum_r_rate:
                branches.append((block, next_tuple))
        for block, next_sums in reversed(branches):
            next_state = (position + 1, next_sums)
            if next_state in recovery_seen:
                continue
            recovery_seen.add(next_state)
            parents[next_state] = ((position, current_sums), block)
            recovery_stack.append(next_state)

    if final_state is None:
        raise RuntimeError("The minimum-rank assignment could not be recovered.")

    assigned_blocks = [0] * len(free_rows)
    state = final_state
    while state != initial_recovery:
        previous, block = parents[state]
        assigned_blocks[previous[0]] = block
        state = previous
    for component_index, block in zip(component_order, assigned_blocks):
        plan[component_index] = block

    return dict(
        minimum_r_rate=minimum_r_rate,
        minimum_plan=tuple((int(block),) for block in plan),
        state_count=modular_states + len(seen_states),
        search_method="exact",
        modular_primes=tuple(modular_primes),
    )


def _assignment_dependence_summary(
    structure: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Minimise s - c_scale + r_rate over admissible scale assignments.

    Args:
        structure (Dict[str, Any]): Assignment structure produced by
            _build_assignment_structure.

    Returns:
        Dict[str, Any]: Exact minimum structural objective, one minimising plan,
        minimum rate rank and search diagnostics.
    """
    s = structure["num_scale_symbols"]
    block_count = structure["num_mandatory_blocks"]
    if s == 0 or block_count == 0:
        return dict(
            minimum_r_rate=0,
            minimum_parameter_objective=0,
            minimum_plan=tuple(),
            state_count=1,
            search_method="trivial",
            modular_primes=tuple(),
        )

    result = _minimum_assignment_rate_rank(structure)
    result["minimum_parameter_objective"] = (
        s - block_count + result["minimum_r_rate"]
    )
    return result


def _assignment_from_structural_plan(
    structure: Dict[str, Any],
    K_fixed: Dict[int, sp.Symbol],
    plan: Tuple[Tuple[int, ...], ...],
) -> Dict[int, sp.Symbol]:
    """
    Construct one species-scale assignment that realises a structural plan.

    Args:
        structure (Dict[str, Any]): Assignment structure produced by
            _build_assignment_structure.
        K_fixed (Dict[int, sp.Symbol]): Pre-fixed species scale symbols.
        plan (Tuple[Tuple[int, ...], ...]): One selected mandatory block for each
            S-based species component.

    Returns:
        Dict[int, sp.Symbol]: Complete map from species indices to scale symbols.
    """
    scale_symbols = structure["scale_symbols"]
    mandatory_blocks = structure["mandatory_blocks"]
    component_data = structure["component_data"]
    assignment = dict(K_fixed)

    for data, selected_blocks in zip(component_data, plan):
        symbol_index = mandatory_blocks[selected_blocks[0]][0]
        symbol = scale_symbols[symbol_index]
        for i in data["free_indices"]:
            assignment[i] = symbol

    return assignment

# ----------------------------- Main analyser ---------------------------------
def analyse_crn(
    S: sp.Matrix,
    V: List[sp.Expr],
    X: List[sp.Symbol],
    K_fixed: Dict[int, sp.Symbol],
) -> Dict[str, Any]:
    """
    Main CRN analyser: connectivity, scale constraints and generator bounds.

    Args:
        S (sp.Matrix): Stoichiometric matrix (n_species x n_reactions).
        V (List[sp.Expr]): Reaction-rate monomials V_j(X) for each reaction.
        X (List[sp.Symbol]): Species symbols corresponding to rows of S.
        K_fixed (Dict[int, sp.Symbol]): Nonempty map from 0-based species index
            to a pre-fixed scale symbol K_i implementing partial nondim.

    Returns:
        Dict[str, Any]: Dictionary containing:
            connected (bool): Whether the S-based species graph is connected.
            components (List[List[int]]): Components as 0-based indices.
            components_species (List[List[sp.Symbol]]): Components as species.
            comp_info (Dict[int, Dict[str, Any]]): Per-component scale data.
            min_theor_gen (int): Minimum number of generators possible.
            _assignment_analysis (Dict[str, Any]): Internal exact assignment
                data, present only when the fast assignment-independent cases do
                not apply.
    """
    if not K_fixed:
        raise ValueError("K_fixed must contain at least one fixed species scale.")

    # First, compute S-based connected components and per-reaction species.
    # Then, build scale constraints per component and the scale-symbol graph.
    # Finally, compute generator lower bounds g_min and g_min_nonunit.
    cc = compute_connected_components(S)
    connected = cc["connected"]
    comps_1b: List[List[int]] = cc["components_1b"]
    # Convert to 0-based for internal computations and for 'components' indexing
    comps_0b: List[List[int]] = [[i - 1 for i in comp1b] for comp1b in comps_1b]
    # Per-component scale constraints (built on S-based components)
    comp_info, used_symbols, scale_edges, comps_as_names = _scale_constraints_for_cscale_max(
        comps_0b, K_fixed, X
    )
    # Compute c_scale_max as # of connected components in the (used scale-symbol) graph
    c_scale_max = _n_components_in_symbol_graph(used_symbols, scale_edges)
    s_used = len(used_symbols)
    # Distinct rate parameters among V, excluding species
    rate_params = _distinct_rate_parameters(V, X)
    # Compute r_rate via stoichiometric rate lattice rank
    per_rxn_0b = [[i - 1 for i in comp] for comp in cc['per_reaction_species_1b']]
    r_rate = _compute_r_rate(V, X, per_rxn_0b, comp_info, used_symbols, scale_edges)
    # Theoretical min generators (with normalising K_T and appropriately chosen scaling factors)
    min_theor_gen = len(rate_params) + (s_used - c_scale_max) + r_rate - 1

    # A minimum is attained without merging the mandatory scale blocks, but
    # merging can tie that minimum if the rate rank drops by the same amount.
    # Such ties must be retained for the subsequent numerical optimisation.
    assignment_structure = None
    assignment_summary = None
    if not connected and s_used > 0 and len(K_fixed) < len(X):
        bar_lambda_rate = _build_bar_lambda_rate(
            V, X, comps_0b, per_rxn_0b
        )
        assignment_structure = _build_assignment_structure(
            V,
            X,
            comps_0b,
            per_rxn_0b,
            K_fixed,
            bar_lambda_rate=bar_lambda_rate,
        )
        has_active_floating_component = any(
            block is None and any(row)
            for row, block in zip(
                assignment_structure["rate_rows"],
                assignment_structure["anchored_blocks"],
            )
        )
        has_splittable_floating_component = any(
            block is None and len(data["free_indices"]) > 1
            for data, block in zip(
                assignment_structure["component_data"],
                assignment_structure["anchored_blocks"],
            )
        )
        if has_active_floating_component or (
            assignment_structure["num_mandatory_blocks"] > 1
            and (has_splittable_floating_component
                 or assignment_structure["bar_lambda_rate_rank"] > 0)
        ):
            assignment_summary = _assignment_dependence_summary(
                assignment_structure
            )
            min_theor_gen = (
                len(rate_params)
                + assignment_summary["minimum_parameter_objective"]
                - 1
            )

    result = dict(
        # Connected-component diagnostics (S-based)
        connected=connected,
        # Components (0-based) + species names (for the examples' prints)
        components=comps_0b,
        components_species=comps_as_names,
        # Scale constraints and symbol-graph summary
        comp_info=comp_info,
        # generator counts
        min_theor_gen=min_theor_gen,
    )
    if assignment_summary is not None:
        result["_assignment_analysis"] = dict(
            structure=assignment_structure,
            minimum_r_rate=assignment_summary["minimum_r_rate"],
            minimum_parameter_objective=(
                assignment_summary["minimum_parameter_objective"]
            ),
            state_count=assignment_summary["state_count"],
            search_method=assignment_summary["search_method"],
            modular_primes=assignment_summary["modular_primes"],
            minimum_plan=assignment_summary["minimum_plan"],
            # When the minimum rate rank is zero, any merge strictly increases
            # s - c_scale + r_rate. Otherwise, enumerate possible tied merges.
            requires_full_assignment_search=(
                assignment_summary["minimum_r_rate"] > 0
                and assignment_structure["num_mandatory_blocks"] > 1
            ),
        )
    return result
