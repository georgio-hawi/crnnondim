from typing import List, Tuple, Dict, Any
import sympy as sp
from .utils import _gen_symbols, _key_to_vec, _vec_to_monom

def factor_generators_columnwise(S: sp.Matrix,
                                 family: Dict[str, Any],
                                 ks: List[sp.Symbol],
                                 nu: sp.Matrix):
    """
    Factor generator weights column-wise into S and V for a given family.

    Args:
        S (sp.Matrix): Stoichiometric matrix of the original CRN.
        family (Dict[str, Any]): Nondimensionalisation family structure with
            keys like "generators", "coeff_classes_by_generators", "K_family",
            "partition", "normalised_block_index".
        ks (List[sp.Symbol]): Rate-constant symbols k_j (one per reaction).
        nu (sp.Matrix): Mass-action exponent matrix nu (n_species x n_reactions).

    Returns:
        Tuple[sp.Matrix, sp.Matrix]: A pair (S_tilde, V_tilde) where
            V_tilde[j] = alpha^{tildez_j} * x^{nu*j}
            S_tilde[i, j] = S[i, j] * alpha^{z_(i,j) - tildez_j},
        so that the transformed system is algebraically identical to the
        original but with generator factors grouped per reaction column.
    """
    # Extract generator names and coefficient-class structure from the family
    # and build a mapping from each edge (i, j) to a generator exponent vector.
    gens = family.get("generators", [])
    classes = family.get("coeff_classes_by_generators", {})
    # Map edge -> exponent vector z_(i,j)
    edge2vec: Dict[Tuple[int,int], Tuple[int,...]] = {}
    for key, edges in classes.items():
        z = _key_to_vec(key, gens)
        for (i, j) in edges:
            edge2vec[(i, j)] = z
    n, r = S.shape
    # Column-wise common factor (componentwise min over the column)
    zbar = [tuple(0 for _ in gens) for _ in range(r)]
    for j in range(r):
        col_vecs = [edge2vec.get((i, j), tuple(0 for _ in gens)) for i in range(n) if S[i, j] != 0]
        if not col_vecs:
            continue
        # componentwise min
        mins = list(col_vecs[0])
        for v in col_vecs[1:]:
            mins = [min(a, b) for a, b in zip(mins, v)]
        zbar[j] = tuple(mins)
    sym_g = _gen_symbols(gens)
    # Build V_tilde
    x_vars = sp.symbols(f'x_1:{nu.rows+1}')
    monom_factors = []
    for j in range(r):
        # product in ascending index order with powers for x's
        factors = []
        for l in range(nu.rows):
            e = int(nu[l, j])
            if e:
                factors.append(x_vars[l]**e)
        monom_factors.append(factors)
    # Ensure generator prints before the x-product by grouping x factors
    V_tilde = sp.Matrix([
        sp.Mul(_vec_to_monom(zbar[j], gens, sym_g), *monom_factors[j])
        for j in range(r)
    ])
    # Build S_tilde with residuals
    S_tilde = sp.zeros(n, r)
    for i in range(n):
        for j in range(r):
            if S[i, j] == 0:
                continue
            z_ij = edge2vec.get((i, j), tuple(0 for _ in gens))
            # residual = z_ij - zbar_j (componentwise)
            dz = tuple(a - b for a, b in zip(z_ij, zbar[j]))
            S_tilde[i, j] = S[i, j] * _vec_to_monom(dz, gens, sym_g)
    return S_tilde, V_tilde
