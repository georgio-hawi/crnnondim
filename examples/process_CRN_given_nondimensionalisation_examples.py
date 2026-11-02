"""
Examples showing how to use process_CRN_given_nondimensionalisation to 
non-dimensionalise a CRN given a specific non-dimensionalisation.
"""
from crnnondim import process_CRN_given_nondimensionalisation, analyse_results, Michaelis_MentenExample, Kim_ForgerExample, McKeithanExample
import sympy as sp

def Michaelis_Menten_example(return_all_families=False):
    # Instantiate the Michaelis-Menten example.
    mm_example = Michaelis_MentenExample()

    # Characteristic concentration scales for S and E.
    s_0, e_0 = sp.symbols('s_0 e_0', positive=True)

    #Give an explicit choice of non-dimensionalisation.
    nondim = {mm_example.S: s_0, mm_example.E: e_0, mm_example.C: e_0, mm_example.P: e_0}

    # Non-dimensionalise the CRN given this particular non-dimensionalisation. 
    # Time scaling, via K_T, can be provided if desired.
    res = process_CRN_given_nondimensionalisation(
        mm_example._S,
        mm_example._V,
        mm_example._X,
        nondim=nondim,
        basis_type="Laurent",
        return_all_families=return_all_families
    )
    return res

def Kim_Forger_example(return_all_families=False):
    # Instantiate the Kim-Forger example.
    kf_example = Kim_ForgerExample()

    # Characteristic concentration scales for X, Y, Z, S.
    x_0, y_0, z_0, s_0 = sp.symbols('x_0 y_0 z_0 s_0', positive=True)

    #Give an explicit choice of non-dimensionalisation.
    nondim1 = {kf_example.X: x_0, kf_example.Y: y_0, kf_example.Z: z_0, kf_example.S: s_0, kf_example.C:z_0}

    # Non-dimensionalise the CRN given this particular non-dimensionalisation. 
    # Time scaling, via K_T, can be provided if desired.
    res1 = process_CRN_given_nondimensionalisation(kf_example._S, kf_example._V, kf_example._X, nondim=nondim1, basis_type="Laurent", return_all_families=return_all_families)

    # Second choice of non-dimensionalisation.
    nondim2 = {kf_example.X: x_0, kf_example.Y: y_0, kf_example.Z: z_0, kf_example.S: s_0, kf_example.C:x_0}   # species-keyed
    res2 = process_CRN_given_nondimensionalisation(kf_example._S, kf_example._V, kf_example._X, nondim=nondim2, basis_type="Laurent", return_all_families=return_all_families)

    return [res1, res2]

def McKeithan_example(return_all_families=False):
    # Instantiate the McKeithan TCR proofreading example.
    m_example = McKeithanExample()

    # Characteristic concentration scales for T and M.
    T_0, M_0 = sp.symbols('T_0 M_0', positive=True)

    # consider a dimensionalisation where the scaling factors for C0, C1, C2, C3, C4, C5, C6 are T_0.
    nondim = {m_example.T: T_0, m_example.M: M_0}
    for sym in [m_example.C0, m_example.C1, m_example.C2, m_example.C3, m_example.C4, m_example.C5, m_example.C6]:
        # All complexes share the same concentration scale T_0.
        nondim[sym] = T_0

    # Non-dimensionalise the CRN given this particular non-dimensionalisation. 
    # Time scaling, via K_T, can be provided if desired.
    res = process_CRN_given_nondimensionalisation(m_example._S, m_example._V, m_example._X, nondim=nondim, basis_type="Laurent", return_all_families=return_all_families)
    return res

# Choose which example to run and whether to return all families. If return_all_families is true,
# all families where K_T is chosen to minimise the number of free parameters will be printed.
example_name = "Michaelis-Menten"
return_all_families = True
match example_name:
    case "Michaelis-Menten":
        all_results = Michaelis_Menten_example(return_all_families=return_all_families)
    case "Kim-Forger":
        all_results = Kim_Forger_example(return_all_families=return_all_families)
    case _:
        all_results = McKeithan_example(return_all_families=return_all_families)

# Analyse the results and print the analysis (if verbose=True).
all_transformed_results = analyse_results(all_results, verbose=True)