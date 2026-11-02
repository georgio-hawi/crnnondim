"""
Examples showing how to use process_CRN to analyse CRNs, derive restrictions 
on scaling factors so that the number of free parameters in the resulting non-dimensional 
system is minimised, and construct an example non-dimensionalisation that attains this minimum.
"""
from crnnondim import process_CRN, analyse_results, Michaelis_MentenExample, Kim_ForgerExample, McKeithanExample
import sympy as sp

def Michaelis_Menten_example(find_example_nondim=False):
    # Instantiate the Michaelis-Menten example.
    mm_example = Michaelis_MentenExample()

    # Characteristic concentration scales for S and E.
    s0, e0 = sp.symbols('s_0 e_0', positive=True)

    # Map species -> scale symbols. These scale symbols are held fixed when analysing the CRN.
    fixed_K = {mm_example.S: s0, mm_example.E: e0}

    # Analyse the CRN, optionally asking it to construct an explicit example 
    # non-dimensionalisation that attains the minimum possible number of 
    # free parameters (find_example_nondim=True).
    res = process_CRN(mm_example._S, mm_example._V, mm_example._X, fixed_K=fixed_K, find_example_nondim=find_example_nondim, verbose=True, basis_type="Laurent")
    return res

def Kim_Forger_example(find_example_nondim=False):
    # Instantiate the Kim-Forger example.
    kf_example = Kim_ForgerExample()

    # Characteristic concentration scales for X, Y, Z, S.
    x0, y0, z0, s0 = sp.symbols('x_0 y_0 z_0 s_0', positive=True)

    # Map species -> scale symbols. These scale symbols are held fixed when analysing the CRN.
    fixed_K = {kf_example.X: x0, kf_example.Y: y0, kf_example.Z: z0, kf_example.S: s0}

    # Analyse the CRN, optionally asking it to construct an explicit example 
    # non-dimensionalisation that attains the minimum possible number of 
    # free parameters (find_example_nondim=True).
    res = process_CRN(kf_example._S, kf_example._V, kf_example._X, fixed_K=fixed_K, find_example_nondim=find_example_nondim, verbose=True, basis_type="Laurent")
    return res

def McKeithan_example(find_example_nondim=False):
    # Instantiate the McKeithan TCR proofreading example.
    m_example = McKeithanExample()

    # Characteristic concentration scales for T and M.
    T_0, M_0 = sp.symbols('T_0 M_0', positive=True)

    # Map species -> scale symbols. These scale symbols are held fixed when analysing the CRN.
    fixed_K = {m_example.T: T_0, m_example.M: M_0}

    # Analyse the CRN, optionally asking it to construct an explicit example 
    # non-dimensionalisation that attains the minimum possible number of 
    # free parameters (find_example_nondim=True).
    res = process_CRN(m_example._S, m_example._V, m_example._X, fixed_K=fixed_K, find_example_nondim=find_example_nondim, verbose=True, basis_type="Laurent")
    return res

# Choose which example to run.
example_name = "McKeithan"
find_example_nondim = True
match example_name:
    case "Michaelis-Menten":
        all_results = Michaelis_Menten_example(find_example_nondim=find_example_nondim)
    case "Kim-Forger":
        all_results = Kim_Forger_example(find_example_nondim=find_example_nondim)
    case _:
        all_results = McKeithan_example(find_example_nondim=find_example_nondim)

# Analyse the results and print the analysis (if verbose=True).
all_transformed_results = analyse_results(all_results, verbose=True)