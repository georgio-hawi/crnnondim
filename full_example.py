# End-to-end example on the Vilar2002 oscillator model.

import sympy as sp
from crnnondim import reaction, reversible, build_mass_action_system, process_CRN, find_optimal_nondim, analyse_results
def Vilar2002_example():
    """
    Reaction list for BIOMD0000000035 (Vilar2002_Oscillator)
    Species:
        A   - activator protein
        C   - A:R complex
        DA  - activator gene promoter (unbound)
        DAp - activator gene promoter bound by A
        DR  - repressor gene promoter (unbound)
        DRp - repressor gene promoter bound by A
        MA  - activator mRNA
        MR  - repressor mRNA
        R   - repressor protein

    The reactions below implement the transcription/translation and binding
    processes for this oscillator in a mass-action form.
    """
    rxns = []
    # 1) DA -> DA + MA      (alphaA)
    rxns.append(reaction("DA -> DA + MA", k="alphaA"))
    # 2) DAp -> DAp + MA    (alphaAp)
    rxns.append(reaction("DAp -> DAp + MA", k="alphaAp"))
    # 3) DR -> DR + MR      (alphaR)
    rxns.append(reaction("DR -> DR + MR", k="alphaR"))
    # 4) DRp -> DRp + MR    (alphaRp)
    rxns.append(reaction("DRp -> DRp + MR", k="alphaRp"))
    # 5) MA -> A + MA       (betaA)
    rxns.append(reaction("MA -> A + MA", k="betaA"))
    # 6) MR -> MR + R       (betaR)
    rxns.append(reaction("MR -> MR + R", k="betaR"))
    # 7) A + DA -> DAp      (gammaA)
    rxns.append(reaction("A + DA -> DAp", k="gammaA"))
    # 8) A + R -> C         (gammaC)
    rxns.append(reaction("A + R -> C", k="gammaC"))
    # 9) A + DR -> DRp      (gammaR)
    rxns.append(reaction("A + DR -> DRp", k="gammaR"))
    # 10) A -> 0            (deltaA)
    rxns.append(reaction("A -> 0", k="deltaA"))
    # 11) C -> R            (deltaA)   same degradation rate as A
    rxns.append(reaction("C -> R", k="deltaA"))
    # 12) MA -> 0           (deltaMA)
    rxns.append(reaction("MA -> 0", k="deltaMA"))
    # 13) MR -> 0           (deltaMR)
    rxns.append(reaction("MR -> 0", k="deltaMR"))
    # 14) R -> 0            (deltaR)
    rxns.append(reaction("R -> 0", k="deltaR"))
    # 15) DAp -> A + DA     (thetaA)
    rxns.append(reaction("DAp -> A + DA", k="thetaA"))
    # 16) DRp -> A + DR     (thetaR)
    rxns.append(reaction("DRp -> A + DR", k="thetaR"))
    return rxns

# Convert the reaction list into objects usable by CRNNondim and print a small summary of the parsed CRN.
parsed_out = build_mass_action_system(Vilar2002_example(), verbose=True)

species_names = parsed_out["species"]
_S = parsed_out['_S']
_X = parsed_out['_X']
_V = parsed_out['_V']
rate_constants = parsed_out['rate_constants']
expanded_ODEs = parsed_out['expanded_ODEs']

# Unpack the species variables for convenience.
A, C, DA, DAp, DR, DRp, MA, MR, R = _X

# Example dimensional initial conditions for the species.
x_0_num = {A: 0.0, C: 0.0, DA: 1.0, DAp: 0.0, DR: 1.0, DRp: 0.0, MA: 0.0, MR: 0.0, R: 0.0}

# Characteristic concentration scales for each species with non-zero initial condition.
fixed_K = {} 
for sym, val in x_0_num.items(): 
    if val > 0.0: 
        fixed_K[sym] = sp.Symbol(f"{str(sym)}_0", positive=True)

# First, analyse the CRN and print basic structural information about the CRN (if verbose=True).
pre_processed = process_CRN(_S, _V, _X, fixed_K=fixed_K, find_example_nondim=False, verbose=True, basis_type="Polynomial")

# Example dimensional rate constants used to evaluate O(1) deviation penalty.
alphaA, alphaAp, alphaR, alphaRp, betaA, betaR, deltaA, deltaMA, deltaMR, deltaR, gammaA, gammaC, gammaR, thetaA, thetaR = rate_constants
k_num = {
    alphaA: 50.0,
    alphaAp: 500.0,
    alphaR: 0.01,
    alphaRp: 50.0,
    betaA: 50.0,
    betaR: 5.0,
    gammaA: 1.0,
    gammaC: 2.0,
    gammaR: 1.0,
    deltaA: 1.0,
    deltaMA: 10.0,
    deltaMR: 0.5,
    deltaR: 0.2,
    thetaA: 50.0,
    thetaR: 100.0,
}

# Time window (in dimensional time) over which we want to consider a penalty.
T_max = 400
T_min = 100

# Non-dimensionalisation metric
nondim_metric = "deviation" # "deviation" or "maximum"

# Basis type
basis_type = "Laurent" # "Laurent" or "Polynomial"

# Dimensioanlly consistent sets
# Optional: list of sets of dimensionally consistent state variables
# default=None assumes all state variables have the same dimensions.
dimensionally_consistent_sets=None

# Compute the optimal non-dimensionalisation for this CRN.
solver_opts=dict(method="Radau", atol=1e-12, rtol=1e-9)
all_results = find_optimal_nondim(
    _S,
    _V,
    _X,
    fixed_K=fixed_K,
    dimensionally_consistent_sets=dimensionally_consistent_sets,
    k_numeric=k_num,
    x0_numeric=x_0_num,
    T_min=T_min,
    T_max=T_max,
    basis_type=basis_type,
    solver_opts=solver_opts,
    pre=pre_processed,
    nondim_metric=nondim_metric,
    n_jobs=None,  # Number of CPU cores to use. Set to None for automatic selection.
    # Optional: export results to a pickle format.
    export_results=True,
    output_path=f"out/BIOMD0000000035_{nondim_metric}_{basis_type.lower()}_nondim.pkl",
)

# Analyse the results and print the analysis (if verbose=True).
all_transformed_results = analyse_results(all_results, verbose=True)
