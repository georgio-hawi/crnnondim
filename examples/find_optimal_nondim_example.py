"""
Example showing how to use find_optimal_nondim to find
the optimal non-dimensionalisation given numerical values for
typical initial conditions and rate constants.
"""

from crnnondim import find_optimal_nondim, analyse_results, Kim_ForgerExample
import sympy as sp

def Kim_Forger_example():
    # Instantiate the Kim-Forger example.
    kf_example = Kim_ForgerExample()

    # Example dimensional initial conditions for the species.
    x_0_num = {kf_example.X: 1.0, kf_example.Y: 1.0, kf_example.Z: 0.06128, kf_example.S: 0.06128, kf_example.C: 0.0}

    # Characteristic concentration scales for each species with non-zero initial condition.
    x0, y0, z0, s0 = sp.symbols('x_0 y_0 z_0 s_0', positive=True)

    # Map species -> scale symbols. These scale symbols are held fixed when searching for
    # optimal non-dimensionalisations.
    fixed_K = {kf_example.X: x0, kf_example.Y: y0, kf_example.Z: z0, kf_example.S: s0}

    # Example dimensional rate constants used to evaluate O(1) deviation penalty.
    k_num = {kf_example.k1: 0.07, kf_example.k2: 0.07, kf_example.k3: 0.05, kf_example.k4: 0.05, kf_example.k5: 0.05, kf_example.k6: 0.05, kf_example.kf: 1.0, kf_example.kb: 1.0, kf_example.ku: 0.05}

    # Time window (in dimensional time) over which we want to consider a penalty.
    T_max = 12
    T_min = T_max/2

    # Non-dimensionalisation metric
    nondim_metric = "deviation" #"maximum"

	# Dimensioanlly consistent sets
	# Optional: list of sets of dimensionally consistent state variables
	# default=None assumes all state variables have the same dimensions.
    dimensionally_consistent_sets=None
	
    # Compute the optimal non-dimensionalisation for this CRN.
    solver_opts=dict(method="Radau", atol=1e-12, rtol=1e-9)
    res = find_optimal_nondim(kf_example._S, kf_example._V, kf_example._X, fixed_K=fixed_K, 
        dimensionally_consistent_sets=dimensionally_consistent_sets,
        k_numeric=k_num, x0_numeric=x_0_num, T_min=T_min, T_max=T_max, nondim_metric=nondim_metric, solver_opts=solver_opts, basis_type="Polynomial", n_jobs=None,
        # export results to a pickle format.
        export_results=True,
        output_path=f"out/Kim_Forger_{nondim_metric}_nondim.pkl")
    return res

# Run the example.
all_results = Kim_Forger_example()

# Analyse the results and print the analysis (if verbose=True).
all_transformed_results = analyse_results(all_results, verbose=True)