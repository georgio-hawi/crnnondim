"""
Examples showing how to use poly_system_to_CRN to convert the
5D SEIRV epidemic model to a mass-action CRN representation (S, V, X, etc),
and use process_CRN to analyse it.
"""
import sympy as sp
from crnnondim import poly_system_to_CRN, process_CRN, analyse_results

# ---------------------------------------------------------------------------
# Example: 5D SEIRV epidemic model
# ---------------------------------------------------------------------------
# STATE VARIABLES (species):
#   S(t): Susceptible individuals
#   E(t): Exposed (infected but not yet infectious)
#   I(t): Infectious
#   R(t): Recovered with immunity
#   V(t): Vaccinated and currently protected
#
# PARAMETERS:
#   Lambda   : recruitment rate (birth/immigration) into S
#   mu       : natural per-capita death rate (applies to all compartments)
#   beta     : infection rate (S becomes E via contact with I)
#   sigma    : progression rate from exposed to infectious (E -> I)
#   gamma    : recovery rate from infectious to recovered (I -> R)
#   nu       : vaccination rate of susceptibles (S -> V)
#   omega    : waning immunity rate in recovered (R -> S)
#   omega_V  : waning vaccine protection rate (V -> S)
#
# MODEL EQUATIONS:
#
#   dS/dt = Lambda - beta*S*I - nu*S + omega*R + omega_V*V - mu*S
#   dE/dt = beta*S*I - sigma*E - mu*E
#   dI/dt = sigma*E - gamma*I - mu*I
#   dR/dt = gamma*I - omega*R - mu*R
#   dV/dt = nu*S - omega_V*V - mu*V
# ---------------------------------------------------------------------------

# Define species (state variables) as SymPy symbols
S, E, I, R, V = sp.symbols("S E I R V", nonnegative=True)
X_syms = [S, E, I, R, V]

# Define parameters as SymPy symbols (positive)
Lambda, mu, beta, sigma, gamma, nu, omega, omega_V = sp.symbols(
    "Lambda mu beta sigma gamma nu omega omega_V", positive=True
)

# ODE right-hand sides
dXdt = [
    Lambda - beta*S*I - nu*S + omega*R + omega_V*V - mu*S,
    beta*S*I - sigma*E - mu*E,
    sigma*E - gamma*I - mu*I,
    gamma*I - omega*R - mu*R,
    nu*S - omega_V*V - mu*V]

# Convert the ODE system into objects usable by CRNNondim and print a small summary of the parsed CRN.
parsed_out = poly_system_to_CRN(
    _X=X_syms,
    ode_rhs=dXdt,
    verbose=True
)

species_names = parsed_out["species"]
_S = parsed_out['_S']
_X = parsed_out['_X']
_V = parsed_out['_V']
rate_constants = parsed_out['rate_constants']

# Map species -> scale symbols. These scale symbols are held fixed when analysing the CRN.
S_0, I_0 = sp.symbols('S_0 I_0', positive=True)
fixed_K = {S: S_0, I: I_0}

# Analyse the CRN, optionally asking it to construct an explicit example 
# non-dimensionalisation that attains the minimum possible number of 
# free parameters (find_example_nondim=True).
all_results = process_CRN(_S, _V, _X, fixed_K=fixed_K, find_example_nondim=True, verbose=True, basis_type="Laurent")

# Analyse the results and print the analysis (if verbose=True).
all_transformed_results = analyse_results(all_results, verbose=True)