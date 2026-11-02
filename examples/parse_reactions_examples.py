"""
Examples showing how to use build_mass_action_system to convert
a list of reactions to a mass-action CRN representation (S, V, X, etc).
"""
from crnnondim import reaction, reversible, build_mass_action_system

def Michaelis_Menten_example():
    """
    Defines the reaction list for the reversible Michaelis-Menten scheme.

    Returns
    -------
    rxns : list
        List of reaction objects (forward + reverse) that can be passed to
        build_mass_action_system.
    """
    rxns = []
    # E + S <-> C with forward rate k1 and reverse rate km1
    rxns += reversible("E + S <-> C", kf="k1",  kr="km1")
    # C <-> P + E with forward rate k2 and reverse rate km2
    rxns += reversible("C <-> P + E", kf="k2",  kr="km2")
    return rxns

def Kim_Forger_example():
    """
    Defines the reaction list for the Kim-Forger (KF) model, a simplified
    circadian clock model.
    """
    rxns = [
        # S -> X + S : basal production of X driven by S (rate constant k1)
        reaction("S -> X + S", k="k1"),
        # X -> 0 : degradation/removal of X (rate constant k2)
        reaction("X -> 0", k="k2"),
        # X -> X + Y : production of Y catalysed by X (rate constant k3)
        reaction("X -> X + Y", k="k3"),
        # Y -> 0 : degradation/removal of Y (rate constant k4)
        reaction("Y -> 0",     k="k4"),
        # Y -> Y + Z : production of Z catalysed by Y (rate constant k5)
        reaction("Y -> Y + Z", k="k5"),
        # Z -> 0 : degradation/removal of Z (rate constant k6)
        reaction("Z -> 0",     k="k6"),
        # C -> S : conversion/decay of complex C back to free S
        reaction("C -> S",     k="ku")]
    # Reversible binding/unbinding between S and Z to form complex C.
    # Forward binding S + Z -> C has rate constant kf,
    # reverse unbinding C -> S + Z has rate constant kb.
    rxns+= reversible("S + Z <-> C", kf="kf", kr="kb")
    return rxns

def McKeithan_example(n=6):
    """
    Defines the reaction list for the McKeithan TCR proofreading chain with n
    intermediate ligand-receptor complexes.

    Parameters
    ----------
    n : int
        Number of proofreading steps (C0 -> C1 -> ... -> Cn).
    """
    rxns = []
    # Association/dissociation of the first complex C0.
    rxns.extend(reversible("T + M <-> C0", kf="k1", kr="km1"))
    # Forward proofreading step through the chain of complexes.
    for i in range(n):
        rxns.append(reaction(f"C{i} -> C{i+1}", k=f"k2"))  # forward proofreading step
    # Each complex can dissociate back to T + M with the same rate km1.
    for i in range(1,n+1):
        rxns.append(reaction(f"C{i} -> T + M", k=f"km1"))
    return rxns

# Select which example to parse and convert into a mass-action CRN.
example_name = "McKeithan"

match example_name:
    case "Michaelis-Menten":
        reactions = Michaelis_Menten_example()
    case "Kim-Forger":
        reactions = Kim_Forger_example()
    case _:
        reactions = McKeithan_example()

# Convert the reaction list into objects usable by CRNNondim and print a small summary of the parsed CRN.
parsed_out = build_mass_action_system(reactions, verbose=True)

species_names = parsed_out["species"]
_S = parsed_out['_S']
_X = parsed_out['_X']
_V = parsed_out['_V']
rate_constants = parsed_out['rate_constants']
expanded_ODEs = parsed_out['expanded_ODEs']