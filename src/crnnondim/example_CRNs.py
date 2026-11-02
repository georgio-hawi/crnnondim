# Symbolic definitions of the example CRNs used throughout example files in CRNNondim.
# Each class provides:
#   - self._X : list of species symbols
#   - self._S : stoichiometry matrix
#   - self._V : reaction rate vector (mass-action)
# and, where convenient, named attributes for individual species / parameters.
import sympy as sp

class CRNExample:
    def __init__(self):
        # subclasses must set: self._S, self._X, self._V
        if not hasattr(self, "_S") or not hasattr(self, "_X") or not hasattr(self, "_V"):
            raise NotImplementedError("Subclass must set _S, _X, and _V.")


class Michaelis_MentenExample(CRNExample):
    def __init__(self):
        # Human-readable name for this CRN.
        self.name = "Michaelis-Menten"

        # Species: substrate S, enzyme E, complex C, product P.
        S, E, C, P = sp.symbols("S E C P", positive=True)
        self.S, self.E, self.C, self.P = S, E, C, P

        # Rate constants for the reversible binding (k1, km1) and product
        # formation / reverse step (k2, km2).
        k1, km1, k2, km2 = sp.symbols("k1 km1 k2 km2", positive=True)
        self.k1, self.km1, self.k2, self.km2 = k1, km1, k2, km2

        # Order of species in the state vector.
        self._X = [S, E, C, P]

        # Stoichiometry matrix _S: rows correspond to dS/dT, dE/dT, dC/dT, dP/dT,
        # where T is dimensional time. Columns correspond to the reactions in self._V.
        self._S = sp.Matrix(
            [
                [-1, 1, 0, 0],
                [-1, 1, 1, -1],
                [1, -1, -1, 1],
                [0, 0, 1, -1],
            ]
        )

        # Reaction vector _V:
        self._V = [
            k1 * S * E, 
            km1 * C, 
            k2 * C, 
            km2 * E * P]


class Kim_ForgerExample(CRNExample):
    def __init__(self):
        # Human-readable name for this CRN.
        self.name = "Kim-Forger"

        # Species: mRNA X, protein Y, repressor protein Z, activator S, sequestered complex C.
        X, Y, Z, S, C = sp.symbols("X Y Z S C", positive=True)
        self.X, self.Y, self.Z, self.S, self.C = X, Y, Z, S, C

        # Rate constants k1,...,k6, kf, kb, ku.
        k1, k2, k3, k4, k5, k6, kf, kb, ku = sp.symbols(
            "k1 k2 k3 k4 k5 k6 kf kb ku", positive=True
        )
        self.k1, self.k2, self.k3, self.k4, self.k5, self.k6, self.kf, self.kb, self.ku = k1, k2, k3, k4, k5, k6, kf, kb, ku

        # Order of species in the state vector.
        self._X = [X, Y, Z, S, C]

        # Stoichiometry matrix _S: rows correspond to dX/dT, dY/dT, dZ/dT, dS/dT, dC/dT,
        # where T is dimensional time. Columns correspond to the reactions in self._V.
        self._S = sp.Matrix(
            [
                [1, -1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, -1, -1, 1, 0],
                [0, 0, 0, 0, 0, 0, -1, 1, 1], 
                [0, 0, 0, 0, 0, 0, 1, -1, -1],
            ]
        )

        # Reaction vector _V:
        self._V = [
            k1 * S,  
            k2 * X,  
            k3 * X,  
            k4 * Y,  
            k5 * Y, 
            k6 * Z, 
            kf * Z * S, 
            kb * C,  
            ku * C, 
        ]

class McKeithanExample(CRNExample):
    def __init__(self):
        # Human-readable name for this CRN.
        self.name = "McKeithan"

        # Species: C0,...,C6 (complexes), M (ligand), T (receptor).
        C0, C1, C2, C3, C4, C5, C6, M, T = sp.symbols("C0 C1 C2 C3 C4 C5 C6 M T", positive=True)
        self.C0, self.C1, self.C2, self.C3, self.C4, self.C5, self.C6, self.M, self.T = (
            C0, C1, C2, C3, C4, C5, C6, M, T
        )
        
        # Rate constants: k1 (association), km1 (dissociation), k2 (forward proofreading).
        k1, km1, k2 = sp.symbols("k1 km1 k2", positive=True)
        self.k1, self.km1, self.k2 = k1, km1, k2

        # State vector ordering: complexes first, then M and T.
        self._X = [C0, C1, C2, C3, C4, C5, C6, M, T]

        # Stoichiometry matrix _S: rows correspond to dC0/dt, dC1/dt, ..., dC6/dt, dM/dt, dT/dt,
        # where T is dimensional time. Columns correspond to the reactions in self._V.
        self._S = sp.Matrix(
            [[1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 1, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0], 
            [0, 0, 0, 1, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0], 
            [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, -1, 0, 0, 0], 
            [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, -1, 0, 0], 
            [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, -1, 0], 
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1], 
            [-1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], 
            [-1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])

        # Reaction vector _V:
        self._V = [None] * 14
        self._V[0]  = k1   * M * T
        self._V[1]  = km1  * C0
        self._V[2]  = k2   * C0
        self._V[3]  = k2   * C1
        self._V[4]  = k2   * C2
        self._V[5]  = k2   * C3
        self._V[6]  = k2   * C4
        self._V[7]  = k2   * C5
        self._V[8]  = km1  * C1
        self._V[9] = km1  * C2
        self._V[10] = km1  * C3
        self._V[11] = km1  * C4
        self._V[12] = km1  * C5
        self._V[13] = km1  * C6