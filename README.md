# CRNNondim
CRNNondim is a Python library for automating the non-dimensionalisation of polynomial dynamical systems via chemical reaction networks (CRNs). It was developed by Georgio Hawi and Martin Wechselberger, with an accompanying paper available soon. Please direct all enquiries to georgio.hawi (at) sydney.edu.au.

Given a polynomial CRN, CRNNondim helps you move from human-readable reaction lists to carefully chosen dimensionless variables and parameters. You can analyse a proposed non-dimensionalisation, or ask the library to search for one that is optimal under a precise objective (for example, keeping parameters O(1) over a chosen time window).

CRNNondim provides tools to automate analysing and non-dimensionalising chemical reaction networks (CRNs). 
CRNNondim includes functions that let you:
- parse readable reaction network descriptions or a polynomial ODE system into a symbolic CRN-like ODE system (stoichiometry matrix `S`, reaction vector `V`, species vector `X`, and rate constants),
- analyse CRNs to derive algebraic constraints on concentration and time scales and construct non-dimensionalisations that minimise the number of free parameters,
- non-dimensionalise a CRN given an explicit choice of species scales, so you can test or refine hand-crafted scalings,
- search for optimal non-dimensionalisations using typical initial conditions and rate constants, targeting O(1) parameters over a specified time window,
- run end-to-end worked examples such as the Vilar2002 oscillator [(BIOMD0000000035)](https://www.ebi.ac.uk/biomodels/BIOMD0000000035) and several well-known CRNs (Michaelis-Menten, Kim-Forger, McKeithan, and SEIRV models) via `full_example.py` and the scripts in `examples/`.

## Installation

From PyPI (recommended):

```bash
pip install crnnondim
```

From source with uv:

```bash
git clone https://github.com/georgio-hawi/crnnondim.git
cd crnnondim
uv sync
```

## Quick start

After setting up the project with uv, run the full end-to-end example:

```bash
cd crnnondim
uv run python full_example.py
```

Or use the library directly:

```python
from crnnondim import (
    reaction,
    reversible,
    build_mass_action_system,
    poly_system_to_CRN,
    process_CRN,
    process_CRN_given_nondimensionalisation,
    find_optimal_nondim,
    analyse_results
)

# Define reactions and build a CRN/polynomial system...
```

More focused examples are in the [`examples/`](examples/) directory:

- `examples/parse_reactions_examples.py`
- `examples/process_CRN_examples.py`
- `examples/process_CRN_given_nondimensionalisation_examples.py`
- `examples/find_optimal_nondim_example.py`
- `examples/5D_SEIRV_example.py`

For `find_optimal_nondim_example`, for example, simply run:

```bash
uv run python examples/find_optimal_nondim_example.py
```

## Project layout

```text
crnnondim/
├── CITATION.cff
├── full_example.py              # full worked Vilar2002 example
├── LICENSE
├── pyproject.toml
├── README.md
├── uv.lock
├── src/
│   └── crnnondim/               # importable package
│       ├── __init__.py
│       ├── _LEXIWrapper.py
│       ├── analyse_results.py
│       ├── CRN_analyser.py
│       ├── example_CRNs.py
│       ├── factor_generators_columnwise.py
│       ├── find_optimal_nondim.py
│       ├── objective_function_utils.py
│       ├── parse_reactions.py
│       ├── poly_system_to_CRN.py
│       ├── process_CRN.py
│       ├── process_CRN_given_nondimensionalisation.py
│       └── utils.py
├── examples/                    # example scripts
│   ├── 5D_SEIRV_example.py
│   ├── find_optimal_nondim_example.py
│   ├── parse_reactions_examples.py
│   ├── process_CRN_examples.py
│   └── process_CRN_given_nondimensionalisation_examples.py
└── out/                         # saved non-dimensionalisation results
    ├── BIOMD0000000035_deviation_laurent_nondim.pkl
    ├── BIOMD0000000035_deviation_polynomial_nondim.pkl
    ├── BIOMD0000000035_maximum_laurent_nondim.pkl
    ├── BIOMD0000000035_maximum_polynomial_nondim.pkl
    ├── Kim_Forger_deviation_nondim.pkl
    └── Kim_Forger_maximum_nondim.pkl
```

## Citation

If you use this software in your research, please cite it as follows:

> Georgio Hawi and Martin Wechselberger. **Algorithmic Non-dimensionalisation of Polynomial Dynamical Systems with CRNNondim**, 2026.

```bibtex
@article{Hawi2026,
  title = {Algorithmic Non-dimensionalisation of Polynomial Dynamical Systems with CRNNondim},
  author = {Georgio Hawi and Martin Wechselberger},
  year = {2026},
}
```

## Licence

This project is licensed under the terms of the MIT licence.
