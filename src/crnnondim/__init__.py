"""
CRNNondim: tools for analysing and nondimensionalising chemical reaction networks.
"""
from __future__ import annotations

import sys as _sys
from types import ModuleType as _ModuleType

__all__ = [
    "reaction",
    "reversible",
    "build_mass_action_system",
    "poly_system_to_CRN",
    "process_CRN",
    "process_CRN_given_nondimensionalisation",
    "find_optimal_nondim",
    "analyse_results",
    "Michaelis_MentenExample",
    "Kim_ForgerExample",
    "McKeithanExample"
]

CITATION = """
************************************************************************************************************************************
Algorithmic non-dimensionalisation of polynomial dynamical systems via CRNNondim.
If you use this software in your research, please cite it as follows:
Georgio Hawi and Martin Wechselberger. Algorithmic Non-dimensionalisation of Polynomial Dynamical Systems with CRNNondim, 2026.
************************************************************************************************************************************
"""

def _print_citation_once() -> None:
    # This ensures the banner only prints once per Python process
    if not getattr(_print_citation_once, "_already_printed", False):
        print(CITATION)
        _print_citation_once._already_printed = True

# Print the citation when the package is first imported
#_print_citation_once()

def __getattr__(name: str):
    """
    Lazy attribute loader for the crnnondim top-level API.

    This keeps import time fast by only importing heavy submodules
    when their symbols are first accessed.
    """

    if name in ("reaction", "reversible", "build_mass_action_system"):
        from . import parse_reactions as _pr

        globals().update(
            {
                "reaction": _pr.reaction,
                "reversible": _pr.reversible,
                "build_mass_action_system": _pr.build_mass_action_system,
            }
        )
        return globals()[name]

    if name in (
        "poly_system_to_CRN", "process_CRN",
        "process_CRN_given_nondimensionalisation", "find_optimal_nondim",
        "analyse_results",
    ):
        # Keep the package attribute a (callable) module, even after lazy loading.
        # Replacing it with its function breaks `import crnnondim.name as module`.
        from importlib import import_module
        return import_module(f".{name}", __name__)

    if name in ("Michaelis_MentenExample", "Kim_ForgerExample", "McKeithanExample"):
        from . import example_CRNs as _ex

        globals().update(
            {
                "Michaelis_MentenExample": _ex.Michaelis_MentenExample,
                "Kim_ForgerExample": _ex.Kim_ForgerExample,
                "McKeithanExample": _ex.McKeithanExample,
            }
        )
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # Make crnnondim and auto-complete aware of the public API
    return sorted(set(list(globals().keys()) + __all__))


class _CallableModule(_ModuleType):
    """Forward calls to the public function without replacing the submodule."""

    @property
    def __wrapped__(self):
        return getattr(self, self.__name__.rsplit(".", 1)[-1])

    def __call__(self, *args, **kwargs):
        return self.__wrapped__(*args, **kwargs)


class _PublicModule(_ModuleType):
    """Preserve callable public names when submodule imports bind package attributes."""

    def __setattr__(self, name: str, value) -> None:
        if name in __all__ and isinstance(value, _ModuleType):
            if callable(getattr(value, name, None)):
                value.__class__ = _CallableModule
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _PublicModule
