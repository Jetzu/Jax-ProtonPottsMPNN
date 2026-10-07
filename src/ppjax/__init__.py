# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX.
# Purpose: public surface of the JAX port of ProtonPottsMPNN.
"""ppjax - a JAX port of ProtonPottsMPNN's generation path.

The model (featurisation -> encoder -> Potts head) and the pH design engine are JAX; structure
parsing / HBPLUS / the FLAML labeller stay upstream and are consumed through :mod:`ppjax.frontend`.
``ppjax.rng`` reproduces torch's CPU RNG stream so the same seed yields the same designs.

Exports are resolved lazily (PEP 562). That is deliberate: the featurisation front end
(:mod:`ppjax.frontend`) runs in a torch environment that need not have jax installed, and the design
engine runs in a jax environment that need not have torch - so neither half may drag the other in at
import time.
"""

from typing import TYPE_CHECKING

__all__ = ["PHDesignCriteria", "PHDesignOutput", "PHDesignSet", "PHContext",
           "PottsMPNNPHEngine", "FeaturisedBackbone", "PottsScorer"]

_LOCATIONS = {
    "PHDesignCriteria": "ppjax.criteria",
    "PHDesignOutput": "ppjax.criteria",
    "PHDesignSet": "ppjax.criteria",
    "PHContext": "ppjax.engine",
    "PottsMPNNPHEngine": "ppjax.engine",
    "FeaturisedBackbone": "ppjax.frontend",
    "PottsScorer": "ppjax.scorer",
}


def __getattr__(name: str):
    module = _LOCATIONS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value          # cache, so this costs one import
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:                     # for type checkers and IDEs only
    from ppjax.criteria import PHDesignCriteria, PHDesignOutput, PHDesignSet
    from ppjax.engine import PHContext, PottsMPNNPHEngine
    from ppjax.frontend import FeaturisedBackbone
    from ppjax.scorer import PottsScorer
