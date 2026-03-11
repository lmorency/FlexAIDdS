"""flexaidds: Python bindings and read-only analysis helpers for FlexAID∆S."""

from .models import BindingModeResult, DockingResult, PoseResult
from .results import load_results
from .__version__ import __version__

try:
    from ._core import (
        BoltzmannLUT,
        ENCoMEngine,
        NormalMode,
        Replica,
        State,
        StatMechEngine,
        Thermodynamics,
        TIPoint,
        VibrationalEntropy,
        WHAMBin,
        kB_kcal,
        kB_SI,
    )
    HAS_CORE_BINDINGS = True
except ImportError:
    HAS_CORE_BINDINGS = False
    # Fall back to pure-Python implementations where available
    from .thermodynamics import StatMechEngine, Thermodynamics, kB_kcal, kB_SI
    from .encom import (
        ENCoMEngine,
        NormalMode,
        VibrationalEntropy,
    )
    # These have no pure-Python fallback
    BoltzmannLUT = None
    Replica = None
    State = None
    TIPoint = None
    WHAMBin = None

__all__ = [
    # Python models & I/O (always available)
    "PoseResult",
    "BindingModeResult",
    "DockingResult",
    "load_results",
    # Physical constants (always available)
    "kB_kcal",
    "kB_SI",
    # Availability flag
    "HAS_CORE_BINDINGS",
    # Thermodynamics (always available via pure-Python fallback)
    "StatMechEngine",
    "Thermodynamics",
    # ENCoM (always available via pure-Python fallback)
    "ENCoMEngine",
    "NormalMode",
    "VibrationalEntropy",
]

# C++-only modules (no pure-Python fallback)
if HAS_CORE_BINDINGS:
    __all__.extend([
        "State",
        "BoltzmannLUT",
        "Replica",
        "WHAMBin",
        "TIPoint",
    ])
