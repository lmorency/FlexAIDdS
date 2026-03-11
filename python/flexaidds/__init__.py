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
    BoltzmannLUT = None
    ENCoMEngine = None
    NormalMode = None
    Replica = None
    State = None
    TIPoint = None
    VibrationalEntropy = None
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
    # Pure-Python fallbacks (always available)
    "StatMechEngine",
    "Thermodynamics",
]

# C++ core modules (only available when compiled)
if HAS_CORE_BINDINGS:
    __all__.extend([
        "State",
        "BoltzmannLUT",
        "Replica",
        "WHAMBin",
        "TIPoint",
        "ENCoMEngine",
        "NormalMode",
        "VibrationalEntropy",
    ])
