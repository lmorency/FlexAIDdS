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
    # C++ core: statistical mechanics
    "StatMechEngine",
    "Thermodynamics",
    "State",
    "BoltzmannLUT",
    # C++ core: parallel tempering & free energy methods
    "Replica",
    "WHAMBin",
    "TIPoint",
    # C++ core: ENCoM vibrational entropy
    "ENCoMEngine",
    "NormalMode",
    "VibrationalEntropy",
    # Physical constants
    "kB_kcal",
    "kB_SI",
    # Python models & I/O
    "PoseResult",
    "BindingModeResult",
    "DockingResult",
    "load_results",
    # Metadata
    "HAS_CORE_BINDINGS",
]
