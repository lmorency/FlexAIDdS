"""flexaidds: Python bindings and read-only analysis helpers for FlexAID∆S."""

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
from .models import BindingModeResult, DockingResult, PoseResult
from .results import load_results

try:
    from ._core import StatMechEngine, Thermodynamics
    HAS_CORE_BINDINGS = True
except ImportError:
    StatMechEngine = None
    Thermodynamics = None
    HAS_CORE_BINDINGS = False

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
]

if HAS_CORE_BINDINGS:
    __all__.extend(["StatMechEngine", "Thermodynamics"])

__version__ = "0.1.0"
