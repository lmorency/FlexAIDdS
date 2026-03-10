"""flexaidds: Python bindings and read-only analysis helpers for FlexAID∆S."""

try:
    from ._core import (
        BoltzmannLUT,
        ENCoMEngine,
        NormalMode,
        Replica,
        State,
        TIPoint,
        VibrationalEntropy,
        WHAMBin,
        kB_kcal,
        kB_SI,
    )
    HAS_CORE_BINDINGS = True
except ImportError:
    BoltzmannLUT = None
    ENCoMEngine = None
    NormalMode = None
    Replica = None
    State = None
    TIPoint = None
    VibrationalEntropy = None
    WHAMBin = None
    kB_kcal = 0.001987206   # kcal mol⁻¹ K⁻¹
    kB_SI = 1.380649e-23    # J K⁻¹
    HAS_CORE_BINDINGS = False

from .thermodynamics import StatMechEngine, Thermodynamics
from .models import BindingModeResult, DockingResult, PoseResult
from .results import load_results

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
    # Feature flag
    "HAS_CORE_BINDINGS",
]

__version__ = "0.1.0"
