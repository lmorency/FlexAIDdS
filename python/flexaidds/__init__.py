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
    BoltzmannLUT = None
    ENCoMEngine = None
    NormalMode = None
    Replica = None
    State = None
    StatMechEngine = None
    Thermodynamics = None
    TIPoint = None
    VibrationalEntropy = None
    WHAMBin = None
    kB_kcal = 0.0019872041
    kB_SI = 1.380649e-23
    HAS_CORE_BINDINGS = False
    # Fall back to pure-Python implementations where available
    from .thermodynamics import StatMechEngine, Thermodynamics, kB_kcal, kB_SI
    from .encom import ENCoMEngine, NormalMode, VibrationalEntropy
    from ._fallback_types import BoltzmannLUT, Replica, State, TIPoint, WHAMBin

from .models import BindingModeResult, DockingResult, PoseResult
from .results import load_results

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
    # Core types (C++ when available, pure-Python fallback otherwise)
    "StatMechEngine",
    "Thermodynamics",
    "State",
    "BoltzmannLUT",
    "Replica",
    "WHAMBin",
    "TIPoint",
    "ENCoMEngine",
    "NormalMode",
    "VibrationalEntropy",
]

__version__ = "0.1.0"
