"""flexaidds: Python bindings and read-only analysis helpers for FlexAID∆S."""

# C++ extension — optional: pure-Python helpers work without it
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
    # Fallback when C++ extension is not built
    BoltzmannLUT = None
    Replica = None
    State = None
    TIPoint = None
    WHAMBin = None
    kB_kcal = 0.001987206   # kcal mol⁻¹ K⁻¹
    kB_SI = 1.380649e-23    # J K⁻¹
    HAS_CORE_BINDINGS = False
    # Fall back to pure-Python implementations where available
    from .encom import ENCoMEngine, NormalMode, VibrationalEntropy

from .thermodynamics import StatMechEngine, Thermodynamics
from .models import BindingModeResult, DockingResult, PoseResult
from .results import load_results
from .tencom_results import FlexModeResult, FlexPopulationResult, parse_tencom_pdb, parse_tencom_json

__all__ = [
    # Python models & I/O (always available)
    "PoseResult",
    "BindingModeResult",
    "DockingResult",
    "load_results",
    # StatMech (always available via pure-Python fallback)
    "StatMechEngine",
    "Thermodynamics",
    # ENCoM (always available via pure-Python fallback)
    "ENCoMEngine",
    "NormalMode",
    "VibrationalEntropy",
    # tENCoM results (always available, pure Python)
    "FlexModeResult",
    "FlexPopulationResult",
    "parse_tencom_pdb",
    "parse_tencom_json",
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

# C++-only modules (no pure-Python fallback)
if HAS_CORE_BINDINGS:
    __all__.extend([
        "State",
        "BoltzmannLUT",
        "Replica",
        "WHAMBin",
        "TIPoint",
    ])
