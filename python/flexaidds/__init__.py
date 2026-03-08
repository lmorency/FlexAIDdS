"""flexaidds: Python bindings and read-only analysis helpers for FlexAID∆S."""

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
    "HAS_CORE_BINDINGS",
    "PoseResult",
    "BindingModeResult",
    "DockingResult",
    "load_results",
]

if HAS_CORE_BINDINGS:
    __all__.extend(["StatMechEngine", "Thermodynamics"])

__version__ = "0.1.0"
