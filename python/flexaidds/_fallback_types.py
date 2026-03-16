"""Pure-Python fallback stubs for C++-only types.

These lightweight dataclass stubs allow ``from flexaidds import ...`` to work
even when the compiled ``_core`` extension is not available.  They mirror the
C++ pybind11 type signatures so that type-checking and basic data passing work
without the compiled extension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WHAMBin:
    """One bin from a WHAM free energy profile."""
    coord_center: float = 0.0
    free_energy: float = 0.0
    count: int = 0


@dataclass
class TIPoint:
    """One lambda point for thermodynamic integration."""
    lambda_val: float = 0.0
    dV_dlambda: float = 0.0


@dataclass
class Replica:
    """Parallel tempering replica."""
    id: int = 0
    temperature: float = 300.0
    beta: float = 0.0
    current_energy: float = 0.0


@dataclass
class State:
    """Microstate in the statistical mechanics ensemble."""
    energy: float = 0.0
    count: int = 1


@dataclass
class BoltzmannLUT:
    """Precomputed Boltzmann factor lookup table."""
    temperature: float = 300.0
    table: List[float] = field(default_factory=list)

    def lookup(self, energy: float) -> float:
        """Placeholder — C++ LUT provides O(1) lookup."""
        import math
        kB = 0.0019872041
        beta = 1.0 / (kB * self.temperature) if self.temperature > 0 else 0.0
        return math.exp(-beta * energy)
