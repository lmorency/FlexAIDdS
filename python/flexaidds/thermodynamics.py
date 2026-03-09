"""Statistical mechanics and thermodynamic analysis for FlexAID∆S.

Provides Pythonic wrappers around C++ StatMechEngine with NumPy integration.
"""

import math
from typing import List, Optional, Tuple
from dataclasses import dataclass

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    from . import _core
except ImportError:
    _core = None

# Physical constants
if _core is not None:
    kB_kcal: float = _core.kB_kcal  # kcal mol⁻¹ K⁻¹
    kB_SI: float = _core.kB_SI      # J K⁻¹
else:
    kB_kcal = 0.001987206
    kB_SI = 1.380649e-23


@dataclass
class Thermodynamics:
    """Complete thermodynamic properties of a conformational ensemble.
    
    Attributes:
        temperature: Temperature in Kelvin
        log_Z: Natural logarithm of partition function
        free_energy: Helmholtz free energy F = -kT ln Z (kcal/mol)
        mean_energy: Boltzmann-weighted average energy ⟨E⟩ (kcal/mol)
        mean_energy_sq: ⟨E²⟩ for variance calculation
        heat_capacity: Cv = (⟨E²⟩ - ⟨E⟩²) / (kT²) (kcal mol⁻¹ K⁻²)
        entropy: Configurational entropy S = (⟨E⟩ - F) / T (kcal mol⁻¹ K⁻¹)
        std_energy: Standard deviation of energy σ_E (kcal/mol)
    """
    temperature: float
    log_Z: float
    free_energy: float
    mean_energy: float
    mean_energy_sq: float
    heat_capacity: float
    entropy: float
    std_energy: float
    
    @property
    def binding_free_energy(self) -> float:
        """Alias for free_energy (common in docking context)."""
        return self.free_energy
    
    @property
    def entropy_term(self) -> float:
        """Entropic contribution to free energy: TΔS (kcal/mol)."""
        return self.temperature * self.entropy
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'temperature_K': self.temperature,
            'log_Z': self.log_Z,
            'free_energy_kcal_mol': self.free_energy,
            'enthalpy_kcal_mol': self.mean_energy,
            'entropy_kcal_mol_K': self.entropy,
            'heat_capacity_kcal_mol_K2': self.heat_capacity,
            'std_energy_kcal_mol': self.std_energy,
        }


class _PyStatMechEngine:
    """Pure-Python canonical-ensemble engine (fallback when C++ _core is absent).

    Uses the log-sum-exp trick for numerical stability.
    """

    def __init__(self, temperature_K: float) -> None:
        self._T = float(temperature_K)
        self._beta = 1.0 / (kB_kcal * self._T)
        self._energies: List[float] = []

    # ------------------------------------------------------------------
    # sample accumulation
    # ------------------------------------------------------------------

    def add_sample(self, energy: float, multiplicity: int = 1) -> None:
        for _ in range(max(1, int(multiplicity))):
            self._energies.append(float(energy))

    def clear(self) -> None:
        self._energies.clear()

    @property
    def size(self) -> int:
        return len(self._energies)

    @property
    def temperature(self) -> float:
        return self._T

    @property
    def beta(self) -> float:
        return self._beta

    # ------------------------------------------------------------------
    # thermodynamic computation
    # ------------------------------------------------------------------

    def compute(self) -> Thermodynamics:
        if not self._energies:
            raise RuntimeError("No samples added to StatMechEngine before compute()")

        e = self._energies
        n = len(e)
        e_min = min(e)

        # log Z via log-sum-exp trick
        shifted = [-self._beta * (ei - e_min) for ei in e]
        log_sum = math.log(sum(math.exp(s) for s in shifted))
        log_Z = -self._beta * e_min + log_sum

        # Boltzmann weights
        log_w = [-self._beta * ei - log_Z for ei in e]
        w = [math.exp(lw) for lw in log_w]

        mean_e = sum(wi * ei for wi, ei in zip(w, e))
        mean_e2 = sum(wi * ei * ei for wi, ei in zip(w, e))
        var_e = mean_e2 - mean_e ** 2
        std_e = math.sqrt(max(0.0, var_e))

        free_energy = -kB_kcal * self._T * log_Z
        heat_capacity = var_e / (kB_kcal * self._T ** 2)
        entropy = (mean_e - free_energy) / self._T

        return Thermodynamics(
            temperature=self._T,
            log_Z=log_Z,
            free_energy=free_energy,
            mean_energy=mean_e,
            mean_energy_sq=mean_e2,
            heat_capacity=heat_capacity,
            entropy=entropy,
            std_energy=std_e,
        )

    def boltzmann_weights(self) -> List[float]:
        if not self._energies:
            return []
        thermo = self.compute()
        log_Z = thermo.log_Z
        return [math.exp(-self._beta * ei - log_Z) for ei in self._energies]

    def delta_G(self, other: "_PyStatMechEngine") -> float:
        return self.compute().free_energy - other.compute().free_energy


class StatMechEngine:
    """Statistical mechanics engine for conformational ensembles.

    Computes partition functions, free energies, entropies, and heat capacities
    from sampled configurations using canonical ensemble formalism.

    Example:
        >>> engine = StatMechEngine(temperature_K=300.0)
        >>> engine.add_samples([-10.5, -9.8, -10.2, -11.0])  # energies in kcal/mol
        >>> thermo = engine.compute()
        >>> print(f"Free energy: {thermo.free_energy:.2f} kcal/mol")
        >>> print(f"Entropy: {thermo.entropy:.5f} kcal/(mol·K)")
    """

    def __init__(self, temperature_K: float = 300.0):
        """Initialize engine at specified temperature.

        Args:
            temperature_K: Simulation temperature in Kelvin (default 300K)
        """
        if _core is not None:
            self._engine = _core.StatMechEngine(temperature_K)
        else:
            self._engine = _PyStatMechEngine(temperature_K)
    
    def add_sample(self, energy: float, multiplicity: int = 1) -> None:
        """Add a single sampled configuration.
        
        Args:
            energy: Configuration energy in kcal/mol (negative = favorable)
            multiplicity: Degeneracy/sampling count (default 1)
        """
        self._engine.add_sample(energy, multiplicity)
    
    def add_samples(self, energies) -> None:
        """Add multiple configurations from a sequence or NumPy array.

        Args:
            energies: Iterable of configuration energies (kcal/mol)
        """
        if np is not None:
            energies = np.asarray(energies, dtype=np.float64)
        for e in energies:
            self._engine.add_sample(float(e))
    
    def compute(self) -> Thermodynamics:
        """Compute full thermodynamics from current ensemble.
        
        Returns:
            Thermodynamics object with F, S, H, Cv, etc.
        """
        thermo_cpp = self._engine.compute()
        return Thermodynamics(
            temperature=thermo_cpp.temperature,
            log_Z=thermo_cpp.log_Z,
            free_energy=thermo_cpp.free_energy,
            mean_energy=thermo_cpp.mean_energy,
            mean_energy_sq=thermo_cpp.mean_energy_sq,
            heat_capacity=thermo_cpp.heat_capacity,
            entropy=thermo_cpp.entropy,
            std_energy=thermo_cpp.std_energy,
        )
    
    def boltzmann_weights(self):
        """Get Boltzmann weights for all samples.

        Returns:
            NumPy array of normalized weights (sum to 1.0), or a plain list
            when NumPy is not available.
        """
        weights = self._engine.boltzmann_weights()
        if np is not None:
            return np.array(weights)
        return list(weights)
    
    def delta_G(self, reference: 'StatMechEngine') -> float:
        """Compute relative free energy to another ensemble.
        
        Args:
            reference: Reference StatMechEngine
        
        Returns:
            ΔG = F_this - F_reference (kcal/mol)
        """
        return self._engine.delta_G(reference._engine)
    
    def clear(self) -> None:
        """Remove all samples from ensemble."""
        self._engine.clear()
    
    @property
    def temperature(self) -> float:
        """Temperature in Kelvin."""
        return self._engine.temperature
    
    @property
    def beta(self) -> float:
        """Thermodynamic beta = 1/(kT) in (kcal/mol)⁻¹."""
        return self._engine.beta
    
    @property
    def n_samples(self) -> int:
        """Number of configurations in ensemble."""
        return self._engine.size
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __repr__(self) -> str:
        return f"<StatMechEngine T={self.temperature:.1f}K n_samples={self.n_samples}>"


class BoltzmannLUT:
    """Pre-tabulated Boltzmann factors for fast inner-loop evaluation.
    
    Provides O(1) lookup for exp(-βE) over a specified energy range.
    
    Example:
        >>> lut = BoltzmannLUT(beta=1.688, e_min=-20.0, e_max=5.0, n_bins=10000)
        >>> weight = lut(-12.5)  # exp(-β × -12.5)
    """
    
    def __init__(self, beta: float, e_min: float, e_max: float, n_bins: int = 10000):
        """Initialize lookup table.
        
        Args:
            beta: 1/(kT) in (kcal/mol)⁻¹
            e_min: Minimum energy (kcal/mol)
            e_max: Maximum energy (kcal/mol)
            n_bins: Number of table entries (default 10000)
        """
        if _core is None:
            raise RuntimeError("C++ bindings not available")
        self._lut = _core.BoltzmannLUT(beta, e_min, e_max, n_bins)
    
    def __call__(self, energy: float) -> float:
        """Look up Boltzmann factor for given energy.
        
        Args:
            energy: Energy in kcal/mol
        
        Returns:
            exp(-βE)
        """
        return self._lut(energy)


def helmholtz_from_energies(energies, temperature: float = 300.0) -> float:
    """Convenience function: Helmholtz free energy from energy array.

    Works with or without C++ bindings (uses pure-Python engine as fallback).

    Args:
        energies: Iterable of configuration energies (kcal/mol)
        temperature: Temperature in Kelvin

    Returns:
        Helmholtz free energy F (kcal/mol)
    """
    engine = StatMechEngine(temperature)
    engine.add_samples(energies)
    return engine.compute().free_energy
