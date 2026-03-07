"""High-level docking interface for FlexAID∆S.

Provides Pythonic API for molecular docking workflows.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .thermodynamics import Thermodynamics, StatMechEngine

try:
    from . import _core
except ImportError:
    _core = None


@dataclass
class Pose:
    """Single docked pose within a binding mode.
    
    Attributes:
        index: Pose index in GA population
        energy: Binding energy (CF score) in kcal/mol
        rmsd: RMSD to reference structure (if available)
        coordinates: Atomic coordinates (Nx3 array)
        boltzmann_weight: Statistical weight in ensemble
    """
    index: int
    energy: float
    rmsd: Optional[float] = None
    coordinates: Optional[np.ndarray] = None
    boltzmann_weight: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'index': self.index,
            'energy_kcal_mol': self.energy,
            'rmsd_angstrom': self.rmsd,
            'boltzmann_weight': self.boltzmann_weight,
        }


class BindingMode:
    """Binding mode: cluster of docked poses with thermodynamic scoring.
    
    A binding mode represents a distinct local minimum on the binding energy
    landscape, characterized by an ensemble of similar poses.
    
    Example:
        >>> mode = results.binding_modes[0]  # top-ranked mode
        >>> thermo = mode.get_thermodynamics()
        >>> print(f"ΔG = {thermo.free_energy:.2f} kcal/mol")
        >>> print(f"ΔH = {thermo.mean_energy:.2f}, TΔS = {thermo.entropy_term:.2f}")
    """
    
    def __init__(self, cpp_binding_mode=None):
        """Initialize from C++ BindingMode object (internal use)."""
        self._cpp_mode = cpp_binding_mode
        self._poses: List[Pose] = []
    
    def get_thermodynamics(self) -> Thermodynamics:
        """Get full thermodynamic properties of this binding mode.
        
        Returns:
            Thermodynamics object with F, S, H, Cv, etc.
        """
        if self._cpp_mode is None:
            raise RuntimeError("C++ binding mode not initialized")
        thermo_cpp = self._cpp_mode.get_thermodynamics()
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
    
    @property
    def free_energy(self) -> float:
        """Helmholtz free energy F = H - TS (kcal/mol)."""
        if self._cpp_mode:
            return self._cpp_mode.get_free_energy()
        return float('inf')
    
    @property
    def enthalpy(self) -> float:
        """Boltzmann-weighted average energy ⟨E⟩ (kcal/mol)."""
        if self._cpp_mode:
            return self._cpp_mode.compute_enthalpy()
        return float('inf')
    
    @property
    def entropy(self) -> float:
        """Configurational entropy S (kcal mol⁻¹ K⁻¹)."""
        if self._cpp_mode:
            return self._cpp_mode.compute_entropy()
        return 0.0
    
    @property
    def n_poses(self) -> int:
        """Number of poses in this binding mode."""
        if self._cpp_mode:
            return self._cpp_mode.get_BindingMode_size()
        return len(self._poses)
    
    def __len__(self) -> int:
        return self.n_poses
    
    def __repr__(self) -> str:
        return (f"<BindingMode n_poses={self.n_poses} "
                f"F={self.free_energy:.2f} H={self.enthalpy:.2f} "
                f"S={self.entropy:.5f}>")


class BindingPopulation:
    """Collection of binding modes from a docking run.
    
    Provides ensemble-level analysis and ranking of binding modes.
    """
    
    def __init__(self):
        self._modes: List[BindingMode] = []
        self._temperature: float = 300.0
    
    def add_mode(self, mode: BindingMode) -> None:
        """Add a binding mode to the population."""
        self._modes.append(mode)
    
    def rank_by_free_energy(self) -> List[BindingMode]:
        """Return binding modes sorted by free energy (best first)."""
        return sorted(self._modes, key=lambda m: m.free_energy)
    
    def compute_global_thermodynamics(self) -> Thermodynamics:
        """Compute thermodynamics over all binding modes.
        
        Returns:
            Global ensemble thermodynamics
        """
        engine = StatMechEngine(self._temperature)
        for mode in self._modes:
            # Aggregate all pose energies from all modes
            for _ in range(mode.n_poses):
                engine.add_sample(mode.enthalpy)  # Simplified: use mode average
        return engine.compute()
    
    @property
    def n_modes(self) -> int:
        """Number of binding modes."""
        return len(self._modes)
    
    def __len__(self) -> int:
        return self.n_modes
    
    def __getitem__(self, index: int) -> BindingMode:
        return self._modes[index]
    
    def __iter__(self):
        return iter(self._modes)
    
    def __repr__(self) -> str:
        return f"<BindingPopulation n_modes={self.n_modes} T={self._temperature}K>"


class Docking:
    """High-level interface for FlexAID∆S molecular docking.
    
    Example:
        >>> docking = Docking("config.inp")
        >>> results = docking.run()
        >>> top_mode = results.binding_modes[0]
        >>> print(f"Best ΔG: {top_mode.free_energy:.2f} kcal/mol")
    """
    
    def __init__(self, config_file: str):
        """Initialize docking from configuration file.
        
        Args:
            config_file: Path to FlexAID .inp config file
        """
        self.config_file = Path(config_file)
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        self._config: Dict[str, Any] = {}
        self._parse_config()
    
    def _parse_config(self) -> None:
        """Parse FlexAID config file.

        The format is fixed-width: the first 6 characters are the keyword,
        character 7 is a space delimiter, and the remainder of the line is the
        value.  Lines that start with '#' or are blank are ignored.

        Keywords that may appear multiple times (OPTIMZ, FLEXSC) are collected
        into lists.  All other keywords map to a single string value (or a
        boolean ``True`` for flag-only keywords such as EXCHET, ROTOBS, etc.).

        After parsing, ``self._config`` is populated with keys matching the
        6-character keyword names used by the C++ FlexAID engine.
        """
        # Keywords whose value is the rest of the line (path / string).
        _string_keys = {
            "PDBNAM", "INPLIG", "RNGOPT", "METOPT", "BPKENM", "COMPLF",
            "VCTSCO", "IMATRX", "DEFTYP", "CONSTR", "NMAAMP", "NMAEIG",
            "RMSDST", "DEPSPA", "STATEP", "TEMPOP", "CLUSTA",
        }
        # Keywords whose value is a float.
        _float_keys = {
            "ACSWEI", "CLRMSD", "PERMEA", "INTRAF", "VARDIS", "VARANG",
            "VARDIH", "VARFLX", "SLVPEN", "DEECLA", "ROTPER", "SPACER",
        }
        # Keywords whose value is an integer.
        _int_keys = {
            "NMAMOD", "MAXRES", "TEMPER", "NRGOUT",
        }
        # Keywords that appear multiple times; values are collected into a list.
        _list_keys = {"OPTIMZ", "FLEXSC"}
        # Flag-only keywords: presence means True, no value expected.
        _flag_keys = {
            "DEEFLX", "ROTOBS", "NORMAR", "USEACS", "EXCHET", "INCHOH",
            "NOINTR", "OMITBU", "VINDEX", "HTPMOD", "OUTRNG", "USEDEE",
            "NRGSUI", "SCOLIG", "SCOOUT", "ROTOUT",
        }

        # Initialise list accumulators so callers can always iterate them.
        for key in _list_keys:
            self._config[key] = []

        with open(self.config_file) as fh:
            for raw_line in fh:
                line = raw_line.rstrip("\n").rstrip("\r")

                # Skip blank lines and comments.
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue

                # The keyword occupies exactly the first 6 characters.
                if len(line) < 6:
                    continue
                keyword = line[:6].strip()
                value_str = line[7:].strip() if len(line) > 7 else ""

                if keyword in _flag_keys:
                    self._config[keyword] = True
                elif keyword in _list_keys:
                    self._config[keyword].append(value_str)
                elif keyword in _float_keys:
                    try:
                        self._config[keyword] = float(value_str.split()[0])
                    except (ValueError, IndexError):
                        self._config[keyword] = value_str
                elif keyword in _int_keys:
                    try:
                        self._config[keyword] = int(value_str.split()[0])
                    except (ValueError, IndexError):
                        self._config[keyword] = value_str
                elif keyword in _string_keys:
                    self._config[keyword] = value_str
                else:
                    # Unknown keyword: store raw value string.
                    self._config[keyword] = value_str

    @property
    def receptor(self) -> Optional[str]:
        """Path to receptor PDB file (PDBNAM keyword)."""
        return self._config.get("PDBNAM")

    @property
    def ligand(self) -> Optional[str]:
        """Path to ligand input file (INPLIG keyword)."""
        return self._config.get("INPLIG")

    @property
    def temperature(self) -> Optional[int]:
        """Simulation temperature in Kelvin (TEMPER keyword)."""
        return self._config.get("TEMPER")

    @property
    def optimization_method(self) -> Optional[str]:
        """Optimization method, e.g. 'GA' (METOPT keyword)."""
        return self._config.get("METOPT")
    
    def run(self, **kwargs) -> BindingPopulation:
        """Execute docking simulation.
        
        Returns:
            BindingPopulation with ranked binding modes
        
        Note:
            Full implementation requires integration with C++ FlexAID GA engine.
            This is a stub for Phase 2 development.
        """
        raise NotImplementedError(
            "Full docking pipeline integration in progress. "
            "For now, use C++ FlexAID binary directly and parse output with "
            "BindingMode/BindingPopulation wrappers."
        )
    
    def __repr__(self) -> str:
        return f"<Docking config={self.config_file.name}>"
