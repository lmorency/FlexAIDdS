"""High-level docking interface for FlexAID∆S.

Provides Pythonic API for molecular docking workflows.
"""

import subprocess
import shutil
import re
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
        # Python-only path: use best pose energy as proxy
        if self._poses:
            return min(p.energy for p in self._poses)
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
    
    def __init__(self, modes: Optional[List[BindingMode]] = None,
                 temperature: float = 300.0):
        self._modes: List[BindingMode] = list(modes) if modes else []
        self._temperature: float = temperature
    
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
    
    def run(self, binary: Optional[str] = None,
            timeout: int = 3600, **kwargs) -> BindingPopulation:
        """Execute docking via the FlexAID C++ binary and parse results.

        Locates the ``FlexAID`` binary (in PATH, project build/, or explicit
        *binary* argument), invokes it with this config file, waits for
        completion, then parses all ``*_N_M.pdb`` output files written by
        ``output_Population()`` to reconstruct a ``BindingPopulation``.

        Args:
            binary:  Path to FlexAID executable.  If *None*, searches PATH and
                     common build locations (``build/FlexAID``,
                     ``../build/FlexAID``).
            timeout: Wall-clock timeout in seconds (default 3600).
            **kwargs: Ignored; reserved for future keyword overrides.

        Returns:
            BindingPopulation populated from the PDB REMARK lines written by
            ``output_BindingMode()`` / ``output_Population()``.

        Raises:
            FileNotFoundError: binary not found.
            RuntimeError:      FlexAID exited non-zero or produced no output.
        """
        # ── 1. Locate binary ─────────────────────────────────────────────────
        exe = self._find_binary(binary)

        # ── 2. Invoke FlexAID ────────────────────────────────────────────────
        cmd = [str(exe), str(self.config_file)]
        try:
            result = subprocess.run(
                cmd,
                cwd=self.config_file.parent,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"FlexAID timed out after {timeout}s"
            ) from exc

        if result.returncode != 0:
            raise RuntimeError(
                f"FlexAID exited with code {result.returncode}.\n"
                f"stdout: {result.stdout[-2000:]}\n"
                f"stderr: {result.stderr[-2000:]}"
            )

        # ── 3. Discover output PDBs ──────────────────────────────────────────
        # output_BindingMode writes files named <prefix>_<minPoints>_<mode>.pdb
        # Collect all candidate PDB files in the working directory.
        work_dir = self.config_file.parent
        pdb_files = sorted(work_dir.glob("*_*.pdb"),
                           key=lambda p: p.stat().st_mtime)

        if not pdb_files:
            raise RuntimeError(
                "FlexAID completed but no PDB output files were found in "
                f"{work_dir}. Check the config file NRGOUT / output settings."
            )

        # ── 4. Parse PDB REMARK lines into BindingModes ──────────────────────
        temperature = self._config.get("TEMPER", 300) or 300
        modes: List[BindingMode] = []
        seen_modes: Dict[int, BindingMode] = {}

        for pdb_path in pdb_files:
            mode_info = self._parse_remark_pdb(pdb_path, temperature)
            if mode_info is None:
                continue
            mode_idx, pose = mode_info
            if mode_idx not in seen_modes:
                seen_modes[mode_idx] = BindingMode()
            seen_modes[mode_idx]._poses.append(pose)

        # Sort modes by free energy (ascending → most favourable first)
        modes = sorted(seen_modes.values(),
                       key=lambda m: m.free_energy)

        return BindingPopulation(modes, temperature=float(temperature))

    # ── helpers ───────────────────────────────────────────────────────────────

    def _find_binary(self, binary: Optional[str]) -> Path:
        """Locate the FlexAID executable."""
        if binary is not None:
            p = Path(binary)
            if not p.is_file():
                raise FileNotFoundError(f"Specified FlexAID binary not found: {binary}")
            return p

        # Search order: PATH → project-relative build dirs
        in_path = shutil.which("FlexAID")
        if in_path:
            return Path(in_path)

        candidates = [
            self.config_file.parent / "FlexAID",
            self.config_file.parent / "build" / "FlexAID",
            Path(__file__).parents[3] / "build" / "FlexAID",
        ]
        for c in candidates:
            if c.is_file():
                return c

        raise FileNotFoundError(
            "FlexAID binary not found in PATH or build/. "
            "Build with 'cmake --build build' or pass binary= argument."
        )

    @staticmethod
    def _parse_remark_pdb(
            pdb_path: Path, temperature: float) -> Optional[tuple]:
        """Parse a single output PDB written by output_BindingMode().

        Extracts mode index, CF, RMSD, and per-pose energy from REMARK lines.
        Returns (mode_index, Pose) or None if the file lacks FlexAID remarks.
        """
        mode_idx   = None
        cf_val     = None
        rmsd_val   = None
        freq       = 1

        try:
            text = pdb_path.read_text(errors="replace")
        except OSError:
            return None

        for line in text.splitlines():
            if not line.startswith("REMARK"):
                continue
            # "REMARK Binding Mode:N Best CF in Binding Mode:X …"
            m = re.search(
                r"Binding Mode:(\d+).*?Best CF in Binding Mode:\s*([-\d.]+)"
                r".*?Binding Mode Frequency:(\d+)",
                line)
            if m:
                mode_idx = int(m.group(1))
                cf_val   = float(m.group(2))
                freq     = int(m.group(3))
            # "REMARK 0.12345 RMSD to ref. structure …"
            m2 = re.search(r"REMARK\s+([\d.]+)\s+RMSD to ref\.", line)
            if m2 and rmsd_val is None:
                rmsd_val = float(m2.group(1))

        if mode_idx is None or cf_val is None:
            return None

        import math
        beta = 1.0 / (0.001987206 * float(temperature))
        bw   = math.exp(-beta * cf_val)

        pose = Pose(
            index=mode_idx,
            energy=cf_val,
            rmsd=rmsd_val,
            boltzmann_weight=bw,
        )
        return mode_idx, pose
    
    def __repr__(self) -> str:
        return f"<Docking config={self.config_file.name}>"
