"""Data models for FlexAID∆S docking results.

This module defines frozen dataclasses that represent docking output at three
levels of granularity:

- :class:`PoseResult` – a single docked pose (one PDB file).
- :class:`BindingModeResult` – a cluster of poses sharing a binding geometry.
- :class:`DockingResult` – the top-level container returned by
  :func:`~flexaidds.results.load_results`.

All three classes are immutable (``frozen=True``) so they can be safely shared
across threads and used as dictionary keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PoseResult:
    """A single docked pose read from one FlexAID∆S output PDB file.

    Attributes:
        path: Absolute path to the PDB file on disk.
        mode_id: Binding-mode (cluster) index this pose belongs to.
        pose_rank: Rank of this pose within its binding mode (1-based).
        cf: NATURaL complementarity-function score (kcal/mol). Lower is better.
        cf_app: Apparent CF score after grid-approximation correction (kcal/mol).
        rmsd_raw: RMSD to reference structure without symmetry correction (Å).
        rmsd_sym: Symmetry-corrected RMSD to reference structure (Å).
        free_energy: Helmholtz free energy F = H − TS (kcal/mol), if present
            in the PDB REMARK section.
        enthalpy: Boltzmann-weighted average energy ⟨E⟩ (kcal/mol).
        entropy: Configurational entropy S = (⟨E⟩ − F) / T
            (kcal mol⁻¹ K⁻¹).
        heat_capacity: Ensemble heat capacity Cv = (⟨E²⟩ − ⟨E⟩²) / (kT²)
            (kcal mol⁻¹ K⁻²).
        std_energy: Standard deviation of ensemble energies σ_E (kcal/mol).
        temperature: Simulation temperature (K) parsed from REMARK section.
        remarks: Raw key→value mapping of all ``REMARK`` fields parsed from the
            PDB header.
    """

    path: Path
    mode_id: int
    pose_rank: int
    cf: Optional[float] = None
    cf_app: Optional[float] = None
    rmsd_raw: Optional[float] = None
    rmsd_sym: Optional[float] = None
    free_energy: Optional[float] = None
    enthalpy: Optional[float] = None
    entropy: Optional[float] = None
    heat_capacity: Optional[float] = None
    std_energy: Optional[float] = None
    temperature: Optional[float] = None
    remarks: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BindingModeResult:
    """A cluster of docked poses that share a common binding geometry.

    A binding mode aggregates :class:`PoseResult` objects that were grouped
    together by the OPTICS/DBSCAN clustering step inside the FlexAID∆S C++
    engine.  Thermodynamic quantities stored here are mode-level aggregates
    derived from the statistical mechanics engine (Helmholtz free energy,
    configurational entropy, heat capacity, etc.).

    Attributes:
        mode_id: Unique integer identifier for this binding mode.
        rank: Rank of this mode among all modes (1 = best free energy).
        poses: Ordered list of individual poses belonging to this mode.
        free_energy: Helmholtz free energy F (kcal/mol) for the mode ensemble.
        enthalpy: Boltzmann-weighted mean energy ⟨E⟩ (kcal/mol).
        entropy: Configurational entropy S (kcal mol⁻¹ K⁻¹).
        heat_capacity: Ensemble heat capacity Cv (kcal mol⁻¹ K⁻²).
        std_energy: Standard deviation of ensemble energies σ_E (kcal/mol).
        best_cf: Lowest (most favourable) individual CF score within the mode.
        frequency: Number of GA chromosomes assigned to this mode; proportional
            to Boltzmann population weight.
        temperature: Simulation temperature (K) associated with this mode.
        metadata: Arbitrary extra fields shared across all poses in the mode
            (e.g. receptor name, ligand SMILES).
    """

    mode_id: int
    rank: int
    poses: List[PoseResult]
    free_energy: Optional[float] = None
    enthalpy: Optional[float] = None
    entropy: Optional[float] = None
    heat_capacity: Optional[float] = None
    std_energy: Optional[float] = None
    best_cf: Optional[float] = None
    frequency: Optional[int] = None
    temperature: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_poses(self) -> int:
        """Number of poses in this binding mode."""
        return len(self.poses)

    def best_pose(self) -> Optional[PoseResult]:
        """Return the pose with the lowest CF (or cf_app) score.

        Selection priority:

        1. Pose with the lowest ``cf`` value.
        2. Pose with the lowest ``cf_app`` value (if no ``cf`` is available).
        3. First pose in :attr:`poses` (fallback when no scores are present).

        Returns:
            The best-scored :class:`PoseResult`, or ``None`` if the mode is
            empty.
        """
        scored = [p for p in self.poses if p.cf is not None]
        if scored:
            return min(scored, key=lambda p: p.cf)
        scored = [p for p in self.poses if p.cf_app is not None]
        if scored:
            return min(scored, key=lambda p: p.cf_app)
        return self.poses[0] if self.poses else None


@dataclass(frozen=True)
class DockingResult:
    """Top-level container for a complete FlexAID∆S docking run.

    Returned by :func:`~flexaidds.results.load_results` after scanning a
    docking output directory.  Provides convenience methods for ranking,
    serialisation, and optional pandas integration.

    Attributes:
        source_dir: Absolute path to the directory that was scanned.
        binding_modes: List of :class:`BindingModeResult` objects, sorted by
            ascending ``mode_id``.
        temperature: Simulation temperature (K) inferred from the output files,
            or ``None`` if not available.
        metadata: Arbitrary extra information collected during loading (e.g.
            ``n_pose_files``).
    """

    source_dir: Path
    binding_modes: List[BindingModeResult]
    temperature: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_modes(self) -> int:
        """Number of binding modes in this result."""
        return len(self.binding_modes)

    def top_mode(self) -> Optional[BindingModeResult]:
        """Return the binding mode with the lowest free energy.

        Falls back to the mode with the lowest :attr:`~BindingModeResult.rank`
        when no free-energy values are available.

        Returns:
            Best :class:`BindingModeResult`, or ``None`` if there are no modes.
        """
        if not self.binding_modes:
            return None
        free_modes = [m for m in self.binding_modes if m.free_energy is not None]
        if free_modes:
            return min(free_modes, key=lambda m: m.free_energy)
        return min(self.binding_modes, key=lambda m: m.rank)

    def to_records(self) -> List[Dict[str, Any]]:
        """Serialise all binding modes to a list of flat dictionaries.

        Each dictionary contains mode-level scalar fields plus the path to the
        best pose.  Suitable for direct conversion to a
        :class:`pandas.DataFrame` via :meth:`to_dataframe`.

        Returns:
            List of dictionaries, one per binding mode, with keys:
            ``mode_id``, ``rank``, ``n_poses``, ``free_energy``,
            ``enthalpy``, ``entropy``, ``heat_capacity``, ``std_energy``,
            ``best_cf``, ``temperature``, ``best_pose_path``.
        """
        records: List[Dict[str, Any]] = []
        for mode in self.binding_modes:
            best_pose = mode.best_pose()
            records.append(
                {
                    "mode_id": mode.mode_id,
                    "rank": mode.rank,
                    "n_poses": mode.n_poses,
                    "free_energy": mode.free_energy,
                    "enthalpy": mode.enthalpy,
                    "entropy": mode.entropy,
                    "heat_capacity": mode.heat_capacity,
                    "std_energy": mode.std_energy,
                    "best_cf": mode.best_cf,
                    "temperature": mode.temperature,
                    "best_pose_path": str(best_pose.path) if best_pose else None,
                }
            )
        return records

    def to_dataframe(self):
        """Convert binding-mode results to a :class:`pandas.DataFrame`.

        Each row corresponds to one binding mode.  Columns match the fields
        returned by :meth:`to_records`.

        Raises:
            ImportError: If ``pandas`` is not installed.  Use
                :meth:`to_records` for a dependency-free alternative.

        Returns:
            :class:`pandas.DataFrame` with one row per binding mode.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for DockingResult.to_dataframe(); use to_records() instead."
            ) from exc
        return pd.DataFrame(self.to_records())
