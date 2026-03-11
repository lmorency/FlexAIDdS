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

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


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

    def __repr__(self) -> str:
        parts = [f"mode={self.mode_id}", f"rank={self.pose_rank}"]
        if self.cf is not None:
            parts.append(f"cf={self.cf:.2f}")
        if self.free_energy is not None:
            parts.append(f"F={self.free_energy:.2f}")
        return f"<PoseResult {' '.join(parts)}>"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PoseResult":
        """Reconstruct a PoseResult from a dictionary.

        Accepts both the internal field names (``cf``, ``rmsd_raw``) and the
        serialised names produced by :meth:`to_records`-style output
        (``best_pose_path``).  Unknown keys are silently ignored.

        Args:
            data: Dictionary with PoseResult field values.

        Returns:
            A new :class:`PoseResult` instance.
        """
        path = data.get("path", data.get("best_pose_path", ""))
        return cls(
            path=Path(path) if not isinstance(path, Path) else path,
            mode_id=data.get("mode_id", 0),
            pose_rank=data.get("pose_rank", 0),
            cf=data.get("cf"),
            cf_app=data.get("cf_app"),
            rmsd_raw=data.get("rmsd_raw"),
            rmsd_sym=data.get("rmsd_sym"),
            free_energy=data.get("free_energy"),
            enthalpy=data.get("enthalpy"),
            entropy=data.get("entropy"),
            heat_capacity=data.get("heat_capacity"),
            std_energy=data.get("std_energy"),
            temperature=data.get("temperature"),
            remarks=data.get("remarks", {}),
        )


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

    def __repr__(self) -> str:
        parts = [f"id={self.mode_id}", f"rank={self.rank}", f"poses={len(self.poses)}"]
        if self.free_energy is not None:
            parts.append(f"F={self.free_energy:.2f}")
        if self.best_cf is not None:
            parts.append(f"best_cf={self.best_cf:.2f}")
        return f"<BindingModeResult {' '.join(parts)}>"

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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BindingModeResult":
        """Reconstruct a BindingModeResult from a dictionary.

        Pose entries under the ``"poses"`` key are deserialised via
        :meth:`PoseResult.from_dict`.  If ``"poses"`` is absent an empty list
        is used.

        Args:
            data: Dictionary with BindingModeResult field values.

        Returns:
            A new :class:`BindingModeResult` instance.
        """
        poses = [PoseResult.from_dict(p) for p in data.get("poses", [])]
        return cls(
            mode_id=data.get("mode_id", 0),
            rank=data.get("rank", 0),
            poses=poses,
            free_energy=data.get("free_energy"),
            enthalpy=data.get("enthalpy"),
            entropy=data.get("entropy"),
            heat_capacity=data.get("heat_capacity"),
            std_energy=data.get("std_energy"),
            best_cf=data.get("best_cf"),
            frequency=data.get("frequency"),
            temperature=data.get("temperature"),
            metadata=data.get("metadata", {}),
        )


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

    def __repr__(self) -> str:
        return f"<DockingResult modes={len(self.binding_modes)} source={self.source_dir.name!r}>"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DockingResult":
        """Reconstruct a DockingResult from a dictionary.

        Accepts the format produced by :meth:`to_json` (flat
        ``binding_modes`` records from :meth:`to_records`) as well as
        nested structures where each mode contains a ``"poses"`` list.

        Args:
            data: Dictionary with DockingResult field values.

        Returns:
            A new :class:`DockingResult` instance.
        """
        raw_modes = data.get("binding_modes", [])
        modes: List[BindingModeResult] = []
        for i, m in enumerate(raw_modes):
            if "poses" in m:
                modes.append(BindingModeResult.from_dict(m))
            else:
                # Flat record from to_records(): wrap into a BindingModeResult
                modes.append(BindingModeResult(
                    mode_id=m.get("mode_id", i),
                    rank=m.get("rank", i + 1),
                    poses=[],
                    free_energy=m.get("free_energy"),
                    enthalpy=m.get("enthalpy"),
                    entropy=m.get("entropy"),
                    heat_capacity=m.get("heat_capacity"),
                    std_energy=m.get("std_energy"),
                    best_cf=m.get("best_cf"),
                    temperature=m.get("temperature"),
                ))
        return cls(
            source_dir=Path(data.get("source_dir", ".")),
            binding_modes=modes,
            temperature=data.get("temperature"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, source: Union[str, Path]) -> "DockingResult":
        """Load a DockingResult from a JSON file or string.

        Accepts either a file path or a raw JSON string.  The JSON format
        is the one produced by :meth:`to_json`.

        Args:
            source: Path to a ``.json`` file, or a JSON-encoded string.

        Returns:
            A new :class:`DockingResult` instance.
        """
        path = Path(source) if not isinstance(source, Path) else source
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            data = json.loads(str(source))
        return cls.from_dict(data)

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

    def to_json(self, path: Union[str, Path, None] = None, **kwargs) -> Optional[str]:
        """Serialise docking results to JSON.

        The output includes the source directory, temperature, metadata, and
        a ``binding_modes`` array produced by :meth:`to_records`.

        Args:
            path: Destination file path.  When *None* the JSON text is returned
                  as a string instead of being written to disk.
            **kwargs: Extra keyword arguments forwarded to :func:`json.dumps`
                (e.g. ``indent``, ``sort_keys``).

        Returns:
            JSON text when *path* is ``None``, otherwise ``None``.
        """
        payload = {
            "source_dir": str(self.source_dir),
            "temperature": self.temperature,
            "n_modes": self.n_modes,
            "metadata": self.metadata,
            "binding_modes": self.to_records(),
        }
        kwargs.setdefault("indent", 2)
        text = json.dumps(payload, **kwargs)

        if path is None:
            return text

        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.write("\n")
        return None

    def to_csv(self, path: Union[str, Path, None] = None) -> Optional[str]:
        """Write binding mode summary to CSV.

        Args:
            path: Destination file path.  When *None* the CSV text is returned
                  as a string instead of being written to disk.

        Returns:
            CSV text when *path* is ``None``, otherwise ``None``.
        """
        records = self.to_records()
        if not records:
            fieldnames: List[str] = []
        else:
            fieldnames = list(records[0].keys())

        if path is None:
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
            return buf.getvalue()

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        return None
