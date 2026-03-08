from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PoseResult:
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
        return len(self.poses)

    def best_pose(self) -> Optional[PoseResult]:
        scored = [p for p in self.poses if p.cf is not None]
        if scored:
            return min(scored, key=lambda p: p.cf)
        scored = [p for p in self.poses if p.cf_app is not None]
        if scored:
            return min(scored, key=lambda p: p.cf_app)
        return self.poses[0] if self.poses else None


@dataclass(frozen=True)
class DockingResult:
    source_dir: Path
    binding_modes: List[BindingModeResult]
    temperature: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_modes(self) -> int:
        return len(self.binding_modes)

    def top_mode(self) -> Optional[BindingModeResult]:
        if not self.binding_modes:
            return None
        free_modes = [m for m in self.binding_modes if m.free_energy is not None]
        if free_modes:
            return min(free_modes, key=lambda m: m.free_energy)
        return min(self.binding_modes, key=lambda m: m.rank)

    def to_records(self) -> List[Dict[str, Any]]:
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
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for DockingResult.to_dataframe(); use to_records() instead."
            ) from exc
        return pd.DataFrame(self.to_records())
