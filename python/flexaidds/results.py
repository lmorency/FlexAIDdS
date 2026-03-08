from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .io import parse_pose_result
from .models import BindingModeResult, DockingResult, PoseResult

_PDB_SUFFIXES = {".pdb", ".ent"}


def _collect_pose_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in _PDB_SUFFIXES:
            files.append(path)
    return sorted(files)


def _mode_temperature(poses: Sequence[PoseResult]) -> Optional[float]:
    for pose in poses:
        if pose.temperature is not None:
            return pose.temperature
    return None


def _mode_metric(poses: Sequence[PoseResult], name: str) -> Optional[float]:
    for pose in poses:
        value = getattr(pose, name)
        if value is not None:
            return value
    return None


def _mode_frequency(poses: Sequence[PoseResult]) -> Optional[int]:
    for pose in poses:
        for key in ("frequency", "nposes", "population", "cluster_size", "size"):
            value = pose.remarks.get(key)
            if isinstance(value, int):
                return value
    return len(poses) if poses else None


def _mode_metadata(poses: Sequence[PoseResult]) -> Dict[str, object]:
    meta: Dict[str, object] = {}
    if not poses:
        return meta
    shared_keys = set(poses[0].remarks)
    for pose in poses[1:]:
        shared_keys &= set(pose.remarks)
    for key in shared_keys:
        values = {pose.remarks.get(key) for pose in poses}
        if len(values) == 1:
            meta[key] = poses[0].remarks[key]
    return meta


def _build_mode(mode_id: int, poses: Sequence[PoseResult]) -> BindingModeResult:
    ordered = sorted(poses, key=lambda p: (p.pose_rank, p.path.name))
    best_pose = None
    scored = [p for p in ordered if p.cf is not None]
    if scored:
        best_pose = min(scored, key=lambda p: p.cf)
    elif ordered:
        best_pose = ordered[0]

    return BindingModeResult(
        mode_id=mode_id,
        rank=mode_id,
        poses=list(ordered),
        free_energy=_mode_metric(ordered, "free_energy"),
        enthalpy=_mode_metric(ordered, "enthalpy"),
        entropy=_mode_metric(ordered, "entropy"),
        heat_capacity=_mode_metric(ordered, "heat_capacity"),
        std_energy=_mode_metric(ordered, "std_energy"),
        best_cf=best_pose.cf if best_pose else None,
        frequency=_mode_frequency(ordered),
        temperature=_mode_temperature(ordered),
        metadata=_mode_metadata(ordered),
    )


def load_results(path: str | Path) -> DockingResult:
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Docking results directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a results directory, got file: {root}")

    pose_files = _collect_pose_files(root)
    if not pose_files:
        raise FileNotFoundError(f"No PDB-like docking result files found under: {root}")

    grouped: Dict[int, List[PoseResult]] = defaultdict(list)
    for pose_file in pose_files:
        pose = parse_pose_result(pose_file)
        grouped[pose.mode_id].append(pose)

    modes = [_build_mode(mode_id, poses) for mode_id, poses in sorted(grouped.items())]
    temperature = next((m.temperature for m in modes if m.temperature is not None), None)

    return DockingResult(
        source_dir=root,
        binding_modes=modes,
        temperature=temperature,
        metadata={"n_pose_files": len(pose_files)},
    )
