from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

try:
    from pymol import cmd
except ImportError as exc:
    raise ImportError("PyMOL not available") from exc

try:
    from flexaidds import BindingModeResult, DockingResult, load_results
except ImportError as exc:
    raise ImportError(
        "flexaidds Python package is required for the read-only PyMOL adapter"
    ) from exc


def _safe_int(value, name: str = "value") -> Optional[int]:
    """Convert to int with a user-friendly error message."""
    try:
        return int(value)
    except (ValueError, TypeError):
        print(f"ERROR: '{value}' is not a valid integer for {name}.")
        return None


_loaded_result: Optional[DockingResult] = None
_loaded_objects: Dict[int, List[str]] = {}
_loaded_prefix: str = "flexaids"


def _object_name(prefix: str, mode_id: int, pose_rank: int) -> str:
    return f"{prefix}_mode{mode_id}_pose{pose_rank}"


def _group_name(prefix: str, mode_id: int) -> str:
    return f"{prefix}_mode{mode_id}"


def _get_mode(mode_id: int) -> Optional[BindingModeResult]:
    if _loaded_result is None:
        return None
    for mode in _loaded_result.binding_modes:
        if mode.mode_id == int(mode_id):
            return mode
    return None


def load_docking_results(results_dir: str, prefix: str = "flexaids") -> None:
    """Load FlexAID∆S result files through the Python read-only loader.

    Args:
        results_dir: Directory containing docking result PDB files.
        prefix: Prefix used to create PyMOL object and group names.

    Example:
        PyMOL> flexaids_load_results /path/to/output
    """
    global _loaded_result, _loaded_objects, _loaded_prefix

    result = load_results(results_dir)
    _loaded_result = result
    _loaded_prefix = prefix
    _loaded_objects = {}

    for mode in result.binding_modes:
        group_name = _group_name(prefix, mode.mode_id)
        object_names: List[str] = []
        best_pose = mode.best_pose()

        for pose in mode.poses:
            obj_name = _object_name(prefix, mode.mode_id, pose.pose_rank)
            cmd.load(str(pose.path), obj_name)
            cmd.group(group_name, obj_name)
            cmd.hide("everything", obj_name)
            cmd.show("sticks", f"{obj_name} and organic")
            cmd.show("lines", obj_name)
            if best_pose is None or pose.path != best_pose.path:
                cmd.disable(obj_name)
            object_names.append(obj_name)

        _loaded_objects[mode.mode_id] = object_names

    print(
        f"Loaded {result.n_modes} binding modes from {Path(results_dir).resolve()} "
        f"with prefix '{prefix}'."
    )
    print("Use 'flexaids_show_mode <mode_id>' to inspect a mode.")


def show_binding_mode(mode_id: int, show_all: int = 0) -> None:
    """Show one loaded binding mode.

    Args:
        mode_id: Numeric binding-mode identifier.
        show_all: 1 to show all poses in the mode, 0 for best pose only.
    """
    mid = _safe_int(mode_id, "mode_id")
    if mid is None:
        return
    mode = _get_mode(mid)
    if mode is None:
        print("ERROR: No loaded result set or mode not found.")
        return

    object_names = _loaded_objects.get(mode.mode_id, [])
    if not object_names:
        print(f"ERROR: No PyMOL objects loaded for mode {mode.mode_id}.")
        return

    best_pose = mode.best_pose()
    best_name = None
    if best_pose is not None:
        best_name = _object_name(_loaded_prefix, mode.mode_id, best_pose.pose_rank)

    for other_mode_id, names in _loaded_objects.items():
        for name in names:
            cmd.disable(name)

    if int(show_all):
        for name in object_names:
            cmd.enable(name)
    elif best_name is not None:
        cmd.enable(best_name)
    else:
        cmd.enable(object_names[0])

    cmd.zoom(_group_name(_loaded_prefix, mode.mode_id))
    print(
        f"Mode {mode.mode_id}: n_poses={mode.n_poses}, "
        f"free_energy={mode.free_energy}, best_cf={mode.best_cf}"
    )


def color_mode_by_score(mode_id: int, metric: str = "cf") -> None:
    """Color poses within one mode using a score gradient.

    Args:
        mode_id: Numeric binding-mode identifier.
        metric: 'cf' or 'free_energy'. Lower values are colored red.
    """
    mid = _safe_int(mode_id, "mode_id")
    if mid is None:
        return
    mode = _get_mode(mid)
    if mode is None:
        print("ERROR: No loaded result set or mode not found.")
        return

    object_names = _loaded_objects.get(mode.mode_id, [])
    if not object_names:
        print(f"ERROR: No PyMOL objects loaded for mode {mode.mode_id}.")
        return

    metric = metric.strip().lower()
    values = []
    for pose in mode.poses:
        if metric == "free_energy":
            values.append(pose.free_energy if pose.free_energy is not None else mode.free_energy)
        else:
            values.append(pose.cf if pose.cf is not None else mode.best_cf)

    finite = [v for v in values if v is not None]
    if not finite:
        print(f"ERROR: No numeric values available for metric '{metric}'.")
        return

    vmin = min(finite)
    vmax = max(finite)
    vrange = (vmax - vmin) if vmax > vmin else 1.0

    for pose, obj_name, value in zip(mode.poses, object_names, values):
        if value is None:
            continue
        t = (value - vmin) / vrange
        t = max(0.0, min(1.0, t))
        color_name = f"{_loaded_prefix}_{metric}_m{mode.mode_id}_p{pose.pose_rank}"
        cmd.set_color(color_name, [1.0 - t, 0.0, t])
        cmd.color(color_name, obj_name)
        cmd.enable(obj_name)

    print(f"Colored mode {mode.mode_id} by {metric} (red=lower, blue=higher).")


def show_mode_details(mode_id: int) -> None:
    """Print thermodynamic summary for one loaded mode."""
    mid = _safe_int(mode_id, "mode_id")
    if mid is None:
        return
    mode = _get_mode(mid)
    if mode is None:
        print("ERROR: No loaded result set or mode not found.")
        return

    temperature = mode.temperature if mode.temperature is not None else (
        _loaded_result.temperature if _loaded_result is not None else None
    )
    entropy_term = None
    if mode.entropy is not None and temperature is not None:
        entropy_term = mode.entropy * temperature

    print(f"Mode {mode.mode_id} details")
    print(f"  rank:          {mode.rank}")
    print(f"  poses:         {mode.n_poses}")
    print(f"  best_cf:       {mode.best_cf}")
    print(f"  free_energy:   {mode.free_energy}")
    print(f"  enthalpy:      {mode.enthalpy}")
    print(f"  entropy:       {mode.entropy}")
    print(f"  temperature:   {temperature}")
    print(f"  T*S:           {entropy_term}")
