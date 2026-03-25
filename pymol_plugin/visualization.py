"""PyMOL visualization functions for FlexAID∆S binding modes.

These functions can be called from PyMOL command line:
    PyMOL> flexaids_load /path/to/output
    PyMOL> flexaids_show_ensemble mode1
    PyMOL> flexaids_color_boltzmann mode1
    PyMOL> flexaids_thermo mode1

This module now delegates result-directory parsing to the canonical
``flexaidds.load_results()`` API instead of maintaining a separate parser.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    from pymol import cmd
    import pymol  # noqa: F401
except ImportError as exc:
    raise ImportError("PyMOL not available") from exc

try:
    from flexaidds import BindingModeResult, DockingResult, PoseResult, load_results
    from flexaidds.thermodynamics import kB_kcal as _kB_kcal
except ImportError as exc:
    raise ImportError(
        "flexaidds Python package is required for PyMOL result loading"
    ) from exc


@dataclass
class _ModeRecord:
    """Thermodynamic and structural data for one FlexAID binding mode."""

    mode_id: int
    pdb_objects: List[str] = field(default_factory=list)
    poses: List[PoseResult] = field(default_factory=list)
    cf_values: List[float] = field(default_factory=list)
    best_cf: Optional[float] = None
    total_cf: Optional[float] = None
    frequency: int = 0
    free_energy: Optional[float] = None
    enthalpy: Optional[float] = None
    entropy: Optional[float] = None
    heat_capacity: Optional[float] = None
    boltzmann_weights: List[float] = field(default_factory=list)


_loaded_modes: Dict[str, _ModeRecord] = {}
_loaded_result: Optional[DockingResult] = None
_output_dir: Optional[Path] = None
_temperature_K: float = 300.0


def _score_value(pose: PoseResult) -> Optional[float]:
    if pose.cf is not None:
        return pose.cf
    if pose.cf_app is not None:
        return pose.cf_app
    if pose.free_energy is not None:
        return pose.free_energy
    return None


def _compute_boltzmann_weights(values: List[float], temperature_K: float) -> List[float]:
    if not values:
        return []
    beta = 1.0 / (_kB_kcal * temperature_K)
    neg_beta_e = [-beta * value for value in values]
    max_val = max(neg_beta_e)
    shifted = [math.exp(v - max_val) for v in neg_beta_e]
    total = sum(shifted)
    if total <= 0.0:
        return []
    return [value / total for value in shifted]


def _make_mode_record(mode: BindingModeResult) -> _ModeRecord:
    cf_values = [value for value in (_score_value(pose) for pose in mode.poses) if value is not None]
    temperature = mode.temperature if mode.temperature is not None else _temperature_K
    weights = _compute_boltzmann_weights(cf_values, temperature) if cf_values else []
    total_cf = sum(cf_values) if cf_values else None
    frequency = mode.frequency if mode.frequency is not None else mode.n_poses
    return _ModeRecord(
        mode_id=mode.mode_id,
        poses=list(mode.poses),
        cf_values=cf_values,
        best_cf=mode.best_cf,
        total_cf=total_cf,
        frequency=frequency,
        free_energy=mode.free_energy,
        enthalpy=mode.enthalpy,
        entropy=mode.entropy,
        heat_capacity=mode.heat_capacity,
        boltzmann_weights=weights,
    )


def _mode_name(mode_id: int) -> str:
    return f"mode{mode_id}"


def _find_mode(mode_name: str) -> Optional[_ModeRecord]:
    return _loaded_modes.get(mode_name)


def load_binding_modes(output_dir: str, temperature: float = 300.0) -> None:
    """Load FlexAID∆S docking results from output directory.

    This legacy PyMOL entrypoint now delegates parsing to ``flexaidds.load_results``
    and only handles PyMOL object creation plus display bookkeeping.
    """

    global _loaded_modes, _loaded_result, _output_dir, _temperature_K

    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"ERROR: Directory not found: {output_dir}")
        return

    _temperature_K = float(temperature)

    try:
        result = load_results(output_path)
    except Exception as exc:
        print(f"ERROR: Could not load docking results: {exc}")
        return

    _loaded_modes.clear()
    _loaded_result = result
    _output_dir = result.source_dir

    for mode in result.binding_modes:
        mode_name = _mode_name(mode.mode_id)
        record = _make_mode_record(mode)

        for pose in mode.poses:
            obj_name = f"flexaids_{mode_name}_{pose.path.stem}"
            cmd.load(str(pose.path), obj_name)
            cmd.disable(obj_name)
            record.pdb_objects.append(obj_name)

        _loaded_modes[mode_name] = record

    # Sync with results_adapter so entropy_heatmap, animation, ITC work
    # regardless of whether user loaded via flexaids_load or flexaids_load_results
    from . import results_adapter
    results_adapter._loaded_result = result

    n_modes = len(_loaded_modes)
    n_poses = sum(len(rec.pdb_objects) for rec in _loaded_modes.values())
    print(f"Loaded {n_modes} binding modes ({n_poses} PDB objects) from {_output_dir}")
    print("Use 'flexaids_show_ensemble modeN' to visualize a binding mode.")


def show_pose_ensemble(mode_name: str, show_all: bool = True) -> None:
    """Display all poses belonging to a binding mode."""
    if not _loaded_modes:
        print("ERROR: No modes loaded. Use 'flexaids_load' first.")
        return

    rec = _find_mode(mode_name)
    if rec is None:
        available = ", ".join(sorted(_loaded_modes))
        print(f"ERROR: Mode '{mode_name}' not found. Available: {available}")
        return

    if not rec.pdb_objects:
        print(f"ERROR: No PDB objects for {mode_name}.")
        return

    if show_all:
        for obj in rec.pdb_objects:
            cmd.enable(obj)
            cmd.show("cartoon", obj)
            cmd.show("sticks", f"{obj} and organic")
    else:
        if rec.boltzmann_weights and len(rec.boltzmann_weights) == len(rec.pdb_objects):
            rep_idx = rec.boltzmann_weights.index(max(rec.boltzmann_weights))
        else:
            rep_idx = 0

        for index, obj in enumerate(rec.pdb_objects):
            if index == rep_idx:
                cmd.enable(obj)
                cmd.show("cartoon", obj)
                cmd.show("sticks", f"{obj} and organic")
            else:
                cmd.disable(obj)

    cmd.zoom(" ".join(rec.pdb_objects))
    label = "all poses" if show_all else "representative pose"
    print(f"Showing {label} for {mode_name} ({len(rec.pdb_objects)} PDB objects).")


def _burgundy_purple_rgb(t: float):
    """Interpolate burgundy red → purple blue.

    t = 0.0 → burgundy red (0.502, 0.0, 0.125)
    t = 1.0 → purple blue  (0.294, 0.0, 0.510)
    """
    t = max(0.0, min(1.0, t))
    r = 0.502 + t * (0.294 - 0.502)
    g = 0.0
    b = 0.125 + t * (0.510 - 0.125)
    return [r, g, b]


def color_by_boltzmann_weight(mode_name: str) -> None:
    """Color poses by Boltzmann weight (burgundy = high probability, purple = low)."""
    if not _loaded_modes:
        print("ERROR: No modes loaded. Use 'flexaids_load' first.")
        return

    rec = _find_mode(mode_name)
    if rec is None:
        available = ", ".join(sorted(_loaded_modes))
        print(f"ERROR: Mode '{mode_name}' not found. Available: {available}")
        return

    if not rec.pdb_objects:
        print(f"ERROR: No poses for {mode_name}.")
        return

    weights = rec.boltzmann_weights
    if not weights or len(weights) != len(rec.pdb_objects):
        n = len(rec.pdb_objects)
        weights = [1.0 / n] * n

    w_min = min(weights)
    w_max = max(weights)
    w_range = w_max - w_min if w_max > w_min else 1.0

    for index, (obj, weight) in enumerate(zip(rec.pdb_objects, weights)):
        t = (weight - w_min) / w_range
        # High weight = burgundy red (t=1 → frac=0), low weight = purple blue (t=0 → frac=1)
        frac = 1.0 - t
        color_name = f"flexaids_bw_{mode_name}_{index}"
        cmd.set_color(color_name, _burgundy_purple_rgb(frac))
        cmd.color(color_name, obj)
        cmd.enable(obj)

    print(
        f"Colored {len(rec.pdb_objects)} poses for {mode_name} by Boltzmann weight "
        "(burgundy=high, purple=low)."
    )


def show_thermodynamics(mode_name: str) -> None:
    """Print thermodynamic properties of a binding mode to PyMOL console."""
    if not _loaded_modes:
        print("ERROR: No modes loaded. Use 'flexaids_load' first.")
        return

    rec = _find_mode(mode_name)
    if rec is None:
        available = ", ".join(sorted(_loaded_modes))
        print(f"ERROR: Mode '{mode_name}' not found. Available: {available}")
        return

    temperature = _temperature_K
    if _loaded_result is not None:
        for mode in _loaded_result.binding_modes:
            if mode.mode_id == rec.mode_id and mode.temperature is not None:
                temperature = mode.temperature
                break
        else:
            if _loaded_result.temperature is not None:
                temperature = _loaded_result.temperature

    entropy_term = (rec.entropy * temperature) if rec.entropy is not None else None

    print(f"\nThermodynamics for {mode_name} (T = {temperature:.1f} K):")
    print(f"  ΔG (Free Energy):     {rec.free_energy:10.4f} kcal/mol" if rec.free_energy is not None else "  ΔG (Free Energy):     N/A")
    print(f"  ΔH (Enthalpy):        {rec.enthalpy:10.4f} kcal/mol" if rec.enthalpy is not None else "  ΔH (Enthalpy):        N/A")
    print(f"  S (Entropy):          {rec.entropy:10.6f} kcal/(mol·K)" if rec.entropy is not None else "  S (Entropy):          N/A")
    print(f"  TΔS (Entropy term):   {entropy_term:10.4f} kcal/mol" if entropy_term is not None else "  TΔS (Entropy term):   N/A")
    print(f"  Heat Capacity (Cv):   {rec.heat_capacity:10.4f} kcal/(mol·K²)" if rec.heat_capacity is not None else "  Heat Capacity (Cv):   N/A")
    print(f"  Best CF score:        {rec.best_cf:10.5f}" if rec.best_cf is not None else "  Best CF score:        N/A")
    print(f"  # Poses:              {rec.frequency:10d}")
    print()


def export_to_nrgsuite(output_dir: str, nrgsuite_file: str) -> None:
    """Export binding modes to NRGSuite-compatible format."""
    if not _loaded_modes:
        load_binding_modes(output_dir)
        if not _loaded_modes:
            print(f"ERROR: Could not load any modes from {output_dir}")
            return

    out_path = Path(nrgsuite_file)
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(
                "# FlexAID∆S → NRGSuite export\n"
                "# mode_id\tbest_cf\tfree_energy_kcal_mol\t"
                "enthalpy_kcal_mol\tentropy_kcal_mol_K\tn_poses\n"
            )
            for mode_name in sorted(_loaded_modes, key=lambda name: _loaded_modes[name].mode_id):
                rec = _loaded_modes[mode_name]
                if None not in (rec.free_energy, rec.enthalpy, rec.entropy):
                    fh.write(
                        f"{rec.mode_id}\t{rec.best_cf:.5f}\t{rec.free_energy:.4f}\t"
                        f"{rec.enthalpy:.4f}\t{rec.entropy:.6f}\t{rec.frequency}\n"
                    )
                else:
                    best_cf = f"{rec.best_cf:.5f}" if rec.best_cf is not None else "N/A"
                    fh.write(f"{rec.mode_id}\t{best_cf}\tN/A\tN/A\tN/A\t{rec.frequency}\n")
    except OSError as exc:
        print(f"ERROR: Could not write NRGSuite file: {exc}")
        return

    print(f"Exported {len(_loaded_modes)} binding modes to {out_path}")
