"""Interactive docking workflow for FlexAID∆S (Phase 3, deliverable 3.2).

Enables docking from within PyMOL using the ``flexaidds.docking.Docking``
API.  Provides binding-site selection from PyMOL selection objects and
real-time progress display.

Usage:
    PyMOL> flexaids_dock receptor_obj, ligand.mol2
    PyMOL> flexaids_dock receptor_obj, ligand.mol2, site_selection=sele
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

try:
    from pymol import cmd, stored
except ImportError as exc:
    raise ImportError("PyMOL not available") from exc

try:
    from flexaidds import load_results
except ImportError as exc:
    raise ImportError(
        "flexaidds Python package is required for interactive docking"
    ) from exc


class DockingProgressCallback:
    """Tracks docking progress and updates PyMOL status."""

    def __init__(self) -> None:
        self.generation: int = 0
        self.best_cf: float = float("inf")
        self.running: bool = False

    def on_generation(self, gen_num: int, best_cf: float, mean_entropy: float = 0.0) -> None:
        """Update progress after each GA generation."""
        self.generation = gen_num
        self.best_cf = best_cf
        print(
            f"  Gen {gen_num}: Best CF = {best_cf:.4f}, "
            f"Mean S = {mean_entropy:.6f}"
        )


def _get_selection_center(selection: str) -> Optional[tuple]:
    """Get the center of mass of a PyMOL selection.

    Returns (x, y, z) tuple or None if selection is empty/invalid.
    """
    try:
        model = cmd.get_model(selection)
        if not model.atom:
            return None
        xs = [a.coord[0] for a in model.atom]
        ys = [a.coord[1] for a in model.atom]
        zs = [a.coord[2] for a in model.atom]
        return (
            sum(xs) / len(xs),
            sum(ys) / len(ys),
            sum(zs) / len(zs),
        )
    except Exception:
        return None


def _get_selection_radius(selection: str) -> float:
    """Estimate the binding site radius from a PyMOL selection.

    Returns the maximum distance from the center of mass to any
    selected atom, plus a 2A padding.
    """
    try:
        model = cmd.get_model(selection)
        if not model.atom:
            return 10.0
        xs = [a.coord[0] for a in model.atom]
        ys = [a.coord[1] for a in model.atom]
        zs = [a.coord[2] for a in model.atom]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        cz = sum(zs) / len(zs)
        max_r = 0.0
        for a in model.atom:
            dx = a.coord[0] - cx
            dy = a.coord[1] - cy
            dz = a.coord[2] - cz
            r = (dx * dx + dy * dy + dz * dz) ** 0.5
            if r > max_r:
                max_r = r
        return max_r + 2.0
    except Exception:
        return 10.0


def _save_receptor_pdb(obj_name: str, output_path: str) -> bool:
    """Save a PyMOL object as a PDB file."""
    try:
        cmd.save(output_path, obj_name)
        return True
    except Exception as exc:
        print(f"ERROR: Could not save receptor: {exc}")
        return False


def _write_minimal_config(
    config_path: str,
    receptor_pdb: str,
    ligand_file: str,
    center: tuple,
    radius: float,
    temperature: int = 300,
    n_results: int = 10,
) -> None:
    """Write a minimal FlexAID configuration file."""
    with open(config_path, "w") as fh:
        fh.write(f"PDBNAM {receptor_pdb}\n")
        fh.write(f"INPLIG {ligand_file}\n")
        fh.write(f"METOPT GA\n")
        fh.write(f"TEMPER {temperature}\n")
        fh.write(f"NRGOUT {n_results}\n")
        fh.write(f"SPACER {radius:.1f}\n")
        fh.write(f"COMPLF {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}\n")
        fh.write(f"CLRMSD 2.0\n")
        fh.write(f"PERMEA 0.05\n")


def dock_interactive(
    receptor_obj: str,
    ligand_file: str,
    site_selection: str = "sele",
    temperature: int = 300,
    timeout: int = 300,
    n_results: int = 10,
) -> None:
    """Run FlexAID∆S docking from PyMOL using a loaded receptor and ligand.

    Workflow:
    1. Export receptor from PyMOL object to temporary PDB
    2. Determine binding site from PyMOL selection (or sphere center)
    3. Generate FlexAID config file
    4. Execute docking via ``flexaidds.docking.Docking``
    5. Load results back into PyMOL

    Args:
        receptor_obj: Name of the receptor PyMOL object.
        ligand_file: Path to ligand file (MOL2 or SDF).
        site_selection: PyMOL selection defining the binding site
                       (default: 'sele').
        temperature: Docking temperature in Kelvin (default: 300).
        timeout: Maximum docking time in seconds (default: 300).
        n_results: Number of output poses to generate (default: 10).

    Example:
        PyMOL> flexaids_dock receptor, /path/to/ligand.mol2
        PyMOL> flexaids_dock receptor, ligand.mol2, site_selection=active_site
    """
    receptor_obj = str(receptor_obj).strip()
    ligand_file = str(ligand_file).strip()
    site_selection = str(site_selection).strip()
    temperature = int(temperature)
    timeout = int(timeout)
    n_results = int(n_results)

    # Validate receptor object exists in PyMOL
    if receptor_obj not in cmd.get_object_list():
        print(f"ERROR: Receptor object '{receptor_obj}' not found in PyMOL.")
        available = ", ".join(cmd.get_object_list())
        if available:
            print(f"  Available objects: {available}")
        return

    # Validate ligand file exists
    ligand_path = Path(ligand_file)
    if not ligand_path.is_file():
        print(f"ERROR: Ligand file not found: {ligand_file}")
        return

    # Get binding site center
    center = _get_selection_center(site_selection)
    if center is None:
        print(
            f"WARNING: Selection '{site_selection}' is empty or invalid. "
            "Using receptor center as binding site."
        )
        center = _get_selection_center(receptor_obj)
        if center is None:
            print("ERROR: Could not determine binding site center.")
            return

    radius = _get_selection_radius(site_selection)

    # Create temporary working directory
    work_dir = tempfile.mkdtemp(prefix="flexaids_dock_")
    receptor_pdb = os.path.join(work_dir, "receptor.pdb")
    config_path = os.path.join(work_dir, "dock.inp")

    # Save receptor
    print(f"Preparing docking...")
    print(f"  Receptor: {receptor_obj}")
    print(f"  Ligand: {ligand_file}")
    print(f"  Binding site center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
    print(f"  Binding site radius: {radius:.1f} A")
    print(f"  Temperature: {temperature} K")

    if not _save_receptor_pdb(receptor_obj, receptor_pdb):
        return

    _write_minimal_config(
        config_path, receptor_pdb, str(ligand_path.resolve()),
        center, radius, temperature, n_results,
    )

    # Run docking
    callback = DockingProgressCallback()
    callback.running = True

    print(f"Starting FlexAID∆S docking (timeout: {timeout}s)...")

    try:
        from flexaidds.docking import Docking

        docking = Docking(config_path)
        population = docking.run(timeout=timeout)

        callback.running = False

        # Load results into PyMOL
        n_modes = len(population)
        print(f"Docking complete: {n_modes} binding mode(s) found.")

        for mode_idx, mode in enumerate(population):
            thermo = mode.get_thermodynamics()
            print(
                f"  Mode {mode_idx + 1}: F={thermo.free_energy:.2f} kcal/mol, "
                f"H={thermo.mean_energy:.2f}, S={thermo.entropy:.6f}, "
                f"n_poses={mode.n_poses}"
            )

        # Load output PDBs into PyMOL
        pdb_files = sorted(Path(work_dir).glob("*_*.pdb"))
        output_pdbs = [p for p in pdb_files if p.name != "receptor.pdb"]

        if output_pdbs:
            from . import results_adapter
            results_adapter.load_docking_results(work_dir, prefix="dock")
            print(f"Results loaded into PyMOL with prefix 'dock'.")
        else:
            print("No output PDB files generated.")

    except FileNotFoundError:
        print(
            "ERROR: FlexAID binary not found. Build with:\n"
            "  cmake --build build --target FlexAID"
        )
    except RuntimeError as exc:
        print(f"ERROR: Docking failed: {exc}")
    except Exception as exc:
        print(f"ERROR: Unexpected error: {exc}")
    finally:
        callback.running = False
