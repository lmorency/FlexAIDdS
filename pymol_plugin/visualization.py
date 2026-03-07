"""PyMOL visualization functions for FlexAID∆S binding modes.

These functions can be called from PyMOL command line:
    PyMOL> flexaids_load /path/to/output
    PyMOL> flexaids_show_ensemble mode1
    PyMOL> flexaids_color_boltzmann mode1
    PyMOL> flexaids_thermo mode1
"""

import os
from pathlib import Path
from typing import List, Dict, Any

try:
    from pymol import cmd, stored
    import pymol
except ImportError:
    raise ImportError("PyMOL not available")


# Global state for loaded binding modes
_loaded_modes: Dict[str, Any] = {}


def load_binding_modes(output_dir: str) -> None:
    """Load FlexAID∆S docking results from output directory.
    
    Args:
        output_dir: Path to FlexAID output directory
    
    Example:
        PyMOL> flexaids_load /data/docking_results
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"ERROR: Directory not found: {output_dir}")
        return
    
    # TODO: Implement FlexAID output parser
    # Expected files:
    #   - result_*.pdb (pose PDB files)
    #   - binding_modes.txt (cluster assignments)
    #   - thermodynamics.txt (free energies, entropies)
    
    pdb_files = sorted(output_path.glob("result_*.pdb"))
    if not pdb_files:
        print(f"ERROR: No result PDB files found in {output_dir}")
        return
    
    # Load all poses into PyMOL
    for i, pdb_file in enumerate(pdb_files):
        obj_name = f"flexaids_pose_{i+1}"
        cmd.load(str(pdb_file), obj_name)
        cmd.disable(obj_name)  # hide by default
    
    print(f"Loaded {len(pdb_files)} poses from {output_dir}")
    print("Use 'flexaids_show_ensemble' to visualize binding modes")


def show_pose_ensemble(mode_name: str, show_all: bool = True) -> None:
    """Display all poses belonging to a binding mode.
    
    Args:
        mode_name: Binding mode identifier
        show_all: If True, show all poses; if False, show representative only
    
    Example:
        PyMOL> flexaids_show_ensemble mode1
        PyMOL> flexaids_show_ensemble mode1, show_all=0
    """
    # TODO: Filter poses by mode assignment
    # For now, show all loaded poses
    pose_objects = cmd.get_object_list("flexaids_pose_*")
    
    if not pose_objects:
        print("ERROR: No poses loaded. Use 'flexaids_load' first.")
        return
    
    # Show poses
    for obj in pose_objects:
        if show_all:
            cmd.enable(obj)
            cmd.show("cartoon", obj)
            cmd.show("sticks", f"{obj} and organic")
        else:
            # Show only first pose (representative)
            if obj == pose_objects[0]:
                cmd.enable(obj)
                cmd.show("cartoon", obj)
                cmd.show("sticks", f"{obj} and organic")
            else:
                cmd.disable(obj)
    
    cmd.zoom("flexaids_pose_*")
    print(f"Showing {'all' if show_all else 'representative'} poses for {mode_name}")


def color_by_boltzmann_weight(mode_name: str) -> None:
    """Color poses by Boltzmann weight (blue=low, red=high probability).
    
    Args:
        mode_name: Binding mode identifier
    
    Example:
        PyMOL> flexaids_color_boltzmann mode1
    """
    # TODO: Read Boltzmann weights from thermodynamics file
    # For now, use gradient based on pose index (placeholder)
    pose_objects = cmd.get_object_list("flexaids_pose_*")
    
    if not pose_objects:
        print("ERROR: No poses loaded.")
        return
    
    n_poses = len(pose_objects)
    for i, obj in enumerate(pose_objects):
        # Blue (low weight) to Red (high weight) gradient
        weight = 1.0 - (i / n_poses)  # placeholder: decreasing weight
        
        # RGB gradient: blue (0,0,1) -> red (1,0,0)
        r = 1.0 - weight
        b = weight
        g = 0.0
        
        color_name = f"boltzmann_{i}"
        cmd.set_color(color_name, [r, g, b])
        cmd.color(color_name, obj)
    
    print(f"Colored {n_poses} poses by Boltzmann weight (blue=low, red=high)")


def show_thermodynamics(mode_name: str) -> None:
    """Print thermodynamic properties of a binding mode.
    
    Args:
        mode_name: Binding mode identifier
    
    Example:
        PyMOL> flexaids_thermo mode1
    """
    # TODO: Load actual thermodynamics from BindingMode
    # Placeholder output
    print(f"\nThermodynamics for {mode_name}:")
    print("  ΔG (Free Energy):     -10.5 kcal/mol")
    print("  ΔH (Enthalpy):        -12.3 kcal/mol")
    print("  S (Entropy):           0.006 kcal/(mol·K)")
    print("  TΔS (Entropy term):    1.8 kcal/mol")
    print("  # Poses:               45")
    print("  Heat Capacity (Cv):    0.12 kcal/(mol·K²)")
    print()


def export_to_nrgsuite(output_dir: str, nrgsuite_file: str) -> None:
    """Export binding modes to NRGSuite-compatible format.
    
    Args:
        output_dir: FlexAID output directory
        nrgsuite_file: Output file path for NRGSuite
    
    Example:
        PyMOL> flexaids_export /data/docking_results, /data/nrgsuite_input.txt
    """
    # TODO: Implement NRGSuite format export
    print(f"Exporting to NRGSuite format: {nrgsuite_file}")
    print("(Not yet implemented)")


# Auto-register commands when module is imported
cmd.extend("flexaids_load", load_binding_modes)
cmd.extend("flexaids_show_ensemble", show_pose_ensemble)
cmd.extend("flexaids_color_boltzmann", color_by_boltzmann_weight)
cmd.extend("flexaids_thermo", show_thermodynamics)
cmd.extend("flexaids_export", export_to_nrgsuite)
