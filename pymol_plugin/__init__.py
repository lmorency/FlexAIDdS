"""FlexAID∆S PyMOL Plugin.

Visualization and analysis interface for molecular docking results.

Features:
- Binding mode cluster rendering
- Thermodynamic property display
- Pose ensemble visualization with Boltzmann weighting
- Integration with NRGSuite workflow
- Read-only loading of docking result ensembles through the flexaidds Python API
- Entropy heatmap visualization (spatial entropy density)
- Interactive docking workflow from PyMOL
- Binding mode animation (coordinate interpolation morph)
- ITC-style thermogram comparison plots

Installation:
    1. PyMOL > Plugin Manager > Install New Plugin
    2. Select this directory or ZIP file
    3. Restart PyMOL

Usage:
    PyMOL> Plugin > FlexAID∆S
"""

from __future__ import absolute_import, print_function

try:
    from pymol import cmd, stored

    PYMOL_AVAILABLE = True
except ImportError:
    PYMOL_AVAILABLE = False
    import warnings

    warnings.warn("PyMOL not available. Plugin functionality disabled.", ImportWarning)

__version__ = "2.0.0-alpha"
__author__ = "Louis-Philippe Morency"
__email__ = "lp@thebonhomme.com"


def __init_plugin__(app=None):
    """PyMOL plugin initialization (called automatically by PyMOL)."""
    if not PYMOL_AVAILABLE:
        print("FlexAID∆S Plugin: PyMOL not available")
        return

    from pymol.plugins import addmenuitemqt

    addmenuitemqt("FlexAID∆S", run_plugin_gui)


def run_plugin_gui():
    """Launch the main FlexAID∆S GUI panel."""
    if not PYMOL_AVAILABLE:
        print("ERROR: PyMOL not available")
        return

    from .gui import FlexAIDSPanel

    dialog = FlexAIDSPanel()
    dialog.show()


# Register PyMOL commands for CLI access
if PYMOL_AVAILABLE:
    from .visualization import (
        load_binding_modes,
        show_pose_ensemble,
        color_by_boltzmann_weight,
        show_thermodynamics,
    )
    from .results_adapter import (
        load_docking_results,
        show_binding_mode,
        color_mode_by_score,
        show_mode_details,
    )
    from .entropy_heatmap import render_entropy_heatmap
    from .mode_animation import animate_binding_modes
    from .itc_comparison import (
        plot_enthalpy_entropy_compensation,
        plot_free_energy_comparison,
    )
    from .interactive_docking import dock_interactive, dock_cancel

    cmd.extend("flexaids_load", load_binding_modes)
    cmd.extend("flexaids_show_ensemble", show_pose_ensemble)
    cmd.extend("flexaids_color_boltzmann", color_by_boltzmann_weight)
    cmd.extend("flexaids_thermo", show_thermodynamics)
    cmd.extend("flexaids_load_results", load_docking_results)
    cmd.extend("flexaids_show_mode", show_binding_mode)
    cmd.extend("flexaids_color_mode", color_mode_by_score)
    cmd.extend("flexaids_mode_details", show_mode_details)
    cmd.extend("flexaids_entropy_heatmap", render_entropy_heatmap)
    cmd.extend("flexaids_animate", animate_binding_modes)
    cmd.extend("flexaids_itc_plot", plot_enthalpy_entropy_compensation)
    cmd.extend("flexaids_itc_compare", plot_free_energy_comparison)
    cmd.extend("flexaids_dock", dock_interactive)
    cmd.extend("flexaids_dock_cancel", dock_cancel)
