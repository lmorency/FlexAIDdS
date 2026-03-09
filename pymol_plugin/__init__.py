"""FlexAID∆S PyMOL Plugin.

Visualization and analysis interface for molecular docking results.

Features:
- Binding mode cluster rendering
- Thermodynamic property display
- Pose ensemble visualization with Boltzmann weighting
- Integration with NRGSuite workflow
- Read-only loading of docking result ensembles through the flexaidds Python API

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

__version__ = "1.0.0-alpha"
__author__ = "Louis-Philippe Morency"
__email__ = "lp.morency@umontreal.ca"


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
    
    cmd.extend("flexaids_load", load_binding_modes)
    cmd.extend("flexaids_show_ensemble", show_pose_ensemble)
    cmd.extend("flexaids_color_boltzmann", color_by_boltzmann_weight)
    cmd.extend("flexaids_thermo", show_thermodynamics)
    cmd.extend("flexaids_load_results", load_docking_results)
    cmd.extend("flexaids_show_mode", show_binding_mode)
    cmd.extend("flexaids_color_mode", color_mode_by_score)
    cmd.extend("flexaids_mode_details", show_mode_details)
