"""Verify the flexaidds package imports gracefully without C++ bindings.

The read-only result loading (models, io, results) must work even when the
compiled _core extension is unavailable.  Only StatMechEngine should raise
at *instantiation* time, not at import time.
"""

import importlib
import sys
import types
from pathlib import Path
from unittest import mock


def test_package_imports_without_core_extension():
    """Importing flexaidds must not crash when _core is missing."""
    # Temporarily remove _core from sys.modules so the import fallback is
    # exercised.  We patch it with a module that raises ImportError on
    # attribute access, simulating an unbuilt extension.
    saved = sys.modules.pop("flexaidds._core", None)
    saved_pkg = sys.modules.pop("flexaidds", None)
    saved_thermo = sys.modules.pop("flexaidds.thermodynamics", None)
    saved_models = sys.modules.pop("flexaidds.models", None)
    saved_io = sys.modules.pop("flexaidds.io", None)
    saved_results = sys.modules.pop("flexaidds.results", None)

    # Make _core raise ImportError when the package tries to import it
    sys.modules["flexaidds._core"] = None  # triggers ImportError on import

    try:
        import flexaidds

        # Read-only classes should be available
        assert hasattr(flexaidds, "PoseResult")
        assert hasattr(flexaidds, "BindingModeResult")
        assert hasattr(flexaidds, "DockingResult")
        assert hasattr(flexaidds, "load_results")
        assert hasattr(flexaidds, "Thermodynamics")
        assert hasattr(flexaidds, "StatMechEngine")
    finally:
        # Restore original module state
        sys.modules.pop("flexaidds._core", None)
        sys.modules.pop("flexaidds", None)
        sys.modules.pop("flexaidds.thermodynamics", None)
        sys.modules.pop("flexaidds.models", None)
        sys.modules.pop("flexaidds.io", None)
        sys.modules.pop("flexaidds.results", None)
        if saved is not None:
            sys.modules["flexaidds._core"] = saved
        if saved_thermo is not None:
            sys.modules["flexaidds.thermodynamics"] = saved_thermo
        if saved_models is not None:
            sys.modules["flexaidds.models"] = saved_models
        if saved_io is not None:
            sys.modules["flexaidds.io"] = saved_io
        if saved_results is not None:
            sys.modules["flexaidds.results"] = saved_results
        if saved_pkg is not None:
            sys.modules["flexaidds"] = saved_pkg


def test_load_results_works_without_core(tmp_path: Path):
    """load_results() should work without C++ bindings."""
    # Write a minimal PDB file
    pdb = tmp_path / "mode_1_pose_1.pdb"
    pdb.write_text(
        "REMARK binding_mode = 1\n"
        "REMARK CF = -5.0\n"
        "ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )

    from flexaidds.results import load_results

    result = load_results(tmp_path)
    assert result.n_modes == 1
    assert result.binding_modes[0].best_cf == -5.0
