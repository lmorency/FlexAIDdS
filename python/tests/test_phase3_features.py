"""Tests for Phase 3 PyMOL plugin features.

Tests cover:
- Entropy heatmap spatial entropy computation
- Mode animation coordinate interpolation
- ITC comparison data parsing
- Interactive docking configuration generation

All tests mock PyMOL to avoid requiring it as a dependency.
"""

import csv
import math
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test files."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture(autouse=True)
def mock_pymol():
    """Mock PyMOL for all tests in this module."""
    pymol_mod = types.ModuleType("pymol")
    pymol_mod.cmd = MagicMock()
    pymol_mod.stored = MagicMock()
    with patch.dict(sys.modules, {"pymol": pymol_mod}):
        yield pymol_mod


@pytest.fixture
def sample_pdb(tmp_dir):
    """Create a minimal PDB file with ATOM records."""
    pdb_path = tmp_dir / "pose1.pdb"
    pdb_path.write_text(
        "REMARK Binding Mode:1 Best CF in Binding Mode:-5.0 Binding Mode Frequency:3\n"
        "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00  0.00           C\n"
        "ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00  0.00           C\n"
        "ATOM      4  O   ALA A   1       4.000   5.000   6.000  1.00  0.00           O\n"
        "ATOM      5  H   ALA A   1       1.500   2.500   3.500  1.00  0.00           H\n"
        "END\n"
    )
    return pdb_path


@pytest.fixture
def sample_pdb2(tmp_dir):
    """Create a second PDB file with different coordinates."""
    pdb_path = tmp_dir / "pose2.pdb"
    pdb_path.write_text(
        "REMARK Binding Mode:1 Best CF in Binding Mode:-4.5 Binding Mode Frequency:3\n"
        "ATOM      1  N   ALA A   1       5.000   6.000   7.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       6.000   7.000   8.000  1.00  0.00           C\n"
        "ATOM      3  C   ALA A   1       7.000   8.000   9.000  1.00  0.00           C\n"
        "ATOM      4  O   ALA A   1       8.000   9.000  10.000  1.00  0.00           O\n"
        "ATOM      5  H   ALA A   1       5.500   6.500   7.500  1.00  0.00           H\n"
        "END\n"
    )
    return pdb_path


@pytest.fixture
def sample_itc_csv(tmp_dir):
    """Create a sample ITC experimental data CSV."""
    csv_path = tmp_dir / "itc_data.csv"
    csv_path.write_text("dG,dH,TdS\n-8.5,-12.3,3.8\n")
    return csv_path


# ---------------------------------------------------------------------------
# Entropy Heatmap Tests
# ---------------------------------------------------------------------------

class TestEntropyHeatmapHelpers:
    """Test entropy heatmap computation."""

    def test_read_pose_coords_filters_hydrogens(self, sample_pdb):
        """Hydrogen atoms should be excluded from coordinate extraction."""
        from pymol_plugin.entropy_heatmap import _read_pose_coords
        coords = _read_pose_coords(str(sample_pdb), ligand_only=False)
        # Should have 4 heavy atoms (N, CA, C, O), not H
        assert len(coords) == 4

    def test_read_pose_coords_returns_xyz(self, sample_pdb):
        """Coordinates should be (x, y, z) tuples."""
        from pymol_plugin.entropy_heatmap import _read_pose_coords
        coords = _read_pose_coords(str(sample_pdb), ligand_only=False)
        assert coords[0] == pytest.approx((1.0, 2.0, 3.0))
        assert coords[1] == pytest.approx((2.0, 3.0, 4.0))

    def test_read_pose_coords_missing_file(self):
        """Non-existent file should return empty list."""
        from pymol_plugin.entropy_heatmap import _read_pose_coords
        coords = _read_pose_coords("/nonexistent/file.pdb")
        assert coords == []

    def test_compute_grid_bounds(self, sample_pdb):
        """Grid bounds should encompass all coordinates with padding."""
        from pymol_plugin.entropy_heatmap import (
            _read_pose_coords, _compute_grid_bounds
        )
        coords = _read_pose_coords(str(sample_pdb), ligand_only=False)
        lo, hi = _compute_grid_bounds(coords, padding=2.0)
        assert lo[0] < 1.0  # x_min - padding
        assert hi[0] > 4.0  # x_max + padding

    def test_entropy_color_low(self):
        """Low entropy should be blue."""
        from pymol_plugin.entropy_heatmap import _entropy_color
        r, g, b = _entropy_color(0.0)
        assert b == 1.0
        assert r == 0.0

    def test_entropy_color_high(self):
        """High entropy should be red."""
        from pymol_plugin.entropy_heatmap import _entropy_color
        r, g, b = _entropy_color(1.0)
        assert r == 1.0
        assert b == 0.0

    def test_entropy_color_mid(self):
        """Mid entropy should be white."""
        from pymol_plugin.entropy_heatmap import _entropy_color
        r, g, b = _entropy_color(0.5)
        assert r == pytest.approx(1.0)
        assert g == pytest.approx(1.0)
        assert b == pytest.approx(1.0)

    def test_ligand_only_fallback(self, sample_pdb):
        """ligand_only=True falls back to all atoms when no HETATM found."""
        from pymol_plugin.entropy_heatmap import _read_pose_coords
        # sample_pdb has only ATOM records, so ligand_only should fall back
        coords = _read_pose_coords(str(sample_pdb), ligand_only=True)
        assert len(coords) == 4  # Falls back to all heavy atoms


# ---------------------------------------------------------------------------
# Mode Animation Tests
# ---------------------------------------------------------------------------

class TestModeAnimationHelpers:
    """Test mode animation coordinate interpolation."""

    def test_read_atom_coords(self, sample_pdb):
        """Should read all ATOM coordinates including hydrogens."""
        from pymol_plugin.mode_animation import _read_atom_coords
        coords = _read_atom_coords(str(sample_pdb))
        assert len(coords) == 5  # All atoms including H

    def test_interpolate_coords_t0(self, sample_pdb, sample_pdb2):
        """t=0 should return coords1."""
        from pymol_plugin.mode_animation import (
            _read_atom_coords, _interpolate_coords
        )
        c1 = _read_atom_coords(str(sample_pdb))
        c2 = _read_atom_coords(str(sample_pdb2))
        result = _interpolate_coords(c1, c2, 0.0)
        for i in range(len(result)):
            assert result[i] == pytest.approx(c1[i])

    def test_interpolate_coords_t1(self, sample_pdb, sample_pdb2):
        """t=1 should return coords2."""
        from pymol_plugin.mode_animation import (
            _read_atom_coords, _interpolate_coords
        )
        c1 = _read_atom_coords(str(sample_pdb))
        c2 = _read_atom_coords(str(sample_pdb2))
        result = _interpolate_coords(c1, c2, 1.0)
        for i in range(len(result)):
            assert result[i] == pytest.approx(c2[i])

    def test_interpolate_coords_midpoint(self, sample_pdb, sample_pdb2):
        """t=0.5 should return the midpoint."""
        from pymol_plugin.mode_animation import (
            _read_atom_coords, _interpolate_coords
        )
        c1 = _read_atom_coords(str(sample_pdb))
        c2 = _read_atom_coords(str(sample_pdb2))
        result = _interpolate_coords(c1, c2, 0.5)
        # First atom: (1,2,3) and (5,6,7) -> midpoint (3,4,5)
        assert result[0] == pytest.approx([3.0, 4.0, 5.0])

    def test_interpolate_unequal_lengths(self):
        """Unequal coordinate lists should use the shorter length."""
        from pymol_plugin.mode_animation import _interpolate_coords
        c1 = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
        c2 = [[10.0, 10.0, 10.0], [11.0, 11.0, 11.0]]
        result = _interpolate_coords(c1, c2, 0.5)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# ITC Comparison Tests
# ---------------------------------------------------------------------------

class TestITCComparison:
    """Test ITC data parsing and comparison helpers."""

    def test_parse_itc_csv(self, sample_itc_csv):
        """Should parse ITC experimental data from CSV."""
        from pymol_plugin.itc_comparison import _parse_itc_csv
        data = _parse_itc_csv(str(sample_itc_csv))
        assert data["dG"] == pytest.approx(-8.5)
        assert data["dH"] == pytest.approx(-12.3)
        assert data["TdS"] == pytest.approx(3.8)

    def test_parse_itc_csv_missing_file(self, tmp_dir):
        """Missing CSV file should raise an error."""
        from pymol_plugin.itc_comparison import _parse_itc_csv
        with pytest.raises((FileNotFoundError, OSError)):
            _parse_itc_csv(str(tmp_dir / "nonexistent.csv"))


# ---------------------------------------------------------------------------
# Interactive Docking Config Tests
# ---------------------------------------------------------------------------

class TestInteractiveDockingHelpers:
    """Test interactive docking configuration generation."""

    def test_write_minimal_config(self, tmp_dir):
        """Should generate a valid FlexAID configuration file."""
        from pymol_plugin.interactive_docking import _write_minimal_config
        config_path = str(tmp_dir / "test.inp")
        _write_minimal_config(
            config_path,
            receptor_pdb="receptor.pdb",
            ligand_file="ligand.mol2",
            center=(10.0, 20.0, 30.0),
            radius=15.0,
            temperature=310,
            n_results=5,
        )
        text = Path(config_path).read_text()
        assert "PDBNAM receptor.pdb" in text
        assert "INPLIG ligand.mol2" in text
        assert "TEMPER 310" in text
        assert "NRGOUT 5" in text
        assert "COMPLF 10.000 20.000 30.000" in text
        assert "SPACER 15.0" in text

    def test_write_minimal_config_defaults(self, tmp_dir):
        """Default temperature and n_results should be used."""
        from pymol_plugin.interactive_docking import _write_minimal_config
        config_path = str(tmp_dir / "defaults.inp")
        _write_minimal_config(
            config_path,
            receptor_pdb="rec.pdb",
            ligand_file="lig.mol2",
            center=(0.0, 0.0, 0.0),
            radius=10.0,
        )
        text = Path(config_path).read_text()
        assert "TEMPER 300" in text
        assert "NRGOUT 10" in text
