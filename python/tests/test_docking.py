"""Tests for flexaidds.docking — exercising Python-only paths."""

import math
import tempfile
import textwrap
from pathlib import Path

import pytest

from flexaidds.docking import Docking, BindingMode, BindingPopulation, Pose


# ── Fixtures ──────────────────────────────────────────────────────────────────

MINIMAL_CONFIG = textwrap.dedent("""\
    PDBNAM receptor.pdb
    INPLIG ligand.mol2
    TEMPER 300
    METOPT GA
    NRGOUT 3
""")

REMARK_PDB_CONTENT = textwrap.dedent("""\
    REMARK Binding Mode:1 Best CF in Binding Mode:-11.23 Binding Mode Frequency:5
    REMARK 0.45678 RMSD to ref. structure
    ATOM      1  N   GLY A   1       1.000   2.000   3.000  1.00  0.00           N
    END
""")


@pytest.fixture
def config_file(tmp_path):
    cfg = tmp_path / "test.inp"
    cfg.write_text(MINIMAL_CONFIG)
    return cfg


@pytest.fixture
def docking(config_file):
    return Docking(str(config_file))


# ── Docking.__init__ / _parse_config ─────────────────────────────────────────

class TestDockingInit:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Docking(str(tmp_path / "nonexistent.inp"))

    def test_receptor_property(self, docking):
        assert docking.receptor == "receptor.pdb"

    def test_ligand_property(self, docking):
        assert docking.ligand == "ligand.mol2"

    def test_temperature_property(self, docking):
        assert docking.temperature == 300

    def test_optimization_method(self, docking):
        assert docking.optimization_method == "GA"

    def test_repr(self, docking):
        assert "test.inp" in repr(docking)

    def test_list_keys_always_lists(self, config_file):
        """OPTIMZ and FLEXSC should always be lists even when absent."""
        d = Docking(str(config_file))
        assert isinstance(d._config["OPTIMZ"], list)
        assert isinstance(d._config["FLEXSC"], list)

    def test_multiple_optimz(self, tmp_path):
        cfg = tmp_path / "multi.inp"
        cfg.write_text("PDBNAM x.pdb\nOPTIMZ res1\nOPTIMZ res2\n")
        d = Docking(str(cfg))
        assert d._config["OPTIMZ"] == ["res1", "res2"]


# ── _parse_remark_pdb ─────────────────────────────────────────────────────────

class TestParseRemarkPdb:
    def test_valid_remark(self, tmp_path):
        pdb = tmp_path / "pose.pdb"
        pdb.write_text(REMARK_PDB_CONTENT)
        result = Docking._parse_remark_pdb(pdb, temperature=300.0)
        assert result is not None
        mode_idx, pose = result
        assert mode_idx == 1
        assert abs(pose.energy - (-11.23)) < 1e-6
        assert abs(pose.rmsd - 0.45678) < 1e-6

    def test_missing_remark_returns_none(self, tmp_path):
        pdb = tmp_path / "plain.pdb"
        pdb.write_text("ATOM      1  N   GLY A   1       1.0   2.0   3.0\nEND\n")
        assert Docking._parse_remark_pdb(pdb, temperature=300.0) is None

    def test_boltzmann_weight_positive(self, tmp_path):
        pdb = tmp_path / "pose.pdb"
        pdb.write_text(REMARK_PDB_CONTENT)
        _, pose = Docking._parse_remark_pdb(pdb, temperature=300.0)
        assert pose.boltzmann_weight > 0.0

    def test_boltzmann_weight_formula(self, tmp_path):
        pdb = tmp_path / "pose.pdb"
        pdb.write_text(REMARK_PDB_CONTENT)
        _, pose = Docking._parse_remark_pdb(pdb, temperature=300.0)
        kB = 0.001987206
        beta = 1.0 / (kB * 300.0)
        expected = math.exp(-beta * (-11.23))
        assert abs(pose.boltzmann_weight - expected) < 1e-10


# ── Pose ──────────────────────────────────────────────────────────────────────

class TestPose:
    def test_to_dict_keys(self):
        p = Pose(index=1, energy=-10.5, rmsd=0.3, boltzmann_weight=0.4)
        d = p.to_dict()
        assert "index" in d
        assert "energy_kcal_mol" in d
        assert "rmsd_angstrom" in d

    def test_optional_fields_default_none(self):
        p = Pose(index=0, energy=-5.0)
        assert p.rmsd is None
        assert p.coordinates is None


# ── BindingMode ───────────────────────────────────────────────────────────────

class TestBindingMode:
    def test_free_energy_no_cpp(self):
        mode = BindingMode(cpp_binding_mode=None)
        mode._poses = [Pose(0, -10.5), Pose(1, -9.8), Pose(2, -11.0)]
        assert abs(mode.free_energy - (-11.0)) < 1e-9

    def test_free_energy_empty_is_inf(self):
        mode = BindingMode()
        assert mode.free_energy == float("inf")

    def test_n_poses_python_path(self):
        mode = BindingMode()
        mode._poses = [Pose(0, -10.0), Pose(1, -9.0)]
        assert mode.n_poses == 2

    def test_len_equals_n_poses(self):
        mode = BindingMode()
        mode._poses = [Pose(i, float(-i)) for i in range(5)]
        assert len(mode) == 5

    def test_repr_contains_n_poses(self):
        mode = BindingMode()
        mode._poses = [Pose(0, -10.0)]
        assert "n_poses=1" in repr(mode)


# ── BindingPopulation ─────────────────────────────────────────────────────────

class TestBindingPopulation:
    def _make_mode(self, energy: float) -> BindingMode:
        m = BindingMode()
        m._poses = [Pose(0, energy)]
        return m

    def test_rank_by_free_energy_sorted(self):
        pop = BindingPopulation()
        for e in [-8.0, -11.0, -9.5]:
            pop.add_mode(self._make_mode(e))
        ranked = pop.rank_by_free_energy()
        energies = [m.free_energy for m in ranked]
        assert energies == sorted(energies)

    def test_n_modes(self):
        pop = BindingPopulation()
        assert pop.n_modes == 0
        pop.add_mode(self._make_mode(-10.0))
        assert pop.n_modes == 1

    def test_len_equals_n_modes(self):
        modes = [self._make_mode(-10.0 - i) for i in range(3)]
        pop = BindingPopulation(modes=modes)
        assert len(pop) == 3

    def test_getitem(self):
        modes = [self._make_mode(-10.0 - i) for i in range(3)]
        pop = BindingPopulation(modes=modes)
        assert pop[0] is modes[0]

    def test_iter(self):
        modes = [self._make_mode(-10.0 - i) for i in range(3)]
        pop = BindingPopulation(modes=modes)
        assert list(pop) == modes

    def test_repr(self):
        pop = BindingPopulation(temperature=310.0)
        assert "310.0K" in repr(pop)
