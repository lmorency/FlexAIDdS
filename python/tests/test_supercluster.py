"""Tests for flexaidds.supercluster — super-cluster extraction module."""

import math

import pytest

from flexaidds.supercluster import SuperCluster
from flexaidds.docking import BindingMode, BindingPopulation, Pose
from flexaidds.tencm import run_shannon_thermo_stack, FullThermoResult


# ── SuperCluster basic tests ────────────────────────────────────────────────

class TestSuperCluster:
    def test_uniform_energies_keeps_most(self):
        energies = [-10.0 + 0.1 * i for i in range(20)]
        sc = SuperCluster(energies)
        filtered = sc.filter_energies()
        # Uniform spread — should keep a substantial fraction
        assert len(filtered) >= 4
        assert len(filtered) <= len(energies)

    def test_bimodal_filters_to_dominant(self):
        # Dense cluster near -10, sparse outliers near 0
        energies = [-10.0 + 0.1 * i for i in range(30)]
        energies += [0.0, 1.0, 2.0, 3.0, 4.0]
        sc = SuperCluster(energies)
        filtered = sc.filter_energies()
        assert len(filtered) < len(energies)
        assert len(filtered) > 0

    def test_empty_input(self):
        sc = SuperCluster([])
        assert sc.n_total == 0
        assert sc.n_selected == 0
        assert sc.filter_energies() == []

    def test_single_energy(self):
        sc = SuperCluster([-5.0])
        assert sc.n_selected == 1
        assert sc.filter_energies() == [-5.0]

    def test_identical_energies(self):
        energies = [-10.0] * 20
        sc = SuperCluster(energies)
        filtered = sc.filter_energies()
        # All identical → all should be kept
        assert len(filtered) >= 4

    def test_indices_are_valid(self):
        energies = [-10.0, -8.0, -12.0, -9.0, -11.0]
        sc = SuperCluster(energies)
        indices = sc.extract()
        for idx in indices:
            assert 0 <= idx < len(energies)

    def test_filter_preserves_values(self):
        energies = [-10.0, -8.0, -12.0, -9.0, -11.0]
        sc = SuperCluster(energies)
        filtered = sc.filter_energies()
        for e in filtered:
            assert e in energies

    def test_repr(self):
        sc = SuperCluster([-10.0, -9.0, -8.0])
        r = repr(sc)
        assert "SuperCluster" in r
        assert "n_selected=" in r

    def test_min_pts_respected(self):
        energies = [-10.0, -5.0, 0.0, 5.0, 10.0]
        sc = SuperCluster(energies, min_pts=3)
        indices = sc.extract()
        # Should return at least min_pts if possible
        assert len(indices) >= 3

    def test_caching(self):
        sc = SuperCluster([-10.0, -9.0, -8.0])
        indices1 = sc.extract()
        indices2 = sc.extract()
        assert indices1 == indices2


# ── Integration with run_shannon_thermo_stack ───────────────────────────────

class TestShannonThermoStackSuperCluster:
    def test_use_super_cluster_flag(self):
        energies = [-10.0, -12.0, -8.0, -11.0, -9.0]
        result = run_shannon_thermo_stack(
            energies, base_deltaG=-10.0, use_super_cluster=True)
        assert isinstance(result, FullThermoResult)
        assert math.isfinite(result.deltaG)
        assert result.shannonEntropy >= 0.0

    def test_super_cluster_report_mentions_filter(self):
        energies = list(range(20))  # 20 values for super-cluster to filter
        result = run_shannon_thermo_stack(
            energies, base_deltaG=0.0, use_super_cluster=True)
        assert "SuperCluster" in result.report

    def test_without_super_cluster_no_mention(self):
        energies = [-10.0, -12.0, -8.0]
        result = run_shannon_thermo_stack(
            energies, base_deltaG=-10.0, use_super_cluster=False)
        assert "SuperCluster" not in result.report

    def test_small_ensemble_skips_filter(self):
        # <= 4 energies should skip super-cluster
        energies = [-10.0, -9.0]
        result = run_shannon_thermo_stack(
            energies, base_deltaG=-10.0, use_super_cluster=True)
        assert "SuperCluster" not in result.report


# ── Integration with BindingPopulation ──────────────────────────────────────

class TestBindingPopulationSuperCluster:
    def _make_mode(self, energies):
        m = BindingMode()
        m._poses = [Pose(i, e) for i, e in enumerate(energies)]
        return m

    def test_compute_super_cluster_thermodynamics(self):
        pop = BindingPopulation()
        pop.add_mode(self._make_mode([-10.0, -9.5, -10.2]))
        pop.add_mode(self._make_mode([-8.0, -7.5]))
        result = pop.compute_super_cluster_thermodynamics()
        assert math.isfinite(result.free_energy)

    def test_empty_population(self):
        pop = BindingPopulation()
        result = pop.compute_super_cluster_thermodynamics()
        assert result.free_energy == float('inf')

    def test_super_cluster_vs_global(self):
        pop = BindingPopulation()
        pop.add_mode(self._make_mode([-10.0 + 0.1 * i for i in range(20)]))
        pop.add_mode(self._make_mode([5.0, 10.0, 15.0]))  # outliers
        global_thermo = pop.compute_global_thermodynamics()
        sc_thermo = pop.compute_super_cluster_thermodynamics()
        # Both should be finite
        assert math.isfinite(global_thermo.free_energy)
        assert math.isfinite(sc_thermo.free_energy)


# ── Docking config parser recognizes SUPCLU ─────────────────────────────────

class TestDockingSuperClusterConfig:
    def test_supclu_flag_parsed(self, tmp_path):
        cfg = tmp_path / "test.inp"
        cfg.write_text("PDBNAM receptor.pdb\nSUPCLU\n")
        from flexaidds.docking import Docking
        d = Docking(str(cfg))
        assert d._config.get("SUPCLU") is True

    def test_supclu_absent(self, tmp_path):
        cfg = tmp_path / "test.inp"
        cfg.write_text("PDBNAM receptor.pdb\n")
        from flexaidds.docking import Docking
        d = Docking(str(cfg))
        assert "SUPCLU" not in d._config
