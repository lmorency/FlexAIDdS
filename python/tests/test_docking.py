"""Tests for flexaidds.docking – Docking config parser, Pose, BindingPopulation.

Priority 5 coverage.  No C++ extension needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flexaidds.docking import BindingMode, BindingPopulation, Docking, Pose


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_config(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ===========================================================================
# Pose
# ===========================================================================

class TestPose:
    def test_to_dict_keys(self):
        pose = Pose(index=0, energy=-10.5, rmsd=1.2, boltzmann_weight=0.3)
        d = pose.to_dict()
        assert set(d.keys()) == {"index", "energy_kcal_mol", "rmsd_angstrom", "boltzmann_weight"}

    def test_to_dict_values(self):
        pose = Pose(index=3, energy=-8.0, rmsd=0.5, boltzmann_weight=0.25)
        d = pose.to_dict()
        assert d["index"] == 3
        assert d["energy_kcal_mol"] == pytest.approx(-8.0)
        assert d["rmsd_angstrom"] == pytest.approx(0.5)
        assert d["boltzmann_weight"] == pytest.approx(0.25)

    def test_to_dict_rmsd_none(self):
        pose = Pose(index=0, energy=-5.0)
        assert pose.to_dict()["rmsd_angstrom"] is None

    def test_default_boltzmann_weight_is_zero(self):
        pose = Pose(index=0, energy=-5.0)
        assert pose.boltzmann_weight == 0.0


# ===========================================================================
# BindingMode – without C++ backend
# ===========================================================================

class TestBindingModeNoCpp:
    def test_n_poses_uses_internal_list_when_no_cpp(self):
        mode = BindingMode(cpp_binding_mode=None)
        assert mode.n_poses == 0

    def test_len_delegates_to_n_poses(self):
        mode = BindingMode(cpp_binding_mode=None)
        assert len(mode) == 0

    def test_free_energy_returns_inf_without_cpp(self):
        mode = BindingMode(cpp_binding_mode=None)
        assert mode.free_energy == float("inf")

    def test_enthalpy_returns_inf_without_cpp(self):
        mode = BindingMode(cpp_binding_mode=None)
        assert mode.enthalpy == float("inf")

    def test_entropy_returns_zero_without_cpp(self):
        mode = BindingMode(cpp_binding_mode=None)
        assert mode.entropy == 0.0

    def test_get_thermodynamics_raises_without_cpp(self):
        mode = BindingMode(cpp_binding_mode=None)
        with pytest.raises(RuntimeError, match="not initialized"):
            mode.get_thermodynamics()

    def test_repr_contains_mode_info(self):
        mode = BindingMode(cpp_binding_mode=None)
        r = repr(mode)
        assert "BindingMode" in r


# ===========================================================================
# BindingPopulation
# ===========================================================================

class TestBindingPopulation:
    def _make_population(self, n: int = 3) -> BindingPopulation:
        pop = BindingPopulation()
        for _ in range(n):
            pop.add_mode(BindingMode(cpp_binding_mode=None))
        return pop

    def test_n_modes_zero_initially(self):
        assert BindingPopulation().n_modes == 0

    def test_add_mode_increments_count(self):
        pop = self._make_population(3)
        assert pop.n_modes == 3

    def test_len_matches_n_modes(self):
        pop = self._make_population(2)
        assert len(pop) == 2

    def test_getitem_returns_correct_mode(self):
        pop = BindingPopulation()
        m1 = BindingMode(cpp_binding_mode=None)
        m2 = BindingMode(cpp_binding_mode=None)
        pop.add_mode(m1)
        pop.add_mode(m2)
        assert pop[0] is m1
        assert pop[1] is m2

    def test_iter_yields_all_modes(self):
        pop = self._make_population(3)
        modes = list(pop)
        assert len(modes) == 3

    def test_rank_by_free_energy_order(self):
        """All modes have free_energy=inf without cpp, so order is stable."""
        pop = self._make_population(3)
        ranked = pop.rank_by_free_energy()
        assert len(ranked) == 3

    def test_repr_contains_population_info(self):
        pop = self._make_population(2)
        r = repr(pop)
        assert "BindingPopulation" in r
        assert "2" in r


# ===========================================================================
# Docking – FileNotFoundError
# ===========================================================================

class TestDockingInit:
    def test_raises_when_config_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            Docking(str(tmp_path / "nonexistent.inp"))

    def test_repr(self, tmp_path):
        cfg = tmp_path / "test.inp"
        _write_config(cfg, ["PDBNAM receptor.pdb"])
        d = Docking(str(cfg))
        assert "Docking" in repr(d)
        assert "test.inp" in repr(d)


# ===========================================================================
# Docking._parse_config – keyword type dispatch
# ===========================================================================

class TestDockingParseConfig:
    def _docking(self, tmp_path: Path, lines: list[str]) -> Docking:
        cfg = tmp_path / "test.inp"
        _write_config(cfg, lines)
        return Docking(str(cfg))

    # --- string keywords ----------------------------------------------------

    def test_string_keyword_pdbnam(self, tmp_path):
        d = self._docking(tmp_path, ["PDBNAM receptor.pdb"])
        assert d.receptor == "receptor.pdb"

    def test_string_keyword_inplig(self, tmp_path):
        d = self._docking(tmp_path, ["INPLIG ligand.mol2"])
        assert d.ligand == "ligand.mol2"

    def test_string_keyword_metopt(self, tmp_path):
        d = self._docking(tmp_path, ["METOPT GA"])
        assert d.optimization_method == "GA"

    # --- float keywords -----------------------------------------------------

    def test_float_keyword_clrmsd(self, tmp_path):
        d = self._docking(tmp_path, ["CLRMSD 1.5"])
        assert d._config["CLRMSD"] == pytest.approx(1.5)
        assert isinstance(d._config["CLRMSD"], float)

    def test_float_keyword_extra_tokens_ignored(self, tmp_path):
        """Only the first token after the keyword is used for float keys."""
        d = self._docking(tmp_path, ["CLRMSD 2.0 # comment"])
        assert d._config["CLRMSD"] == pytest.approx(2.0)

    def test_float_keyword_bad_value_stored_as_string(self, tmp_path):
        d = self._docking(tmp_path, ["CLRMSD NOT_A_FLOAT"])
        assert isinstance(d._config["CLRMSD"], str)

    # --- integer keywords ---------------------------------------------------

    def test_int_keyword_temper(self, tmp_path):
        d = self._docking(tmp_path, ["TEMPER 310"])
        assert d.temperature == 310
        assert isinstance(d._config["TEMPER"], int)

    def test_int_keyword_bad_value_stored_as_string(self, tmp_path):
        d = self._docking(tmp_path, ["TEMPER NOT_INT"])
        assert isinstance(d._config["TEMPER"], str)

    # --- list keywords (OPTIMZ / FLEXSC) ------------------------------------

    def test_list_keyword_multiple_entries(self, tmp_path):
        d = self._docking(tmp_path, [
            "OPTIMZ option1",
            "OPTIMZ option2",
            "OPTIMZ option3",
        ])
        assert d._config["OPTIMZ"] == ["option1", "option2", "option3"]

    def test_list_keyword_initialised_empty(self, tmp_path):
        d = self._docking(tmp_path, ["PDBNAM receptor.pdb"])
        assert d._config["OPTIMZ"] == []
        assert d._config["FLEXSC"] == []

    def test_flexsc_collected(self, tmp_path):
        d = self._docking(tmp_path, ["FLEXSC chainA", "FLEXSC chainB"])
        assert d._config["FLEXSC"] == ["chainA", "chainB"]

    # --- flag keywords ------------------------------------------------------

    def test_flag_keyword_exchet(self, tmp_path):
        d = self._docking(tmp_path, ["EXCHET"])
        assert d._config["EXCHET"] is True

    def test_flag_keyword_rotobs(self, tmp_path):
        d = self._docking(tmp_path, ["ROTOBS"])
        assert d._config["ROTOBS"] is True

    def test_flag_keyword_usedee(self, tmp_path):
        d = self._docking(tmp_path, ["USEDEE"])
        assert d._config["USEDEE"] is True

    def test_absent_flag_not_in_config(self, tmp_path):
        d = self._docking(tmp_path, ["PDBNAM receptor.pdb"])
        assert "EXCHET" not in d._config

    # --- comment and blank line handling ------------------------------------

    def test_hash_comment_lines_ignored(self, tmp_path):
        d = self._docking(tmp_path, [
            "# This is a comment",
            "PDBNAM receptor.pdb",
        ])
        assert d._config["PDBNAM"] == "receptor.pdb"

    def test_blank_lines_ignored(self, tmp_path):
        d = self._docking(tmp_path, ["", "   ", "PDBNAM receptor.pdb", ""])
        assert d._config["PDBNAM"] == "receptor.pdb"

    def test_line_shorter_than_6_chars_ignored(self, tmp_path):
        d = self._docking(tmp_path, ["AB", "PDBNAM receptor.pdb"])
        assert d._config["PDBNAM"] == "receptor.pdb"

    # --- unknown keywords ---------------------------------------------------

    def test_unknown_keyword_stored_as_raw_string(self, tmp_path):
        d = self._docking(tmp_path, ["UNKNWN some_value"])
        assert d._config["UNKNWN"] == "some_value"

    def test_unknown_keyword_no_value(self, tmp_path):
        d = self._docking(tmp_path, ["UNKNWN"])
        assert d._config["UNKNWN"] == ""

    # --- full realistic config ----------------------------------------------

    def test_realistic_config(self, tmp_path):
        lines = [
            "# FlexAID config",
            "PDBNAM /data/receptor.pdb",
            "INPLIG /data/ligand.mol2",
            "METOPT GA",
            "TEMPER 300",
            "CLRMSD 2.0",
            "OPTIMZ ROTRAN",
            "OPTIMZ DIHANG",
            "EXCHET",
            "USEDEE",
        ]
        d = self._docking(tmp_path, lines)
        assert d.receptor == "/data/receptor.pdb"
        assert d.ligand == "/data/ligand.mol2"
        assert d.optimization_method == "GA"
        assert d.temperature == 300
        assert d._config["CLRMSD"] == pytest.approx(2.0)
        assert d._config["OPTIMZ"] == ["ROTRAN", "DIHANG"]
        assert d._config["EXCHET"] is True
        assert d._config["USEDEE"] is True

    # --- property accessors when keys absent --------------------------------

    def test_receptor_none_when_absent(self, tmp_path):
        d = self._docking(tmp_path, ["METOPT GA"])
        assert d.receptor is None

    def test_ligand_none_when_absent(self, tmp_path):
        d = self._docking(tmp_path, ["PDBNAM receptor.pdb"])
        assert d.ligand is None

    def test_temperature_none_when_absent(self, tmp_path):
        d = self._docking(tmp_path, ["PDBNAM receptor.pdb"])
        assert d.temperature is None

    def test_optimization_method_none_when_absent(self, tmp_path):
        d = self._docking(tmp_path, ["PDBNAM receptor.pdb"])
        assert d.optimization_method is None


# ===========================================================================
# Docking.run – raises NotImplementedError
# ===========================================================================

class TestDockingRun:
    def test_run_raises_not_implemented(self, tmp_path):
        cfg = tmp_path / "test.inp"
        _write_config(cfg, ["PDBNAM receptor.pdb"])
        d = Docking(str(cfg))
        with pytest.raises(NotImplementedError):
            d.run()


# ===========================================================================
# BindingPopulation.compute_global_thermodynamics – needs C++ core
# ===========================================================================

_core_available: bool
try:
    from flexaidds.thermodynamics import StatMechEngine as _SME
    _SME(300.0)
    _core_available = True
except Exception:
    _core_available = False


@pytest.mark.skipif(not _core_available, reason="C++ _core extension not built")
class TestComputeGlobalThermodynamics:
    """compute_global_thermodynamics() aggregates all mode pose energies via
    StatMechEngine and returns a Thermodynamics object."""

    def _pop_with_real_modes(self) -> BindingPopulation:
        """Return a BindingPopulation backed by real C++ BindingMode stubs.

        Since we cannot easily instantiate C++ BindingMode objects here we
        monkey-patch the BindingMode instances so that n_poses and enthalpy
        delegate to Python values only.
        """
        from flexaidds.thermodynamics import Thermodynamics

        pop = BindingPopulation()
        # Use real BindingMode instances but override their C++-dependent attrs
        for energy in (-10.0, -9.0):
            mode = BindingMode(cpp_binding_mode=None)
            # Manually push a fake pose so n_poses = 1
            mode._poses.append(object())  # just needs to be countable
            # Patch enthalpy property to return a known value
            type(mode).enthalpy = property(lambda self, e=energy: e)
            pop.add_mode(mode)
        return pop

    def test_returns_thermodynamics_object(self):
        from flexaidds.thermodynamics import Thermodynamics
        pop = self._pop_with_real_modes()
        result = pop.compute_global_thermodynamics()
        assert isinstance(result, Thermodynamics)

    def test_free_energy_is_finite(self):
        pop = self._pop_with_real_modes()
        result = pop.compute_global_thermodynamics()
        import math
        assert math.isfinite(result.free_energy)
