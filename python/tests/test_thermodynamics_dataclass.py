"""Tests for the pure-Python Thermodynamics dataclass.

No C++ extension required – these run in every environment.
"""

from __future__ import annotations

import pytest

from flexaidds.thermodynamics import Thermodynamics


def _make(**overrides) -> Thermodynamics:
    defaults = dict(
        temperature=300.0,
        log_Z=-16.8,
        free_energy=-10.0,
        mean_energy=-9.5,
        mean_energy_sq=90.5,
        heat_capacity=0.05,
        entropy=0.001667,
        std_energy=0.3,
    )
    defaults.update(overrides)
    return Thermodynamics(**defaults)


class TestThermodynamicsDataclass:
    def test_binding_free_energy_is_alias(self):
        t = _make(free_energy=-10.0)
        assert t.binding_free_energy == t.free_energy

    def test_entropy_term_is_T_times_S(self):
        T, S = 300.0, 0.002
        t = _make(temperature=T, entropy=S)
        assert t.entropy_term == pytest.approx(T * S)

    def test_to_dict_has_expected_keys(self):
        d = _make().to_dict()
        expected = {
            "temperature_K", "log_Z", "free_energy_kcal_mol",
            "enthalpy_kcal_mol", "entropy_kcal_mol_K",
            "heat_capacity_kcal_mol_K2", "std_energy_kcal_mol",
        }
        assert expected == set(d.keys())

    def test_to_dict_values_match_fields(self):
        t = _make()
        d = t.to_dict()
        assert d["temperature_K"] == t.temperature
        assert d["free_energy_kcal_mol"] == t.free_energy
        assert d["enthalpy_kcal_mol"] == t.mean_energy
        assert d["entropy_kcal_mol_K"] == t.entropy
        assert d["heat_capacity_kcal_mol_K2"] == t.heat_capacity
        assert d["std_energy_kcal_mol"] == t.std_energy

    def test_entropy_term_zero_when_entropy_zero(self):
        t = _make(entropy=0.0)
        assert t.entropy_term == 0.0

    def test_fields_are_readable(self):
        """All declared fields are readable (basic sanity)."""
        t = _make()
        for attr in ("temperature", "log_Z", "free_energy", "mean_energy",
                     "mean_energy_sq", "heat_capacity", "entropy", "std_energy"):
            assert hasattr(t, attr)


class TestThermodynamicsFromDict:
    def test_round_trip(self):
        """from_dict(to_dict()) should reconstruct all fields."""
        original = _make()
        d = original.to_dict()
        # Add fields not in to_dict() output for exact round-trip
        d["log_Z"] = original.log_Z
        d["mean_energy_sq"] = original.mean_energy_sq
        restored = Thermodynamics.from_dict(d)
        assert restored.temperature == original.temperature
        assert restored.log_Z == pytest.approx(original.log_Z)
        assert restored.free_energy == pytest.approx(original.free_energy)
        assert restored.mean_energy == pytest.approx(original.mean_energy)
        assert restored.mean_energy_sq == pytest.approx(original.mean_energy_sq)
        assert restored.heat_capacity == pytest.approx(original.heat_capacity)
        assert restored.entropy == pytest.approx(original.entropy)
        assert restored.std_energy == pytest.approx(original.std_energy)

    def test_infers_log_Z_when_missing(self):
        """log_Z should be recomputed from free_energy and temperature."""
        original = _make()
        d = original.to_dict()
        d["mean_energy_sq"] = original.mean_energy_sq
        # Don't include log_Z — it should be inferred
        restored = Thermodynamics.from_dict(d)
        assert restored.log_Z == pytest.approx(original.log_Z, rel=1e-6)

    def test_infers_mean_energy_sq_when_missing(self):
        """mean_energy_sq should be estimated from std_energy and mean_energy."""
        original = _make()
        d = original.to_dict()
        d["log_Z"] = original.log_Z
        # Don't include mean_energy_sq — it should be inferred
        restored = Thermodynamics.from_dict(d)
        expected = original.std_energy ** 2 + original.mean_energy ** 2
        assert restored.mean_energy_sq == pytest.approx(expected)

    def test_missing_required_key_raises(self):
        with pytest.raises(KeyError):
            Thermodynamics.from_dict({"temperature_K": 300.0})
