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
