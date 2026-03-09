"""Tests for python/flexaidds/thermodynamics.py.

These tests cover the pure-Python numerical logic and the module's public API.
Tests that require the compiled C++ extension are marked with @requires_core.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import requires_core


# ─────────────────────────────────────────────────────────────────────────────
# Module-level imports (no C++ needed)
# ─────────────────────────────────────────────────────────────────────────────

def test_import_thermodynamics():
    """The thermodynamics module must import without the C++ extension."""
    import importlib
    mod = importlib.import_module("flexaidds.thermodynamics")
    assert hasattr(mod, "Thermodynamics")
    assert hasattr(mod, "StatMechEngine")
    assert hasattr(mod, "BoltzmannLUT")
    assert hasattr(mod, "helmholtz_from_energies")
    assert hasattr(mod, "kB_kcal")


def test_kB_kcal_value():
    """kB_kcal must be within 0.1% of the NIST value."""
    from flexaidds.thermodynamics import kB_kcal
    assert abs(kB_kcal - 0.001987206) < 1e-8


def test_thermodynamics_dataclass():
    """Thermodynamics dataclass properties and serialisation."""
    from flexaidds.thermodynamics import Thermodynamics

    thermo = Thermodynamics(
        temperature=300.0,
        log_Z=10.0,
        free_energy=-5.0,
        mean_energy=-4.5,
        mean_energy_sq=20.5,
        heat_capacity=0.1,
        entropy=0.001667,
        std_energy=0.3,
    )

    assert thermo.binding_free_energy == thermo.free_energy
    assert math.isclose(thermo.entropy_term, 300.0 * 0.001667, rel_tol=1e-9)

    d = thermo.to_dict()
    assert d["temperature_K"] == 300.0
    assert d["free_energy_kcal_mol"] == -5.0
    assert d["entropy_kcal_mol_K"] == 0.001667


# ─────────────────────────────────────────────────────────────────────────────
# Tests that require the compiled C++ extension
# ─────────────────────────────────────────────────────────────────────────────

@requires_core
def test_statmech_engine_basic(sample_energies):
    """StatMechEngine returns sensible thermodynamics for a small ensemble."""
    from flexaidds.thermodynamics import StatMechEngine

    engine = StatMechEngine(temperature_K=300.0)
    engine.add_samples(sample_energies)

    thermo = engine.compute()

    # Free energy must be ≤ min(energies) at any temperature
    assert thermo.free_energy <= min(sample_energies) + 1e-9

    # Entropy must be non-negative
    assert thermo.entropy >= 0.0

    # Heat capacity must be non-negative
    assert thermo.heat_capacity >= 0.0

    # Mean energy must be between min and max
    assert min(sample_energies) - 1e-9 <= thermo.mean_energy <= max(sample_energies) + 1e-9


@requires_core
def test_statmech_boltzmann_weights_sum_to_one(sample_energies):
    """Boltzmann weights must sum to 1.0."""
    from flexaidds.thermodynamics import StatMechEngine

    engine = StatMechEngine(temperature_K=300.0)
    engine.add_samples(sample_energies)

    weights = engine.boltzmann_weights()
    assert abs(sum(weights) - 1.0) < 1e-10


@requires_core
def test_statmech_single_sample():
    """With one sample, F = E and S = 0."""
    from flexaidds.thermodynamics import StatMechEngine

    engine = StatMechEngine(temperature_K=300.0)
    engine.add_sample(-10.0)

    thermo = engine.compute()
    assert math.isclose(thermo.free_energy, -10.0, abs_tol=1e-9)
    assert abs(thermo.entropy) < 1e-10


@requires_core
def test_statmech_clear():
    """clear() resets the ensemble."""
    from flexaidds.thermodynamics import StatMechEngine

    engine = StatMechEngine(300.0)
    engine.add_sample(-10.0)
    assert len(engine) == 1
    engine.clear()
    assert len(engine) == 0


@requires_core
def test_statmech_delta_G():
    """delta_G computes a reasonable relative free energy."""
    from flexaidds.thermodynamics import StatMechEngine

    eng1 = StatMechEngine(300.0)
    for e in [-12.0, -11.5]:
        eng1.add_sample(e)

    eng2 = StatMechEngine(300.0)
    for e in [-10.0, -9.5]:
        eng2.add_sample(e)

    dG = eng1.delta_G(eng2)
    # eng1 has lower energies → more negative ΔG
    assert dG < 0.0


@requires_core
def test_boltzmann_lut():
    """BoltzmannLUT result approximates direct exp(-βE) within 1%."""
    from flexaidds.thermodynamics import BoltzmannLUT, kB_kcal

    T = 300.0
    beta = 1.0 / (kB_kcal * T)
    lut = BoltzmannLUT(beta=beta, e_min=-20.0, e_max=5.0, n_bins=10000)

    for E in [-15.0, -10.0, -5.0, 0.0]:
        expected = math.exp(-beta * E)
        got = lut(E)
        assert abs(got - expected) / (expected + 1e-30) < 0.01


@requires_core
def test_helmholtz_from_energies():
    """helmholtz_from_energies matches StatMechEngine.compute()."""
    from flexaidds.thermodynamics import StatMechEngine, helmholtz_from_energies

    energies = [-12.5, -11.8, -12.1, -10.9]
    T = 300.0

    engine = StatMechEngine(T)
    engine.add_samples(energies)
    expected_F = engine.compute().free_energy

    got_F = helmholtz_from_energies(energies, T)
    assert math.isclose(expected_F, got_F, rel_tol=1e-9)
