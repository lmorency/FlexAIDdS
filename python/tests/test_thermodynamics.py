"""Tests for flexaidds.thermodynamics — Python-only paths exercised without C++."""

import math
import pytest
import numpy as np

# ── helpers ───────────────────────────────────────────────────────────────────

kB = 0.001987206  # kcal mol⁻¹ K⁻¹


def _boltzmann_weights(energies, temperature=300.0):
    """Reference implementation of Boltzmann weighting."""
    beta = 1.0 / (kB * temperature)
    shifted = [e - min(energies) for e in energies]
    weights = [math.exp(-beta * e) for e in shifted]
    z = sum(weights)
    return [w / z for w in weights]


def _helmholtz(energies, temperature=300.0):
    """Reference Helmholtz free energy via log-sum-exp."""
    beta = 1.0 / (kB * temperature)
    e_min = min(energies)
    log_z = math.log(sum(math.exp(-beta * (e - e_min)) for e in energies)) - beta * e_min
    return -kB * temperature * log_z


# ── Thermodynamics dataclass ──────────────────────────────────────────────────

class TestThermodynamicsDataclass:
    def _make(self, **kw):
        from flexaidds.thermodynamics import Thermodynamics
        defaults = dict(
            temperature=300.0, log_Z=0.0, free_energy=-10.0,
            mean_energy=-9.0, mean_energy_sq=82.0,
            heat_capacity=0.05, entropy=0.003, std_energy=0.5,
        )
        defaults.update(kw)
        return Thermodynamics(**defaults)

    def test_binding_free_energy_alias(self):
        t = self._make(free_energy=-12.3)
        assert t.binding_free_energy == t.free_energy == -12.3

    def test_entropy_term(self):
        t = self._make(temperature=300.0, entropy=0.01)
        assert abs(t.entropy_term - 3.0) < 1e-9

    def test_to_dict_keys(self):
        t = self._make()
        d = t.to_dict()
        for key in ("temperature_K", "free_energy_kcal_mol",
                    "entropy_kcal_mol_K", "heat_capacity_kcal_mol_K2"):
            assert key in d

    def test_to_dict_values(self):
        t = self._make(temperature=310.0, free_energy=-11.5)
        d = t.to_dict()
        assert d["temperature_K"] == 310.0
        assert d["free_energy_kcal_mol"] == -11.5


# ── Python-level StatMechEngine (skipped when C++ missing) ───────────────────

_CORE_AVAILABLE = False
try:
    import flexaidds._core  # noqa: F401
    _CORE_AVAILABLE = True
except ImportError:
    pass

needs_core = pytest.mark.skipif(not _CORE_AVAILABLE,
                                reason="C++ _core module not built")


@needs_core
class TestStatMechEngine:
    def test_empty_engine(self):
        from flexaidds.thermodynamics import StatMechEngine
        eng = StatMechEngine(300.0)
        assert eng.n_samples == 0
        assert abs(eng.temperature - 300.0) < 1e-9

    def test_add_samples_length(self):
        from flexaidds.thermodynamics import StatMechEngine
        eng = StatMechEngine(300.0)
        energies = [-10.5, -9.8, -10.2, -11.0]
        eng.add_samples(np.array(energies))
        assert eng.n_samples == len(energies)

    def test_free_energy_single_sample(self):
        """With one sample, F = E (partition function = 1)."""
        from flexaidds.thermodynamics import StatMechEngine
        eng = StatMechEngine(300.0)
        eng.add_sample(-10.0)
        thermo = eng.compute()
        assert abs(thermo.free_energy - (-10.0)) < 1e-6

    def test_boltzmann_weights_sum_to_one(self):
        from flexaidds.thermodynamics import StatMechEngine
        eng = StatMechEngine(300.0)
        eng.add_samples(np.array([-10.0, -9.0, -11.0]))
        weights = eng.boltzmann_weights()
        assert abs(sum(weights) - 1.0) < 1e-9

    def test_clear_resets_count(self):
        from flexaidds.thermodynamics import StatMechEngine
        eng = StatMechEngine(300.0)
        eng.add_sample(-8.0)
        eng.clear()
        assert eng.n_samples == 0

    def test_delta_g_same_ensemble_is_zero(self):
        from flexaidds.thermodynamics import StatMechEngine
        eng1 = StatMechEngine(300.0)
        eng2 = StatMechEngine(300.0)
        for e in [-10.0, -9.5, -10.5]:
            eng1.add_sample(e)
            eng2.add_sample(e)
        assert abs(eng1.delta_G(eng2)) < 1e-9

    def test_thermodynamics_entropy_nonnegative(self):
        """Configurational entropy should be ≥ 0 for a spread ensemble."""
        from flexaidds.thermodynamics import StatMechEngine
        eng = StatMechEngine(300.0)
        eng.add_samples(np.linspace(-12.0, -8.0, 20))
        thermo = eng.compute()
        assert thermo.entropy >= 0.0


# ── BoltzmannLUT (pure math check) ───────────────────────────────────────────

@needs_core
class TestBoltzmannLUT:
    def test_lookup_matches_exact(self):
        from flexaidds.thermodynamics import BoltzmannLUT, kB_kcal
        beta = 1.0 / (kB_kcal * 300.0)
        lut = BoltzmannLUT(beta=beta, e_min=-20.0, e_max=5.0, n_bins=100000)
        energy = -12.5
        exact = math.exp(-beta * energy)
        approx = lut(energy)
        # LUT resolution: within 0.01% of exact value
        assert abs(approx - exact) / exact < 1e-4
