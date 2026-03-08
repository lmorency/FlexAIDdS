"""Tests for ENCoM vibrational entropy — Python-level checks + C++ integration."""

import math
import pytest

# ── C++ availability guard ────────────────────────────────────────────────────

_CORE_AVAILABLE = False
try:
    import flexaidds._core  # noqa: F401
    _CORE_AVAILABLE = True
except ImportError:
    pass

needs_core = pytest.mark.skipif(not _CORE_AVAILABLE,
                                reason="C++ _core module not built")


# ── NormalMode ────────────────────────────────────────────────────────────────

@needs_core
class TestNormalMode:
    def _make(self, eigenvalue=1.0, frequency=100.0):
        from flexaidds import NormalMode
        return NormalMode(eigenvalue=eigenvalue, frequency=frequency)

    def test_eigenvalue_stored(self):
        nm = self._make(eigenvalue=2.5)
        assert abs(nm.eigenvalue - 2.5) < 1e-12

    def test_frequency_stored(self):
        nm = self._make(frequency=150.0)
        assert abs(nm.frequency - 150.0) < 1e-9

    def test_zero_eigenvalue_allowed(self):
        nm = self._make(eigenvalue=0.0, frequency=0.0)
        assert nm.eigenvalue == 0.0


# ── VibrationalEntropy ────────────────────────────────────────────────────────

@needs_core
class TestVibrationalEntropy:
    def _make(self, s_vib=0.01, dG_vib=-3.0, temperature=300.0):
        from flexaidds import VibrationalEntropy
        return VibrationalEntropy(
            S_vib_kcal_mol_K=s_vib,
            dG_vib_kcal_mol=dG_vib,
            temperature=temperature,
        )

    def test_s_vib_stored(self):
        ve = self._make(s_vib=0.02)
        assert abs(ve.S_vib_kcal_mol_K - 0.02) < 1e-12

    def test_dG_vib_stored(self):
        ve = self._make(dG_vib=-5.0)
        assert abs(ve.dG_vib_kcal_mol - (-5.0)) < 1e-12

    def test_temperature_stored(self):
        ve = self._make(temperature=310.0)
        assert abs(ve.temperature - 310.0) < 1e-9

    def test_ts_vib_equals_t_times_s(self):
        """−TΔS_vib should equal temperature × S_vib."""
        ve = self._make(s_vib=0.01, temperature=300.0)
        assert abs(ve.TS_vib_kcal_mol - 300.0 * 0.01) < 1e-9


# ── ENCoMEngine ───────────────────────────────────────────────────────────────

@needs_core
class TestENCoMEngine:
    def test_default_cutoff(self):
        from flexaidds import ENCoMEngine
        eng = ENCoMEngine()
        # Default eigenvalue_cutoff should be small positive
        assert eng.eigenvalue_cutoff > 0.0

    def test_custom_cutoff(self):
        from flexaidds import ENCoMEngine
        eng = ENCoMEngine(eigenvalue_cutoff=1e-5)
        assert abs(eng.eigenvalue_cutoff - 1e-5) < 1e-12

    def test_compute_with_empty_modes_zero(self):
        """No modes → zero vibrational entropy."""
        from flexaidds import ENCoMEngine
        eng = ENCoMEngine()
        result = eng.compute_vibrational_entropy([], temperature=300.0)
        assert result.S_vib_kcal_mol_K == 0.0

    def test_compute_with_positive_eigenvalues(self):
        """Positive eigenvalues → positive vibrational entropy."""
        from flexaidds import ENCoMEngine, NormalMode
        eng = ENCoMEngine()
        modes = [
            NormalMode(eigenvalue=1.0, frequency=100.0),
            NormalMode(eigenvalue=2.0, frequency=141.4),
            NormalMode(eigenvalue=0.5, frequency=70.7),
        ]
        result = eng.compute_vibrational_entropy(modes, temperature=300.0)
        assert result.S_vib_kcal_mol_K >= 0.0

    def test_more_modes_higher_entropy(self):
        """Adding more vibrational modes should not decrease entropy."""
        from flexaidds import ENCoMEngine, NormalMode
        eng = ENCoMEngine()

        def make_modes(n):
            return [NormalMode(eigenvalue=float(i + 1), frequency=float((i + 1) * 50))
                    for i in range(n)]

        s3 = eng.compute_vibrational_entropy(make_modes(3), 300.0).S_vib_kcal_mol_K
        s6 = eng.compute_vibrational_entropy(make_modes(6), 300.0).S_vib_kcal_mol_K
        assert s6 >= s3

    def test_higher_temperature_higher_entropy(self):
        """Vibrational entropy increases with temperature."""
        from flexaidds import ENCoMEngine, NormalMode
        eng = ENCoMEngine()
        modes = [NormalMode(eigenvalue=1.0, frequency=100.0)]
        s_low = eng.compute_vibrational_entropy(modes, 200.0).S_vib_kcal_mol_K
        s_high = eng.compute_vibrational_entropy(modes, 400.0).S_vib_kcal_mol_K
        assert s_high >= s_low


# ── Python-level ENCoM smoke test (no C++ needed) ────────────────────────────

class TestENCoMPythonFallback:
    """Verify that the ENCoM symbols are None (not missing) when C++ absent."""

    def test_encom_symbols_accessible(self):
        import flexaidds as fds
        # Should be importable either as a class or None
        assert hasattr(fds, "ENCoMEngine")
        assert hasattr(fds, "NormalMode")
        assert hasattr(fds, "VibrationalEntropy")

    def test_encom_symbols_none_when_no_core(self):
        if _CORE_AVAILABLE:
            pytest.skip("C++ core is built; checking None path not applicable")
        import flexaidds as fds
        assert fds.ENCoMEngine is None
        assert fds.NormalMode is None
        assert fds.VibrationalEntropy is None
