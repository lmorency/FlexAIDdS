"""Tests for python/flexaidds/encom.py.

Pure-Python path tests run without the C++ extension.
Tests that specifically validate C++ agreement are marked @requires_core.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from conftest import requires_core


# ─────────────────────────────────────────────────────────────────────────────
# NormalMode dataclass
# ─────────────────────────────────────────────────────────────────────────────

def test_normalmode_repr():
    from flexaidds.encom import NormalMode

    nm = NormalMode(index=7, eigenvalue=3.14)
    assert "NormalMode" in repr(nm)
    assert "7" in repr(nm)


def test_normalmode_defaults():
    from flexaidds.encom import NormalMode

    nm = NormalMode()
    assert nm.index == 0
    assert nm.eigenvalue == 0.0
    assert nm.eigenvector == []


# ─────────────────────────────────────────────────────────────────────────────
# VibrationalEntropy dataclass
# ─────────────────────────────────────────────────────────────────────────────

def test_vibrational_entropy_free_energy_correction():
    from flexaidds.encom import VibrationalEntropy

    vs = VibrationalEntropy(S_vib_kcal_mol_K=0.002, temperature=300.0)
    expected = -300.0 * 0.002  # = -0.6 kcal/mol
    assert math.isclose(vs.free_energy_correction, expected, abs_tol=1e-12)


def test_vibrational_entropy_repr():
    from flexaidds.encom import VibrationalEntropy

    vs = VibrationalEntropy(S_vib_kcal_mol_K=0.001, n_modes=12, temperature=298.0)
    r = repr(vs)
    assert "VibrationalEntropy" in r
    assert "12" in r


# ─────────────────────────────────────────────────────────────────────────────
# ENCoMEngine – pure-Python path (no _core)
# ─────────────────────────────────────────────────────────────────────────────

def test_encom_total_entropy_additive():
    from flexaidds.encom import ENCoMEngine

    S_conf = 0.0030
    S_vib  = 0.0015
    total  = ENCoMEngine.total_entropy(S_conf, S_vib)
    assert math.isclose(total, S_conf + S_vib, abs_tol=1e-15)


def test_encom_free_energy_with_vibrations():
    from flexaidds.encom import ENCoMEngine

    F_elec = -10.0
    S_vib  = 0.002
    T      = 300.0
    F_total = ENCoMEngine.free_energy_with_vibrations(F_elec, S_vib, T)
    expected = F_elec - T * S_vib  # = -10.6
    assert math.isclose(F_total, expected, abs_tol=1e-10)


def test_encom_compute_vibrational_entropy_all_trivial():
    """All-zero eigenvalues (translations/rotations only) → S_vib = 0."""
    from flexaidds.encom import ENCoMEngine, NormalMode

    modes = [NormalMode(index=i, eigenvalue=0.0, frequency=0.0) for i in range(6)]
    vs = ENCoMEngine.compute_vibrational_entropy(modes, temperature_K=300.0)
    assert vs.n_modes == 0
    assert vs.S_vib_kcal_mol_K == 0.0


def test_encom_compute_vibrational_entropy_single_mode():
    """One non-trivial mode → n_modes == 1, S_vib > 0 at 300 K."""
    from flexaidds.encom import ENCoMEngine, NormalMode

    modes = [
        NormalMode(index=1, eigenvalue=0.0),
        NormalMode(index=2, eigenvalue=0.0),
        NormalMode(index=3, eigenvalue=4.5),
    ]
    vs = ENCoMEngine.compute_vibrational_entropy(modes, temperature_K=300.0)
    assert vs.n_modes == 1
    assert vs.S_vib_kcal_mol_K > 0.0


def test_encom_compute_vibrational_entropy_higher_T_more_entropy():
    """S_vib increases with temperature (more accessible states)."""
    from flexaidds.encom import ENCoMEngine, NormalMode

    modes = [NormalMode(index=i + 1, eigenvalue=float(i + 1)) for i in range(6)]

    vs_low  = ENCoMEngine.compute_vibrational_entropy(modes, temperature_K=200.0)
    vs_high = ENCoMEngine.compute_vibrational_entropy(modes, temperature_K=400.0)
    assert vs_high.S_vib_kcal_mol_K > vs_low.S_vib_kcal_mol_K


def test_encom_load_modes_pure_python(encom_files):
    """load_modes parses eigenvalue/eigenvector files into NormalMode list."""
    from flexaidds.encom import ENCoMEngine

    ev_file, evc_file = encom_files
    modes = ENCoMEngine.load_modes(str(ev_file), str(evc_file))

    # 9 eigenvalues → 9 modes; first 6 are zero (trivial)
    assert len(modes) == 9
    assert all(m.eigenvalue == 0.0 for m in modes[:6])
    assert modes[6].eigenvalue == pytest.approx(1.23, abs=1e-9)


def test_encom_load_modes_frequencies_match_eigenvalues(encom_files):
    """Frequency = sqrt(eigenvalue) for each mode."""
    from flexaidds.encom import ENCoMEngine

    ev_file, evc_file = encom_files
    modes = ENCoMEngine.load_modes(str(ev_file), str(evc_file))

    for m in modes:
        expected_freq = math.sqrt(max(m.eigenvalue, 0.0))
        assert math.isclose(m.frequency, expected_freq, abs_tol=1e-12)
