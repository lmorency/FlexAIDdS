import math

import pytest

import flexaidds as fd

_core_available: bool
try:
    fd.StatMechEngine(300.0)
    _core_available = True
except Exception:
    _core_available = False


@pytest.mark.skipif(not _core_available, reason="C++ _core extension not built")
def test_statmech_smoke():
    engine = fd.StatMechEngine(300.0)
    engine.add_sample(-7.0)
    engine.add_sample(-6.0)
    engine.add_sample(-5.5, multiplicity=2.0)

    thermo = engine.compute()

    assert math.isfinite(thermo.free_energy)
    assert math.isfinite(thermo.mean_energy)
    assert math.isfinite(thermo.entropy)
    assert math.isfinite(thermo.heat_capacity)
    assert math.isfinite(thermo.std_energy)
    assert thermo.free_energy <= thermo.mean_energy
