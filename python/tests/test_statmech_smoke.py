import math

import pytest

import flexaidds as fd

pytestmark = pytest.mark.skipif(
    not fd.HAS_CORE_BINDINGS or fd.StatMechEngine is None,
    reason="pybind11 core extension not built",
)


def test_statmech_engine_smoke() -> None:
    engine = fd.StatMechEngine(temperature=300.0)
    engine.add_sample(-7.5)
    engine.add_sample(-6.0)
    engine.add_sample(-5.5, multiplicity=2.0)

    thermo = engine.compute()

    assert math.isfinite(thermo.free_energy)
    assert math.isfinite(thermo.mean_energy)
    assert math.isfinite(thermo.entropy)
    assert math.isfinite(thermo.heat_capacity)
    assert math.isfinite(thermo.std_energy)
    assert thermo.free_energy <= thermo.mean_energy
