import flexaidds as fd


def test_statmech_smoke():
    engine = fd.StatMechEngine(300.0)
    engine.add_sample(-7.0)
    engine.add_sample(-6.0)
    thermo = engine.compute()

    assert hasattr(thermo, "free_energy")
    assert hasattr(thermo, "mean_energy")
    assert hasattr(thermo, "entropy")
    assert hasattr(thermo, "heat_capacity")
    assert hasattr(thermo, "std_energy")
