# flexaidds Python Package

Python bindings for the FlexAID∆S thermodynamic core.

This first Python phase exposes the canonical statistical mechanics layer cleanly and with minimal build risk:

- `Thermodynamics`: free energy, mean energy, entropy, heat capacity, and energy dispersion
- `StatMechEngine`: canonical ensemble builder and analyzer

Higher-level bindings for `BindingMode`, `BindingPopulation`, and full docking orchestration are intentionally deferred until the Phase 1 C++ thermodynamic bridge is merged cleanly.

## Installation

From the repository root:

```bash
cd python
pip install -e .
```

## Quick example

```python
import flexaidds as fd

engine = fd.StatMechEngine(temperature=300.0)
engine.add_sample(-7.5)
engine.add_sample(-6.0)
engine.add_sample(-5.5, multiplicity=2.0)
thermo = engine.compute()

print("F =", thermo.free_energy)
print("E =", thermo.mean_energy)
print("S =", thermo.entropy)
print("Cv =", thermo.heat_capacity)
print("sigma_E =", thermo.std_energy)
```

## Scope

This package is designed to be additive and low-conflict:

- no edits to `LIB/` sources
- no edits to root `CMakeLists.txt`
- no interference with ongoing `BindingMode` / `StatMechEngine` integration work

That makes it safe to develop in parallel while the deeper C++ refactor lands.
