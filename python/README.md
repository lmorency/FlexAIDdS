# flexaidds Python Package

Python bindings and analysis helpers for the FlexAID∆S thermodynamic core.

The current Python package now covers two complementary layers:

- `Thermodynamics`: free energy, mean energy, entropy, heat capacity, and energy dispersion from the canonical ensemble engine
- `StatMechEngine`: canonical ensemble builder and analyzer exposed through the compiled extension
- `load_results(...)`: a read-only loader for existing docking result directories, grouping PDB outputs into binding-mode ensembles from `REMARK` metadata and filename heuristics
- `PoseResult`, `BindingModeResult`, `DockingResult`: immutable Python-side data models for downstream analysis

Higher-level live docking orchestration is still intentionally staged behind the ongoing C++ integration work, but read-only ensemble inspection is now available immediately.

## Installation

From the repository root:

```bash
cd python
pip install -e .
```

## Quick example: thermodynamics

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

## Quick example: load docking results

```python
import flexaidds as fd

run = fd.load_results("path/to/output_dir")
print(run.n_modes)
print(run.binding_modes[0].best_cf)
print(run.binding_modes[0].free_energy)
print(run.to_records())
```

The loader scans PDB-like files recursively, parses `REMARK` lines such as `binding_mode`, `pose_rank`, `CF`, `free_energy`, `enthalpy`, `entropy`, and falls back to filename heuristics like `binding_mode_1_pose_2.pdb` when metadata are sparse.

## Command-line inspection

You can inspect a result directory directly without writing Python code:

```bash
python -m flexaidds path/to/output_dir
python -m flexaidds path/to/output_dir --json
```

This is useful for smoke-checking ensemble structure before pushing results into notebooks, pandas, or PyMOL-side tooling.

## Scope

This package is designed to stay additive and low-conflict while the deeper C++ refactor lands:

- no edits to `LIB/` sources for the read-only results layer
- no edits to root `CMakeLists.txt`
- no interference with ongoing `BindingMode` / `StatMechEngine` engine-side integration work

That makes it safe to evolve the Python analysis surface in parallel with the core thermodynamic plumbing.
