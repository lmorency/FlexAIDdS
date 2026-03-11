# CLAUDE.md — FlexAIDdS Development Guide

## Project Overview

FlexAIDdS (FlexAID with ΔS Entropy) is an entropy-driven molecular docking engine combining genetic algorithms with statistical mechanics thermodynamics. It targets real-world psychopharmacology and drug discovery applications.

- **Languages**: C++20 (core engine), Python (bindings/analysis), Objective-C++ (Metal GPU), CUDA (optional GPU)
- **License**: Apache-2.0 (no GPL dependencies allowed — see `THIRD_PARTY_LICENSES.md`)
- **Lead**: Louis-Philippe Morency, PhD (Candidate), Université de Montréal, NRGlab

## Repository Structure

```
FlexAIDdS/
├── LIB/                    # Core C++ library (~100+ files)
│   ├── flexaid.h            # Main header: constants, data structures
│   ├── gaboom.cpp/h         # Genetic algorithm (GA) engine
│   ├── Vcontacts.cpp/h      # Voronoi contact function scoring
│   ├── statmech.cpp/h       # StatMechEngine: partition function, free energy, entropy
│   ├── BindingMode.cpp/h    # Pose clustering & thermodynamic integration
│   ├── encom.cpp/h          # Elastic network contact model (vibrational entropy)
│   ├── tencm.cpp/h          # Torsional ENCoM backbone flexibility
│   ├── Mol2Reader.cpp/h     # MOL2 file reader
│   ├── SdfReader.cpp/h      # SDF file reader
│   ├── CleftDetector.cpp/h  # Binding-site cleft detection
│   ├── ShannonThermoStack/  # Shannon configurational entropy + HW acceleration
│   ├── LigandRingFlex/      # Non-aromatic ring & sugar pucker sampling
│   ├── ChiralCenter/        # R/S stereocenter discrimination
│   ├── NATURaL/             # Co-translational assembly module
│   └── CavityDetect/        # SURFNET cavity detection (Metal GPU support)
├── src/                    # Entry point (gaboom.cpp)
├── tests/                  # C++ unit tests (GoogleTest)
├── python/                 # Python package & bindings
│   ├── flexaidds/           # Python package (API, models, CLI)
│   ├── bindings/            # pybind11 C++ bridge (core_bindings.cpp)
│   ├── tests/               # Pytest test suite
│   ├── setup.py             # setuptools + pybind11 (builds statmech & encom only)
│   └── pyproject.toml       # Python project metadata (setuptools backend)
├── pymol_plugin/           # PyMOL integration (GUI, visualization, results adapter)
├── docs/                   # Documentation (architecture, implementation, licensing)
├── cmake/                  # CMake helpers
├── .github/workflows/      # CI/CD (GitHub Actions)
├── CMakeLists.txt          # Primary build configuration
├── WRK/                    # Working directory for builds
└── BIN/                    # Binary output directory
```

## Build System

### Requirements

- C++20 compiler (GCC >= 10, Clang >= 10, MSVC)
- CMake >= 3.18
- Optional: Boost, Eigen3, OpenMP, CUDA Toolkit, Metal framework, pybind11

### Build Commands

```bash
# Standard release build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target FlexAID -j $(nproc)

# With tests
cmake .. -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . -j $(nproc)
ctest --test-dir .

# With Python bindings
cmake .. -DBUILD_PYTHON_BINDINGS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . -j $(nproc)
```

### Key CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `FLEXAIDS_USE_CUDA` | OFF | CUDA GPU evaluation |
| `FLEXAIDS_USE_METAL` | OFF | Metal acceleration (macOS) |
| `FLEXAIDS_USE_AVX2` | ON | AVX2 SIMD acceleration |
| `FLEXAIDS_USE_AVX512` | OFF | AVX-512 SIMD |
| `FLEXAIDS_USE_OPENMP` | ON | OpenMP parallelism |
| `FLEXAIDS_USE_EIGEN` | ON | Eigen3 linear algebra |
| `BUILD_PYTHON_BINDINGS` | OFF | pybind11 extensions |
| `BUILD_TESTING` | OFF | GoogleTest unit tests |
| `ENABLE_TENCOM_BENCHMARK` | OFF | tENCoM benchmark binary |
| `ENABLE_TENCOM_TOOL` | OFF | tENCoM vibrational entropy tool |
| `ENABLE_VCFBATCH_BENCHMARK` | OFF | VoronoiCFBatch benchmark binary |

## Testing

### C++ Tests (GoogleTest)

```bash
# Build and run
cmake -DBUILD_TESTING=ON .. && cmake --build . -j $(nproc)
ctest --test-dir build
```

Key test files in `tests/`:
- `test_statmech.cpp` — StatMechEngine correctness
- `test_binding_mode_statmech.cpp` — BindingMode ↔ StatMechEngine integration
- `test_ga_validation.cpp` — Genetic algorithm validation
- `test_hardware_dispatch.cpp` — ShannonThermoStack hardware dispatch layer
- `test_tencom_diff.cpp` — tENCoM differential engine

### Python Tests (pytest)

```bash
cd python
pip install -e .
pytest tests/
```

Key test files in `python/tests/`:
- `test_results_io.py` — Result file parsing (pure Python, no C++ needed)
- `test_results_loader_models.py` — Data model tests (pure Python)
- `test_statmech.py` — StatMechEngine accuracy (requires C++ bindings)
- `test_statmech_smoke.py` — Smoke test for CI
- `test_cli.py` — CLI entry point tests
- `test_docking.py` — High-level docking interface
- `test_encom.py` — ENCoM normal-mode analysis
- `test_io.py` — PDB I/O and REMARK parsing
- `test_thermodynamics.py` / `test_thermodynamics_dataclass.py` — Thermodynamics module
- `test_models.py` — Data model validation
- `test_py_statmech.py` — Pure-Python StatMech fallback
- `test_import_fallback.py` — Graceful import without C++ bindings
- `test_results.py` — Result loading integration
- `test_version.py` — Version string checks

**Marker**: `@requires_core` — marks tests that need the compiled C++ `_core` extension. These skip gracefully if bindings are not built.

### CI Structure (.github/workflows/ci.yml)

1. **Pure Python tests** — always run, no C++ needed
2. **C++ core build** — multi-platform matrix (Linux GCC/Clang, macOS Clang)
3. **Python bindings smoke test** — builds `_core` extension + runs smoke test

## Code Conventions

### Commit Messages

- Use prefix: `Fix:`, `Add:`, `Update:`, `Refactor:`, etc.
- Concise title line describing the change
- Example: `Fix: Define missing _rel_lib variable in setup.py`

### C++ Style

- C++20 standard
- Compiler flags: `-Wall -O3 -ffast-math` + SIMD flags
- Core data structures: `chromosome` (GA gene encoding), `Pose`, `BindingMode`, `BindingPopulation`
- Use log-sum-exp for numerical stability in partition functions

### Python Style

- Python >= 3.9
- Package: `flexaidds`
- Key exports: `StatMechEngine`, `Thermodynamics`, `ENCoMEngine`, `load_results()`
- Conditional exports (require C++ `_core`): `State`, `BoltzmannLUT`, `Replica`, `WHAMBin`, `TIPoint`, `NormalMode`, `VibrationalEntropy`
- Data classes: `PoseResult`, `BindingModeResult`, `DockingResult`
- Constants: `kB_kcal`, `kB_SI`, `HAS_CORE_BINDINGS`
- Modules: `models.py`, `results.py`, `thermodynamics.py`, `docking.py`, `encom.py`, `io.py`, `visualization.py`

### Licensing Rules

- **Allowed**: Apache-2.0, BSD, MIT, MPL-2.0, PSF
- **Forbidden**: GPL/AGPL dependencies — never introduce them
- Clean-room policy: No GPL code even as inspiration (see `docs/licensing/clean-room-policy.md`)

## Architecture

### Core Pipeline

1. **Genetic Algorithm** (`gaboom.cpp`) — explores conformational space
2. **Scoring** (`Vcontacts.cpp`) — Voronoi-based contact function
3. **Statistical Mechanics** (`statmech.cpp`) — partition function, free energy (F), entropy (S), heat capacity (C_v)
4. **Binding Mode Clustering** (`BindingMode.cpp`) — groups poses, integrates thermodynamics
5. **Vibrational Entropy** (`encom.cpp`) — elastic network model for protein flexibility

### Usage Modes

- **Legacy**: `./FlexAID config.inp ga.inp`
- **CLI**: `./flexaids dock receptor.pdb ligand.mol2`
- **Python**: `import flexaidds` (Phase 2, partially complete)

### Development Phases

- **Phase 1** (Complete): StatMechEngine integration
- **Phase 2** (In Progress): Python bindings & result I/O
- **Phase 3** (Planned): ENCoM vibrational entropy integration

## Key Files to Know

| File | Purpose |
|------|---------|
| `LIB/flexaid.h` | Central header — constants, structs, macros |
| `LIB/gaboom.cpp` | Genetic algorithm core |
| `LIB/statmech.cpp` | Statistical mechanics engine |
| `LIB/Vcontacts.cpp` | Voronoi contact scoring |
| `LIB/BindingMode.cpp` | Pose clustering + thermodynamics |
| `python/flexaidds/__init__.py` | Python API surface |
| `LIB/Mol2Reader.cpp` | MOL2 molecular file reader |
| `LIB/SdfReader.cpp` | SDF molecular file reader |
| `LIB/CleftDetector.cpp` | Binding-site cleft detection |
| `python/flexaidds/docking.py` | High-level docking interface |
| `python/flexaidds/encom.py` | ENCoM normal-mode analysis |
| `python/flexaidds/io.py` | PDB I/O and REMARK parsing |
| `python/flexaidds/visualization.py` | PyMOL integration helpers |
| `python/bindings/core_bindings.cpp` | pybind11 bridge code |
| `pymol_plugin/` | PyMOL plugin (GUI, visualization, results adapter) |
| `CMakeLists.txt` | Build configuration (all targets, options) |
| `.github/workflows/ci.yml` | CI pipeline definition |

## Common Tasks

### Adding a new C++ source file

1. Add the `.cpp`/`.h` files under `LIB/`
2. Add the source to the `FLEXAID_SOURCES` list in `CMakeLists.txt`
3. Write tests in `tests/` using GoogleTest
4. Enable `BUILD_TESTING=ON` and run `ctest`

### Adding a new Python module

1. Add module under `python/flexaidds/`
2. Export in `__init__.py` if it's part of the public API
3. Write tests in `python/tests/`
4. If it needs C++ bindings, add pybind11 wrappers in `python/bindings/`

### Cross-platform considerations

- Test with both GCC and Clang on Linux, Clang on macOS
- MSVC support exists but is not in CI matrix
- Metal code only compiles on macOS (`FLEXAIDS_USE_METAL=ON`)
- CUDA code requires CUDA toolkit (`FLEXAIDS_USE_CUDA=ON`)
