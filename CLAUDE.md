# CLAUDE.md — FlexAIDdS Development Guide

## Project Overview

FlexAIDdS (FlexAID with ΔS Entropy) is an entropy-driven molecular docking engine combining genetic algorithms with statistical mechanics thermodynamics. It targets real-world psychopharmacology and drug discovery applications.

- **Languages**: C++20 (core engine), Python (bindings/analysis), Objective-C++ (Metal GPU), CUDA (optional GPU)
- **License**: Apache-2.0 (no GPL dependencies allowed — see `THIRD_PARTY_LICENSES.md`)
- **Lead**: Louis-Philippe Morency, PhD (Candidate), Université de Montréal, NRGlab
- **Version**: 1.0.0-alpha (Python package), v1.76 (core engine)

## Repository Structure

```
FlexAIDdS/
├── LIB/                    # Core C++ library (~138 source files)
│   ├── flexaid.h            # Main header: constants, data structures
│   ├── gaboom.cpp/h         # Genetic algorithm (GA) engine
│   ├── Vcontacts.cpp/h      # Voronoi contact function scoring
│   ├── VoronoiCFBatch.h     # Batched Voronoi CF evaluation
│   ├── statmech.cpp/h       # StatMechEngine: partition function, free energy, entropy
│   ├── BindingMode.cpp/h    # Pose clustering & thermodynamic integration
│   ├── encom.cpp/h          # Elastic network contact model (vibrational entropy)
│   ├── tencm.cpp/h          # Torsional ENCoM backbone flexibility
│   ├── tencom_main.cpp      # tENCoM standalone tool entry point
│   ├── tencom_diff.cpp/h    # tENCoM differential engine
│   ├── tencom_output.cpp/h  # tENCoM output formatting
│   ├── CleftDetector.cpp/h  # Binding-site cleft detection
│   ├── Mol2Reader.cpp/h     # MOL2 file format reader
│   ├── SdfReader.cpp/h      # SDF file format reader
│   ├── FOPTICS.cpp/h        # Fast-OPTICS clustering algorithm
│   ├── DensityPeak_Cluster.cpp  # Density-peak clustering
│   ├── metal_eval.mm/h      # Metal GPU evaluation bridge
│   ├── cuda_eval.cu/cuh     # CUDA GPU evaluation
│   ├── simd_distance.h      # SIMD-accelerated distance functions
│   ├── ShannonThermoStack/  # Shannon configurational entropy + HW acceleration
│   │   ├── ShannonThermoStack.cpp/h  # Core Shannon entropy engine
│   │   ├── ShannonMetalBridge.mm/h   # Metal GPU bridge
│   │   ├── shannon_cuda.cu/cuh       # CUDA implementation
│   │   └── shannon_metal.metal       # Metal shader kernel
│   ├── LigandRingFlex/      # Non-aromatic ring & sugar pucker sampling
│   │   ├── LigandRingFlex.cpp/h      # Ring flexibility engine
│   │   ├── RingConformerLibrary.cpp/h # Conformer library
│   │   └── SugarPucker.cpp/h         # Pyranose/furanose sampling
│   ├── ChiralCenter/        # R/S stereocenter discrimination
│   │   └── ChiralCenterGene.cpp/h
│   ├── NATURaL/             # Co-translational assembly module
│   │   ├── NATURaLDualAssembly.cpp/h # Dual-assembly engine
│   │   ├── RibosomeElongation.cpp/h  # Ribosome speed model
│   │   └── TransloconInsertion.cpp/h # Sec translocon TM insertion
│   └── CavityDetect/        # SURFNET cavity detection (Metal GPU support)
│       ├── CavityDetect.cpp/h        # CPU implementation
│       ├── CavityDetect.metal        # Metal shader
│       └── CavityDetectMetalBridge.mm/h  # Metal bridge
├── src/                    # Entry point
│   └── gaboom.cpp           # Main executable entry point
├── tests/                  # C++ unit tests (GoogleTest, 5 files)
├── python/                 # Python package & bindings
│   ├── flexaidds/           # Python package (API, models, CLI)
│   │   ├── __init__.py      # Package root with C++ fallback
│   │   ├── __version__.py   # Version info (1.0.0-alpha)
│   │   ├── __main__.py      # CLI entry point
│   │   ├── _core.cpp        # pybind11 core bindings source
│   │   ├── models.py        # Data classes (PoseResult, etc.)
│   │   ├── results.py       # Result loading and parsing
│   │   ├── io.py            # I/O and file handling
│   │   ├── docking.py       # High-level docking API
│   │   ├── encom.py         # ENCoM vibrational entropy wrapper
│   │   ├── thermodynamics.py # Pure-Python StatMechEngine fallback
│   │   └── visualization.py # Visualization utilities
│   ├── bindings/            # pybind11 C++ bridge
│   │   └── core_bindings.cpp
│   ├── tests/               # Pytest test suite (16 files)
│   ├── setup.py             # setuptools config
│   └── pyproject.toml       # Python project metadata
├── pymol_plugin/           # PyMOL visualization plugin
│   ├── __init__.py          # Plugin initialization
│   ├── gui.py               # PyMOL GUI integration
│   ├── results_adapter.py   # FlexAID result parsing
│   └── visualization.py    # 3D visualization in PyMOL
├── docs/                   # Documentation
│   ├── IMPLEMENTATION_ROADMAP.md
│   ├── PHASE1_SUMMARY_AND_DELIVERABLES.md
│   ├── architecture/        # Architecture diagrams
│   ├── implementation/      # Implementation details
│   └── licensing/           # License compliance docs
├── cmake/                  # CMake helpers
│   └── MetalAcceleration.cmake
├── .github/workflows/      # CI/CD (GitHub Actions)
│   └── ci.yml               # 3-job CI pipeline
├── CMakeLists.txt          # Primary build configuration
├── VERSION                 # Release notes (v1.76)
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
| `ENABLE_TENCOM_BENCHMARK` | OFF | TeNCoM benchmark binary |
| `ENABLE_TENCOM_TOOL` | OFF | tENCoM standalone tool |

### Build Targets

| Target | Description |
|--------|-------------|
| `FlexAID` | Main docking executable (~109 source files) |
| `_core` | Python pybind11 extension module (requires `BUILD_PYTHON_BINDINGS=ON`) |
| `tENCoM` | Standalone tENCoM vibrational entropy tool (requires `ENABLE_TENCOM_TOOL=ON`) |
| `benchmark_tencom` | TeNCoM benchmark (requires `ENABLE_TENCOM_BENCHMARK=ON`) |
| `benchmark_vcfbatch` | VoronoiCFBatch benchmark |
| `test_statmech` | StatMechEngine unit tests (requires `BUILD_TESTING=ON`) |
| `test_hardware_dispatch` | ShannonThermoStack dispatch tests (requires `BUILD_TESTING=ON`) |
| `test_tencom_diff` | tENCoM differential engine tests (requires `BUILD_TESTING=ON`) |

## Testing

### C++ Tests (GoogleTest)

```bash
# Build and run
cmake -DBUILD_TESTING=ON .. && cmake --build . -j $(nproc)
ctest --test-dir build --output-on-failure
```

Test files in `tests/`:
- `test_statmech.cpp` — StatMechEngine correctness
- `test_binding_mode_statmech.cpp` — BindingMode ↔ StatMechEngine integration
- `test_ga_validation.cpp` — Genetic algorithm validation
- `test_hardware_dispatch.cpp` — ShannonThermoStack hardware dispatch layer
- `test_tencom_diff.cpp` — tENCoM differential engine correctness

### Python Tests (pytest)

```bash
cd python
pip install -e .
pytest tests/
```

Test files in `python/tests/` (16 files):
- `test_results_io.py` — Result file parsing (pure Python, no C++ needed)
- `test_results_loader_models.py` — Data model tests (pure Python)
- `test_models.py` — PoseResult/BindingModeResult/DockingResult tests
- `test_io.py` — I/O module tests
- `test_statmech.py` — StatMechEngine accuracy (requires C++ bindings)
- `test_statmech_smoke.py` — Smoke test for CI
- `test_thermodynamics.py` — Pure-Python thermodynamics engine
- `test_thermodynamics_dataclass.py` — Thermodynamics data classes
- `test_py_statmech.py` — Python StatMechEngine tests
- `test_encom.py` — ENCoM wrapper tests
- `test_docking.py` — Docking API tests
- `test_cli.py` — CLI entry point tests
- `test_import_fallback.py` — Graceful fallback when C++ bindings absent
- `test_version.py` — Version string tests
- `test_results.py` — Result loading integration tests

**Marker**: `@requires_core` — marks tests that need the compiled C++ `_core` extension. These skip gracefully if bindings are not built.

### CI Structure (.github/workflows/ci.yml)

Three jobs run on push, pull_request, and workflow_dispatch:

1. **pure_python_results** — Pure Python result I/O tests (ubuntu-latest, Python 3.11)
2. **cxx_core_build** — C++ core matrix build + `ctest` (Linux GCC, Linux Clang, macOS Clang)
3. **python_bindings_smoke** — Builds `_core` pybind11 extension + runs `test_statmech_smoke.py`

Concurrency: cancels in-progress runs for the same branch.

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
- Data classes: `PoseResult`, `BindingModeResult`, `DockingResult`
- Pure-Python fallbacks available when C++ `_core` bindings are not compiled

### Licensing Rules

- **Allowed**: Apache-2.0, BSD, MIT, MPL-2.0, PSF
- **Forbidden**: GPL/AGPL dependencies — never introduce them
- Clean-room policy: No GPL code even as inspiration (see `docs/licensing/clean-room-policy.md`)

## Architecture

### Core Pipeline

1. **Genetic Algorithm** (`gaboom.cpp`) — explores conformational space with OpenMP parallelism
2. **Scoring** (`Vcontacts.cpp`, `VoronoiCFBatch.h`) — Voronoi-based contact function scoring
3. **Statistical Mechanics** (`statmech.cpp`) — partition function, free energy (F), entropy (S), heat capacity (C_v)
4. **Binding Mode Clustering** (`BindingMode.cpp`) — groups poses, integrates thermodynamics
5. **Clustering Algorithms** (`FOPTICS.cpp`, `DensityPeak_Cluster.cpp`) — advanced pose clustering
6. **Vibrational Entropy** (`encom.cpp`, `tencm.cpp`) — elastic network model + torsional ENCoM
7. **Shannon Entropy** (`ShannonThermoStack/`) — configurational entropy with GPU acceleration
8. **Cavity Detection** (`CavityDetect/`) — SURFNET-based binding site identification

### Hardware Acceleration

| Technology | Location | Purpose |
|-----------|----------|---------|
| **CUDA** | `LIB/cuda_eval.cu`, `LIB/ShannonThermoStack/shannon_cuda.cu` | GPU scoring & entropy |
| **Metal** | `LIB/metal_eval.mm`, `LIB/ShannonThermoStack/`, `LIB/CavityDetect/` | macOS GPU acceleration |
| **SIMD** | `-mavx2`/`-mavx512` flags, `LIB/simd_distance.h` | Vectorized distance calculations |
| **OpenMP** | Throughout core engine | Multi-threaded parallelism |

### Python Package Architecture

The `flexaidds` Python package uses a two-tier design:

- **Always available** (pure Python): `PoseResult`, `BindingModeResult`, `DockingResult`, `load_results()`, `kB_kcal`, `kB_SI`
- **C++ bindings** (when `_core` is compiled): `StatMechEngine`, `Thermodynamics`, `State`, `BoltzmannLUT`, `Replica`, `WHAMBin`, `TIPoint`, `ENCoMEngine`, `NormalMode`, `VibrationalEntropy`
- **Fallback**: Pure-Python `StatMechEngine` and `Thermodynamics` from `thermodynamics.py` when C++ is unavailable

Check `HAS_CORE_BINDINGS` flag to determine which tier is active.

### Usage Modes

- **Legacy**: `./FlexAID config.inp ga.inp`
- **CLI**: `./flexaids dock receptor.pdb ligand.mol2`
- **Python**: `import flexaidds`
- **PyMOL Plugin**: Interactive visualization via `pymol_plugin/`

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
| `LIB/encom.cpp` | ENCoM vibrational entropy |
| `LIB/tencm.cpp` | Torsional ENCoM |
| `LIB/FOPTICS.cpp` | Fast-OPTICS clustering |
| `LIB/ShannonThermoStack/` | Shannon entropy + GPU acceleration |
| `LIB/CavityDetect/` | Cavity detection + Metal GPU |
| `python/flexaidds/__init__.py` | Python API surface |
| `python/flexaidds/_core.cpp` | pybind11 bindings source |
| `python/bindings/core_bindings.cpp` | Extended pybind11 bridge |
| `python/flexaidds/thermodynamics.py` | Pure-Python StatMechEngine fallback |
| `pymol_plugin/` | PyMOL visualization plugin |
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
4. If it needs C++ bindings, add pybind11 wrappers in `python/bindings/core_bindings.cpp`
5. Run `pytest python/tests/` to verify

### Adding a new C++ test

1. Create `tests/test_<name>.cpp` using GoogleTest
2. Register the test target in `CMakeLists.txt` (follow existing `add_executable` + `target_link_libraries` + `add_test` pattern)
3. Build with `BUILD_TESTING=ON` and run with `ctest --test-dir build --output-on-failure`

### Cross-platform considerations

- Test with both GCC and Clang on Linux, Clang on macOS (matches CI matrix)
- MSVC support exists but is not in CI matrix
- Metal code only compiles on macOS (`FLEXAIDS_USE_METAL=ON`)
- CUDA code requires CUDA toolkit (`FLEXAIDS_USE_CUDA=ON`)
- OpenMP may need Homebrew `libomp` on macOS
