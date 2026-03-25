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
│   ├── Mol2Reader.cpp/h     # MOL2 file reader
│   ├── SdfReader.cpp/h      # SDF file reader
│   ├── CleftDetector.cpp/h  # Binding-site cleft detection
│   ├── VoronoiCFBatch.h     # Batch Voronoi CF evaluation (header-only, std::span)
│   ├── fileio.cpp/h         # File I/O utilities
│   ├── tENCoM/              # Torsional ENCoM backbone flexibility module
│   ├── ShannonThermoStack/  # Shannon configurational entropy + HW acceleration
│   ├── LigandRingFlex/      # Non-aromatic ring & sugar pucker sampling
│   ├── ChiralCenter/        # R/S stereocenter discrimination
│   ├── NATURaL/             # Co-translational assembly module
│   └── CavityDetect/        # SURFNET cavity detection (Metal GPU support)
├── src/                    # Entry point (gaboom.cpp stub/template)
├── tests/                  # C++ unit tests (GoogleTest)
├── python/                 # Python package & bindings
│   ├── flexaidds/           # Python package (API, models, CLI)
│   ├── bindings/            # pybind11 C++ bridge (core_bindings.cpp)
│   ├── tests/               # Pytest test suite (15 test files)
│   ├── conftest.py          # Pytest fixtures & markers
│   ├── setup.py             # setuptools + pybind11 (builds statmech & encom only)
│   └── pyproject.toml       # Python project metadata (setuptools backend)
├── pymol_plugin/           # PyMOL integration (GUI, visualization, results adapter)
│   ├── __init__.py          # Plugin registration & command mapping
│   ├── gui.py               # FlexAIDSPanel widget interface
│   ├── visualization.py     # Pose rendering & Boltzmann coloring
│   └── results_adapter.py   # Bridge to flexaidds.load_results()
├── docs/                   # Documentation
│   ├── IMPLEMENTATION_ROADMAP.md  # Development roadmap
│   ├── PHASE1_SUMMARY_AND_DELIVERABLES.md  # Phase 1 deliverables
│   ├── architecture/        # Architecture diagrams
│   ├── implementation/      # Phase summaries & corrected docs
│   └── licensing/           # License matrix, clean-room policy, GPL isolation
├── cmake/                  # CMake helpers
│   └── MetalAcceleration.cmake  # Metal GPU build helper
├── .github/workflows/      # CI/CD (GitHub Actions)
│   └── ci.yml               # Three-job CI pipeline
├── CMakeLists.txt          # Primary build configuration
├── THIRD_PARTY_LICENSES.md # Dependency licensing matrix
├── CONTRIBUTING.md         # Contribution guidelines
├── CODE_OF_CONDUCT.md      # Community guidelines
├── WRK/                    # Working directory for builds
└── BIN/                    # Binary output directory
```

### LIB/ Subdirectory Modules

**tENCoM/** — Torsional ENCoM backbone flexibility:
- `tencm.cpp/h` — TorsionalENM class (Hessian assembly, Jacobi diagonalization, sampling)
- `tencom_diff.cpp/h` — Differential engine (eigenvalue/eigenvector overlaps, ΔS_vib)
- `tencom_output.cpp/h` — Output formatter (PDB/JSON/CSV with REMARK metadata)
- `tencom_main.cpp` — Standalone CLI entry point
- `tencom_entropy_diff.cpp` — Entropy differential standalone tool
- `pdb_calpha.cpp/h` — Lightweight Cα/C4' PDB backbone reader
- `benchmark_tencom.cpp` — Performance benchmark
- `tencm_cuda.cu/cuh` — CUDA GPU contact discovery + Hessian assembly
- `tencm_metal.h/mm` — Metal GPU bridge

**ShannonThermoStack/** — Shannon configurational entropy with hardware acceleration:
- `ShannonThermoStack.cpp/h` — Hardware dispatch layer (CPU/CUDA/Metal)
- `shannon_cuda.cu/cuh` — CUDA kernel for histogram binning
- `shannon_metal.metal` — Metal shader for GPU histogram
- `ShannonMetalBridge.h/mm` — Objective-C++ Metal bridge

**LigandRingFlex/** — Non-aromatic ring and sugar pucker sampling:
- `LigandRingFlex.cpp/h` — Main ring flexibility engine
- `RingConformerLibrary.cpp/h` — Ring conformer database
- `SugarPucker.cpp/h` — Sugar pucker pseudorotation sampling

**ChiralCenter/** — Stereocenter discrimination:
- `ChiralCenterGene.cpp/h` — R/S chiral center GA gene encoding

**NATURaL/** — Co-translational assembly:
- `NATURaLDualAssembly.cpp/h` — Dual ribosomal assembly simulation
- `RibosomeElongation.cpp/h` — Ribosomal elongation dynamics
- `TransloconInsertion.cpp/h` — Translocon insertion model

**CavityDetect/** — SURFNET-based cavity detection:
- `CavityDetect.cpp/h` — Native cavity detection engine
- `CavityDetect.metal` — Metal GPU shader for probe-sphere generation
- `CavityDetectMetalBridge.h/mm` — Objective-C++ Metal bridge

### LIB/ Additional Source Files (by category)

**Coordinate & structure management:**
`ic2cf.cpp`, `ic_bounds.cpp`, `read_coor.cpp`, `calc_rmsd.cpp`, `calc_rmsd_chrom.cpp`, `calc_rmsp.cpp`, `geometry.cpp`, `modify_pdb.cpp`, `write_pdb.cpp`, `read_pdb.cpp`, `read_conect.cpp`, `residue_conect.cpp`, `shortest_path.cpp`

**Ligand & rotamer handling:**
`read_lig.cpp`, `build_rotamers.cpp`, `read_rotlib.cpp`, `read_rotobs.cpp`, `buildcc.cpp`, `buildcc_point.cpp`, `buildic.cpp`, `buildic_point.cpp`, `build_close.cpp`, `bondedlist.cpp`, `update_bonded.cpp`

**Scoring & evaluation:**
`vcfunction.cpp`, `cffunction.cpp`, `spfunction.cpp`, `set_intprob.cpp`

**GPU evaluation:**
`cuda_eval.cu/cuh` (CUDA batch kernel), `metal_eval.h/mm` (Metal batch evaluation)

**Atom typing & assignment:**
`assign_radii.cpp`, `assign_radii_types.cpp`, `assign_radius.cpp`, `assign_types.cpp`, `assign_shift.cpp`, `assign_shortflex.cpp`, `assign_eigen.cpp`, `assign_constraint.cpp`, `update_constraint.cpp`

**Grid & spatial operations:**
`read_grid.cpp`, `read_normalgrid.cpp`, `generate_grid.cpp`, `slice_grid.cpp`, `partition_grid.cpp`, `write_grid.cpp`, `write_sphere.cpp`, `write_rrg.cpp`, `write_rrd.cpp`, `read_spheres.cpp`

**Clustering:**
`cluster.cpp`, `DensityPeak_Cluster.cpp`, `FastOPTICS_cluster.cpp`, `FOPTICS.cpp/h`

**Input/config:**
`read_input.cpp`, `read_emat.cpp`, `read_flexscfile.cpp`, `read_constraints.cpp`, `read_rmsdst.cpp`, `top.cpp`

**Benchmarks:**
`benchmark_vcfbatch.cpp`

**Legacy/internal:**
`python_bindings.cpp` (legacy pybind11 stub), `add2_optimiz_vec.cpp`, `alter_mode.cpp`, `buildlist.cpp`, `create_rebuild_list.cpp`, `print_surfmat.cpp`, `read_eigen.cpp`, `update_optres.cpp`, `wif083.cpp`

**Other:**
`rna_structure.cpp`, `dee_pivot.cpp`, `dee_print.cpp`, `maps.cpp`, `check_clash.cpp`, `calc_center.cpp`, `calc_cleftic.cpp`, `number_of_dihedrals.cpp`

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

# Python-only development (no CMake needed)
cd python
pip install -e .
pytest tests/
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

### Build Targets

| Target | Description |
|--------|-------------|
| `FlexAID` | Main docking executable (~50 source files) |
| `_core` | pybind11 Python extension (statmech + encom bindings) |
| `tENCoM` | Standalone tENCoM CLI tool (vibrational entropy differential) |
| `benchmark_tencom` | tENCoM performance benchmark |
| `benchmark_vcfbatch` | VoronoiCFBatch batch interface benchmark |

### Build System Details

- **SIMD**: Custom `flexaids_configure_simd` function for AVX2/AVX512 detection
- **Eigen3**: Dual detection via pkg-config and CMake module; warns on fallback
- **CUDA**: Separable compilation, targets architectures 70/75/80/86/89/90
- **Metal**: `xcrun metal` compiler for .metal shaders → .metallib; Objective-C++ for bridges
- **GoogleTest**: Auto-downloaded via `FetchContent` when `BUILD_TESTING=ON`

## Testing

### C++ Tests (GoogleTest)

```bash
# Build and run
cmake -DBUILD_TESTING=ON .. && cmake --build . -j $(nproc)
ctest --test-dir build
```

Key test files in `tests/`:
- `test_statmech.cpp` — StatMechEngine: partition function, Boltzmann probabilities, WHAM, parallel tempering, TI, log-sum-exp stability, edge cases
- `test_binding_mode_statmech.cpp` — BindingMode ↔ StatMechEngine integration, pose clustering, thermodynamic integration with GA ensemble
- `test_binding_mode_vibrational.cpp` — BindingMode ENCoM vibrational correction (Phase 3)
- `test_ga_validation.cpp` — GA correctness: selection, crossover, mutation operators, fitness landscape convergence
- `test_hardware_dispatch.cpp` — ShannonThermoStack: entropy computation, hardware backend reporting (CPU/CUDA/Metal), distribution edge cases
- `test_tencom_diff.cpp` — tENCoM differential engine: Cα PDB reader, mode overlap, B-factor diffs, synthetic PDB generation

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
- `test_cli.py` — CLI entry point tests (`python -m flexaidds`)
- `test_docking.py` — Docking API and BindingMode thermodynamics
- `test_encom.py` — ENCoM vibrational entropy
- `test_tencm.py` — TorsionalENM and ShannonThermoStack (pure Python)
- `test_io.py` — PDB I/O and REMARK parsing
- `test_thermodynamics.py` / `test_thermodynamics_dataclass.py` — Thermodynamics module
- `test_models.py` — Data model validation
- `test_py_statmech.py` — Pure-Python StatMech fallback
- `test_import_fallback.py` — Graceful import without C++ bindings
- `test_results.py` — Result loading integration
- `test_version.py` — Version string checks

**Marker**: `@requires_core` — marks tests that need the compiled C++ `_core` extension. These skip gracefully if bindings are not built.

**Fixtures** (in `python/conftest.py`): `requires_core`, `tmp_dir`, `sample_energies`, `simple_pdb_file`, `simple_mol2_file`, `simple_rrd_file`, `rrd_with_rmsd_file`, `flexaid_config_file`, `encom_files`

### CI Structure (.github/workflows/ci.yml)

1. **pure_python_results** (ubuntu-latest, Python 3.11) — runs `test_results_io.py` and `test_results_loader_models.py` only; no C++ needed
2. **cxx_core_build** — multi-platform matrix (Linux GCC/Clang, macOS Clang); builds with `BUILD_TESTING=ON`, ninja-build, Eigen3, OpenMP; runs full `ctest`
3. **python_bindings_smoke** (ubuntu-latest) — builds `_core` extension with `BUILD_PYTHON_BINDINGS=ON`; runs `test_statmech_smoke.py`

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
6. **Shannon Entropy** (`ShannonThermoStack/`) — configurational entropy with GPU acceleration
7. **Cavity Detection** (`CavityDetect/`) — SURFNET-based binding site identification

### Python Package Architecture

| Module | Purpose |
|--------|---------|
| `__init__.py` | API surface with conditional C++ binding imports |
| `__main__.py` | CLI entry point: `python -m flexaidds <dir> [--json\|--csv\|--top N]` |
| `__version__.py` | Version string management |
| `models.py` | `PoseResult`, `BindingModeResult`, `DockingResult` dataclasses with JSON/CSV serialization |
| `results.py` | `load_results()` — recursive PDB scanning, REMARK parsing |
| `io.py` | PDB/MOL2/FlexAID config readers; REMARK parser; mode/rank inference from filenames |
| `thermodynamics.py` | Pure-Python fallback `StatMechEngine` and `Thermodynamics` dataclass |
| `docking.py` | High-level docking: `Pose`, `BindingMode`, `BindingPopulation` classes |
| `encom.py` | `NormalMode`, `ENCoMEngine` wrappers with quasi-harmonic entropy fallback |
| `visualization.py` | PyMOL integration helpers: `load_binding_mode`, `show_pose_ensemble`, `color_by_boltzmann_weight` |
| `_core.cpp` | pybind11 source → compiles to `_core.so` extension |

### PyMOL Plugin

The `pymol_plugin/` package registers PyMOL commands for interactive visualization:

| Command | Function | Description |
|---------|----------|-------------|
| `flexaids_load` | `load_binding_modes()` | Load binding modes from results |
| `flexaids_show_ensemble` | `show_pose_ensemble()` | Render pose ensemble |
| `flexaids_color_boltzmann` | `color_by_boltzmann_weight()` | Color by Boltzmann weight |
| `flexaids_thermo` | `show_thermodynamics()` | Display thermodynamic properties |
| `flexaids_load_results` | `load_docking_results()` | Load full docking results |
| `flexaids_show_mode` | `show_binding_mode()` | Show single binding mode |
| `flexaids_color_mode` | `color_mode_by_score()` | Color mode by score |
| `flexaids_mode_details` | `show_mode_details()` | Display mode details |

Plugin files: `gui.py` (FlexAIDSPanel widget), `visualization.py` (rendering), `results_adapter.py` (bridge to `flexaidds.load_results()`)

### pybind11 Bindings

`python/bindings/core_bindings.cpp` exposes:
- **Data structures**: `State`, `Thermodynamics`, `Replica`, `WHAMBin`, `TIPoint`
- **Engines**: `StatMechEngine` (partition function, WHAM, parallel tempering, TI), `ENCoMEngine`
- **Normal modes**: `NormalMode`, `VibrationalEntropy`
- **Utilities**: `BoltzmannLUT` (fast lookup table), `kB_kcal`, `kB_SI` constants

`python/setup.py` compiles only `statmech.cpp` and `encom.cpp` from LIB/ (not the full FlexAID engine).

### Usage Modes

- **Legacy**: `./FlexAID config.inp ga.inp`
- **CLI**: `./flexaids dock receptor.pdb ligand.mol2`
- **Python**: `import flexaidds` (Phase 2, complete)
- **Python CLI**: `python -m flexaidds <results_dir> [--json|--csv|--top N]`
- **PyMOL**: Load plugin from `pymol_plugin/`

### Development Phases

- **Phase 1** (Complete): StatMechEngine integration
- **Phase 2** (Complete): Python bindings, result I/O, `dock()` API, ENCoM `compute_delta_s()`
- **Phase 3** (Complete): ENCoM vibrational entropy integration into docking pipeline

## Key Files to Know

| File | Purpose |
|------|---------|
| `LIB/flexaid.h` | Central header — constants, structs, macros |
| `LIB/gaboom.cpp` | Genetic algorithm core |
| `LIB/statmech.cpp` | Statistical mechanics engine |
| `LIB/Vcontacts.cpp` | Voronoi contact scoring |
| `LIB/BindingMode.cpp` | Pose clustering + thermodynamics |
| `LIB/encom.cpp` | ENCoM vibrational entropy |
| `LIB/tENCoM/tencm.cpp` | Torsional ENCoM backbone flexibility |
| `LIB/tENCoM/tencom_diff.cpp` | tENCoM differential engine |
| `LIB/Mol2Reader.cpp` | MOL2 molecular file reader |
| `LIB/SdfReader.cpp` | SDF molecular file reader |
| `LIB/CleftDetector.cpp` | Binding-site cleft detection |
| `LIB/VoronoiCFBatch.h` | Batch Voronoi CF (header-only, std::span) |
| `LIB/ShannonThermoStack/ShannonThermoStack.cpp` | Shannon entropy dispatch layer |
| `LIB/CavityDetect/CavityDetect.cpp` | SURFNET cavity detection |
| `python/flexaidds/__init__.py` | Python API surface |
| `python/flexaidds/models.py` | PoseResult, BindingModeResult, DockingResult |
| `python/flexaidds/results.py` | load_results() loader |
| `python/flexaidds/io.py` | PDB I/O and REMARK parsing |
| `python/flexaidds/docking.py` | High-level docking interface |
| `python/flexaidds/encom.py` | ENCoM normal-mode analysis |
| `python/flexaidds/thermodynamics.py` | Pure-Python StatMech fallback |
| `python/flexaidds/visualization.py` | PyMOL integration helpers |
| `python/flexaidds/tencm.py` | TorsionalENM + ShannonThermoStack Python wrappers |
| `python/bindings/core_bindings.cpp` | pybind11 bridge code |
| `pymol_plugin/__init__.py` | PyMOL plugin registration |
| `CMakeLists.txt` | Build configuration (all targets, options) |
| `cmake/MetalAcceleration.cmake` | Metal GPU build helper |
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
4. Add fixtures in `python/conftest.py` if needed
5. If it needs C++ bindings, add pybind11 wrappers in `python/bindings/core_bindings.cpp`

### Running Python CLI

```bash
# Load and display docking results
python -m flexaidds /path/to/results/

# Export as JSON or CSV
python -m flexaidds /path/to/results/ --json
python -m flexaidds /path/to/results/ --csv

# Show top N results
python -m flexaidds /path/to/results/ --top 5
```

### Cross-platform considerations

- Test with both GCC and Clang on Linux, Clang on macOS
- MSVC support exists but is not in CI matrix
- Metal code only compiles on macOS (`FLEXAIDS_USE_METAL=ON`)
- CUDA code requires CUDA toolkit (`FLEXAIDS_USE_CUDA=ON`)
- No `.clang-format`, `.clang-tidy`, or `.editorconfig` — follow existing code style
