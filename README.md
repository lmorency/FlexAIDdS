<div align="center">

# FlexAID∆S

**Entropy-Driven Molecular Docking**

*Combining genetic algorithms with Shannon information theory*
*and statistical mechanics for accurate binding free energy prediction*

[![CI](https://github.com/lmorency/FlexAIDdS/actions/workflows/ci.yml/badge.svg)](https://github.com/lmorency/FlexAIDdS/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Python](https://img.shields.io/badge/python-%E2%89%A5%203.9-3776AB.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)](#)
[![Version](https://img.shields.io/badge/version-2.0.0-brightgreen.svg)](VERSION.md)
[![DOI](https://img.shields.io/badge/DOI-10.1021%2Facs.jcim.5b00078-blue)](https://doi.org/10.1021/acs.jcim.5b00078)

</div>

**[Installation](docs/INSTALLATION.md)** · **[User Guide](docs/USERGUIDE.md)** · **[Support Matrix](docs/SUPPORT_MATRIX.md)** · **[Reproducibility](docs/REPRODUCIBILITY.md)** · **[Benchmarks](docs/BENCHMARKS.md)** · **[Changelog](VERSION.md)** · **[Website](https://lmorency.github.io/FlexAIDdS/)** · **[Documentation Hub](#documentation)**

---

FlexAID∆S extends the [FlexAID](https://doi.org/10.1021/acs.jcim.5b00078) docking engine with a full **canonical ensemble thermodynamics** layer based on Shannon information theory and statistical mechanics. Where conventional docking programs rank poses by enthalpy alone, FlexAID∆S computes the Helmholtz free energy *F* = *H* - *TS* from the partition function over the GA conformational ensemble, accounting for configurational and vibrational entropy contributions that are critical for correct binding mode identification.

**Key capabilities:**
- Genetic algorithm docking with Voronoi contact function scoring
- Canonical ensemble partition function, free energy, entropy, and heat capacity
- Torsional elastic network model (tENCoM) for backbone vibrational entropy
- Full ligand flexibility: torsions, ring conformers, chiral center discrimination
- Unified hardware dispatch (CUDA, Metal, AVX-512, AVX2, OpenMP)
- Python package with docking API, result analysis, and PyMOL visualization
- Cross-platform: Linux (GCC/Clang), macOS (Clang + Apple Silicon Metal), Windows (MSVC)

---

## Core 1.0 Support Boundary

The repository contains a broad research platform. **Not every visible feature is part of the supported 1.0 contract.**

### Supported for Core 1.0

- `FlexAIDdS` command-line executable (LTO-optimized)
- `FlexAID` legacy-compatible command-line executable
- `tENCoM` command-line executable (vibrational entropy)
- JSON configuration workflows documented for the core engine
- `flexaidds` Python package (pure-Python + optional C++ accelerated mode)
- Core repository documentation (install, validation, reproducibility)
- Benchmark bundles under [`benchmarks/`](benchmarks/)

### Experimental (not part of Core 1.0)

- Swift packages and Apple-device integration layers
- TypeScript / PWA / browser-facing dashboards
- Bonhomme Fleet and iCloud-based distributed execution
- NATURaL co-translational / co-transcriptional workflows
- Backend-specific accelerator paths not covered by the support matrix

See: [`docs/VALIDATED_CAPABILITIES.md`](docs/VALIDATED_CAPABILITIES.md) · [`docs/EXPERIMENTAL_CAPABILITIES.md`](docs/EXPERIMENTAL_CAPABILITIES.md) · [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md)

---

## Quick Start

```bash
git clone https://github.com/lmorency/FlexAIDdS.git && cd FlexAIDdS
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

```bash
# Dock with full flexibility and entropy at 300 K (default)
./build/FlexAIDdS receptor.pdb ligand.mol2

# Argument order doesn't matter -- auto-detected from file content
./build/FlexAIDdS ligand.mol2 receptor.pdb

# Dock directly from a SMILES string (3D coordinates built automatically)
./build/FlexAIDdS receptor.pdb "CC(=O)Oc1ccccc1C(=O)O"

# CIF/mmCIF input, JSON config override, rigid screening
./build/FlexAIDdS receptor.cif ligand.sdf
./build/FlexAIDdS receptor.pdb ligand.mol2 -c config.json
./build/FlexAIDdS receptor.pdb ligand.mol2 --rigid
```

```python
import flexaidds as fd

results = fd.dock(
    receptor="receptor.pdb",
    ligand="ligand.mol2",
    compute_entropy=True,
)

for mode in results.rank_by_free_energy():
    print(f"Mode: dG={mode.free_energy:.2f} kcal/mol")
```

---

## What's New in v2.0

Released 2026-04-04 — complete rewrite of the FlexAID engine.

- **Entropy-driven scoring** — full canonical ensemble thermodynamics via Shannon information theory (ΔG, ΔH, -TΔS, Cv)
- **Multi-format input with SMILES** — dock directly from SMILES strings with automatic 3D coordinate generation; pure C++20 parser, no RDKit/Boost
- **tENCoM vibrational entropy** — torsional elastic network contact model for backbone vibrational entropy differentials
- **Unified hardware dispatch** — automatic runtime backend: CUDA > Metal > AVX-512 > AVX2 > OpenMP > scalar
- **Full ligand flexibility by default** — ring conformer sampling, sugar pucker, R/S chiral center discrimination
- **Python package & PyMOL plugin** — `flexaidds` with docking API, result I/O, CLI inspector, and interactive visualization
- **GIST desolvation & H-bond scoring** — grid-based water thermodynamics and angular-dependent hydrogen bond potential
- **GA diversity monitoring** — Shannon entropy collapse detection with adaptive catastrophic mutation
- **MPI distributed docking** — grid domain decomposition across compute nodes
- **ML rescoring bridge** — Voronoi graph + Shannon profile feature extraction for hybrid physics/ML scoring

See the full changelog in [VERSION.md](VERSION.md).

---

## Architecture

```
+--------------+    +----------------+    +--------------+    +-------------------+    +-----------------+
|   Input      |    | ProcessLigand  |    |   Genetic    |    |     Scoring       |    | Thermodynamics  |
|              |--->|                |--->|  Algorithm   |--->|                   |--->|                 |
| PDB/CIF +   |    | SMILES parse,  |    |  (gaboom)    |    | Voronoi CF + DEE  |    | StatMech + S    |
| MOL2/SDF/   |    | 3D build,      |    +------+-------+    +-------------------+    +--------+--------+
| SMILES      |    | atom typing    |          |                                              |
+--------------+    +----------------+          v                                              v
                                         +--------------+                              +-----------------+
                                         |  Flexibility |                              |  Binding Modes  |
                                         |              |                              |                 |
                                         | Torsions     |                              | Clustering +    |
                                         | Rings        |                              | dG, dH, -TdS   |
                                         | Chirality    |                              | Cv, Boltzmann   |
                                         | tENCoM       |                              +-----------------+
                                         +--------------+

Hardware: CUDA > Metal > AVX-512 > AVX2 > OpenMP > scalar
```

---

## Features

#### Docking Engine
- **Genetic algorithm** with configurable population, crossover, mutation, selection, and GA diversity monitoring
- **Voronoi contact function (CF)** for shape complementarity scoring
- **GIST water-displacement scoring** -- grid-based explicit solvation term (`GISTEvaluator`, `GISTGrid`)
- **Directional H-bond scoring** -- geometry-aware hydrogen bond potential (`HBondEvaluator`, `hbond_potential.h`)
- **Dead-end elimination (DEE)** reduces ligand conformational search space
- **Batch evaluation** via `VoronoiCFBatch` (AoS) and `VoronoiCFBatch_SoA` (SIMD-friendly) with OpenMP parallelism
- **Multiple clustering** methods: centroid-first, FastOPTICS, Density Peak
- **MPI distributed docking** -- grid domain decomposition across compute nodes (`FLEXAIDS_USE_MPI`)
- **Metal ion scoring** -- receptor-bound ions (Mg2+, Zn2+, Ca2+, Fe2+/3+, Cu2+, Mn2+, Na+, K+, Cl-, Br-, and more) receive crystallographic VdW radii and SYBYL atom types
- **Structural water retention** -- ordered crystallographic waters (B-factor < 20 A^2) participate in Voronoi CF scoring

#### Thermodynamics
- **Canonical ensemble** -- partition function *Z*, Helmholtz free energy *F*, entropy *S*, heat capacity *C_v*
- **Shannon entropy** -- S = -k_B sum(p_i ln p_i) with log-sum-exp numerical stability
- **Torsional ENCoM** (tENCoM) -- backbone vibrational entropy without full rotamer rebuilds
- **ShannonThermoStack** -- combined configurational + vibrational entropy pipeline
- **Thermodynamic integration** and WHAM free energy profiles
- **Boltzmann weight** normalization and fast lookup table

#### Molecular Flexibility
- **Full flexibility by default** -- ligand torsions, ring conformers, chirality, intramolecular scoring at 300 K
- **Non-aromatic ring sampling** -- chair/boat/twist for 6-membered, envelope/twist for 5-membered rings, sugar pucker
- **Chiral center discrimination** -- explicit R/S sampling with stereochemical energy penalty
- **Multi-format input** -- PDB, CIF/mmCIF, MOL2, SDF/MOL V2000, SMILES (with automatic 3D build)

#### Hardware Acceleration
- **Unified hardware dispatch** -- automatic backend selection at runtime
- **CUDA** -- batch CF evaluation and Shannon entropy histograms (Volta through Blackwell, sm_70–sm_120; Blackwell requires CUDA ≥ 12.6)
- **Metal** -- Apple Silicon GPU for Shannon entropy, cavity detection, and evaluation
- **SIMD** -- AVX-512 and AVX2 vectorised geometric primitives
- **OpenMP + Eigen3** -- thread parallelism and vectorised linear algebra
- **LTO binaries** -- link-time optimized executables

#### Python Package
- **`flexaidds`** -- docking API, result I/O, thermodynamics, CLI inspector
- **Pure-Python fallback** -- works without C++ compilation
- **pybind11 C++ bridge** -- `StatMechEngine`, `ENCoMEngine`, `Thermodynamics`, `VibrationalEntropy`
- **PyMOL plugin** -- binding mode visualization, entropy heatmaps, ITC comparison

---

## Build

### Requirements

- **Required**: C++20 compiler (GCC >= 10, Clang >= 10, MSVC >= 19.30), CMake >= 3.18
- **Recommended**: Eigen3 (`libeigen3-dev` / `brew install eigen`)
- **Optional**: OpenMP, CUDA Toolkit, Metal framework (macOS), pybind11, MPI

### Output Binaries

| Binary | Description |
|:-------|:------------|
| `FlexAID` | Standard docking executable |
| `FlexAIDdS` | Optimized docking (LTO + `-march=native`) |
| `tENCoM` | Vibrational entropy differential tool |

### Build Variants

```bash
# Standard release
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# With tests (46 C++ test files via GoogleTest)
cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel && ctest --test-dir build --output-on-failure

# With Python bindings
cmake -S . -B build -DBUILD_PYTHON_BINDINGS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# HPC deployment (AVX-512 + OpenMP)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DFLEXAIDS_USE_AVX512=ON -DFLEXAIDS_USE_OPENMP=ON

# Apple Silicon with Metal GPU
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFLEXAIDS_USE_METAL=ON
```

```bash
# Distributed validation (parallel C++ tests + MPI benchmark smoke run)
MPI_PROCS=4 CTEST_JOBS=8 ./scripts/run_distributed_validation.sh
```

### CMake Options

| Option | Default | Description |
|:-------|:--------|:------------|
| `BUILD_FLEXAIDDS_FAST` | **ON** | LTO-optimized FlexAIDdS binary |
| `BUILD_TESTING` | OFF | GoogleTest unit tests (46 test files) |
| `BUILD_PYTHON_BINDINGS` | OFF | pybind11 Python extension (`_core`) |
| `BUILD_SWIFT_BRIDGE` | OFF | Swift bridge (macOS only, experimental) |
| `ENABLE_TENCOM_TOOL` | **ON** | tENCoM vibrational entropy tool |
| `ENABLE_TENCOM_BENCHMARK` | OFF | tENCoM benchmark binary |
| `ENABLE_VCFBATCH_BENCHMARK` | OFF | VoronoiCFBatch benchmark binary |
| `FLEXAIDS_USE_CUDA` | OFF | CUDA GPU acceleration |
| `FLEXAIDS_USE_METAL` | OFF | Metal GPU acceleration (macOS) |
| `FLEXAIDS_USE_AVX2` | **ON** | AVX2 SIMD acceleration |
| `FLEXAIDS_USE_AVX512` | OFF | AVX-512 SIMD acceleration |
| `FLEXAIDS_USE_OPENMP` | **ON** | OpenMP thread parallelism |
| `FLEXAIDS_USE_EIGEN` | **ON** | Eigen3 vectorised linear algebra |
| `FLEXAIDS_USE_256_MATRIX` | **ON** | 256x256 soft contact matrix system |
| `FLEXAIDS_USE_MPI` | OFF | MPI distributed parallel docking |

---

## Usage

### Command Line

```bash
# Full flexibility dock (entropy at 300 K by default)
./build/FlexAIDdS receptor.pdb ligand.mol2

# JSON config override
./build/FlexAIDdS receptor.pdb ligand.mol2 -c config.json

# Rigid screening (no flexibility, no entropy)
./build/FlexAIDdS receptor.pdb ligand.mol2 --rigid
```

All parameters have built-in defaults. Override only what you need via JSON:

```json
{
  "thermodynamics": { "temperature": 310, "clustering_algorithm": "DP" },
  "ga": { "num_chromosomes": 2000, "num_generations": 1000 },
  "flexibility": { "ligand_torsions": true, "ring_conformers": true }
}
```

### Vibrational Entropy (tENCoM)

```bash
tENCoM reference.pdb target1.pdb [target2.pdb ...] [-T temp] [-r cutoff] [-k k0] [-o prefix]
```

### Python Package

```bash
cd python && pip install -e .
```

The `flexaidds` package works in two modes: **pure Python** (always available) and **C++ accelerated** (when built with `BUILD_PYTHON_BINDINGS=ON`).

```python
import flexaidds as fd

# High-level docking
results = fd.dock(receptor="receptor.pdb", ligand="ligand.mol2", compute_entropy=True)

# Load existing results
docking = fd.load_results("output_prefix")
for mode in docking.binding_modes:
    print(f"Mode {mode.rank}: dG={mode.free_energy:.2f}, S={mode.entropy:.3f}")
```

```python
from flexaidds import StatMechEngine

engine = StatMechEngine(temperature=300)
engine.add_energies(pose_energies)
thermo = engine.compute()
print(f"F = {thermo.free_energy:.2f} kcal/mol")
print(f"S = {thermo.entropy:.4f} kcal/(mol*K)")
```

### CLI Inspector

```bash
python -m flexaidds /path/to/results/              # summary table
python -m flexaidds /path/to/results/ --top 5      # top 5 modes
python -m flexaidds /path/to/results/ --json        # JSON output
python -m flexaidds /path/to/results/ --csv out.csv  # CSV export
```

<details>
<summary><strong>PyMOL Plugin</strong></summary>

**Installation**: PyMOL > Plugin Manager > Install New Plugin > select `pymol_plugin/` > restart.

| Command | Description |
|:--------|:------------|
| `flexaids_load <dir> [temp]` | Load results from output directory |
| `flexaids_show_ensemble <mode>` | Display all poses in a binding mode |
| `flexaids_color_boltzmann <mode>` | Color by Boltzmann weight |
| `flexaids_thermo <mode>` | Print thermodynamic properties |
| `flexaids_entropy_heatmap <mode>` | Spatial entropy density heatmap |
| `flexaids_dock <obj> <lig>` | Interactive docking from PyMOL |

Requires: `pip install -e python/`

</details>

<details>
<summary><strong>Legacy Mode</strong></summary>

```bash
./FlexAID config.inp ga.inp output_prefix
./FlexAIDdS --legacy config.inp ga.inp output_prefix
```

</details>

---

## Testing

### C++ (GoogleTest)

```bash
cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

46 test files covering: StatMechEngine, BindingMode, GA operators, tENCoM, Voronoi contacts, FastOPTICS clustering, hardware dispatch, cavity detection, ion handling, ring conformers, chiral centers, file readers, grid decomposition, MIF scoring, GIST/HBond scoring, GA diversity, distributed backend, and more.

### Python (pytest)

```bash
cd python && pip install -e . && pytest tests/
```

32 test files. Tests marked `@requires_core` skip gracefully when the C++ extension is not built.

### CI

GitHub Actions pipeline: Linux GCC, Linux Clang, macOS Clang (allow-fail), Windows MSVC (allow-fail). Python smoke tests on all platforms. Additional workflows: `license-scan.yml` (Apache compliance), `perf.yml` (benchmark regression), `sanitizers.yml` (ASan/UBSan).

See [`docs/SUPPORT_MATRIX.md`](docs/SUPPORT_MATRIX.md).

---

## Reproducibility

Benchmark and scientific performance claims should be interpreted through the repository reproducibility policy.

**A claim is not repository-reproducible merely because it appears in documentation.** It becomes reproducible only when the corresponding bundle exists under [`benchmarks/`](benchmarks/) with dataset provenance, commands, expected outputs, and metric definitions.

See: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) · [`benchmarks/README.md`](benchmarks/README.md)

---

## Scientific Background

FlexAID∆S treats the GA conformational ensemble as a **canonical ensemble** (*N*, *V*, *T* fixed):

```
Z = sum exp[-beta * E_i]              (partition function)
F = -k_B * T * ln(Z)                  (Helmholtz free energy)
<E> = sum p_i * E_i                    (mean energy)
S = -k_B * sum p_i * ln(p_i)          (Shannon configurational entropy)
C_v = k_B * beta^2 * (<E^2> - <E>^2)  (heat capacity)
```

**Scoring** uses two complementary layers: a **Voronoi contact function** for geometry-based shape complementarity, weighted by a **2-term LJ+Coulomb potential** parameterised over 40 SYBYL atom types. The StatMechEngine converts the GA ensemble into thermodynamic quantities via the canonical partition function with log-sum-exp numerical stability.

---

<details>
<summary><strong>Repository Structure</strong></summary>

```
FlexAIDdS/
+-- LIB/                    # Core C++ library (~100+ source files)
|   +-- flexaid.h            # Main header: constants, data structures
|   +-- gaboom.cpp/h         # Genetic algorithm engine
|   +-- Vcontacts.cpp/h      # Voronoi contact function
|   +-- statmech.cpp/h       # StatMechEngine
|   +-- BindingMode.cpp/h    # Pose clustering + thermodynamics
|   +-- encom.cpp/h          # ENCoM vibrational entropy
|   +-- config_defaults.h    # Default parameter schema
|   +-- config_parser.cpp/h  # JSON config system
|   +-- hardware_dispatch.*  # Unified HW backend selection
|   +-- ShannonThermoStack/  # Shannon entropy + HW acceleration
|   +-- tENCoM/              # Torsional ENCoM module
|   +-- LigandRingFlex/      # Ring conformer sampling
|   +-- ChiralCenter/        # R/S discrimination
|   +-- NATURaL/             # Co-translational assembly (experimental)
|   +-- CavityDetect/        # SURFNET cavity detection
|   +-- Mol2Reader.cpp/h     # MOL2 file reader
|   +-- SdfReader.cpp/h      # SDF file reader
|   +-- CleftDetector.cpp/h  # Binding-site detection
|   +-- VoronoiCFBatch.h     # Batch Voronoi CF (header-only)
+-- src/                    # Entry points
+-- tests/                  # GoogleTest suite (46 test files)
+-- python/                 # Python package + pybind11 bindings
|   +-- flexaidds/           # Python package (22+ modules)
|   +-- bindings/            # C++ bridge
|   +-- tests/               # pytest suite (32 test files)
|   +-- setup.py
+-- pymol_plugin/           # PyMOL visualization plugin
+-- swift/                  # Swift package (experimental)
+-- typescript/             # TypeScript SDK (experimental)
+-- docs/                   # Documentation
+-- benchmarks/             # Reproducibility bundles
+-- cmake/                  # CMake helpers
+-- .github/workflows/      # CI/CD
+-- CMakeLists.txt          # Build configuration
+-- VERSION.md              # Version history and changelog
```

</details>

---

## Documentation

Browse the full documentation at **[lmorency.github.io/FlexAIDdS](https://lmorency.github.io/FlexAIDdS/)**.

### Core Guides

| Document | Description |
|:---------|:------------|
| [Installation Guide](docs/INSTALLATION.md) | Prerequisites, build instructions, platform-specific notes |
| [User Guide](docs/USERGUIDE.md) | Full parameter reference, Python API, PyMOL plugin |
| [Support Matrix](docs/SUPPORT_MATRIX.md) | Supported platforms, compilers, GPU backends, CI coverage |
| [Benchmarks](docs/BENCHMARKS.md) | ITC-187, CASF-2016, LIT-PCBA, cross-docking validation |
| [Reproducibility](docs/REPRODUCIBILITY.md) | Benchmark claim policy, dataset provenance |
| [VERSION.md](VERSION.md) | Version history and changelog |
| [Contributing](CONTRIBUTING.md) | Development setup, license policy, PR guidelines |

### Technical Deep Dives

| Document | Description |
|:---------|:------------|
| [Scoring Functions](docs/docs/scoring/overview.md) | Multi-component scoring: Voronoi CF, H-bond, GIST, electrostatics |
| [H-Bond Potential](docs/docs/scoring/hbond.md) | Angular-dependent Gaussian hydrogen bond scoring |
| [GIST Desolvation](docs/docs/scoring/gist.md) | Grid Inhomogeneous Solvation Theory integration |
| [Genetic Algorithm](docs/docs/ga/overview.md) | Fitness models (SMFREE, PSHARE, LINEAR), key parameters |
| [GA Diversity](docs/docs/ga/diversity.md) | Entropy collapse detection and adaptive mutation |
| [GA Optimizer](docs/docs/ga/optimize.md) | Automated hyperparameter tuning with GAOptimizer |
| [Python API](docs/docs/api/python.md) | StatMechEngine, ENCoMEngine, dock(), load_results() |
| [ML Rescoring](docs/docs/ml-rescore.md) | Feature extraction bridge for hybrid physics/ML scoring |
| [Configuration](docs/docs/configuration.md) | JSON config reference for scoring, GA, and distributed settings |

### Benchmark Suites

| Document | Description |
|:---------|:------------|
| [CASF-2016](docs/docs/benchmarks/casf2016.md) | Scoring and docking power on 285 diverse complexes |
| [LIT-PCBA](docs/docs/benchmarks/litpcba.md) | Unbiased virtual screening across 15 PubChem targets |
| [Cross-Docking](docs/docs/benchmarks/crossdock.md) | Non-native receptor docking validation |

### Internal

| Document | Description |
|:---------|:------------|
| [Implementation Roadmap](docs/IMPLEMENTATION_ROADMAP.md) | Development phases and status |
| [Validated Capabilities](docs/VALIDATED_CAPABILITIES.md) | Core 1.0 supported features |
| [Experimental Capabilities](docs/EXPERIMENTAL_CAPABILITIES.md) | Experimental / non-1.0 features |
| [Known Limitations](docs/KNOWN_LIMITATIONS.md) | Current limitations and caveats |

---

## Publications

If you use FlexAID∆S in your research, please cite:

> Gaudreault F & Najmanovich RJ (2015). FlexAID: Revisiting Docking on Non-Native-Complex Structures.
> *J. Chem. Inf. Model.* 55(7):1323-36.
> [DOI:10.1021/acs.jcim.5b00078](https://doi.org/10.1021/acs.jcim.5b00078)

**Related publications:**

- Morency LP & Najmanovich RJ (2026). FlexAID∆S: Information-Theoretic Entropy Improves Molecular Docking Accuracy and Binding Mode Prediction. *Manuscript in preparation.*
- Gaudreault F, Morency LP & Najmanovich RJ (2015). NRGsuite: a PyMOL plugin to perform docking simulations in real time. *Bioinformatics* 31(23):3856-8. [DOI:10.1093/bioinformatics/btv458](https://doi.org/10.1093/bioinformatics/btv458)
- Frappier V et al. (2015). A Coarse-Grained Elastic Network Atom Contact Model. *Proteins* 83(11):2073-82. [DOI:10.1002/prot.24922](https://doi.org/10.1002/prot.24922)

Full list: [Google Scholar](https://scholar.google.ca/citations?user=amFCT0oAAAAJ&hl=en)

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting pull requests.

| | |
|:--|:--|
| **Accepted licenses** | Apache-2.0, BSD, MIT, MPL-2.0 |
| **Not accepted** | GPL / AGPL -- see [clean-room policy](docs/licensing/clean-room-policy.md) |
| **CLA** | Required for all contributions |

---

## License

[Apache License 2.0](LICENSE) -- free for academic and commercial use.

See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for dependency licenses.

---

<div align="center">

Louis-Philippe Morency
[NRGlab](http://biophys.umontreal.ca/nrg), Departement de biochimie et medecine moleculaire
Universite de Montreal

[Repository](https://github.com/lmorency/FlexAIDdS) · [Issues](https://github.com/lmorency/FlexAIDdS/issues) · [NRGlab GitHub](https://github.com/NRGlab)

</div>
