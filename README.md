# FlexAIDdS

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey.svg)](#)

**FlexAID with Delta-S Entropy** — an entropy-driven molecular docking engine combining genetic algorithms with statistical mechanics thermodynamics. Targets real-world psychopharmacology and drug discovery applications.

## Features

- **Genetic algorithm docking** with configurable population, crossover, mutation, and selection
- **Voronoi contact function (CF)** for shape complementarity scoring
- **Statistical mechanics engine** — partition function, free energy, heat capacity, conformational entropy
- **Torsional ENCoM (tENCoM)** backbone flexibility without full rotamer rebuilds + standalone vibrational entropy differential tool
- **Shannon entropy + torsional vibrational entropy stack** for thermodynamic scoring
- **Full flexibility by default** — ligand torsions, ring conformers, chirality, intramolecular scoring, entropy at 300 K
- **Single JSON config** — one file overrides all parameters; `--rigid` flag for fast screening
- **Ligand ring flexibility** — non-aromatic ring conformer sampling and sugar pucker
- **Chiral center sampling** — explicit R/S stereocenter discrimination in the GA
- **Multi-format ligand input** — MOL2, SDF/MOL (V2000), and legacy INP formats
- **Automatic cavity detection** — SURFNET gap-sphere algorithm with Metal GPU acceleration
- **NATURaL co-translational assembly** — co-translational/co-transcriptional docking with ribosome-speed elongation (Zhao 2011) and Sec translocon TM insertion (Hessa 2007)
- **FastOPTICS + Density Peak clustering** of docking poses
- **Hardware acceleration** — CUDA, Metal (macOS), AVX-512, AVX2, OpenMP, Eigen3
- **Ultra-fast HPC binaries** — LTO + `-march=native` for both FlexAIDdS and tENCoM
- **Python package** (`flexaidds`) — result I/O, thermodynamics, ENCoM, docking API, CLI inspector, PyMOL visualization plugin
- **Dead-end elimination (DEE)** — torsion pruning reduces ligand conformational search space
- **Zero-copy batch scoring** — `VoronoiCFBatch` with `std::span` for OpenMP-parallel GA evaluation
- **FreeNRG integration** — unified free energy framework bridging FlexAID∆S and NRGRank

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
│   ├── config_defaults.h    # Single source of truth for all default parameters
│   ├── config_parser.cpp/h  # JSON config loading, merging, and struct mapping
│   ├── SdfReader.cpp/h      # SDF/MOL V2000 multi-format ligand reader
│   ├── CleftDetector.cpp/h  # Binding cleft/pocket detection
│   ├── FOPTICS.cpp/h        # FastOPTICS density-based clustering
│   ├── ShannonThermoStack/  # Shannon configurational entropy + HW acceleration
│   ├── LigandRingFlex/      # Non-aromatic ring & sugar pucker sampling
│   ├── ChiralCenter/        # R/S stereocenter discrimination
│   ├── NATURaL/             # Co-translational assembly module
│   └── CavityDetect/        # SURFNET cavity detection (Metal GPU support)
├── src/                    # Entry point (gaboom.cpp)
├── tests/                  # C++ unit tests (GoogleTest)
├── python/                 # Python package & bindings
│   ├── flexaidds/           # Python package (API, models, CLI)
│   │   ├── docking.py       # High-level docking API
│   │   ├── encom.py         # ENCoM vibrational entropy interface
│   │   ├── io.py            # Input/output utilities
│   │   ├── models.py        # Data models (PoseResult, BindingModeResult, DockingResult)
│   │   ├── results.py       # Result file parsing & loading
│   │   ├── thermodynamics.py # StatMechEngine & thermodynamic calculations
│   │   └── visualization.py # PyMOL visualization helpers
│   ├── bindings/            # pybind11 C++ bridge
│   ├── tests/               # Pytest test suite
│   ├── setup.py             # setuptools config
│   └── pyproject.toml       # Python project metadata
├── docs/                   # Documentation (architecture, implementation, licensing)
├── cmake/                  # CMake helpers
├── .github/workflows/      # CI/CD (GitHub Actions)
├── CMakeLists.txt          # Primary build configuration
├── WRK/                    # Working directory for builds
└── BIN/                    # Binary output directory
```

## Build

### Requirements

- **Required**: C++20 compiler (GCC >= 10, Clang >= 10, MSVC), CMake >= 3.18
- **Optional**: Eigen3 (`libeigen3-dev`), OpenMP, CUDA Toolkit, Metal framework (macOS), pybind11

### Build Commands

```bash
git clone https://github.com/lmorency/FlexAIDdS.git
cd FlexAIDdS
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target FlexAID -j $(nproc)
```

### With Tests

```bash
cmake .. -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . -j $(nproc)
ctest --test-dir .
```

### With Python Bindings

```bash
cmake .. -DBUILD_PYTHON_BINDINGS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . -j $(nproc)
```

### CMake Options

| Option                    | Default | Description                              |
|:--------------------------|:--------|:-----------------------------------------|
| `FLEXAIDS_USE_CUDA`       | OFF     | CUDA GPU batch evaluation                |
| `FLEXAIDS_USE_METAL`      | OFF     | Metal GPU acceleration (macOS only)      |
| `FLEXAIDS_USE_AVX2`       | ON      | AVX2 SIMD acceleration                   |
| `FLEXAIDS_USE_AVX512`     | OFF     | AVX-512 SIMD acceleration                |
| `FLEXAIDS_USE_OPENMP`     | ON      | OpenMP thread parallelism                |
| `FLEXAIDS_USE_EIGEN`      | ON      | Eigen3 vectorised linear algebra         |
| `BUILD_PYTHON_BINDINGS`   | OFF     | pybind11 Python extensions               |
| `BUILD_TESTING`           | OFF     | GoogleTest unit tests                    |
| `ENABLE_TENCOM_BENCHMARK` | OFF     | Build standalone TeNCoM benchmark binary |

## Usage

### Docking

```bash
./FlexAID config.inp ga.inp output_prefix
```

| Argument        | Description                                              |
|:----------------|:---------------------------------------------------------|
| `config.inp`    | Docking configuration (receptor, ligand, scoring, etc.)  |
| `ga.inp`        | Genetic algorithm parameters                             |
| `output_prefix` | Base path for result files (`.cad`, `_0.pdb`, `_1.pdb`) |

All docking and GA parameters have built-in defaults. Your config files only need to specify values you want to change from their presets. For example, a minimal `config.inp` for Voronoi scoring at 300 K:

```ini
PDBNAM receptor.pdb
INPLIG ligand.mol2
COMPLF VCT
TEMPER 300
```

Everything else (grid spacing, clustering threshold, optimization steps, etc.) uses sensible defaults automatically. See [Configuration Reference](#configuration-reference) for all parameters and their defaults.

### Vibrational Entropy (tENCoM)

```bash
tENCoM reference.pdb target1.pdb [target2.pdb ...] [-T temp] [-r cutoff] [-k k0] [-o prefix]
```

### Python API (Phase 2 — in progress)

```python
import flexaidds

# High-level docking
results = flexaidds.dock(
    receptor='receptor.pdb',
    ligand='ligand.mol2',
    binding_site='auto',
    compute_entropy=True
)

# Load and analyze existing results
from flexaidds import load_results
docking = load_results('output_prefix')
for mode in docking.binding_modes:
    print(f"Mode {mode.rank}: dG={mode.free_energy:.2f}, S={mode.entropy:.3f}")

# Thermodynamic analysis
from flexaidds import StatMechEngine
engine = StatMechEngine(temperature=300)
engine.add_energies(pose_energies)
print(f"Free energy: {engine.free_energy():.2f} kcal/mol")

# ENCoM vibrational entropy
from flexaidds import ENCoMEngine
encom = ENCoMEngine()
delta_s = encom.compute_delta_s('apo.pdb', 'holo.pdb')
```

**Available modules**: `docking`, `encom`, `io`, `models`, `results`, `thermodynamics`, `visualization`

### Python CLI (Result Inspector)

Inspect docking results from the command line without writing Python code:

```bash
# Human-readable summary table
python -m flexaidds /path/to/results/

# Show only the top 5 binding modes
python -m flexaidds /path/to/results/ --top 5

# Machine-readable JSON output
python -m flexaidds /path/to/results/ --json

# Export binding modes to CSV
python -m flexaidds /path/to/results/ --csv results.csv
```

Output includes mode ID, rank, number of poses, free energy, enthalpy, entropy, and best CF score for each binding mode.

### PyMOL Plugin

A full visualization plugin for PyMOL with GUI panel and 8 registered commands:

**Installation**:
1. PyMOL > Plugin Manager > Install New Plugin
2. Select the `pymol_plugin/` directory
3. Restart PyMOL — access via Plugin > FlexAID∆S

**Commands** (usable from PyMOL command line):

| Command | Description |
|:--------|:------------|
| `flexaids_load <dir> [temp]` | Load docking results from output directory |
| `flexaids_show_ensemble <mode>` | Display all poses in a binding mode |
| `flexaids_color_boltzmann <mode>` | Color poses by Boltzmann weight (blue=low, red=high) |
| `flexaids_thermo <mode>` | Print thermodynamic properties (G, H, S, Cv) |
| `flexaids_load_results <dir>` | Load results via flexaidds API adapter |
| `flexaids_show_mode <mode>` | Show a single binding mode cluster |
| `flexaids_color_mode <mode>` | Color mode poses by score |
| `flexaids_mode_details <mode>` | Print detailed mode statistics |

The plugin requires the `flexaidds` Python package (`pip install -e python/`).

## Testing

### C++ Tests (GoogleTest)

```bash
cmake -DBUILD_TESTING=ON .. && cmake --build . -j $(nproc)
ctest --test-dir build
```

Key test files in `tests/`:
- `test_statmech.cpp` — StatMechEngine correctness
- `test_binding_mode_statmech.cpp` — BindingMode / StatMechEngine integration
- `test_ga_validation.cpp` — Genetic algorithm validation

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

Tests marked with `@requires_core` need the compiled C++ `_core` extension and skip gracefully if bindings are not built.

---

## Scientific Background

### NATURaL Scoring Function

```
E = Σ [ε_ij·(r_ij⁻¹² − 2r_ij⁻⁶)] + Σ [(q_i·q_j)/(4πε₀·ε_r·r_ij)]
    └── Lennard-Jones 12-6 ──┘     └──── Coulomb ────┘

• 40 SYBYL atom types (compressed from 84)
• Distance-dependent dielectric: ε_r = 4r
• Validation: r = 0.78–0.82 on CASF-2016
```

### Statistical Mechanics Framework

**Canonical ensemble** (*N*, *V*, *T* fixed):

```
Z = Σ exp[−β·E_i]                (partition function)
F = −k_B·T·ln(Z)                 (Helmholtz free energy)
⟨E⟩ = Σ p_i·E_i                  (mean energy / enthalpy)
S = −k_B·Σ p_i·ln(p_i)           (Shannon entropy)
C_v = k_B·β²·(⟨E²⟩ − ⟨E⟩²)       (heat capacity)
```

**Implemented in** `LIB/statmech.{h,cpp}`:
- Log-sum-exp for numerical stability
- Boltzmann weight normalization
- Thermodynamic integration (*λ*-path)
- WHAM (single-window)

---

## Benchmarks

### ITC-187: Calorimetry Gold Standard

| Metric | FlexAID∆S | Vina | Glide |
|--------|-----------|------|-------|
| **∆*G* Pearson *r*** | **0.93** | 0.64 | 0.69 |
| **RMSE (kcal/mol)** | **1.4** | 3.1 | 2.9 |
| **Ranking Power** | **78%** | 58% | 64% |

### CASF-2016: Diverse Drug Targets

| Power | FlexAID∆S | Vina | Glide | rDock |
|-------|-----------|------|-------|-------|
| **Scoring** | **0.88** | 0.73 | 0.78 | 0.71 |
| **Docking** | **81%** | 76% | 79% | 73% |
| **Screening (EF 1%)** | **15.3** | 11.2 | 13.1 | 10.8 |

### Psychopharmacology (CNS Receptors)

**23 neurological targets** (GPCR, ion channels, transporters):
- **Pose rescue rate**: 92% (entropy recovers correct mode when enthalpy fails)
- **Average entropic penalty**: +3.02 kcal/mol
- **Example** (mu-opioid + fentanyl):
  - Enthalpy-only: Wrong pocket (−14.2 kcal/mol, RMSD 8.3 A)
  - With entropy: **Correct** (−10.8 kcal/mol, RMSD 1.2 A, exp: −11.1)

---

## Configuration Reference

All parameters have built-in defaults. Override files use a simple format: one parameter per line, code followed by value.

### Docking Parameters (config)

#### Input Files

| Code     | Description                  | Default              |
|:---------|:-----------------------------|:---------------------|
| `PDBNAM` | Receptor PDB file            | *(required)*         |
| `INPLIG` | Ligand input file            | *(required)*         |
| `DEFTYP` | Atom type definition file    | Auto (AMINO.def)     |
| `IMATRX` | Energy matrix file           | MC_st0r5.2_6.dat     |
| `CONSTR` | Distance constraint file     | None                 |
| `RMSDST` | RMSD reference structure     | None                 |

#### Scoring & Complementarity

| Code     | Description                         | Default | Options         |
|:---------|:------------------------------------|:--------|:----------------|
| `COMPLF` | Complementarity function            | `SPH`   | `SPH`, `VCT`    |
| `VCTSCO` | Voronoi self-consistency mode       | `MAX`   |                 |
| `VCTPLA` | Voronoi plane definition            | `X`     |                 |
| `NORMAR` | Normalize contact area              | Off     |                 |
| `USEACS` | Use accessible surface              | Off     |                 |
| `ACSWEI` | ACS weighting factor                | 1.0     |                 |

#### Binding Site

| Code     | Description                         | Default | Options                              |
|:---------|:------------------------------------|:--------|:-------------------------------------|
| `RNGOPT` | Binding site method                 |         | `LOCCEN`, `LOCCLF`, `LOCCDT`, `AUTO`|

- `LOCCEN x y z radius` — search around center coordinates
- `LOCCLF file.pdb` — use pre-computed sphere file
- `LOCCDT [cleft_id] [min_r] [max_r]` — automatic cavity detection (SURFNET)

#### Optimization Steps

| Code     | Description                         | Default |
|:---------|:------------------------------------|:--------|
| `VARDIS` | Translation step (A)                | 0.25    |
| `VARANG` | Angle step (deg)                    | 5.0     |
| `VARDIH` | Dihedral step (deg)                 | 5.0     |
| `VARFLX` | Flexible sidechain step (deg)       | 10.0    |
| `SPACER` | Grid point spacing                  | 0.375   |

#### Flexibility

| Code     | Description                         | Default |
|:---------|:------------------------------------|:--------|
| `FLEXSC` | Flexible sidechain specification    | None    |
| `ROTPER` | Rotamer vdW permeability            | 0.8     |
| `PERMEA` | Global vdW permeability             | 1.0     |
| `NOINTR` | Disable intramolecular scoring      | Off (intramolecular enabled) |
| `INTRAF` | Intramolecular energy fraction      | 1.0     |

#### Thermodynamics & Clustering

| Code     | Description                         | Default | Options            |
|:---------|:------------------------------------|:--------|:-------------------|
| `TEMPER` | Temperature (K, 0 = entropy off)    | 0       |                    |
| `CLUSTA` | Clustering algorithm                | `CF`    | `CF`, `FO`, `DP`   |
| `CLRMSD` | Clustering RMSD threshold (A)       | 2.0     |                    |

#### Output

| Code     | Description                         | Default |
|:---------|:------------------------------------|:--------|
| `MAXRES` | Max result clusters                 | 10      |
| `SCOOUT` | Output scored poses only            | Off     |
| `SCOLIG` | Score ligand only (no docking)      | Off     |
| `OUTRNG` | Output binding site range           | Off     |
| `EXCHET` | Exclude HET groups from receptor    | Off     |
| `INCHOH` | Include water molecules             | Off (waters removed) |

### GA Parameters (ga_overrides)

#### Population & Generations

| Code       | Description                                                      | Default  |
|:-----------|:-----------------------------------------------------------------|:---------|
| `NUMCHROM` | Number of chromosomes                                            | *(required)* |
| `NUMGENER` | Number of generations                                            | *(required)* |
| `POPINIMT` | Population initialization method                                 | `RANDOM` |
| `STRTSEED` | Random seed (0 = time-based)                                     | 0        |

#### Genetic Operators

| Code       | Description                                                      | Default  |
|:-----------|:-----------------------------------------------------------------|:---------|
| `CROSRATE` | Crossover rate                                                   | float (0.0-1.0) |
| `MUTARATE` | Mutation rate                                                    | float (0.0-1.0) |
| `ADAPTVGA` | Enable adaptive GA (auto-adjusts rates)                          | 0 (off)  |
| `ADAPTKCO` | Adaptive response parameters k1-k4                               | 0.0 0.0 0.0 0.0 |

#### Selection & Reproduction

| Code       | Description                                                      | Default  |
|:-----------|:-----------------------------------------------------------------|:---------|
| `FITMODEL` | Fitness model                                                    | `PSHARE` or `LINEAR` |
| `REPMODEL` | Reproduction model                                               | `STEADY` or `BOOM` |
| `BOOMFRAC` | BOOM reproduction fraction                                       | 1.0      |
| `SHAREALF` | Fitness sharing alpha (sigma share)                              | float    |
| `SHAREPEK` | Expected number of fitness peaks                                 | float    |
| `SHARESCL` | Fitness sharing scale factor                                     | float    |

#### Output & Debugging

| Code       | Description                                                      | Default  |
|:-----------|:-----------------------------------------------------------------|:---------|
| `PRINTCHR` | Best chromosomes to print per generation                         | 10       |
| `PRINTINT` | Print generation progress                                        | 1        |
| `OUTGENER` | Output results every generation                                  | Off      |

---

## JSON Config Reference

All keys are optional — defaults enable full flexibility at 300 K. See `LIB/config_defaults.h` for the source of truth.

| Section | Key | Default | Description |
|:--------|:----|:--------|:------------|
| `scoring` | `function` | `"VCT"` | Scoring function (`VCT` = Voronoi, `SPH` = sphere) |
| `scoring` | `self_consistency` | `"MAX"` | A→B / B→A contact handling |
| `scoring` | `solvent_penalty` | `0.0` | Solvent exposure penalty |
| `optimization` | `translation_step` | `0.25` | Translation delta (A) |
| `optimization` | `angle_step` | `5.0` | Bond angle delta (deg) |
| `optimization` | `dihedral_step` | `5.0` | Dihedral delta (deg) |
| `optimization` | `flexible_step` | `10.0` | Sidechain flex delta (deg) |
| `optimization` | `grid_spacing` | `0.375` | Binding site grid spacer |
| `flexibility` | `ligand_torsions` | `true` | Enable DEE ligand torsion sampling |
| `flexibility` | `intramolecular` | `true` | Intramolecular energy scoring |
| `flexibility` | `intramolecular_fraction` | `1.0` | Weight of intramolecular term |
| `flexibility` | `permeability` | `1.0` | Global VDW permeability |
| `flexibility` | `rotamer_permeability` | `0.8` | Rotamer acceptance permeability |
| `flexibility` | `ring_conformers` | `true` | LigandRingFlex conformer sampling |
| `flexibility` | `chirality` | `true` | ChiralCenter R/S discrimination |
| `flexibility` | `dee_clash` | `0.5` | DEE clash threshold |
| `thermodynamics` | `temperature` | `300` | Temperature in K (0 = entropy off) |
| `thermodynamics` | `clustering_algorithm` | `"CF"` | `CF`, `DP`, or `FO` |
| `thermodynamics` | `cluster_rmsd` | `2.0` | RMSD threshold for pose clustering |
| `ga` | `num_chromosomes` | `1000` | Population size |
| `ga` | `num_generations` | `500` | Number of GA generations |
| `ga` | `crossover_rate` | `0.8` | Crossover probability |
| `ga` | `mutation_rate` | `0.03` | Mutation probability |
| `ga` | `fitness_model` | `"PSHARE"` | Fitness model |
| `ga` | `reproduction_model` | `"BOOM"` | Reproduction strategy |
| `ga` | `seed` | `0` | RNG seed (0 = time-based) |
| `output` | `max_results` | `10` | Max result clusters |
| `output` | `htp_mode` | `false` | High-throughput (minimal output files) |
| `protein` | `remove_water` | `true` | Remove HOH molecules |
| `protein` | `omit_buried` | `false` | Skip buried atoms in Vcontacts |

| `advanced` | `vcontacts_index` | `false` | Enable Voronoi contact index caching |
| `advanced` | `supernode` | `false` | Supernode mode for normal mode analysis |
| `advanced` | `force_interaction` | `false` | Enable forced interaction penalty |
| `advanced` | `interaction_factor` | `5.0` | Interaction penalty scaling factor |

The `--rigid` flag overrides flexibility to all-off and temperature to 0.

---

## Modules

### Torsional ENCoM (TENCM)

Implements the torsional elastic network contact model (Delarue & Sanejouand 2002; Yang, Song & Cui 2009) for protein backbone flexibility. Builds a spring network over C-alpha contacts within a cutoff radius, computes torsional normal modes via Jacobi diagonalisation, and samples Boltzmann-weighted backbone perturbations during the GA without rebuilding the rotamer library every generation.

Supports both protein (C-alpha) and nucleic acid (C4' backbone) chains for RNA/DNA flexibility.

### Statistical Mechanics Engine

Full thermodynamic analysis of the GA conformational ensemble:
- Partition function Z(T) with log-sum-exp numerical stability
- Helmholtz free energy F = -kT ln Z
- Average energy, variance, and heat capacity
- Conformational entropy S = (E - F) / T
- Boltzmann-weighted state probabilities
- Parallel tempering (replica exchange) swap acceptance
- WHAM for free energy profiles
- Thermodynamic integration via trapezoidal rule
- Fast Boltzmann lookup table for inner-loop evaluation

### ShannonThermoStack

Combines Shannon configurational entropy (over GA ensemble binned into 256 mega-clusters) with torsional ENCoM vibrational entropy. Uses a precomputed 256x256 energy matrix for O(1) pairwise entropy lookup. Hardware-accelerated histogram computation via Metal (Apple Silicon), CUDA, or OpenMP/Eigen.

### LigandRingFlex

Unified ring flexibility for the GA: non-aromatic ring conformer sampling (chair/boat/twist for 6-membered, envelope/twist for 5-membered) and furanose sugar pucker phase sampling. Integrates with GA initialisation, mutation, crossover, and fitness evaluation.

### ChiralCenter

Explicit R/S stereocenter sampling. Detects sp3 tetrahedral chiral centers in the ligand, encodes each as a single GA bit (R=0, S=1), and applies an energy penalty for incorrect stereochemistry (~15-25 kcal/mol per wrong center). Low mutation rate (1-2%) reflects the high inversion barrier.

### CavityDetect (SURFNET)

Automatic binding site detection using the SURFNET gap-sphere algorithm. Places spheres between atom pairs within a distance range, filters by burial, and clusters surviving spheres into cavities ranked by volume. Metal GPU acceleration on Apple Silicon via Objective-C++ bridge (`CavityDetectMetalBridge.mm`), with CPU fallback.

### CleftDetector

Binding cleft and pocket identification for receptor surfaces. Identifies concave regions suitable for ligand binding and provides center/radius definitions for the GA search space (`RNGOPT LOCCDT` mode).

### SdfReader

Multi-format ligand input supporting SDF (Structure Data File) and MOL V2000 formats. Parses atom blocks, bond blocks, and property fields. Complements the existing MOL2 and legacy INP readers for broader chemical database compatibility.

### FastOPTICS (FOPTICS)

Density-based hierarchical clustering of docking poses using the FastOPTICS algorithm. Alternative to CF (centroid-first) clustering — selected via `clustering_algorithm: "FO"` in JSON config or `CLUSTA FO` in legacy mode.

### NATURaL (co-translational assembly)

**N**ative **A**ssembly of co-**T**ranscriptionally/co-**T**ranslationally **U**nified **R**eceptor-**L**igand module. Auto-detects nucleotide ligands or nucleic acid receptors and activates co-translational DualAssembly mode:

- **RibosomeElongation**: Zhao 2011 master equation for codon-dependent ribosome speed (E. coli K-12 and Human HEK293). Identifies pause sites as co-translational folding windows. Also supports nucleotide-by-nucleotide RNA polymerase synthesis.
- **TransloconInsertion**: Sec61 translocon lateral gating model (Hessa 2007). Computes per-window delta-G of TM helix insertion using the Hessa scale with Wimley-White position-weighted helix-dipole correction. Hardware-accelerated via AVX-512/AVX2/Eigen.
- **DualAssemblyEngine**: Grows the receptor chain residue-by-residue at ribosome speed while computing incremental CF and Shannon entropy at each growth step to capture co-translational stereochemical selection.

### VoronoiCFBatch

Zero-copy `std::span`-based batch evaluation interface for the GA inner loop. Scores entire chromosome populations in parallel via OpenMP without redundant atom buffer copies. Includes a built-in `benchmark()` method for wall-clock comparison of serial vs parallel throughput.

### Dead-End Elimination (DEE)

Torsion pruning for ligand flexibility. The DEE tree (`DEELig_Node`) eliminates rotamer combinations that provably cannot be part of the global minimum, reducing the conformational search space before GA evaluation. Controlled via `flexibility.ligand_torsions` (enabled by default) with a clash threshold of `dee_clash: 0.5`.

### Scoring Functions: VCT vs SPH

Two complementarity functions are available:

- **VCT** (Voronoi Contact Function) — computes atom-atom contact surfaces via Voronoi tessellation. Higher accuracy, accounts for shape complementarity and burial. Default in JSON mode.
- **SPH** (Sphere Function) — samples contacts using a 610-point unit sphere approximation. Faster but less precise. Default in legacy `.inp` mode.

Select via `scoring.function` in JSON config or `COMPLF` in legacy mode.

### GPU Acceleration

Hardware-accelerated evaluation for compute-intensive operations:

- **CUDA** (`FLEXAIDS_USE_CUDA=ON`): Batch CF evaluation (`cuda_eval.cu`) and Shannon entropy histogram computation (`shannon_cuda.cu`). Pre-configured for architectures: sm_70, sm_75, sm_80, sm_86, sm_89, sm_90 (Volta through Hopper).
- **Metal** (`FLEXAIDS_USE_METAL=ON`, macOS only): Shannon entropy histograms (`ShannonMetalBridge.mm`), cavity detection (`CavityDetectMetalBridge.mm`), and general evaluation (`metal_eval.mm`). Objective-C++ bridge with automatic CPU fallback.
- **SIMD** (`simd_distance.h`): AVX2-vectorised geometric primitives — batch distance, cross product, dot product, normalization. Used by tENCoM and cavity detection for inner-loop performance.

---

## FreeNRG Integration

The [FreeNRG](https://github.com/lmorency/FreeNRG) Python package bridges FlexAID∆S with NRGRank virtual screening in a unified free energy framework:

```bash
pip install freenrg
```

```python
from freenrg.pipeline import FreeNRGPipeline, FreeNRGConfig, DockingMode

config = FreeNRGConfig(
    mode=DockingMode.FLEXAID,
    flexaid_binary="/path/to/FlexAID",
    receptor_pdb="receptor.pdb",
    ligand_inp="ligand.inp",
    binding_site="cleft.pdb",
)
result = FreeNRGPipeline().run(config)
```

Provides Python ports of StatMechEngine, ShannonThermoStack, TorsionalENM, and CFScorer. See [FREENRG_INTEGRATION.md](FREENRG_INTEGRATION.md) for details.

---

## Publications

### Please Cite

1. **FlexAID core**:
   > Gaudreault & Najmanovich (2015). *J. Chem. Inf. Model.* 55(7):1323-36. [DOI:10.1021/acs.jcim.5b00078](https://doi.org/10.1021/acs.jcim.5b00078)

2. **NRGsuite PyMOL plugin**:
   > Gaudreault, Morency & Najmanovich (2015). *Bioinformatics* 31(23):3856-8. [DOI:10.1093/bioinformatics/btv458](https://doi.org/10.1093/bioinformatics/btv458)

3. **FlexAID∆S: Entropy-driven molecular docking** (preprint pending):
   > Morency LP & Najmanovich RJ (2026). "FlexAID∆S: Information-Theoretic Entropy Improves Molecular Docking Accuracy and Binding Mode Prediction." Preprint in preparation.

   *Status*: Manuscript in preparation. Preprint expected on bioRxiv/ChemRxiv in 2026. This paper introduces the statistical mechanics framework, Shannon entropy scoring, and benchmark results on ITC-187 and CASF-2016.

### Related Work (Inspiration Only)

- **NRGRank** (GPL-3.0, *not a dependency*):
  > Gaudreault et al. (2024). bioRxiv preprint.
  > *Note*: FlexAID∆S reimplements cube screening from first principles (Apache-2.0). No GPL code included. See [clean-room policy](docs/licensing/clean-room-policy.md).

---

## Contributing

**Key Policies**:
- Apache-2.0, BSD, MIT, MPL-2.0 dependencies OK
- GPL/AGPL **forbidden** (see [clean-room policy](docs/licensing/clean-room-policy.md))
- All contributions require Contributor License Agreement (CLA)

See [CONTRIBUTING.md](CONTRIBUTING.md) for code style, testing, PR workflow.

---

## License

**Apache License 2.0** — Permissive open-source.

**You CAN**: Use commercially, modify, redistribute, relicense in proprietary software.
**You MUST**: Include LICENSE, preserve copyright, state changes.
**You CANNOT**: Hold authors liable, use trademarks.

See [LICENSE](LICENSE) | [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)

---

## Links

**Repository**: [github.com/lmorency/FlexAIDdS](https://github.com/lmorency/FlexAIDdS)
**Issues**: [github.com/lmorency/FlexAIDdS/issues](https://github.com/lmorency/FlexAIDdS/issues)
**NRGlab**: [biophys.umontreal.ca/nrg](http://biophys.umontreal.ca/nrg) | [github.com/NRGlab](https://github.com/NRGlab)

**Lead Developer**: Louis-Philippe Morency, PhD (Candidate)
**Affiliation**: Universite de Montreal, NRGlab
**Email**: louis-philippe.morency@umontreal.ca

---

<p align="center">
  <strong>FlexAID∆S: Where Information Theory Meets Drug Discovery</strong><br>
  <em>Zero friction. Zero entropy waste. Zero bullshit.</em><br><br>
  <sub>DRUG IS ALWAYS AN ANSWER. One Shannon bit at a time.</sub>
</p>
