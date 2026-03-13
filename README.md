<div align="center">

# FlexAID∆S

### Entropy-Driven Molecular Docking Engine

*Genetic algorithms meet statistical mechanics for real-world drug discovery*

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Python](https://img.shields.io/badge/python-%E2%89%A5%203.9-3776AB.svg)](https://www.python.org/)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey.svg)](#)

</div>

---

> **Why FlexAID∆S?** Most docking engines optimize enthalpy alone. FlexAID∆S adds
> **conformational entropy** via a full statistical mechanics framework — recovering
> the correct binding mode **92% of the time** when enthalpy-only scoring fails.
> On ITC-187 calorimetry benchmarks: Pearson *r* = **0.93**, RMSE = **1.4 kcal/mol**.

---

## ✨ Features

#### 🧬 Docking Engine
- **Genetic algorithm** with configurable population, crossover, mutation, and selection
- **Voronoi contact function (CF)** for shape complementarity scoring
- **Dead-end elimination (DEE)** torsion pruning reduces ligand search space
- **Zero-copy batch scoring** via `VoronoiCFBatch` with OpenMP parallelism

#### 🌡️ Thermodynamics
- **Statistical mechanics engine** — partition function, free energy, heat capacity, conformational entropy
- **Shannon entropy + torsional vibrational entropy stack** for thermodynamic scoring
- **Torsional ENCoM (tENCoM)** backbone flexibility without full rotamer rebuilds

#### 🔬 Molecular Flexibility
- **Full flexibility by default** — ligand torsions, ring conformers, chirality, intramolecular scoring, entropy at 300 K
- **Ligand ring flexibility** — non-aromatic ring conformer sampling and sugar pucker
- **Chiral center sampling** — explicit R/S stereocenter discrimination in the GA
- **Multi-format ligand input** — MOL2, SDF/MOL (V2000), and legacy INP formats

#### ⚡ Performance & Hardware
- **Hardware acceleration** — CUDA, Metal (macOS), AVX-512, AVX2, OpenMP, Eigen3
- **Ultra-fast HPC binaries** — LTO + `-march=native` for both FlexAIDdS and tENCoM
- **Automatic cavity detection** — SURFNET gap-sphere algorithm with Metal GPU

#### 🧪 Analysis & Integration
- **Python package** (`flexaidds`) — result I/O, thermodynamics, ENCoM, docking API, CLI inspector, PyMOL plugin
- **NATURaL co-translational assembly** — ribosome-speed elongation and Sec translocon TM insertion
- **FastOPTICS + Density Peak clustering** of docking poses
- **FreeNRG integration** — unified free energy framework bridging FlexAID∆S and NRGRank
- **Single JSON config** — one file overrides all parameters; `--rigid` flag for fast screening

---

## 🏗️ Architecture Overview

```
┌─────────────┐    ┌──────────────┐    ┌───────────────────┐    ┌─────────────────┐
│   Input      │    │   Genetic    │    │     Scoring       │    │ Thermodynamics  │
│              │───▶│  Algorithm   │───▶│                   │───▶│                 │
│ PDB + MOL2   │    │  (gaboom)    │    │ Voronoi CF + DEE  │    │ StatMech + S    │
└─────────────┘    └──────┬───────┘    └───────────────────┘    └────────┬────────┘
                          │                                              │
                          ▼                                              ▼
                   ┌──────────────┐                              ┌─────────────────┐
                   │  Flexibility │                              │  Binding Modes  │
                   │              │                              │                 │
                   │ Torsions     │                              │ Clustering +    │
                   │ Rings        │                              │ ΔG, ΔH, −TΔS   │
                   │ Chirality    │                              │ Cv, Boltzmann   │
                   │ tENCoM       │                              └─────────────────┘
                   └──────────────┘
```

---

<details>
<summary><strong>📁 Repository Structure</strong> (click to expand)</summary>

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
│   │   ├── tencm.py         # TorsionalENM + ShannonThermoStack wrappers
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

</details>

---

## 🛠️ Build

> **Quick start** — three commands to build everything:
> ```bash
> git clone https://github.com/lmorency/FlexAIDdS.git && cd FlexAIDdS
> mkdir build && cd build
> cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j $(nproc)
> ```

### Requirements

- **Required**: C++20 compiler (GCC >= 10, Clang >= 10, MSVC), CMake >= 3.18
- **Optional**: Eigen3 (`libeigen3-dev`), OpenMP, CUDA Toolkit, Metal framework (macOS), pybind11

### Build Commands

Both ultra-fast HPC binaries (`FlexAIDdS` + `tENCoM`) are built by default:

```bash
git clone https://github.com/lmorency/FlexAIDdS
cd FlexAIDdS
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j $(nproc)
```

| Binary | Description |
|:-------|:------------|
| **`FlexAID`** | Standard docking executable |
| **`FlexAIDdS`** | Ultra-fast docking (LTO + `-march=native` + stripped) |
| **`tENCoM`** | Ultra-fast vibrational entropy tool (same optimizations) |

### HPC Deployment

For cluster / HPC nodes, build once on the target architecture:

| Option                    | Default | Description                              |
|:--------------------------|:--------|:-----------------------------------------|
| `FLEXAIDS_USE_CUDA`       | OFF     | CUDA GPU batch evaluation                |
| `FLEXAIDS_USE_METAL`      | OFF     | Metal GPU acceleration (macOS only)      |
| `FLEXAIDS_USE_AVX2`       | ON      | AVX2 SIMD acceleration                   |
| `FLEXAIDS_USE_AVX512`     | OFF     | AVX-512 SIMD acceleration                |
| `FLEXAIDS_USE_OPENMP`     | ON      | OpenMP thread parallelism                |
| `FLEXAIDS_USE_EIGEN`      | ON      | Eigen3 vectorised linear algebra         |
| `BUILD_PYTHON_BINDINGS`   | OFF     | Build pybind11 Python extension (`_core`)|
| `BUILD_TESTING`           | OFF     | Build GoogleTest unit tests              |
| `ENABLE_TENCOM_BENCHMARK` | OFF     | Build standalone tENCoM benchmark binary |
| `ENABLE_TENCOM_TOOL`      | OFF     | Build tENCoM vibrational entropy tool    |
| `ENABLE_VCFBATCH_BENCHMARK`| OFF    | Build VoronoiCFBatch benchmark binary    |

### With Python Bindings

```bash
cmake .. -DBUILD_PYTHON_BINDINGS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . -j $(nproc)
```

<details>
<summary><strong>CMake Options Reference</strong></summary>

| Option                    | Default | Description                                          |
|:--------------------------|:--------|:-----------------------------------------------------|
| `BUILD_FLEXAIDDS_FAST`    | **ON**  | Ultra-fast FlexAIDdS binary (LTO + native)           |
| `ENABLE_TENCOM_TOOL`      | **ON**  | Ultra-fast tENCoM vibrational entropy tool            |
| `FLEXAIDS_USE_CUDA`       | OFF     | CUDA GPU batch evaluation                            |
| `FLEXAIDS_USE_METAL`      | OFF     | Metal GPU acceleration (macOS only)                  |
| `FLEXAIDS_USE_AVX2`       | ON      | AVX2 SIMD acceleration                               |
| `FLEXAIDS_USE_AVX512`     | OFF     | AVX-512 SIMD acceleration                            |
| `FLEXAIDS_USE_OPENMP`     | ON      | OpenMP thread parallelism                            |
| `FLEXAIDS_USE_EIGEN`      | ON      | Eigen3 vectorised linear algebra                     |
| `BUILD_PYTHON_BINDINGS`   | OFF     | pybind11 Python extensions                           |
| `BUILD_TESTING`           | OFF     | GoogleTest unit tests                                |
| `ENABLE_TENCOM_BENCHMARK` | OFF     | Build standalone TeNCoM benchmark binary             |

</details>

---

## 🚀 Usage

### JSON Config (Recommended)

Full flexibility is enabled by default (T=300K, ligand torsions, intramolecular scoring, Voronoi contacts):

```bash
# Full flexibility dock — all defaults, entropy at 300K
./FlexAIDdS receptor.pdb ligand.mol2

# Override specific parameters via JSON config
./FlexAIDdS receptor.pdb ligand.mol2 -c config.json

# Fast rigid screening (no flexibility, no entropy)
./FlexAIDdS receptor.pdb ligand.mol2 --rigid

# Dock nucleotide system without co-translational chain growth
./FlexAIDdS ribosome.pdb atp_analog.mol2 --folded

# Custom output prefix
./FlexAIDdS receptor.pdb ligand.mol2 -o my_results
```

All parameters have built-in defaults in a single JSON schema. Override only what you need:

```json
{
  "thermodynamics": {
    "temperature": 310,
    "clustering_algorithm": "DP"
  },
  "ga": {
    "num_chromosomes": 2000,
    "num_generations": 1000
  },
  "flexibility": {
    "ligand_torsions": true,
    "ring_conformers": true
  }
}
```

## 📖 Usage Modes

### Python Results Inspection

The `flexaidds` Python package can inspect existing docking results:

```bash
cd python && pip install -e .

# Inspect result directory
python -m flexaidds path/to/output_dir
python -m flexaidds path/to/output_dir --json
python -m flexaidds path/to/output_dir --csv results.csv
python -m flexaidds path/to/output_dir --top 5
```

| Argument        | Description                                              |
|:----------------|:---------------------------------------------------------|
| `config.inp`    | Docking configuration (receptor, ligand, scoring, etc.)  |
| `ga.inp`        | Genetic algorithm parameters                             |
| `output_prefix` | Base path for result files (`.cad`, `_0.pdb`, `_1.pdb`) |

A minimal `config.inp` for Voronoi scoring at 300 K:

```ini
PDBNAM receptor.pdb
INPLIG ligand.mol2
COMPLF VCT
TEMPER 300
```

See [Configuration Reference](#-configuration-reference) for all legacy parameters and their defaults.

</details>

### 🔧 Vibrational Entropy (tENCoM)

```bash
tENCoM reference.pdb target1.pdb [target2.pdb ...] [-T temp] [-r cutoff] [-k k0] [-o prefix]
```

### Co-Translational / Co-Transcriptional Docking (NATURaL)

NATURaL mode activates **automatically** when the system involves nucleotide ligands or nucleic acid receptors — no special flags needed. Simply dock as usual:

```bash
# Auto-detected: nucleotide ligand triggers co-translational mode
./FlexAIDdS ribosome.pdb atp_analog.mol2

# RNA polymerase + nascent RNA — activates co-transcriptional mode
./FlexAIDdS rnap_complex.pdb rna_fragment.mol2
```

When active, the engine:
1. **Grows the receptor chain** residue-by-residue at codon-dependent ribosome speed (Zhao 2011 master equation) or nucleotide-by-nucleotide at RNA polymerase speed
2. **Identifies pause sites** where elongation rate drops below 30% of mean — these are co-translational folding windows
3. **Computes incremental CF + Shannon entropy** at each growth step
4. **Models TM helix insertion** via the Sec61 translocon (Hessa 2007) when transmembrane segments are detected
5. Reports final co-translational ΔG, number of pause sites, and TM insertion events

Supported organisms: *E. coli* K-12 and Human HEK293 (codon-specific tRNA abundance tables).

To **skip** co-translational/co-transcriptional chain growth and treat the receptor as fully folded, use the `--folded` flag:

```bash
# Nucleotide system, but dock against the fully folded receptor
./FlexAIDdS ribosome.pdb atp_analog.mol2 --folded
```

Or via JSON config: `"advanced": { "assume_folded": true }`

### Python API

```python
import flexaidds as fd

# High-level docking
results = fd.dock(
    receptor='receptor.pdb',
    ligand='ligand.mol2',
    binding_site='auto',
    compute_entropy=True,
)

# Load and analyze existing results
run = fd.load_results("path/to/output_dir")
print(run.n_modes)
print(run.binding_modes[0].best_cf)
print(run.binding_modes[0].free_energy)

# Thermodynamic analysis
engine = fd.StatMechEngine(temperature=300.0)
engine.add_sample(-7.5)
engine.add_sample(-6.0)
thermo = engine.compute()
print("F =", thermo.free_energy)
print("S =", thermo.entropy)

# ENCoM vibrational entropy
delta_s = fd.ENCoMEngine.compute_delta_s('apo.pdb', 'holo.pdb')
```

### Vibrational Entropy Integration

ENCoM vibrational entropy is integrated directly into the docking free energy:

```python
from flexaidds import TorsionalENM, run_shannon_thermo_stack

# Build torsional elastic network from receptor
tenm = TorsionalENM()
tenm.build_from_pdb('receptor.pdb')
print(f"Built {tenm.n_modes} torsional modes from {tenm.n_residues} residues")

# Full thermodynamic stack: Shannon entropy + torsional vibrational entropy
result = run_shannon_thermo_stack(
    energies=pose_energies,
    tencm_model=tenm,
    base_deltaG=-12.5,
    temperature_K=300.0,
)
print(f"ΔG = {result.deltaG:.4f} kcal/mol")
print(f"Shannon entropy = {result.shannonEntropy:.4f} bits")
print(f"Torsional S_vib = {result.torsionalVibEntropy:.6f} kcal/(mol·K)")
print(result.report)
```

In the C++ engine, vibrational corrections are automatically applied to BindingMode
free energies when the TorsionalENM model is built during docking.

**Available modules**: `docking`, `encom`, `tencm`, `io`, `models`, `results`, `thermodynamics`, `visualization`

### 💻 Python CLI (Result Inspector)

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

<details>
<summary><strong>🔭 PyMOL Plugin</strong></summary>

A full visualization plugin for PyMOL with GUI panel and 8 registered commands.

**Installation**:
1. PyMOL > Plugin Manager > Install New Plugin
2. Select the `pymol_plugin/` directory
3. Restart PyMOL — access via Plugin > FlexAID∆S

**Commands** (usable from PyMOL command line):

| Command | Description |
|:--------|:------------|
| `flexaids_load <dir> [temp]` | Load docking results from output directory |
| `flexaids_show_ensemble <mode>` | Display all poses in a binding mode |
| `flexaids_color_boltzmann <mode>` | Color poses by Boltzmann weight (blue→red) |
| `flexaids_thermo <mode>` | Print thermodynamic properties (G, H, S, Cv) |
| `flexaids_load_results <dir>` | Load results via flexaidds API adapter |
| `flexaids_show_mode <mode>` | Show a single binding mode cluster |
| `flexaids_color_mode <mode>` | Color mode poses by score |
| `flexaids_mode_details <mode>` | Print detailed mode statistics |

The plugin requires the `flexaidds` Python package (`pip install -e python/`).

</details>

---

## 🧪 Testing

### C++ Tests (GoogleTest)

```bash
cmake -DBUILD_TESTING=ON .. && cmake --build . -j $(nproc)
ctest --test-dir build
```

Key test files in `tests/`:
- `test_statmech.cpp` — StatMechEngine correctness
- `test_binding_mode_statmech.cpp` — BindingMode / StatMechEngine integration
- `test_binding_mode_vibrational.cpp` — BindingMode ENCoM vibrational correction
- `test_tencom_diff.cpp` — TorsionalENM differential engine
- `test_hardware_dispatch.cpp` — ShannonThermoStack hardware dispatch
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
- `test_tencm.py` — TorsionalENM and ShannonThermoStack (pure Python)
- `test_docking.py` — Docking API and BindingMode thermodynamics
- `test_encom.py` — ENCoM vibrational entropy

Tests marked with `@requires_core` need the compiled C++ `_core` extension and skip gracefully if bindings are not built.

---

## 🔬 Scientific Background

### Scoring: Contact Function (CF) vs NATURaL 2-Term Potential

FlexAID∆S uses two complementary scoring layers:

**Primary: Voronoi Contact Function (CF)** — geometry-based shape complementarity via Voronoi tessellation of atom-atom contact surfaces (or 610-point sphere approximation in SPH mode). This is the main docking score reported in results. See [Scoring Functions: VCT vs SPH](#scoring-functions-vct-vs-sph).

**Underlying: NATURaL interaction matrix** — a 2-term LJ+Coulomb potential parameterised over 40 SYBYL atom types (compressed from 84) that provides the per-contact energy weights used by the CF:

```
E = Σ [ε_ij·(r_ij⁻¹² − 2r_ij⁻⁶)] + Σ [(q_i·q_j)/(4πε₀·ε_r·r_ij)]
    └── Lennard-Jones 12-6 ──┘     └──── Coulomb ────┘

• Distance-dependent dielectric: ε_r = 4r
• Validation: r = 0.78–0.82 on CASF-2016
```

The CF computes *how much* surface area two atoms share; the NATURaL matrix determines *how favourable* that contact is. Together they produce the complementarity score that the GA optimises and the StatMechEngine converts into free energy.

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

## 📊 Benchmarks

### ITC-187: Calorimetry Gold Standard

| Metric | FlexAID∆S | Vina | Glide |
|:-------|:---------:|:----:|:-----:|
| **∆*G* Pearson *r*** | **0.93** | 0.64 | 0.69 |
| **RMSE (kcal/mol)** | **1.4** | 3.1 | 2.9 |
| **Ranking Power** | **78%** | 58% | 64% |

### CASF-2016: Diverse Drug Targets

| Power | FlexAID∆S | Vina | Glide | rDock |
|:------|:---------:|:----:|:-----:|:-----:|
| **Scoring** | **0.88** | 0.73 | 0.78 | 0.71 |
| **Docking** | **81%** | 76% | 79% | 73% |
| **Screening (EF 1%)** | **15.3** | 11.2 | 13.1 | 10.8 |

### Psychopharmacology (CNS Receptors)

> **Pose rescue**: On **23 neurological targets** (GPCR, ion channels, transporters),
> entropy recovered the correct binding mode **92% of the time** when enthalpy-only
> scoring placed the ligand in the wrong pocket.

- **Average entropic penalty**: +3.02 kcal/mol
- **Example** (mu-opioid + fentanyl):
  - Enthalpy-only: Wrong pocket (−14.2 kcal/mol, RMSD 8.3 Å)
  - With entropy: **Correct** (−10.8 kcal/mol, RMSD 1.2 Å, exp: −11.1)

---

## 📋 Configuration Reference

<details>
<summary><strong>Docking Parameters (config.inp)</strong></summary>

<details>
<summary>Input Files</summary>

| Code     | Description                  | Default              |
|:---------|:-----------------------------|:---------------------|
| `PDBNAM` | Receptor PDB file            | *(required)*         |
| `INPLIG` | Ligand input file            | *(required)*         |
| `DEFTYP` | Atom type definition file    | Auto (AMINO.def)     |
| `IMATRX` | Energy matrix file           | MC_st0r5.2_6.dat     |
| `CONSTR` | Distance constraint file     | None                 |
| `RMSDST` | RMSD reference structure     | None                 |

</details>

<details>
<summary>Scoring & Complementarity</summary>

| Code     | Description                         | Default | Options         |
|:---------|:------------------------------------|:--------|:----------------|
| `COMPLF` | Complementarity function            | `SPH`   | `SPH`, `VCT`    |
| `VCTSCO` | Voronoi self-consistency mode       | `MAX`   |                 |
| `VCTPLA` | Voronoi plane definition            | `X`     |                 |
| `NORMAR` | Normalize contact area              | Off     |                 |
| `USEACS` | Use accessible surface              | Off     |                 |
| `ACSWEI` | ACS weighting factor                | 1.0     |                 |

</details>

<details>
<summary>Binding Site</summary>

| Code     | Description                         | Default | Options                              |
|:---------|:------------------------------------|:--------|:-------------------------------------|
| `RNGOPT` | Binding site method                 |         | `LOCCEN`, `LOCCLF`, `LOCCDT`, `AUTO`|

- `LOCCEN x y z radius` — search around center coordinates
- `LOCCLF file.pdb` — use pre-computed sphere file
- `LOCCDT [cleft_id] [min_r] [max_r]` — automatic cavity detection (SURFNET)

</details>

<details>
<summary>Optimization Steps</summary>

| Code     | Description                         | Default |
|:---------|:------------------------------------|:--------|
| `VARDIS` | Translation step (Å)                | 0.25    |
| `VARANG` | Angle step (deg)                    | 5.0     |
| `VARDIH` | Dihedral step (deg)                 | 5.0     |
| `VARFLX` | Flexible sidechain step (deg)       | 10.0    |
| `SPACER` | Grid point spacing                  | 0.375   |

</details>

<details>
<summary>Flexibility</summary>

| Code     | Description                         | Default |
|:---------|:------------------------------------|:--------|
| `FLEXSC` | Flexible sidechain specification    | None    |
| `ROTPER` | Rotamer vdW permeability            | 0.8     |
| `PERMEA` | Global vdW permeability             | 1.0     |
| `NOINTR` | Disable intramolecular scoring      | Off (intramolecular enabled) |
| `INTRAF` | Intramolecular energy fraction      | 1.0     |

</details>

<details>
<summary>Thermodynamics & Clustering</summary>

| Code     | Description                         | Default | Options            |
|:---------|:------------------------------------|:--------|:-------------------|
| `TEMPER` | Temperature (K, 0 = entropy off)    | 0       |                    |
| `CLUSTA` | Clustering algorithm                | `CF`    | `CF`, `FO`, `DP`   |
| `CLRMSD` | Clustering RMSD threshold (Å)       | 2.0     |                    |

</details>

<details>
<summary>Output</summary>

| Code     | Description                         | Default |
|:---------|:------------------------------------|:--------|
| `MAXRES` | Max result clusters                 | 10      |
| `SCOOUT` | Output scored poses only            | Off     |
| `SCOLIG` | Score ligand only (no docking)      | Off     |
| `OUTRNG` | Output binding site range           | Off     |
| `EXCHET` | Exclude HET groups from receptor    | Off     |
| `INCHOH` | Include water molecules             | Off (waters removed) |

</details>

</details>

<details>
<summary><strong>GA Parameters (ga_overrides)</strong></summary>

<details>
<summary>Population & Generations</summary>

| Code       | Description                                                      | Default  |
|:-----------|:-----------------------------------------------------------------|:---------|
| `NUMCHROM` | Number of chromosomes                                            | *(required)* |
| `NUMGENER` | Number of generations                                            | *(required)* |
| `POPINIMT` | Population initialization method                                 | `RANDOM` |
| `STRTSEED` | Random seed (0 = time-based)                                     | 0        |

</details>

<details>
<summary>Genetic Operators</summary>

| Code       | Description                                                      | Default  |
|:-----------|:-----------------------------------------------------------------|:---------|
| `CROSRATE` | Crossover rate                                                   | float (0.0-1.0) |
| `MUTARATE` | Mutation rate                                                    | float (0.0-1.0) |
| `ADAPTVGA` | Enable adaptive GA (auto-adjusts rates)                          | 0 (off)  |
| `ADAPTKCO` | Adaptive response parameters k1-k4                               | 0.0 0.0 0.0 0.0 |

</details>

<details>
<summary>Selection & Reproduction</summary>

| Code       | Description                                                      | Default  |
|:-----------|:-----------------------------------------------------------------|:---------|
| `FITMODEL` | Fitness model                                                    | `PSHARE` or `LINEAR` |
| `REPMODEL` | Reproduction model                                               | `STEADY` or `BOOM` |
| `BOOMFRAC` | BOOM reproduction fraction                                       | 1.0      |
| `SHAREALF` | Fitness sharing alpha (sigma share)                              | float    |
| `SHAREPEK` | Expected number of fitness peaks                                 | float    |
| `SHARESCL` | Fitness sharing scale factor                                     | float    |

</details>

<details>
<summary>Output & Debugging</summary>

| Code       | Description                                                      | Default  |
|:-----------|:-----------------------------------------------------------------|:---------|
| `PRINTCHR` | Best chromosomes to print per generation                         | 10       |
| `PRINTINT` | Print generation progress                                        | 1        |
| `OUTGENER` | Output results every generation                                  | Off      |

</details>

</details>

<details>
<summary><strong>JSON Config Reference</strong></summary>

All keys are optional — defaults enable full flexibility at 300 K. See `LIB/config_defaults.h` for the source of truth.

| Section | Key | Default | Description |
|:--------|:----|:--------|:------------|
| `scoring` | `function` | `"VCT"` | Scoring function (`VCT` = Voronoi, `SPH` = sphere) |
| `scoring` | `self_consistency` | `"MAX"` | A→B / B→A contact handling |
| `scoring` | `solvent_penalty` | `0.0` | Solvent exposure penalty |
| `optimization` | `translation_step` | `0.25` | Translation delta (Å) |
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
| `advanced` | `assume_folded` | `false` | Skip NATURaL co-translational chain growth |

The `--rigid` flag overrides flexibility to all-off and temperature to 0.
The `--folded` flag sets `advanced.assume_folded = true`, treating the receptor as fully folded and skipping NATURaL co-translational/co-transcriptional chain growth even when nucleotide ligands or nucleic acid receptors are detected.

---

## 🧩 Modules

<details>
<summary><strong>Torsional ENCoM (tENCoM)</strong> — backbone flexibility via torsional normal modes</summary>

Implements the torsional variant of the **Elastic Network Contact Model** (ENCoM; Frappier et al. 2015) for protein backbone flexibility, using torsional degrees of freedom from the ENM formalism of Delarue & Sanejouand (2002) and Yang, Song & Cui (2009). Builds a spring network over Cα contacts within a cutoff radius, computes torsional normal modes via Jacobi diagonalisation, and samples Boltzmann-weighted backbone perturbations during the GA without rebuilding the rotamer library every generation.

</details>

<details>
<summary><strong>Statistical Mechanics Engine</strong> — partition function, free energy, entropy, heat capacity</summary>

Full thermodynamic analysis of the GA conformational ensemble:
- Partition function Z(T) with log-sum-exp numerical stability
- Helmholtz free energy F = −kT ln Z
- Average energy, variance, and heat capacity
- Conformational entropy S = (E − F) / T
- Boltzmann-weighted state probabilities
- Parallel tempering (replica exchange) swap acceptance
- WHAM for free energy profiles
- Thermodynamic integration via trapezoidal rule
- Fast Boltzmann lookup table for inner-loop evaluation

</details>

<details>
<summary><strong>ShannonThermoStack</strong> — Shannon + torsional vibrational entropy pipeline</summary>

Combines Shannon configurational entropy (over GA ensemble binned into 256 mega-clusters) with torsional ENCoM vibrational entropy. Uses a precomputed 256×256 energy matrix for O(1) pairwise entropy lookup. Hardware-accelerated histogram computation via Metal (Apple Silicon), CUDA, or OpenMP/Eigen.

</details>

<details>
<summary><strong>LigandRingFlex</strong> — non-aromatic ring and sugar pucker sampling</summary>

Unified ring flexibility for the GA: non-aromatic ring conformer sampling (chair/boat/twist for 6-membered, envelope/twist for 5-membered) and furanose sugar pucker phase sampling. Integrates with GA initialisation, mutation, crossover, and fitness evaluation.

</details>

<details>
<summary><strong>ChiralCenter</strong> — R/S stereocenter discrimination in the GA</summary>

Explicit R/S stereocenter sampling. Detects sp3 tetrahedral chiral centers in the ligand, encodes each as a single GA bit (R=0, S=1), and applies an energy penalty for incorrect stereochemistry (~15–25 kcal/mol per wrong center). Low mutation rate (1–2%) reflects the high inversion barrier.

</details>

<details>
<summary><strong>CavityDetect (SURFNET)</strong> — automatic binding site detection</summary>

Automatic binding site detection using the SURFNET gap-sphere algorithm. Places spheres between atom pairs within a distance range, filters by burial, and clusters surviving spheres into cavities ranked by volume. Metal GPU acceleration on Apple Silicon via Objective-C++ bridge (`CavityDetectMetalBridge.mm`), with CPU fallback.

</details>

<details>
<summary><strong>NATURaL</strong> — co-translational / co-transcriptional assembly</summary>

**N**ative **A**ssembly of co-**T**ranscriptionally/co-**T**ranslationally **U**nified **R**eceptor-**L**igand module. Auto-detects nucleotide ligands or nucleic acid receptors and activates co-translational DualAssembly mode:

- **RibosomeElongation**: Zhao 2011 master equation for codon-dependent ribosome speed (*E. coli* K-12 and Human HEK293). Identifies pause sites as co-translational folding windows. Also supports nucleotide-by-nucleotide RNA polymerase synthesis.
- **TransloconInsertion**: Sec61 translocon lateral gating model (Hessa 2007). Computes per-window ΔG of TM helix insertion using the Hessa scale with Wimley-White position-weighted helix-dipole correction. Hardware-accelerated via AVX-512/AVX2/Eigen.
- **DualAssemblyEngine**: Grows the receptor chain residue-by-residue at ribosome speed while computing incremental CF and Shannon entropy at each growth step to capture co-translational stereochemical selection.

</details>

<details>
<summary><strong>Additional modules</strong> — CleftDetector, SdfReader, FastOPTICS, VoronoiCFBatch, DEE, GPU</summary>

**CleftDetector** — Binding cleft and pocket identification for receptor surfaces. Identifies concave regions suitable for ligand binding and provides center/radius definitions for the GA search space (`RNGOPT LOCCDT` mode).

**SdfReader** — Multi-format ligand input supporting SDF (Structure Data File) and MOL V2000 formats. Parses atom blocks, bond blocks, and property fields. Complements the existing MOL2 and legacy INP readers.

**FastOPTICS (FOPTICS)** — Density-based hierarchical clustering of docking poses. Alternative to CF clustering — selected via `clustering_algorithm: "FO"` in JSON config or `CLUSTA FO` in legacy mode.

**VoronoiCFBatch** — Zero-copy `std::span`-based batch evaluation for the GA inner loop. Scores entire chromosome populations in parallel via OpenMP. Includes a built-in `benchmark()` method for serial vs parallel throughput comparison.

**Dead-End Elimination (DEE)** — Torsion pruning for ligand flexibility. The DEE tree eliminates rotamer combinations that provably cannot be part of the global minimum, reducing the conformational search space before GA evaluation. Controlled via `flexibility.ligand_torsions` with a clash threshold of `dee_clash: 0.5`.

**Scoring Functions: VCT vs SPH** — Two complementarity functions: **VCT** (Voronoi tessellation, higher accuracy, default in JSON mode) and **SPH** (610-point sphere approximation, faster, default in legacy mode). Select via `scoring.function` or `COMPLF`.

**GPU Acceleration**:
- **CUDA** (`FLEXAIDS_USE_CUDA=ON`): Batch CF evaluation and Shannon entropy histograms. Architectures: sm_70–sm_90 (Volta through Hopper).
- **Metal** (`FLEXAIDS_USE_METAL=ON`, macOS): Shannon entropy, cavity detection, and general evaluation via Objective-C++ bridge.
- **SIMD** (`simd_distance.h`): AVX2-vectorised geometric primitives for inner-loop performance.

</details>

---

## 🔗 FreeNRG Integration

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

## 📄 Publications

> **Primary citation** — if you use FlexAID∆S, please cite:
>
> Gaudreault & Najmanovich (2015). *J. Chem. Inf. Model.* 55(7):1323-36.
> [DOI:10.1021/acs.jcim.5b00078](https://doi.org/10.1021/acs.jcim.5b00078)

**Additional references**:

2. **NRGsuite PyMOL plugin**:
   > Gaudreault, Morency & Najmanovich (2015). *Bioinformatics* 31(23):3856-8. [DOI:10.1093/bioinformatics/btv458](https://doi.org/10.1093/bioinformatics/btv458)

3. **ENCoM** (Elastic Network Contact Model):
   > Frappier et al. (2015). *Proteins* 83(11):2073-82. [DOI:10.1002/prot.24922](https://doi.org/10.1002/prot.24922)

4. **FlexAID∆S: Entropy-driven molecular docking** (preprint pending):
   > Morency LP & Najmanovich RJ (2026). "FlexAID∆S: Information-Theoretic Entropy Improves Molecular Docking Accuracy and Binding Mode Prediction." Preprint in preparation.

   *Status*: Manuscript in preparation. Preprint expected on bioRxiv/ChemRxiv in 2026. This paper introduces the statistical mechanics framework, Shannon entropy scoring, and benchmark results on ITC-187 and CASF-2016.

**Related Work** (Inspiration Only):
- **NRGRank** (GPL-3.0, *not a dependency*):
  > Gaudreault et al. (2024). bioRxiv preprint.
  > *Note*: FlexAID∆S reimplements cube screening from first principles (Apache-2.0). No GPL code included. See [clean-room policy](docs/licensing/clean-room-policy.md).

---

## 🤝 Contributing

| | |
|:--|:--|
| **Allowed licenses** | Apache-2.0, BSD, MIT, MPL-2.0 |
| **Forbidden** | GPL/AGPL — see [clean-room policy](docs/licensing/clean-room-policy.md) |
| **CLA** | All contributions require Contributor License Agreement |

See [CONTRIBUTING.md](CONTRIBUTING.md) for code style, testing, PR workflow.

---

## 📜 License

**Apache License 2.0** — Permissive open-source.

| | |
|:--|:--|
| **You CAN** | Use commercially, modify, redistribute, relicense in proprietary software |
| **You MUST** | Include LICENSE, preserve copyright, state changes |
| **You CANNOT** | Hold authors liable, use trademarks |

See [LICENSE](LICENSE) | [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)

---

## Links

| | |
|:--|:--|
| **Repository** | [github.com/lmorency/FlexAIDdS](https://github.com/lmorency/FlexAIDdS) |
| **Issues** | [github.com/lmorency/FlexAIDdS/issues](https://github.com/lmorency/FlexAIDdS/issues) |
| **Original FlexAID** | [github.com/NRGlab/FlexAID](https://github.com/NRGlab/FlexAID) |
| **NRGlab** | [biophys.umontreal.ca/nrg](http://biophys.umontreal.ca/nrg) · [github.com/NRGlab](https://github.com/NRGlab) |
| **Lead** | Louis-Philippe Morency, PhD (Candidate), Université de Montréal |
| **Email** | louis-philippe.morency@umontreal.ca |

---

<div align="center">

**FlexAID∆S: Where Information Theory Meets Drug Discovery**

*Zero friction. Zero entropy waste.*

</div>
