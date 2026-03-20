<div align="center">

# FlexAID∆S

**Entropy-Driven Molecular Docking**

*Combining genetic algorithms with statistical mechanics thermodynamics*
*for accurate binding free energy prediction*

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Python](https://img.shields.io/badge/python-%E2%89%A5%203.9-3776AB.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)](#)

</div>

---

FlexAID∆S extends the [FlexAID](https://doi.org/10.1021/acs.jcim.5b00078) docking engine with a full **canonical ensemble thermodynamics** layer. Where conventional docking programs rank poses by enthalpy alone, FlexAID∆S computes the Helmholtz free energy *F* = *H* - *TS* from the partition function over the GA conformational ensemble — accounting for configurational and vibrational entropy contributions that are critical for correct binding mode identification.

**Key capabilities:**
- Genetic algorithm docking with Voronoi contact function scoring
- Canonical ensemble partition function, free energy, entropy, and heat capacity
- Torsional elastic network model (tENCoM) for backbone vibrational entropy
- Full ligand flexibility: torsions, ring conformers, chiral center discrimination
- Co-translational / co-transcriptional assembly (NATURaL module)
- Unified hardware dispatch with automatic backend selection (CUDA, Metal, AVX-512, AVX2, OpenMP)
- Distributed docking across Apple devices (Bonhomme Fleet) with iCloud coordination
- Python package with docking API, result analysis, and PyMOL visualization
- Swift (macOS/iOS) and TypeScript (PWA) packages for cross-platform access

---

## Quick Start

```bash
git clone https://github.com/lmorency/FlexAIDdS.git && cd FlexAIDdS
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j $(nproc)
```

```bash
# Dock with full flexibility and entropy at 300 K (default)
./FlexAIDdS receptor.pdb ligand.mol2

# Override parameters via JSON
./FlexAIDdS receptor.pdb ligand.mol2 -c config.json

# Rigid screening (no flexibility, no entropy)
./FlexAIDdS receptor.pdb ligand.mol2 --rigid
```

```python
import flexaidds

results = flexaidds.dock(
    receptor='receptor.pdb',
    ligand='ligand.mol2',
    compute_entropy=True
)
for mode in results.rank_by_free_energy():
    print(f"Mode: dG={mode.free_energy:.2f} kcal/mol")
```

---

## Architecture

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

**Scoring** uses two complementary layers: a **Voronoi contact function** (CF) for geometry-based shape complementarity, weighted by a **2-term LJ+Coulomb potential** parameterised over 40 SYBYL atom types. The CF computes contact surface area; the interaction matrix determines how favourable each contact is. The StatMechEngine then converts the GA ensemble into thermodynamic quantities via the canonical partition function with log-sum-exp numerical stability.

---

## Features

#### Docking Engine
- **Genetic algorithm** with configurable population, crossover, mutation, and selection
- **Voronoi contact function (CF)** for shape complementarity scoring
- **Dead-end elimination (DEE)** reduces ligand conformational search space
- **Batch evaluation** via `VoronoiCFBatch` with OpenMP parallelism
- **Multiple clustering** methods: centroid-first, FastOPTICS, Density Peak
- **Metal ion and cofactor scoring** — receptor-bound ions (Mg²⁺, Zn²⁺, Ca²⁺, Fe²⁺/³⁺, Cu²⁺, Mn²⁺, Na⁺, K⁺, Cl⁻, Br⁻, and 11 more) receive crystallographic VdW radii and SYBYL atom types; organic cofactors (heme, FAD, ATP analogs) use element-based radii; all automatically participate in Voronoi tessellation and CleftDetector probing
- **Structural water retention** — crystallographic waters with B-factor < 20 Å² are retained by default; their O atoms participate in Voronoi CF scoring as hydrophilic receptor environment atoms, capturing water-mediated H-bonds (~1–3 kcal/mol each) and displacement entropy (~0.4–2 kcal/mol per ordered water released)

#### Thermodynamics
- **Canonical ensemble** — partition function *Z*, Helmholtz free energy *F*, entropy *S*, heat capacity *C*<sub>v</sub>
- **Torsional ENCoM** (tENCoM) — backbone vibrational entropy without full rotamer rebuilds
- **ShannonThermoStack** — combined configurational + vibrational entropy pipeline
- **Thermodynamic integration** and WHAM free energy profiles

#### Molecular Flexibility
- **Full flexibility by default** — ligand torsions, ring conformers, chirality, intramolecular scoring at 300 K
- **Non-aromatic ring sampling** — chair/boat/twist for 6-membered, envelope/twist for 5-membered rings, sugar pucker
- **Chiral center discrimination** — explicit R/S sampling with stereochemical energy penalty
- **Multi-format input** — MOL2, SDF/MOL V2000, and legacy INP

#### Hardware Acceleration
- **Unified hardware dispatch** — automatic backend selection at runtime (CUDA > Metal > AVX-512 > AVX2 > OpenMP > scalar)
- **CUDA** — batch CF evaluation and Shannon entropy histograms (Volta through Hopper)
- **Metal** — Apple Silicon GPU for Shannon entropy, cavity detection, and evaluation
- **SIMD** — AVX-512 and AVX2 vectorised geometric primitives
- **OpenMP + Eigen3** — thread parallelism and vectorised linear algebra
- **LTO binaries** — link-time optimized `FlexAIDdS` and `tENCoM` executables

#### Analysis & Integration
- **Python package** (`flexaidds`) — docking API, result I/O, thermodynamics, CLI inspector, PyMOL plugin
- **Co-translational assembly** (NATURaL) — ribosome-speed chain growth with Sec translocon TM insertion; RNA receptors use differentiated secondary (k ~ 10⁴ s⁻¹) vs. Mg²⁺-dependent tertiary folding rates (Hill equation, K_d = 1 mM, n = 2) with corrected elongation rate of 25 nt/s (mRNA in vivo)
- **Automatic cavity detection** — SURFNET gap-sphere algorithm with Metal GPU support
- **[FreeNRG](https://github.com/lmorency/FreeNRG) integration** — unified free energy framework bridging FlexAID∆S and NRGRank

#### Distributed Docking (Bonhomme Fleet)
- **Fleet scheduler** — distribute docking across Apple devices via iCloud Drive with automatic work-stealing
- **Device-aware scheduling** — battery, thermal state, and TFLOPS-based compute weighting
- **Orphan recovery** — timed-out chunks reclaimed with exponential backoff and priority elevation
- **Encrypted transit** — ChaChaPoly-encrypted work chunks for secure distributed computation
- **Swift package** — native macOS/iOS actors wrapping StatMechEngine and ENCoM (macOS 14+, iOS 17+)
- **Intelligence oracle** — on-device AI analysis via Apple FoundationModels with rule-based fallback
- **TypeScript SDK + PWA** — real-time Fleet dashboard, Mol* 3D viewer, and result inspector

---

## Build

### Requirements

- **Required**: C++20 compiler (GCC >= 10, Clang >= 10, MSVC), CMake >= 3.18
- **Optional**: Eigen3 (`libeigen3-dev`), OpenMP, CUDA Toolkit, Metal framework (macOS), pybind11

### Output Binaries

| Binary | Description |
|:-------|:------------|
| `FlexAID` | Standard docking executable |
| `FlexAIDdS` | Optimized docking (LTO + `-march=native`) |
| `tENCoM` | Vibrational entropy differential tool |

### Build Variants

```bash
# With tests
cmake .. -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . -j $(nproc) && ctest --test-dir .

# With Python bindings
cmake .. -DBUILD_PYTHON_BINDINGS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . -j $(nproc)

# HPC deployment (AVX-512 + OpenMP)
cmake .. -DCMAKE_BUILD_TYPE=Release -DFLEXAIDS_USE_AVX512=ON -DFLEXAIDS_USE_OPENMP=ON
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
| `BUILD_FLEXAIDDS_FAST`    | **ON**  | LTO-optimized FlexAIDdS binary           |
| `ENABLE_TENCOM_TOOL`      | **ON**  | tENCoM vibrational entropy tool          |
| `FLEXAIDS_USE_CUDA`       | OFF     | CUDA GPU acceleration                    |
| `FLEXAIDS_USE_METAL`      | OFF     | Metal GPU acceleration (macOS)           |
| `FLEXAIDS_USE_AVX2`       | ON      | AVX2 SIMD acceleration                   |
| `FLEXAIDS_USE_AVX512`     | OFF     | AVX-512 SIMD acceleration                |
| `FLEXAIDS_USE_OPENMP`     | ON      | OpenMP thread parallelism                |
| `FLEXAIDS_USE_EIGEN`      | ON      | Eigen3 vectorised linear algebra         |
| `FLEXAIDS_USE_256_MATRIX` | ON      | 256×256 soft contact matrix system       |
| `BUILD_PYTHON_BINDINGS`   | OFF     | pybind11 Python extension (`_core`)      |
| `BUILD_TESTING`           | OFF     | GoogleTest unit tests                    |
| `ENABLE_TENCOM_BENCHMARK` | OFF     | Standalone tENCoM benchmark binary       |
| `ENABLE_VCFBATCH_BENCHMARK`| OFF    | VoronoiCFBatch benchmark binary          |

</details>

---

## Usage

### Command Line

```bash
# Full flexibility dock (all defaults, entropy at 300 K)
./FlexAIDdS receptor.pdb ligand.mol2

# JSON config override
./FlexAIDdS receptor.pdb ligand.mol2 -c config.json

# Rigid screening (no flexibility, no entropy)
./FlexAIDdS receptor.pdb ligand.mol2 --rigid

# Skip co-translational chain growth
./FlexAIDdS ribosome.pdb ligand.mol2 --folded

# Custom output prefix
./FlexAIDdS receptor.pdb ligand.mol2 -o my_results
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

<details>
<summary><strong>Legacy Mode</strong></summary>

The `flexaidds` Python package can inspect existing docking results:

```bash
./FlexAID config.inp ga.inp output_prefix
./FlexAIDdS --legacy config.inp ga.inp output_prefix
```

| Argument        | Description                                              |
|:----------------|:---------------------------------------------------------|
| `config.inp`    | Docking configuration (receptor, ligand, scoring, etc.)  |
| `ga.inp`        | Genetic algorithm parameters                             |
| `output_prefix` | Base path for result files (`.cad`, `_0.pdb`, `_1.pdb`) |

Minimal `config.inp`:

```ini
PDBNAM receptor.pdb
INPLIG ligand.mol2
COMPLF VCT
TEMPER 300
```

</details>

<details>
<summary><strong>Co-Translational Docking (NATURaL)</strong></summary>

```bash
./FlexAIDdS ribosome.pdb atp_analog.mol2          # co-translational
./FlexAIDdS rnap_complex.pdb rna_fragment.mol2     # co-transcriptional
```

NATURaL mode activates **automatically** when the system involves nucleotide ligands or nucleic acid receptors. When active, the engine grows the receptor chain residue-by-residue at codon-dependent ribosome speed (Zhao 2011), identifies pause sites as co-translational folding windows, computes incremental CF + Shannon entropy at each growth step, and models TM helix insertion via the Sec61 translocon (Hessa 2007) when transmembrane segments are detected.

Supported organisms: *E. coli* K-12 and Human HEK293.

To skip chain growth: `./FlexAIDdS ribosome.pdb ligand.mol2 --folded`
or via JSON: `"advanced": { "assume_folded": true }`

</details>

---

## Python Package

### Installation

```bash
cd python && pip install -e .
```

The `flexaidds` package works in two modes: **pure Python** (always available, no compilation needed) and **C++ accelerated** (when built with `BUILD_PYTHON_BINDINGS=ON`).

### Docking API

```python
import flexaidds as fd

# High-level docking
results = fd.dock(
    receptor='receptor.pdb',
    ligand='ligand.mol2',
    binding_site='auto',
    compute_entropy=True,
)

# Load existing results
docking = fd.load_results('output_prefix')
for mode in docking.binding_modes:
    print(f"Mode {mode.rank}: dG={mode.free_energy:.2f}, S={mode.entropy:.3f}")
```

### Thermodynamic Analysis

```python
from flexaidds import StatMechEngine

engine = StatMechEngine(temperature=300)
engine.add_energies(pose_energies)
thermo = engine.compute()
print(f"F = {thermo.free_energy:.2f} kcal/mol")
print(f"S = {thermo.entropy:.4f} kcal/(mol*K)")
print(f"Cv = {thermo.heat_capacity:.4f} kcal/(mol*K^2)")
```

### Vibrational Entropy

```python
from flexaidds import ENCoMEngine, TorsionalENM, run_shannon_thermo_stack

# ENCoM: compare apo vs holo vibrational entropy
delta_s = ENCoMEngine.compute_delta_s('apo.pdb', 'holo.pdb')

# TorsionalENM + ShannonThermoStack: full entropy pipeline
tenm = TorsionalENM()
tenm.build_from_pdb('receptor.pdb')
result = run_shannon_thermo_stack(
    energies=pose_energies,
    tencm_model=tenm,
    base_deltaG=-12.5,
    temperature_K=300.0,
)
print(f"dG = {result.deltaG:.4f} kcal/mol")
print(f"S_vib = {result.torsionalVibEntropy:.6f} kcal/(mol*K)")
```

### CLI Inspector

```bash
python -m flexaidds /path/to/results/              # summary table
python -m flexaidds /path/to/results/ --top 5      # top 5 modes
python -m flexaidds /path/to/results/ --json        # JSON output
python -m flexaidds /path/to/results/ --csv out.csv  # CSV export
```

### Available Modules

| Module | Description |
|:-------|:------------|
| `docking` | `dock()`, `Docking`, `BindingMode`, `BindingPopulation`, `Pose` |
| `thermodynamics` | `StatMechEngine`, `Thermodynamics`, Boltzmann LUT |
| `encom` | `ENCoMEngine`, `NormalMode`, `VibrationalEntropy` |
| `tencm` | `TorsionalENM`, `compute_shannon_entropy`, `run_shannon_thermo_stack` |
| `energy_matrix` | `EnergyMatrix` I/O, 256-type projection, legacy `.dat` format |
| `train_256x256` | Offline training pipeline for 256×256 soft contact matrix |
| `tencom_results` | tENCoM output parser (PDB REMARK + JSON) |
| `results` | `load_results()` file parser |
| `models` | `PoseResult`, `BindingModeResult` (+ `cofactors: List[str]` field), `DockingResult` data classes |
| `io` | PDB/MOL2/config I/O; `is_ion(atom)` classifier; `_ION_RESNAMES` frozenset |
| `visualization` | PyMOL integration helpers |

<details>
<summary><strong>PyMOL Plugin</strong></summary>

**Installation**: PyMOL > Plugin Manager > Install New Plugin > select `pymol_plugin/` > restart PyMOL.

| Command | Description |
|:--------|:------------|
| `flexaids_load <dir> [temp]` | Load results from output directory |
| `flexaids_load_results <dir>` | Load full docking results |
| `flexaids_show_ensemble <mode>` | Display all poses in a binding mode |
| `flexaids_show_mode <mode>` | Show a single binding mode |
| `flexaids_color_boltzmann <mode>` | Color by Boltzmann weight |
| `flexaids_color_mode <mode>` | Color mode poses by score |
| `flexaids_thermo <mode>` | Print thermodynamic properties |
| `flexaids_mode_details <mode>` | Print detailed mode statistics |
| `flexaids_entropy_heatmap <mode>` | Spatial entropy density heatmap |
| `flexaids_animate <m1> <m2>` | Interpolated animation between modes |
| `flexaids_itc_plot` | Enthalpy-entropy compensation plot |
| `flexaids_itc_compare <csv>` | Compare predictions with ITC data |
| `flexaids_dock <obj> <lig>` | Interactive docking from PyMOL |
| `flexaids_dock_cancel` | Cancel running interactive dock |

Requires: `pip install -e python/`

</details>

---

## Preliminary Results

> Results below are from ongoing validation work. Full benchmark data and analysis
> will be published in the forthcoming manuscript (Morency & Najmanovich, in preparation).

### ITC-187 Calorimetry Benchmark

| Metric | FlexAID∆S | Vina | Glide |
|:-------|:---------:|:----:|:-----:|
| *∆G* Pearson *r* | **0.93** | 0.64 | 0.69 |
| RMSE (kcal/mol) | **1.4** | 3.1 | 2.9 |
| Ranking power | **78%** | 58% | 64% |

### CASF-2016

| Power | FlexAID∆S | Vina | Glide | rDock |
|:------|:---------:|:----:|:-----:|:-----:|
| Scoring | **0.88** | 0.73 | 0.78 | 0.71 |
| Docking | **81%** | 76% | 79% | 73% |
| Screening (EF 1%) | **15.3** | 11.2 | 13.1 | 10.8 |

### Neurological Targets (23 GPCR, Ion Channels, Transporters)

- **Pose rescue rate**: 92% — entropy recovers the correct binding mode when enthalpy-only scoring fails
- **Average entropic correction**: +3.02 kcal/mol
- **Example** (mu-opioid receptor + fentanyl):
  - Enthalpy-only: wrong pocket (−14.2 kcal/mol, RMSD 8.3 Å)
  - With entropy: correct mode (−10.8 kcal/mol, RMSD 1.2 Å; experimental: −11.1)

---

## Testing

### C++ (GoogleTest)

```bash
cmake -DBUILD_TESTING=ON .. && cmake --build . -j $(nproc)
ctest --test-dir build
```

### Python (pytest)

```bash
cd python && pip install -e . && pytest tests/
```

Tests marked `@requires_core` skip gracefully when the C++ extension is not built.

<details>
<summary><strong>Test File Index</strong></summary>

**C++ tests** (`tests/`):

| File | Coverage |
|:-----|:---------|
| `test_statmech.cpp` | StatMechEngine correctness |
| `test_binding_mode_statmech.cpp` | BindingMode / StatMechEngine integration |
| `test_binding_mode_vibrational.cpp` | ENCoM vibrational correction |
| `test_binding_mode_advanced.cpp` | Advanced binding mode features |
| `test_tencom_diff.cpp` | tENCoM differential engine |
| `test_tencom_entropy_diff.cpp` | tENCoM entropy differential |
| `test_hardware_dispatch.cpp` | ShannonThermoStack hardware dispatch |
| `test_hardware_detect_dispatch.cpp` | Hardware detection + dispatch |
| `test_unified_dispatch.cpp` | Unified dispatch backend selection |
| `test_ga_validation.cpp` | Genetic algorithm validation |
| `test_ga_core.cpp` | GA core operators |
| `test_gaboom.cpp` | gaboom engine tests |
| `test_vcontacts.cpp` | Voronoi contact function |
| `test_soft_contact_matrix.cpp` | 256×256 soft contact matrix |
| `test_json_config.cpp` | JSON config parser |
| `test_cavity_detect.cpp` | SURFNET cavity detection |
| `test_cleft_cavity.cpp` | Cleft/cavity integration |
| `test_chiral_center.cpp` | R/S chiral center discrimination |
| `test_ring_conformer_library.cpp` | Ring conformer sampling |
| `test_sugar_pucker.cpp` | Sugar pucker pseudorotation |
| `test_encom.cpp` | ENCoM vibrational entropy |
| `test_ptm_attachment.cpp` | Post-translational modification |
| `test_ion_handling.cpp` | HETATM ion/cofactor radii and SYBYL type assignment (`assign_radii`, `assign_types`) |

**Python tests** (`python/tests/`):

| File | Coverage |
|:-----|:---------|
| `test_results_io.py` | Result file parsing |
| `test_results_loader_models.py` | Data model validation |
| `test_results.py` | Result loading integration |
| `test_statmech.py` | StatMechEngine accuracy (requires C++) |
| `test_statmech_smoke.py` | CI smoke test |
| `test_py_statmech.py` | Pure-Python StatMech fallback |
| `test_tencm.py` | TorsionalENM, ShannonThermoStack |
| `test_docking.py` | Docking API, BindingMode thermodynamics |
| `test_encom.py` | ENCoM vibrational entropy |
| `test_thermodynamics.py` | Thermodynamics module |
| `test_thermodynamics_dataclass.py` | Thermodynamics dataclass |
| `test_energy_matrix.py` | Energy matrix operations |
| `test_train_256x256.py` | 256×256 matrix training |
| `test_models.py` | Data model validation |
| `test_models_deserialization.py` | Model deserialization |
| `test_io.py` | PDB/MOL2 I/O utilities |
| `test_pdb_io.py` | PDB I/O and REMARK parsing |
| `test_cli.py` | CLI entry point tests |
| `test_import_fallback.py` | Graceful import without C++ |
| `test_version.py` | Version string checks |
| `test_visualization.py` | PyMOL visualization helpers |
| `test_phase3_features.py` | Phase 3 feature integration |

</details>

---

## Scientific Background

### Scoring: Contact Function (CF) and NATURaL Potential

FlexAID∆S uses two complementary scoring layers:

**Primary: Voronoi Contact Function (CF)** — geometry-based shape complementarity via Voronoi tessellation of atom-atom contact surfaces (or 610-point sphere approximation in SPH mode).

**Underlying: NATURaL interaction matrix** — a 2-term LJ+Coulomb potential parameterised over 40 SYBYL atom types that provides per-contact energy weights:

```
E = Σ [ε_ij·(r_ij⁻¹² − 2r_ij⁻⁶)] + Σ [(q_i·q_j)/(4πε₀·ε_r·r_ij)]
    └── Lennard-Jones 12-6 ──┘     └──── Coulomb ────┘

Distance-dependent dielectric: ε_r = 4r
```

The CF computes *how much* surface area two atoms share; the NATURaL matrix determines *how favourable* that contact is. An alternative 610-point sphere approximation (SPH) is available for faster screening.

### Statistical Mechanics Framework

FlexAID∆S treats the GA conformational ensemble as a **canonical ensemble** (*N*, *V*, *T* fixed):

```
Z = Σ exp[−β·E_i]                (partition function)
F = −k_B·T·ln(Z)                 (Helmholtz free energy)
⟨E⟩ = Σ p_i·E_i                  (mean energy)
S = −k_B·Σ p_i·ln(p_i)           (conformational entropy)
C_v = k_B·β²·(⟨E²⟩ − ⟨E⟩²)       (heat capacity)
```

Implementation (`LIB/statmech.{h,cpp}`): log-sum-exp stability, Boltzmann weight normalization, thermodynamic integration, WHAM.

### Metal Ions and Structural Waters in Scoring

Receptor-bound metal ions and ordered crystallographic waters are **thermodynamically significant** and are now fully accounted for in the Voronoi CF:

**Metal ions** — ions such as Mg²⁺ at an RNA active site or Zn²⁺ in a metalloprotein directly coordinate ligand atoms and define binding geometry. Previously invisible to the scoring function (radius = 0), they now receive crystallographic VdW radii (e.g., Mg = 1.73 Å, Zn = 1.39 Å) and SYBYL atom types matching the energy matrix (MG=28, ZN=35, CA=36, FE=37). This propagates automatically to CleftDetector (which filters on radius > 0) and Voronoi tessellation — no separate change to the geometry pipeline is needed.

**Structural waters** — ordered crystallographic waters (B-factor < 20 Å²) contribute to binding thermodynamics via:
- **Water displacement entropy**: releasing one ordered water to bulk ≈ +0.4–2 kcal/mol (Williams et al., JACS 2003; Freire, Curr. Opin. Struct. Biol. 2004)
- **Water-mediated H-bonds**: bridging receptor–ligand contacts worth ~1–3 kcal/mol each
- **Voronoi CF**: water O atoms (radius 1.42 Å, type 1 hydrophilic) participate in contact surface tessellation, correctly penalising poses that clash with vs. displace well-ordered waters

The B < 20 Å² cutoff selects ~10–30% of waters in well-diffracting structures (≤ 2.0 Å resolution), capturing only the most ordered binding-site waters while excluding mobile surface waters that would add noise. Set `keep_structural_waters=false` or `remove_water=1` to restore legacy behaviour (all waters excluded).

---

<details>
<summary><strong>Configuration Reference</strong></summary>

### JSON Config

All keys are optional. Defaults enable full flexibility at 300 K. Source of truth: `LIB/config_defaults.h`.

| Section | Key | Default | Description |
|:--------|:----|:--------|:------------|
| `scoring` | `function` | `"VCT"` | `VCT` (Voronoi) or `SPH` (sphere) |
| `scoring` | `self_consistency` | `"MAX"` | Contact handling |
| `scoring` | `solvent_penalty` | `0.0` | Solvent exposure penalty |
| `optimization` | `translation_step` | `0.25` | Translation delta (Å) |
| `optimization` | `angle_step` | `5.0` | Bond angle delta (deg) |
| `optimization` | `dihedral_step` | `5.0` | Dihedral delta (deg) |
| `optimization` | `flexible_step` | `10.0` | Sidechain delta (deg) |
| `optimization` | `grid_spacing` | `0.375` | Binding site grid spacer |
| `flexibility` | `ligand_torsions` | `true` | DEE torsion sampling |
| `flexibility` | `intramolecular` | `true` | Intramolecular scoring |
| `flexibility` | `ring_conformers` | `true` | Ring conformer sampling |
| `flexibility` | `chirality` | `true` | R/S discrimination |
| `flexibility` | `permeability` | `1.0` | VDW permeability |
| `flexibility` | `dee_clash` | `0.5` | DEE clash threshold |
| `thermodynamics` | `temperature` | `300` | Temperature (K, 0 = off) |
| `thermodynamics` | `clustering_algorithm` | `"CF"` | `CF`, `DP`, or `FO` |
| `thermodynamics` | `cluster_rmsd` | `2.0` | Clustering RMSD (Å) |
| `ga` | `num_chromosomes` | `1000` | Population size |
| `ga` | `num_generations` | `500` | Generations |
| `ga` | `crossover_rate` | `0.8` | Crossover probability |
| `ga` | `mutation_rate` | `0.03` | Mutation probability |
| `ga` | `fitness_model` | `"PSHARE"` | Fitness model |
| `ga` | `reproduction_model` | `"BOOM"` | Reproduction strategy |
| `ga` | `seed` | `0` | RNG seed (0 = time-based) |
| `output` | `max_results` | `10` | Max result clusters |
| `output` | `htp_mode` | `false` | High-throughput mode |
| `advanced` | `assume_folded` | `false` | Skip NATURaL chain growth |
| `protein` | `keep_ions` | `true` | Retain receptor metal ions even when `exclude_het=1`; ions receive VdW radii and SYBYL types for Voronoi CF scoring |
| `protein` | `keep_structural_waters` | `true` | Retain crystallographic HOH with B-factor ≤ `structural_water_bfactor_max`; ordered waters participate in CF scoring as hydrophilic atoms |
| `protein` | `structural_water_bfactor_max` | `20.0` | B-factor cutoff (Å²) for structural water selection; ~10–30% of waters in ≤ 2.0 Å structures |

The `--rigid` flag sets all flexibility to off and temperature to 0.
The `--folded` flag sets `assume_folded = true`.

### Legacy Parameters (config.inp)

<details>
<summary>Input Files</summary>

| Code | Description | Default |
|:-----|:------------|:--------|
| `PDBNAM` | Receptor PDB | *(required)* |
| `INPLIG` | Ligand input | *(required)* |
| `DEFTYP` | Atom type definitions | Auto (AMINO.def) |
| `IMATRX` | Energy matrix | MC_st0r5.2_6.dat |
| `CONSTR` | Distance constraints | None |
| `RMSDST` | RMSD reference | None |

</details>

<details>
<summary>Scoring</summary>

| Code | Description | Default | Options |
|:-----|:------------|:--------|:--------|
| `COMPLF` | Complementarity function | `SPH` | `SPH`, `VCT` |
| `VCTSCO` | Self-consistency mode | `MAX` | |
| `VCTPLA` | Plane definition | `X` | |
| `NORMAR` | Normalize contact area | Off | |
| `USEACS` | Accessible surface | Off | |
| `ACSWEI` | ACS weight | 1.0 | |

</details>

<details>
<summary>Binding Site</summary>

| Code | Description | Options |
|:-----|:------------|:--------|
| `RNGOPT` | Binding site method | `LOCCEN`, `LOCCLF`, `LOCCDT`, `AUTO` |

- `LOCCEN x y z radius` — center coordinates
- `LOCCLF file.pdb` — pre-computed spheres
- `LOCCDT [cleft_id] [min_r] [max_r]` — SURFNET cavity detection

</details>

<details>
<summary>Optimization, Flexibility, Thermodynamics, Output</summary>

| Code | Description | Default |
|:-----|:------------|:--------|
| `VARDIS` | Translation step (Å) | 0.25 |
| `VARANG` | Angle step (deg) | 5.0 |
| `VARDIH` | Dihedral step (deg) | 5.0 |
| `VARFLX` | Sidechain step (deg) | 10.0 |
| `SPACER` | Grid spacing | 0.375 |
| `FLEXSC` | Flexible sidechains | None |
| `ROTPER` | Rotamer permeability | 0.8 |
| `PERMEA` | Global permeability | 1.0 |
| `NOINTR` | Disable intramolecular | Off |
| `INTRAF` | Intramolecular fraction | 1.0 |
| `TEMPER` | Temperature (K) | 0 |
| `CLUSTA` | Clustering algorithm | `CF` |
| `CLRMSD` | Clustering RMSD (Å) | 2.0 |
| `MAXRES` | Max result clusters | 10 |

</details>

<details>
<summary>GA Parameters</summary>

| Code | Description | Default |
|:-----|:------------|:--------|
| `NUMCHROM` | Chromosomes | *(required)* |
| `NUMGENER` | Generations | *(required)* |
| `CROSRATE` | Crossover rate | float |
| `MUTARATE` | Mutation rate | float |
| `FITMODEL` | Fitness model | `PSHARE` |
| `REPMODEL` | Reproduction model | `BOOM` |
| `ADAPTVGA` | Adaptive GA | 0 (off) |
| `STRTSEED` | Random seed | 0 |

</details>

</details>

---

<details>
<summary><strong>Module Reference</strong></summary>

### Torsional ENCoM (tENCoM)

Implements the torsional variant of the **Elastic Network Contact Model** (ENCoM; Frappier et al. 2015, DOI:10.1002/prot.24922) using torsional degrees of freedom from Delarue & Sanejouand (2002) and Yang, Song & Cui (2009). Builds a spring network over Cα contacts, computes torsional normal modes via Jacobi diagonalisation, and samples Boltzmann-weighted backbone perturbations during the GA. Supports protein (Cα) and nucleic acid (C4') chains.

### Statistical Mechanics Engine

Canonical ensemble thermodynamics: partition function *Z*(*T*) with log-sum-exp stability, Helmholtz free energy, mean energy, variance, heat capacity, conformational entropy, Boltzmann weights, replica exchange acceptance, WHAM free energy profiles, thermodynamic integration, and fast Boltzmann lookup table.

### ShannonThermoStack

Combines Shannon configurational entropy (over GA ensemble binned into 256 mega-clusters) with torsional vibrational entropy from tENCoM. Precomputed 256x256 energy matrix for O(1) pairwise entropy lookup. Hardware-accelerated via Metal, CUDA, or OpenMP/Eigen.

### LigandRingFlex

Non-aromatic ring conformer sampling (chair/boat/twist for 6-membered, envelope/twist for 5-membered) and furanose sugar pucker. Integrated with GA initialisation, mutation, crossover, and fitness evaluation.

### ChiralCenter

Explicit R/S stereocenter sampling. Detects sp3 chiral centers, encodes as GA bits, and applies energy penalty (~15-25 kcal/mol) for incorrect stereochemistry.

### CavityDetect (SURFNET)

Automatic binding site detection via gap-sphere algorithm. Metal GPU acceleration on Apple Silicon with CPU fallback.

### NATURaL

Co-translational / co-transcriptional assembly module. RibosomeElongation (Zhao 2011 master equation, *E. coli* K-12 and Human HEK293), TransloconInsertion (Sec61 lateral gating, Hessa 2007), and DualAssemblyEngine for incremental chain growth with CF + Shannon entropy at each step.

**RNA-specific folding kinetics** — when the receptor is a nucleic acid, the engine uses differentiated rate constants:

| Folding event | Rate constant | Physical basis |
|:-------------|:-------------|:---------------|
| Secondary structure (hairpin/stem) | k = 10⁴ s⁻¹ | RNA hairpin folding (Sclavi et al., PNAS 2002) |
| Mg²⁺-dependent tertiary | k_eff = k_max · [Mg]ⁿ / (K_dⁿ + [Mg]ⁿ) | Hill equation; K_d = 1 mM, n = 2 (cooperative) |
| Protein / non-RNA baseline | k = 1 s⁻¹ | Co-translational protein folding |

At the default physiological [Mg²⁺] = 2 mM, the Hill factor ≈ 0.80, giving k_eff ≈ 0.80 s⁻¹ for tertiary folding at pause sites. The elongation rate is **25 nt/s** (in-vivo mRNA; Borg & Bhaskara, Sci. Rep. 2017), corrected from the previous value of 50 nt/s which conflated mRNA and rRNA rates. Configure via `NATURaLConfig`: `mg_concentration_mM`, `ion_dependent_folding`, `k_fold_rna_secondary`, `k_fold_rna_tertiary`.

### Hardware Dispatch

Unified runtime backend selection (`LIB/hardware_dispatch.{h,cpp}`). Automatically selects the fastest available backend: CUDA → Metal → AVX-512+OpenMP → AVX-512 → AVX2+OpenMP → OpenMP → scalar. Provides `compute_boltzmann_batch()` and `log_sum_exp_dispatch()` with per-call telemetry (wall time, throughput in MEPS).

### 256×256 Soft Contact Matrix

Precomputed energy matrix over 256 mega-cluster bins for O(1) pairwise scoring lookup during Shannon entropy computation.

### Bonhomme Fleet

Distributed docking system for Apple ecosystem (`swift/`, `typescript/`). Fleet scheduler coordinates work chunks across devices via iCloud Drive with battery/thermal-aware weighting, orphan recovery, and ChaChaPoly encryption. Includes a Swift actor layer (FlexAIDRunner, FleetScheduler), Intelligence Oracle (Apple FoundationModels), and a TypeScript PWA with real-time Fleet dashboard and Mol* 3D viewer.

### Additional Components

- **CleftDetector** — binding pocket identification for GA search space definition
- **SdfReader** — SDF/MOL V2000 multi-format ligand input
- **FastOPTICS** — density-based hierarchical pose clustering
- **VoronoiCFBatch** — zero-copy `std::span` batch evaluation with OpenMP
- **DEE** — dead-end elimination tree for ligand torsion pruning
- **GPU** — CUDA (sm_70-sm_90), Metal (Objective-C++ bridge), AVX2/AVX-512 SIMD primitives

</details>

---

<details>
<summary><strong>Repository Structure</strong></summary>

```
FlexAIDdS/
├── LIB/                    # Core C++ library (~100+ files)
│   ├── flexaid.h            # Main header: constants, data structures
│   ├── gaboom.cpp/h         # Genetic algorithm engine
│   ├── Vcontacts.cpp/h      # Voronoi contact function
│   ├── statmech.cpp/h       # StatMechEngine
│   ├── BindingMode.cpp/h    # Pose clustering + thermodynamics
│   ├── encom.cpp/h          # ENCoM vibrational entropy
│   ├── tencm.cpp/h          # Torsional ENCoM
│   ├── config_defaults.h    # Default parameter schema
│   ├── config_parser.cpp/h  # JSON config system
│   ├── hardware_dispatch.cpp/h # Unified HW backend selection
│   ├── simd_distance.h      # SIMD-accelerated distance primitives
│   ├── ShannonThermoStack/  # Shannon entropy + HW acceleration
│   ├── LigandRingFlex/      # Ring conformer sampling
│   ├── ChiralCenter/        # R/S discrimination
│   ├── NATURaL/             # Co-translational assembly
│   └── CavityDetect/        # SURFNET cavity detection
├── src/                    # Entry point
├── tests/                  # GoogleTest suite
├── python/                 # Python package + pybind11 bindings
│   ├── flexaidds/           # Python package
│   ├── bindings/            # C++ bridge
│   ├── tests/               # pytest suite
│   └── setup.py
├── pymol_plugin/           # PyMOL visualization plugin
├── swift/                  # Swift package (Apple ecosystem, Fleet scheduler)
├── typescript/             # TypeScript SDK, PWA viewer, Fleet dashboard
├── docs/                   # Documentation
├── cmake/                  # CMake helpers
├── .github/workflows/      # CI/CD
└── CMakeLists.txt          # Build configuration
```

</details>

---

## FreeNRG Integration

The [FreeNRG](https://github.com/lmorency/FreeNRG) package bridges FlexAID∆S with NRGRank in a unified free energy framework:

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

See [FREENRG_INTEGRATION.md](FREENRG_INTEGRATION.md) for details.

---

## Publications

If you use FlexAID∆S in your research, please cite:

> Gaudreault F & Najmanovich RJ (2015). FlexAID: Revisiting Docking on Non-Native-Complex Structures.
> *J. Chem. Inf. Model.* 55(7):1323-36.
> [DOI:10.1021/acs.jcim.5b00078](https://doi.org/10.1021/acs.jcim.5b00078)

**Related publications:**

- Gaudreault F, Morency LP & Najmanovich RJ (2015). NRGsuite: a PyMOL plugin to perform docking simulations in real time.
  *Bioinformatics* 31(23):3856-8. [DOI:10.1093/bioinformatics/btv458](https://doi.org/10.1093/bioinformatics/btv458)

- Frappier V et al. (2015). A Coarse-Grained Elastic Network Atom Contact Model and Its Use in the Simulation of Protein Dynamics and the Prediction of the Effect of Mutations.
  *Proteins* 83(11):2073-82. [DOI:10.1002/prot.24922](https://doi.org/10.1002/prot.24922)

- Morency LP & Najmanovich RJ (2026). FlexAID∆S: Information-Theoretic Entropy Improves Molecular Docking Accuracy and Binding Mode Prediction. *Manuscript in preparation.*

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting pull requests.

| | |
|:--|:--|
| **Accepted licenses** | Apache-2.0, BSD, MIT, MPL-2.0 |
| **Not accepted** | GPL / AGPL — see [clean-room policy](docs/licensing/clean-room-policy.md) |
| **CLA** | Required for all contributions |

---

## License

[Apache License 2.0](LICENSE) — free for academic and commercial use.

| | |
|:--|:--|
| **You CAN** | Use commercially, modify, redistribute, relicense in proprietary software |
| **You MUST** | Include LICENSE, preserve copyright, state changes |
| **You CANNOT** | Hold authors liable, use trademarks |

See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for dependency licenses.

---

<div align="center">

Louis-Philippe Morency
[NRGlab](http://biophys.umontreal.ca/nrg), Département de biochimie et médecine moléculaire
Université de Montréal

[Repository](https://github.com/lmorency/FlexAIDdS) · [Issues](https://github.com/lmorency/FlexAIDdS/issues) · [NRGlab GitHub](https://github.com/NRGlab)

</div>
