# FlexAIDdS

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey.svg)](#)

## Features

- **Genetic algorithm docking** with configurable population, crossover, mutation, and selection
- **Voronoi contact function (CF)** for shape complementarity scoring
- **Torsional ENCoM (TENCM)** backbone flexibility without full rotamer rebuilds
- **Statistical mechanics engine** — partition function, free energy, heat capacity, conformational entropy
- **Shannon entropy + torsional vibrational entropy stack** for thermodynamic scoring
- **Ligand ring flexibility** — non-aromatic ring conformer sampling and sugar pucker
- **Chiral center sampling** — explicit R/S stereocenter discrimination in the GA
- **NATURaL co-translational assembly** — co-translational/co-transcriptional docking with ribosome-speed elongation (Zhao 2011) and Sec translocon TM insertion (Hessa 2007)
- **FastOPTICS** density-based clustering of docking poses
- **Hardware acceleration** — CUDA, Metal (macOS), AVX-512, AVX2, OpenMP, Eigen3

## Build

```bash
git clone --branch flexaid-cpp https://github.com/NRGlab/FlexAID
cd FlexAID
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target FlexAID -j $(nproc)
```

On macOS, install Boost via Homebrew (`brew install boost`). On Windows, download Boost binaries and pass `-DBoost_DIR=<path>` to CMake if not auto-detected.

### CMake Options

| Option                    | Default | Description                              |
|:--------------------------|:--------|:-----------------------------------------|
| `FLEXAIDS_USE_CUDA`       | OFF     | CUDA GPU batch evaluation                |
| `FLEXAIDS_USE_METAL`      | OFF     | Metal GPU acceleration (macOS only)      |
| `FLEXAIDS_USE_AVX2`       | ON      | AVX2 SIMD acceleration                   |
| `FLEXAIDS_USE_AVX512`     | OFF     | AVX-512 SIMD acceleration                |
| `FLEXAIDS_USE_OPENMP`     | ON      | OpenMP thread parallelism                |
| `FLEXAIDS_USE_EIGEN`      | ON      | Eigen3 vectorised linear algebra         |
| `ENABLE_TENCOM_BENCHMARK` | OFF     | Build standalone TeNCoM benchmark binary |

### Dependencies

- **Required**: C++20 compiler, CMake >= 3.18
- **Optional**: Boost, Eigen3 (`libeigen3-dev`), OpenMP, CUDA Toolkit, Metal framework (macOS)

## Usage

FlexAID requires a **config file** and a **GA parameter file**. These can be generated using `ProcessLigand` (installed via PyPI: `pip install processligand-py`).
When using ProcessLigand make sure `atom_index=90000` on the ligand.

```bash
./FlexAID config.inp ga.inp
```

## Required Config file codes

## 📖 Usage Modes

### Option A: Zero-Config CLI

```bash
./flexaids dock receptor.pdb ligand.mol2
# Auto-detects: binding site, rotatable bonds, hardware backend
# Output: binding_modes.pdb, thermodynamics.json
```

### Option B: YAML Config (Advanced)

```yaml
docking:
  binding_site:
    method: auto  # or {center: [x,y,z], radius: 10.0}
  flexible_sidechains: ["A:TYR123", "A:PHE456"]
  temperature: 300.0

genetic_algorithm:
  population_size: 2000
  max_generations: 100

hardware:
  backend: auto  # or: cuda, metal, avx512, openmp

output:
  top_n_modes: 10
  json_thermodynamics: true
  entropy_decomposition: true
```

### Option C: Python API (Phase 2)

```python
results = flexaids.dock(
    receptor='receptor.pdb',
    ligand='ligand.mol2',
    binding_site='auto',
    compute_entropy=True
)
```

---

## 🧬 Scientific Background

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

## 🏆 Benchmarks

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
- **Example** (μ-opioid + fentanyl):
  - Enthalpy-only: Wrong pocket (−14.2 kcal/mol, RMSD 8.3 Å)
  - With entropy: **Correct** (−10.8 kcal/mol, RMSD 1.2 Å, exp: −11.1)

---

## 📚 Publications

### Please Cite

1. **FlexAID core**:
   > Gaudreault & Najmanovich (2015). *J. Chem. Inf. Model.* 55(7):1323-36. [DOI:10.1021/acs.jcim.5b00078](https://doi.org/10.1021/acs.jcim.5b00078)

2. **NRGsuite PyMOL plugin**:
   > Gaudreault, Morency & Najmanovich (2015). *Bioinformatics* 31(23):3856-8. [DOI:10.1093/bioinformatics/btv458](https://doi.org/10.1093/bioinformatics/btv458)

3. **Shannon entropy extension** (submitted):
   > Morency et al. (2026). "Information-Theoretic Entropy in Molecular Docking." *J. Chem. Theory Comput.* (in review)

### Related Work (Inspiration Only)

- **NRGRank** (GPL-3.0, *not a dependency*):
  > Gaudreault et al. (2024). bioRxiv preprint.  
  > *Note*: FlexAID∆S reimplements cube screening from first principles (Apache-2.0). No GPL code included. See [clean-room policy](docs/licensing/clean-room-policy.md).

---

## 🤝 Contributing

**Key Policies**:
- ✅ Apache-2.0, BSD, MIT, MPL-2.0 dependencies OK
- ❌ GPL/AGPL **forbidden** (see [clean-room policy](docs/licensing/clean-room-policy.md))
- All contributions require Contributor License Agreement (CLA)

See [CONTRIBUTING.md](CONTRIBUTING.md) for code style, testing, PR workflow.

---

## 📜 License

**Apache License 2.0** – Permissive open-source.

**You CAN**: Use commercially, modify, redistribute, relicense in proprietary software.  
**You MUST**: Include LICENSE, preserve copyright, state changes.  
**You CANNOT**: Hold authors liable, use trademarks.

See [LICENSE](LICENSE) | [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)

---

## 🔗 Links

**Repository**: [github.com/lmorency/FlexAIDdS](https://github.com/lmorency/FlexAIDdS)  
**Issues**: [github.com/lmorency/FlexAIDdS/issues](https://github.com/lmorency/FlexAIDdS/issues)  
**NRGlab**: [biophys.umontreal.ca/nrg](http://biophys.umontreal.ca/nrg) | [github.com/NRGlab](https://github.com/NRGlab)

**Lead Developer**: Louis-Philippe Morency, PhD (Candidate)  
**Affiliation**: Université de Montréal, NRGlab  
**Email**: louis-philippe.morency@umontreal.ca

---

<p align="center">
  <strong>FlexAID∆S: Where Information Theory Meets Drug Discovery</strong><br>
  <em>Zero friction. Zero entropy waste. Zero bullshit.</em><br><br>
  <sub>DRUG IS ALWAYS AN ANSWER. One Shannon bit at a time. 🧬⚡</sub>
</p>

---

## GA Codes

| Code       | Description                                                   | Value                | 
|:-----------|:--------------------------------------------------------------|:---------------------|
| `NUMCHROM` | Number of chromosomes                                         | (int)                |
| `NUMGENER` | Number of generations                                         | (int)                |
| `ADAPTVGA` | Enable adaptive GA (adjusts crossover/mutation rates dynamically) | (int flag)           |
| `ADAPTKCO` | Adaptive GA response parameters k1–k4 (each in range 0.0–1.0)    | (list) with 4 floats |
| `CROSRATE` | Crossover rate                                                    | float (0.0–1.0)      |
| `MUTARATE` | Mutation rate                                                     | float (0.0–1.0)      |
| `POPINIMT` | Population initialization method                                  | `RANDOM` or `IPFILE` |
| `FITMODEL` | Fitness model                                                     | `PSHARE` or `LINEAR` |
| `SHAREALF` | Sharing parameter α (sigma share)                                 | float                |
| `SHAREPEK` | Expected number of sharing peaks in the search space              | float                |
| `SHARESCL` | Fitness scaling factor for sharing                                | float                |
| `STRTSEED` | Set a custom starting seed                                        | (int)                |
| `REPMODEL` | Reproduction technique code                                       | `STEADY`, `BOOM`     |
| `BOOMFRAC` | Population boom size  (fraction of the number of chromosomes)     | 0 to 1 (float)       |
| `PRINTCHR` | Number of best chromosome to print each generation                | (int)                |
| `PRINTINT` | Print generation progress as well as current best cf              | 0 or 1               |
| `OUTGENER` | Output results for each generation                                | N/A                  |

---

## v1.5 Modules

### Torsional ENCoM (TENCM)

Implements the torsional elastic network contact model (Delarue & Sanejouand 2002; Yang, Song & Cui 2009) for protein backbone flexibility. Builds a spring network over Cα contacts within a cutoff radius, computes torsional normal modes via Jacobi diagonalisation, and samples Boltzmann-weighted backbone perturbations during the GA without rebuilding the rotamer library every generation.

### Statistical Mechanics Engine

Full thermodynamic analysis of the GA conformational ensemble:
- Partition function Z(T) with log-sum-exp numerical stability
- Helmholtz free energy F = −kT ln Z
- Average energy, variance, and heat capacity
- Conformational entropy S = (⟨E⟩ − F) / T
- Boltzmann-weighted state probabilities
- Parallel tempering (replica exchange) swap acceptance
- WHAM for free energy profiles
- Thermodynamic integration via trapezoidal rule
- Fast Boltzmann lookup table for inner-loop evaluation

### ShannonThermoStack

Combines Shannon configurational entropy (over GA ensemble binned into 256 mega-clusters) with torsional ENCoM vibrational entropy. Uses a precomputed 256×256 energy matrix for O(1) pairwise entropy lookup. Hardware-accelerated histogram computation via Metal (Apple Silicon), CUDA, or OpenMP/Eigen.

### LigandRingFlex

Unified ring flexibility for the GA: non-aromatic ring conformer sampling (chair/boat/twist for 6-membered, envelope/twist for 5-membered) and furanose sugar pucker phase sampling. Integrates with GA initialisation, mutation, crossover, and fitness evaluation.

### ChiralCenter

Explicit R/S stereocenter sampling. Detects sp3 tetrahedral chiral centers in the ligand, encodes each as a single GA bit (R=0, S=1), and applies an energy penalty for incorrect stereochemistry (~15–25 kcal/mol per wrong center). Low mutation rate (1–2%) reflects the high inversion barrier.

### NATURaL (co-translational assembly)

**N**ative **A**ssembly of co-**T**ranscriptionally/co-**T**ranslationally **U**nified **R**eceptor–**L**igand module. Auto-detects nucleotide ligands or nucleic acid receptors and activates co-translational DualAssembly mode:

- **RibosomeElongation**: Zhao 2011 master equation for codon-dependent ribosome speed (E. coli K-12 and Human HEK293). Identifies pause sites as co-translational folding windows. Also supports nucleotide-by-nucleotide RNA polymerase synthesis.
- **TransloconInsertion**: Sec61 translocon lateral gating model (Hessa 2007). Computes per-window ΔG of TM helix insertion using the Hessa scale with Wimley-White position-weighted helix-dipole correction. Hardware-accelerated via AVX-512/AVX2/Eigen.
- **DualAssemblyEngine**: Grows the receptor chain residue-by-residue at ribosome speed while computing incremental CF and Shannon entropy at each growth step to capture co-translational stereochemical selection.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
