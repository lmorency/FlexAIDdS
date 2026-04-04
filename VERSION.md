# FlexAID∆S Version History

**Current Version**: 1.9.0-beta
**Release Date**: 2026-04-04
**Python Package**: 1.0.0-alpha (`python/flexaidds/__version__.py`)
**Repository**: [github.com/lmorency/FlexAIDdS](https://github.com/lmorency/FlexAIDdS)
**License**: Apache-2.0

---

## v1.9.0-beta (2026-04-04) — Current

Feature-complete pre-release of FlexAID∆S, the entropy-driven rewrite of FlexAID. Phases 1–3 and 5 complete; Phase 4 at ~75%. 640+ commits ahead of the v1.5 legacy tag.

### Core Thermodynamics (Phase 1)

- **StatMechEngine** — canonical ensemble partition function, Helmholtz free energy, Shannon entropy, heat capacity
- **BindingMode** — pose clustering with intra-mode thermodynamics and Boltzmann weights
- **BindingPopulation** — multi-mode free energy aggregation via log-sum-exp
- **Log-sum-exp stability** — numerically stable for energy differences > 100 kT
- **WHAM** — single-window weighted histogram free energy profiles
- **Thermodynamic integration** — lambda-path free energy perturbation
- **Replica exchange** — Metropolis swap criterion for parallel tempering

### Python Bindings (Phase 2)

- **pybind11 core** — `StatMechEngine`, `ENCoMEngine`, `Thermodynamics`, `VibrationalEntropy` exposed to Python
- **Data models** — `PoseResult`, `BindingModeResult`, `DockingResult` with JSON/dict round-trip serialization
- **Pure-Python fallback** — full `flexaidds` package works without C++ compilation
- **CLI inspector** — `python -m flexaidds <dir> [--json|--csv|--top N]`
- **22 Python modules** including benchmark, energy matrix training, visualization, dataset adapters

### PyMOL/NRGSuite GUI (Phase 3)

- **PyMOL plugin** — 8 modules: entropy heatmap, mode animation, ITC comparison, interactive docking
- **11 bug fixes** — async docking, NumPy bulk conversion, multi-ligand ITC support
- **14 PyMOL commands** — load, visualize, color, animate binding modes

### Hardware Acceleration (Phase 5)

- **Unified dispatch** — CUDA > ROCm/HIP > Metal > AVX-512 > AVX2 > OpenMP > scalar
- **CUDA kernels** — batch CF evaluation, Shannon histograms, tENCoM Hessian, FastOPTICS k-NN
- **Metal shaders** — Shannon entropy, cavity detection, CF evaluation (macOS/Apple Silicon)
- **SIMD** — AVX-512 and AVX2 vectorised geometric primitives
- **OpenMP + Eigen3** — thread parallelism and vectorised linear algebra
- **LTO binaries** — link-time optimized `FlexAIDdS` and `tENCoM` executables
- **Cross-platform** — Linux (GCC/Clang), macOS (Clang + Metal), Windows (MSVC)

### Grid Optimization & Docking Intelligence (Phase 4, ~75%)

- **GIST water-displacement scoring** — grid-based explicit solvation term (`GISTEvaluator`, `GISTGrid`)
- **Directional H-bond scoring** — geometry-aware hydrogen bond potential (`HBondEvaluator`, `hbond_potential.h`)
- **Parallel docking** — grid domain decomposition with MPI transport
- **GA diversity** — population entropy monitoring and adaptive diversity pressure (`ga_diversity.h`)
- **GAContext** — structured GA run context for reproducible experiments
- **Molecular interaction fields** — MIF-guided docking, reference ligand seeding, Spectrophore descriptors
- **SMFREE fitness** — Shannon entropy as direct GA fitness component with adaptive pressure
- **256×256 energy matrix** — extended atom typing with continuous training pipeline
- **GA bug fixes** — QuickSort ordering, fitness_stats boundary, Swift bridge cleanup
- **DistributedBackend / ThreadBackend / GPUContextPool** — unified parallel execution layer
- **AtomSoA / VoronoiCFBatch_SoA** — Structure-of-Arrays layouts for SIMD-friendly batching

### Additional Features

- **tENCoM** — torsional elastic network model for backbone vibrational entropy
- **ShannonThermoStack** — combined configurational + vibrational entropy with HW acceleration
- **LigandRingFlex** — non-aromatic ring conformer sampling (chair/boat/twist, sugar pucker)
- **ChiralCenter** — explicit R/S stereocenter discrimination with energy penalty
- **CavityDetect** — SURFNET gap-sphere cavity detection with Metal GPU support
- **NATURaL** — co-translational/co-transcriptional assembly with NucleationDetector (E. coli K-12, Human HEK293)
- **CleftDetector** — binding site identification with ion/water awareness
- **Metal ion scoring** — crystallographic VdW radii for 20+ ion types in Voronoi CF
- **Structural water** — ordered waters (B < 20 A^2) participate in contact scoring
- **Multi-format input** — PDB, CIF/mmCIF, MOL2, SDF, SMILES (auto 3D build)
- **JSON config** — single config file with sensible defaults for all parameters
- **Bonhomme Fleet** — distributed docking across Apple devices via iCloud
- **Swift package** — macOS/iOS actors wrapping StatMechEngine and ENCoM
- **TypeScript SDK** — PWA dashboard with Mol* 3D viewer
- **TurboQuant** — vector quantization for contact and energy vector compression

### New in this update (2026-04-04)

- `ml_rescore.py` — machine-learning rescoring module
- `optimize.py` — docking parameter optimization
- `flexmolview.py` — experimental molecule viewer integration (Python)
- Swift `FlexMolViewPrototype` and TypeScript `flexmolview-prototype` — experimental multi-language viewer
- GIL-release bindings for thread-safe parallel C++ calls from Python
- License scanner CI workflow — automated GPL/AGPL dependency detection
- Performance CI workflow — benchmark regression tracking
- Sanitizers CI workflow — ASan/UBSan checks on every commit
- Windows CI parity + `docs/WINDOWS_BUILD_ROADMAP.md`
- CASF-2016, CrossDock, LIT-PCBA benchmark harnesses (`tests/benchmarks/`)
- Smoke validation bundle (`benchmarks/smoke/`) with manifest and run script

### Codebase Statistics

- 640+ commits since v1.5 tag
- 40K+ lines C++, 19K+ lines Python, 10K lines Swift, 4K lines TypeScript, 12K+ lines tests
- 46 C++ test files (GoogleTest), 32 Python test files (pytest) — 78 total
- CI: Linux GCC, Linux Clang, macOS Clang (allow-fail), Windows MSVC (allow-fail)
- 4 CI workflows: `ci.yml`, `license-scan.yml`, `perf.yml`, `sanitizers.yml`

---

## v1.5 (2026-03-06) — Legacy tag

Original FlexAID release with Voronoi contact function scoring. Tagged as `v1.5` in git history.

- Genetic algorithm docking engine
- Voronoi contact function (VCT) and sphere approximation (SPH)
- Dead-end elimination for torsion pruning
- Legacy INP/GA config file format
- No thermodynamics, no entropy, no hardware acceleration

---

## Versioning Policy

FlexAID∆S follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward-compatible
- **PATCH**: Bug fixes, backward-compatible

Pre-release tags: `-alpha` (unstable), `-beta` (feature-complete, testing), `-rc` (release candidate)
