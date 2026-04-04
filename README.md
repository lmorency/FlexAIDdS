<div align="center">

# FlexAID∆S

**Entropy-aware molecular docking with a production-focused Core 1.0 surface**

[![CI](https://github.com/lmorency/FlexAIDdS/actions/workflows/ci.yml/badge.svg)](https://github.com/lmorency/FlexAIDdS/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Python](https://img.shields.io/badge/python-%E2%89%A5%203.9-3776AB.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)](#)
[![DOI](https://img.shields.io/badge/DOI-10.1021%2Facs.jcim.5b00078-blue)](https://doi.org/10.1021/acs.jcim.5b00078)

</div>

**[Installation](docs/INSTALLATION.md)** · **[User Guide](docs/USERGUIDE.md)** · **[Support Matrix](docs/SUPPORT_MATRIX.md)** · **[Reproducibility](docs/REPRODUCIBILITY.md)** · **[Benchmarks](docs/BENCHMARKS.md)** · **[Website](https://lmorency.github.io/FlexAIDdS/)**

---

FlexAID∆S extends the original [FlexAID](https://doi.org/10.1021/acs.jcim.5b00078) docking engine with an entropy-aware thermodynamics layer. The core idea is simple: conventional docking often behaves like an enthalpy-only ranker, while FlexAID∆S treats the genetic-algorithm ensemble as a thermodynamic population and computes free-energy-relevant quantities from that ensemble.

## Core 1.0 support boundary

The repository contains a broad research platform. **Not every visible feature is part of the supported 1.0 contract.**

The supported product for 1.0 is defined in [`PRODUCT.md`](PRODUCT.md).

### Supported for Core 1.0

- `FlexAIDdS` command-line executable
- `FlexAID` legacy-compatible command-line executable
- `tENCoM` command-line executable
- JSON configuration workflows documented for the core engine
- `flexaidds` Python package
- core repository documentation required for install, validation, and reproducibility
- benchmark bundles that exist under [`benchmarks/`](benchmarks/)

### Experimental and not part of the Core 1.0 support contract

- Swift packages and Apple-device integration layers
- TypeScript / PWA / browser-facing dashboards
- Bonhomme Fleet and iCloud-based distributed execution
- NATURaL and co-translational / co-transcriptional workflows
- backend-specific accelerator paths not covered by the support matrix
- benchmark claims that are not backed by a replayable repository bundle

See also:

- [`docs/VALIDATED_CAPABILITIES.md`](docs/VALIDATED_CAPABILITIES.md)
- [`docs/EXPERIMENTAL_CAPABILITIES.md`](docs/EXPERIMENTAL_CAPABILITIES.md)
- [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md)

---

## What the core engine does

- genetic-algorithm docking with Voronoi contact-function scoring
- canonical-ensemble thermodynamic quantities from the docking ensemble
- configurational and vibrational entropy plumbing for ranking and analysis
- ligand flexibility, ring conformers, chirality handling, and multi-format input
- Python access for result loading, thermodynamics, and workflow scripting

The repository still contains broader research components, but those should not be interpreted as release-guaranteed unless explicitly promoted into the validated surface.

---

## Quick start

### Build the supported CLI surface

```bash
git clone https://github.com/lmorency/FlexAIDdS.git
cd FlexAIDdS
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

> Core executable builds (`FlexAID`, `FlexAIDdS`, `tENCoM`) require Eigen3 headers.
> If you are in a restricted/offline environment, you can still build benchmark tooling only:
>
> ```bash
> cmake -S . -B build -DFLEXAIDS_BUILD_CORE=OFF -DBUILD_TESTING=OFF
> cmake --build build --parallel --target benchmark_datasets
> ```

### Run a basic docking workflow

```bash
./build/FlexAIDdS receptor.pdb ligand.mol2
./build/FlexAIDdS ligand.mol2 receptor.pdb
./build/FlexAIDdS receptor.pdb ligand.mol2 -c config.json
./build/FlexAIDdS receptor.pdb ligand.mol2 --rigid
```

### Use the Python package

```bash
cd python
pip install -e .
```

```python
import flexaidds as fd

results = fd.dock(
    receptor="receptor.pdb",
    ligand="ligand.mol2",
    compute_entropy=True,
)

for mode in results.rank_by_free_energy():
    print(mode.free_energy)
```

---

## Build and test

### C++

```bash
cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

### Python

```bash
cd python
pip install -e .
pytest tests/
```

### CI scope

The repository already runs cross-platform core builds and Python smoke coverage in CI. Supported vs experimental interpretation is governed by the support matrix, not by mere code presence.

See [`docs/SUPPORT_MATRIX.md`](docs/SUPPORT_MATRIX.md).

---

## Reproducibility and benchmark claims

Benchmark and scientific performance claims should be interpreted through the repository reproducibility policy.

**A claim is not repository-reproducible merely because it appears in documentation.** It becomes repository-reproducible only when the corresponding bundle exists under [`benchmarks/`](benchmarks/) with dataset provenance, commands, expected outputs, and metric definitions.

See:

- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
- [`benchmarks/README.md`](benchmarks/README.md)
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)

At this stage, benchmark tables and claims that are not backed by a replayable bundle should be treated as **preliminary**.

---

## Repository layout

```text
FlexAIDdS/
├── LIB/                  # core C++ engine and thermodynamics
├── src/                  # entry points
├── tests/                # C++ tests
├── python/               # Python package and bindings
├── pymol_plugin/         # PyMOL integration helpers
├── docs/                 # documentation
├── benchmarks/           # reproducibility bundles and smoke validation
├── .github/workflows/    # CI, release, and analysis workflows
├── swift/                # experimental
└── typescript/           # experimental
```

---

## Documentation map

- [`PRODUCT.md`](PRODUCT.md) — supported 1.0 product boundary
- [`docs/SUPPORT_MATRIX.md`](docs/SUPPORT_MATRIX.md) — supported platform and backend matrix
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — benchmark claim policy
- [`docs/VALIDATED_CAPABILITIES.md`](docs/VALIDATED_CAPABILITIES.md) — positive support inventory
- [`docs/EXPERIMENTAL_CAPABILITIES.md`](docs/EXPERIMENTAL_CAPABILITIES.md) — experimental surface inventory
- [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) — current limitations that matter
- [`SECURITY.md`](SECURITY.md) — security reporting and hardening priorities
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — contribution policy

---

## Publications and citation

If you use FlexAID∆S in research, cite the original FlexAID paper:

> Gaudreault F & Najmanovich RJ (2015). FlexAID: Revisiting Docking on Non-Native-Complex Structures.
> *J. Chem. Inf. Model.* 55(7):1323-36.
> [DOI:10.1021/acs.jcim.5b00078](https://doi.org/10.1021/acs.jcim.5b00078)

Related project context remains listed in the repository documentation and publication history, but manuscript-in-preparation material should not be treated as repository-validated by default.

---

## License

[Apache License 2.0](LICENSE)

See also [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

---

<div align="center">

Louis-Philippe Morency  
[NRGlab](http://biophys.umontreal.ca/nrg), Université de Montréal  

[Repository](https://github.com/lmorency/FlexAIDdS) · [Issues](https://github.com/lmorency/FlexAIDdS/issues) · [NRGlab GitHub](https://github.com/NRGlab)

</div>
