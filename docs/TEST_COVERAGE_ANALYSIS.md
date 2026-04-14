# Test Coverage Analysis — FlexAIDdS

**Date**: 2026-04-05  
**Scope**: Full codebase (C++ core, Python package, PyMOL plugin, CI pipeline)

---

## Executive Summary

The codebase has **strong unit test coverage for thermodynamics, statistical mechanics, and GA core logic**, but significant gaps exist in file I/O, coordinate geometry, ligand processing, and integration testing. Approximately **18% of C++ source files** have dedicated tests. The Python test suite is more comprehensive (~233 tests across 32 modules) but leaves key modules like `dataset_adapters.py`, `energy_matrix_cli.py`, and the entire `pymol_plugin/` untested. CI runs primarily on Linux with no GPU, macOS, or Windows functional testing.

---

## Coverage by Area

### Well-Tested (Production-Ready)

| Area | Test Files | Notes |
|------|-----------|-------|
| StatMechEngine (C++) | `test_statmech.cpp`, `test_grand_partition.cpp` | Partition functions, Boltzmann, WHAM, TI — analytically validated |
| BindingMode + Thermo | `test_binding_mode_statmech.cpp`, `test_binding_mode_vibrational.cpp`, `test_binding_mode_advanced.cpp` | Pose clustering, thermodynamic integration, vibrational corrections |
| GA Core | `test_ga_core.cpp`, `test_ga_validation.cpp`, `test_ga_diversity.cpp`, `test_entropy_ga.cpp` | Selection, crossover, mutation, fitness convergence, diversity metrics |
| ENCoM | `test_encom.cpp` (C++ & Python) | Mode filtering, temperature effects, entropy additivity |
| tENCoM Differential | `test_tencom_diff.cpp`, `test_tencom_entropy_diff.cpp` | Eigenvalue overlaps, B-factor diffs, entropy differentials |
| Mol2/SDF Readers | `test_mol2_sdf_reader.cpp` | Atom types, charges, bonds, PDB numbering, error cases |
| Chiral Centers | `test_chiral_center.cpp` | R/S detection, mutation, crossover, inversion penalties |
| Python Data Models | `test_models.py` (101 tests), `test_models_deserialization.py` | Frozen dataclasses, JSON/CSV round-trips, DataFrame conversion |
| Python I/O | `test_io.py` (76 tests), `test_pdb_io.py` | REMARK parsing, key normalization, filename inference |
| Python CLI | `test_cli.py` (38 tests) | All output formats, argument parsing, top-N filtering |
| Python Benchmarking | `test_benchmark.py` (64 tests) | Affinity conversions, RMSD, correlation metrics, enrichment |
| Energy Matrix | `test_energy_matrix.py`, `test_soft_contact_matrix.cpp` | 256-type encoding, SYBYL projection, .DAT parsing |

### Moderately Tested (Good Foundation, Gaps Remain)

| Area | Status | Missing |
|------|--------|---------|
| Cavity Detection | Basic box geometries tested | No real PDB structures, no parameter sensitivity |
| CleftDetector | Probe generation, clustering tested | No real-world binding sites |
| Ring Flexibility | Conformer library verified | No integration with GA docking, no ring constraint generation |
| Grid Operations | `test_cube_grid.cpp`, `test_gist_grid.cpp` | No tests for `generate_grid`, `partition_grid`, `slice_grid` |
| Clustering | FastOPTICS tested | DensityPeak_Cluster, base FOPTICS untested |
| SharedPosePool | Move semantics, serialization tested | No real concurrent stress testing |
| Hardware Dispatch | Shannon entropy dispatch tested | No GPU fallback paths tested |

### Not Tested At All — Critical Gaps

These are the areas that most need attention, prioritized by risk.

---

## Priority 1: High-Risk Untested Areas

### 1. File I/O Pipeline (C++) — 20+ files, 0 tests

The entire read/write layer has no dedicated tests:

- **Readers**: `read_pdb.cpp`, `read_coor.cpp`, `read_grid.cpp`, `read_lig.cpp`, `read_input.cpp`, `read_eigen.cpp`, `read_emat.cpp`, `read_normalgrid.cpp`, `read_spheres.cpp`, `read_conect.cpp`, `read_constraints.cpp`, `read_rotlib.cpp`, `read_rotobs.cpp`, `read_flexscfile.cpp`
- **Writers**: `write_pdb.cpp`, `write_grid.cpp`, `write_sphere.cpp`, `write_rrd.cpp`, `write_rrg.cpp`
- **Utilities**: `fileio.cpp`, `modify_pdb.cpp`, `residue_conect.cpp`

**Risk**: Silently corrupt data on format changes. File parsing bugs are a top source of runtime errors.

**Recommended tests**:
- Round-trip read/write for each format (PDB, grid, spheres, RRD)
- Malformed input handling (truncated files, missing fields, wrong line lengths)
- Edge cases: empty files, very large structures, non-standard residues

### 2. Coordinate Geometry & RMSD (C++) — 18 files, 0 tests

Core geometric operations are completely untested:

- `geometry.cpp` — fundamental transforms
- `calc_rmsd.cpp`, `calc_rmsd_chrom.cpp`, `calc_rmsp.cpp` — RMSD calculations
- `ic2cf.cpp`, `ic_bounds.cpp` — internal-to-Cartesian coordinate conversion
- `buildic.cpp`, `buildic_point.cpp`, `buildcc.cpp`, `buildcc_point.cpp` — coordinate building
- `calc_center.cpp`, `calc_cleftic.cpp` — center-of-mass, cleft metrics

**Risk**: Incorrect docking poses and scoring. Geometry bugs propagate silently through the entire pipeline.

**Recommended tests**:
- RMSD against known reference values (identity = 0, known rotation = expected value)
- Internal↔Cartesian coordinate round-trips
- Degenerate cases: collinear atoms, single atom, coincident points

### 3. ProcessLigand Subsystem (C++) — 9 files, 0 tests

The complete ligand preparation pipeline is untested:

- `SmilesParser.cpp` — SMILES string parsing
- `RingPerception.cpp` — ring detection
- `Aromaticity.cpp` — aromaticity assignment
- `RotatableBonds.cpp` — rotatable bond identification
- `ValenceChecker.cpp` — valence validation
- `SybylTyper.cpp` — atom type assignment
- `FlexAIDWriter.cpp` — output formatting
- `CoordBuilder.cpp` — 3D coordinate generation

**Risk**: Incorrect ligand representation → wrong docking results. SMILES parsing bugs are notoriously subtle.

**Recommended tests**:
- Known drug molecules (aspirin, caffeine, ibuprofen) through full pipeline
- SMILES edge cases: aromatic rings, stereochemistry, charged groups, metals
- Round-trip: SMILES → internal representation → FlexAID format → verify atom count/bonds

### 4. Dataset Adapters (Python) — `dataset_adapters.py`, 0 tests

Seven adapter classes and the contact table builder are completely untested:

- `PDBbindAdapter`, `ITC187Adapter`, `BindingMOADAdapter`, `BindingDBAdapter`, `ChEMBLAdapter`, `DUDEAdapter`, `DEKOIS2Adapter`
- `create_adapter()` factory function
- `complexes_to_contact_table()`, `get_or_build_contact_table()`
- `normalize_affinity()` — unit conversions (Ki/IC50/Kd → ΔG)

**Risk**: Silent data corruption in training/benchmarking pipelines.

**Recommended tests**:
- Each adapter with minimal synthetic fixture files
- Affinity normalization edge cases (zero, negative, very large values)
- Factory function with all valid/invalid adapter names
- Contact table checksum stability

### 5. Energy Matrix CLI (Python) — `energy_matrix_cli.py`, 0 tests

All 10 subcommands are untested:

- `_cmd_train`, `_cmd_optimize`, `_cmd_evaluate`, `_cmd_convert`
- `_cmd_continuous_train`, `_cmd_build_contacts`, `_cmd_validate_gates`
- `_cmd_compare_runs`, `_cmd_list_runs`

**Recommended tests**: Argument parsing validation, dry-run smoke tests for each subcommand.

---

## Priority 2: Medium-Risk Gaps

### 6. PyMOL Plugin — 8 modules, 0 tests

The entire `pymol_plugin/` package lacks any tests:

- `gui.py` — FlexAIDSPanel widget
- `visualization.py` — pose rendering, Boltzmann coloring
- `results_adapter.py` — bridge to `flexaidds.load_results()`
- `entropy_heatmap.py`, `mode_animation.py`, `itc_comparison.py`, `interactive_docking.py`

**Recommended**: Mock-based unit tests (mock the PyMOL `cmd` module). At minimum test `results_adapter.py` which has no GUI dependency.

### 7. NATURaL RNA Module (C++) — 3 files, 0 tests

- `RibosomeElongation.cpp`, `TransloconInsertion.cpp`, `NucleationDetector.cpp`

**Recommended**: Basic unit tests for each class with synthetic RNA sequences.

### 8. Atom Assignment (C++) — 8+ files, 0 tests

- `assign_types.cpp`, `assign_radius.cpp`, `assign_radii.cpp`, `assign_constraint.cpp`, `assign_shift.cpp`, `assign_eigen.cpp`, `assign_shortflex.cpp`

**Recommended**: Verify known atom types produce correct radii/parameters. Test unknown atom type handling.

### 9. End-to-End Integration Tests — None exist

No test exercises the full pipeline: **GA → Scoring → StatMech → Binding Modes → Output**

**Recommended**:
- One small integration test with a synthetic receptor + ligand (< 50 atoms each)
- Validate that output binding modes have physically reasonable free energies
- Ensure deterministic output with fixed random seed

### 10. Scoring Functions (C++) — `vcfunction.cpp`, `cffunction.cpp`, `spfunction.cpp` — 0 tests

Core scoring functions used by Vcontacts have no standalone tests.

**Recommended**: Known atom pair at known distance → verify expected score.

---

## Priority 3: CI & Platform Gaps

| Gap | Current State | Recommendation |
|-----|--------------|----------------|
| **macOS C++ tests** | Disabled (`-DBUILD_TESTING=OFF`) | Fix build issues, enable tests |
| **Windows** | Excluded entirely | At minimum, compile-only CI job |
| **CUDA** | Always OFF in CI | Add optional GPU runner or mock dispatch |
| **Metal** | Always OFF in CI | Test on macOS runner when tests re-enabled |
| **AVX-512** | Builds but tests disabled | Enable tests on AVX-512 capable runner |
| **MPI** | Builds but never tested as MPI | Add `mpirun -np 2` test execution |
| **Real molecular data** | Only synthetic fixtures | Add small benchmark set (e.g., 3 PDBbind complexes) |

---

## Quantitative Summary

| Category | Files/Modules | Tested | Coverage |
|----------|--------------|--------|----------|
| C++ source files (LIB/) | ~236 | ~47 test files covering ~60 sources | ~25% |
| Python source modules | ~25 | 32 test files | ~85% by module, gaps in depth |
| PyMOL plugin | 8 modules | 0 | 0% |
| CI platforms | Linux/macOS/Win | Linux only (functional) | 33% |
| GPU backends | CUDA/Metal/ROCm | 0 | 0% |
| Integration (end-to-end) | Full pipeline | 0 | 0% |

---

## Recommended Action Plan

### Phase A — Fill Critical Gaps (High ROI)
1. Add C++ file I/O round-trip tests (PDB, grid, ligand formats)
2. Add RMSD and coordinate geometry tests with known reference values
3. Add ProcessLigand subsystem tests (SMILES → typed atoms)
4. Add Python `dataset_adapters.py` tests
5. Add Python `energy_matrix_cli.py` argument parsing tests

### Phase B — Integration & Platform
6. Create one small end-to-end integration test (GA → binding modes)
7. Add scoring function unit tests (`vcfunction`, `cffunction`)
8. Fix and re-enable macOS C++ testing in CI
9. Add PyMOL plugin mock tests for `results_adapter.py`

### Phase C — Completeness
10. Add atom assignment tests
11. Add NATURaL RNA module tests
12. Add MPI transport test with `mpirun -np 2`
13. Add small real-molecule benchmark fixtures
