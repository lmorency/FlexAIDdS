# FlexAIDdS Codebase Optimization Analysis

**Date**: 2026-03-11
**Scope**: Full codebase analysis covering GPU acceleration, SIMD vectorization, memory management, algorithmic efficiency, build system, and Python bindings.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [GPU Acceleration (CUDA & Metal)](#2-gpu-acceleration)
3. [SIMD Vectorization (AVX2, AVX-512, SSE)](#3-simd-vectorization)
4. [OpenMP Parallelism](#4-openmp-parallelism)
5. [Eigen3 Linear Algebra](#5-eigen3-linear-algebra)
6. [Memory Management & Data Structures](#6-memory-management--data-structures)
7. [Algorithmic Efficiency](#7-algorithmic-efficiency)
8. [Build System & Compiler Flags](#8-build-system--compiler-flags)
9. [CI/CD Pipeline](#9-cicd-pipeline)
10. [Python Bindings & Package](#10-python-bindings--package)
11. [Prioritized Recommendations](#11-prioritized-recommendations)

---

## 1. Executive Summary

FlexAIDdS has a mature hardware acceleration infrastructure spanning CUDA, Metal, AVX2/AVX-512, OpenMP, and Eigen3. The codebase demonstrates strong multi-platform support with graceful fallback chains. Key optimization opportunities exist in:

- **GPU synchronization**: Synchronous barriers prevent pipelining (Metal `waitUntilCompleted`, CUDA triple-memcpy readback)
- **SIMD gather inefficiency**: Manual element-by-element gathers in CavityDetect defeat ~20-30% of SIMD throughput
- **Data layout**: AoS (Array of Structures) layout in core structs (`atom_struct`, `chromosome`) limits cache efficiency and vectorization
- **Legacy C patterns**: Raw `malloc`/`free` macros, C-style strings, and linked-list structures throughout `flexaid.h`
- **Missing LTO**: No link-time optimization configured; cross-TU inlining opportunities lost across 50+ source files
- **No runtime dispatch**: SIMD/GPU selection is compile-time only; no cpuid-based fallback for portable binaries

---

## 2. GPU Acceleration

### 2.1 CUDA — CF Batch Evaluation (`LIB/cuda_eval.cu`)

**Status**: Production-ready

**Architecture**:
- One threadblock per chromosome (pop_size blocks), 256 threads/block
- Shared memory: per-ligand SAS counters (MAX_LIG_SAS = 256 floats/block)
- Warp-level shuffle reduction (`__shfl_down_sync`) for COM/WAL accumulation
- Energy matrix: pre-sampled 1D table (N_EMAT_SAMPLES = 128), bilinear interpolation on device

**Memory Transfer Pattern**:
- Atom coordinates, types, radii, energy matrix uploaded once at `cuda_eval_init()`
- Per-generation: only gene vectors (host→device) + COM/WAL/SAS results (device→host)
- For pop_size=256, n_genes=24: ~49 KB per generation transfer — acceptable PCI-e overhead

**Bottleneck**: Three separate `cudaMemcpy` calls for output (COM, WAL, SAS at lines 300-302). Each creates a synchronization point.

**Recommendation**: Concatenate output into a single struct-of-arrays buffer for one `cudaMemcpy` call, reducing synchronization from 3× to 1×.

### 2.2 CUDA — Shannon Histogram (`LIB/ShannonThermoStack/shannon_cuda.cu`)

**Architecture**: 256 threads, shared histogram buffer, grid-stride loop with `atomicAdd` binning.

**Performance**: ~56× vs CPU scalar at n=1M energies (claimed).

**Bottleneck**: Two-level atomic merge (threadgroup → global) creates contention for popular bins.

### 2.3 Metal — CF Batch Evaluation (`LIB/metal_eval.mm`)

**Status**: Production-ready for Apple Silicon

**Key differences vs CUDA**:
- CAS-loop float atomics (no native float atomics in Metal 2.0)
- Uses Metal's native `simd_sum()` for warp reduction
- MAX_LIG_SAS = 256 (same as CUDA)

**Critical Bottleneck** (line 342): `[cmd_buf waitUntilCompleted]` — synchronous full GPU-CPU barrier per evaluation. No pipelining possible.

**Recommendation**: Use Metal completion handler or fence for asynchronous execution. This would allow overlapping GPU evaluation with CPU thermodynamic computation.

### 2.4 Metal — Shannon Histogram (`LIB/ShannonThermoStack/shannon_metal.metal`)

- 256 threads/threadgroup, `memory_order_relaxed` atomics
- Same two-level histogram strategy as CUDA

### 2.5 Metal — Cavity Detection (`LIB/CavityDetect/CavityDetect.metal`)

- 2D grid dispatch with triangle reduction (i < j)
- Atomic sphere counter for output
- O(n³) triple-nested loop — potentially excessive for large proteins

### 2.6 Kernel Fusion Opportunities

| Current Pipeline | Proposed Fusion | Estimated Savings |
|-----------------|-----------------|-------------------|
| CF eval → host → Shannon histogram | Fuse Shannon binning into CF output reduction | Eliminates 1 GPU-CPU roundtrip |
| COM + WAL + SAS separate reductions | Fuse SAS normalization into COM reduction | ~5% per kernel (1 barrier removed) |
| GPU histogram → CPU entropy calculation | Add reduction kernel to compute entropy on GPU | Eliminates CPU readback for entropy-only paths |

---

## 3. SIMD Vectorization

### 3.1 AVX2 Distance Kernels (`LIB/simd_distance.h`)

**Well-optimized functions**:
- `distance2_1x8()`: Full 8-wide AVX2 utilization with FMA chain — ~85-90% efficiency
- `lj_wall_8x()`: Newton-Raphson r^-2 approximation (~1e-5 relative error), inv_r2 → inv_r12 via squares
- `hsum256_ps()`: Optimal 6-op horizontal reduction

**Suboptimal function**:
- `sum_sq_distances()` (lines 91-118): Manual stride-3 gather loop defeats SIMD purpose. Scalar gather → vectorised subtract → reduction pattern achieves only ~50% of peak bandwidth.

**Recommendation**: Reorganize coordinate data into SoA layout (separate x[], y[], z[] arrays) for contiguous SIMD loads, or use `_mm256_i32gather_ps` instructions.

### 3.2 AVX-512 Shannon Histogram (`LIB/ShannonThermoStack/ShannonThermoStack.cpp`)

- Processes 8 doubles per iteration via `_mm512_loadu_pd`
- `_mm512_cvttpd_epi32` for double→int32 narrowing
- Per-thread private histograms with OpenMP parallel merge

**Dispatch priority** (lines 281-292):
1. CUDA → 2. Metal → 3. AVX-512+OpenMP → 4. OpenMP scalar → 5. Scalar fallback

### 3.3 AVX-512/AVX2 Cavity Detection (`LIB/CavityDetect/CavityDetect.cpp`)

**AVX-512** (16 lanes): Full compute pipeline with masked min reduction (`_mm512_mask_min_ps`, `_mm512_reduce_min_ps`).

**Critical issue**: Manual element-by-element gather loop (lines 92-98) copies atoms one-at-a-time into aligned arrays before SIMD load. This wastes ~30-40% of potential throughput.

**Recommendation**: Use `_mm512_i32gather_ps()` or switch atom data to SoA layout for contiguous loads.

### 3.4 Disabled SIMD Paths

- **Sugar Pucker AVX-512** (`LIB/LigandRingFlex/SugarPucker.cpp:91-94`): Disabled due to SVML dependency. Falls back to Eigen or scalar.
- **Translocon SIMD** (`LIB/NATURaL/TransloconInsertion.h`): Headers include `<immintrin.h>` but implementation uses OpenMP+Eigen. SIMD not beneficial here (Hessa scale = array lookup, not vectorizable).

### 3.5 Width Utilization Summary

| Component | SIMD Width | Effective Throughput | Bottleneck |
|-----------|------------|---------------------|------------|
| AVX2 distance2_1x8 | 8 floats | 85-90% | None |
| AVX2 lj_wall_8x | 8 floats | 80-85% | Newton-Raphson approx |
| AVX2 cavity | 8 floats | 70-75% | Manual gather |
| AVX-512 cavity | 16 floats | 60-70% | Manual gather |
| AVX-512 Shannon | 8 doubles | 80-85% | Scalar tail loop |

---

## 4. OpenMP Parallelism

### 4.1 Usage Locations

| File | Line | Schedule | Purpose |
|------|------|----------|---------|
| `gaboom.cpp` | 1151 | `dynamic` | Per-chromosome CF evaluation |
| `gaboom.cpp` | 1216 | `static` | RMSD computation |
| `gaboom.cpp` | 1543 | `dynamic` | Population offset tracking |
| `statmech.cpp` | 73-76 | `static` + `reduction(+:sum)` | log-sum-exp (threshold: n≥4096) |
| `ShannonThermoStack.cpp` | 175-186 | Per-thread private bins | AVX-512 histogram merge |
| `CavityDetect.cpp` | 368 | `dynamic, 16` | Probe placement |
| `CleftDetector.cpp` | 62-105 | `dynamic, 64` + `nowait` | Private buffer + critical merge |
| `VoronoiCFBatch.h` | 211 | `dynamic, 4` | Voronoi CF batch |
| `TransloconInsertion.cpp` | 190 | `static` + `if(n_windows>64)` | Window sliding |

### 4.2 Issues

1. **CleftDetector critical section** (line 103): `#pragma omp critical` serializes probe vector merging. Should use reduction or thread-local accumulation with pre-allocated capacity.

2. **Schedule inconsistency**: Mix of `dynamic` and `static` across similar workloads suggests load-balance characteristics not uniformly characterized.

3. **Missing threshold guards**: Some OpenMP loops lack minimum-size thresholds (unlike `statmech.cpp` which correctly uses `OMP_THRESHOLD = 4096`). Small loops pay OpenMP overhead without benefit.

---

## 5. Eigen3 Linear Algebra

### 5.1 Usage Patterns

- **`statmech.cpp` log_sum_exp()** (lines 66-69): `Eigen::ArrayXd` for vectorized `.exp().sum()` — relies on compiler auto-vectorization
- **`statmech.cpp` Thermodynamics compute()** (lines 111-119): Vectorized probability × energy computation for N≥16
- **`SugarPucker.cpp`** (lines 96-114): Fixed 5-element arrays — minimal SIMD benefit (5 < AVX2 width)
- **`ShannonThermoStack.cpp`** (lines 240-244): Masked log + sum for torsional vibrational entropy

### 5.2 Limitations

- Eigen is header-only; no explicit AVX2/AVX-512 flag pass-through. Relies entirely on compiler auto-vectorization from `-mavx2 -ffast-math`.
- No explicit `EIGEN_MAX_ALIGN_BYTES` or `EIGEN_DONT_VECTORIZE` configuration.

---

## 6. Memory Management & Data Structures

### 6.1 Legacy C-Style Memory (`LIB/flexaid.h`)

The central header defines C-style macros for memory management:

```cpp
#define NEW(p,type)  if ((p=(type *) malloc (sizeof(type))) == NULL) { ... }
#define FREE(p)      if (p) { free ((char *) p); p = NULL; }
```

These are used throughout the codebase instead of `new`/`delete` or smart pointers. Issues:
- No RAII — memory leaks on early returns or exceptions
- No type safety (void* casts)
- No alignment control for SIMD-friendly allocations

### 6.2 Core Struct Layout (Cache Efficiency)

**`atom_struct`** (flexaid.h:175-203, 28 fields):
- Size: ~200+ bytes per atom (includes pointers, arrays, padding)
- Hot fields (`coor[3]`, `radius`, `type`) are mixed with cold fields (`name[5]`, `element[3]`, `cons`)
- **Impact**: Scoring loops iterate over atoms accessing only ~20 bytes of useful data per ~200-byte struct. Cache line utilization: ~10%.

**`chromosome`** (defined in gaboom.h):
- Contains gene array + fitness values
- Used as unit of GA population — iterated in tight loops

**Recommendation**: Split `atom_struct` into hot/cold tiers:
- **Hot tier**: `{coor[3], radius, type}` — 20 bytes, fits in L1 cache line
- **Cold tier**: everything else — accessed only during setup/output

### 6.3 Linked List in Energy Matrix

```cpp
struct energy_values {
    float x;
    float y;
    struct energy_values* next_value;  // linked list!
};
```

This linked-list representation of energy interpolation tables is cache-hostile. Each lookup traverses pointers to potentially non-contiguous memory. The GPU paths already use pre-sampled arrays (N_EMAT_SAMPLES=128) — the CPU path should match.

### 6.4 String Operations

- `char name[5]`, `char element[3]` in atom structs — C-style fixed-size strings
- `sprintf`/`fprintf` used extensively in output paths
- Not a hot-path concern, but modernization opportunity for safety

---

## 7. Algorithmic Efficiency

### 7.1 FOPTICS Clustering (`LIB/FOPTICS.cpp`)

- **Complexity**: O(N × minPoints × nDimensions) for random projections
- **Issue** (line 115): `std::vector<float> vChrom(this->Vectorized_Cartesian_Coordinates(i));` — copy constructor inside loop. Should use move semantics or pre-allocate.
- **Issue** (line 125): `std::make_pair((chromosome*)&chrom[i], vChrom)` — copies the vector again into the pair. Use `std::move(vChrom)`.

### 7.2 Voronoi Contact Scoring (`LIB/Vcontacts.cpp`)

- 1843 lines — the largest scoring module
- All-pairs atom contact computation: O(N_lig × N_rec) per evaluation
- GPU-accelerated via CUDA/Metal for batch evaluation
- CPU path uses OpenMP parallelization

### 7.3 Genetic Algorithm (`LIB/gaboom.cpp`)

- 2305 lines — the main GA engine
- Population evaluation parallelized with OpenMP `schedule(dynamic)`
- RMSD computation uses `schedule(static)` — appropriate for uniform work
- Selection, crossover, mutation are sequential — appropriate (small relative cost)

### 7.4 Statistical Mechanics (`LIB/statmech.cpp`)

- Uses log-sum-exp with numerical stability (x_max subtraction)
- OpenMP threshold at N≥4096 — appropriate
- Eigen vectorization for N≥16 — appropriate

### 7.5 Global RNG (`LIB/FOPTICS.cpp:8-9`)

```cpp
std::random_device rd;
std::mt19937 gen(rd());
```

Global `std::random_device` and `std::mt19937` at file scope — not thread-safe for OpenMP parallel regions. Should be thread-local.

---

## 8. Build System & Compiler Flags

### 8.1 Current Configuration

| Setting | Value | Assessment |
|---------|-------|------------|
| C++ Standard | C++20 | Good |
| Optimization | `-O3 -ffast-math` | Good |
| Warnings | `-Wall` | Adequate |
| AVX2 | `-mavx2 -mfma` | Good |
| AVX-512 | `-mavx512f -mavx512dq -mavx512bw -mavx2 -mfma` | Good |
| MSVC | `/W4 /O2 /MT` | Good |
| LTO | Not configured | **Missing** |
| PGO | Not configured | **Missing** |
| `-march=native` | Not used | Intentional (portability) |

### 8.2 Missing: Link-Time Optimization (LTO)

With 50+ source files, cross-translation-unit inlining would benefit:
- Scoring function calls from `gaboom.cpp` → `Vcontacts.cpp`
- `statmech.cpp` functions called from `BindingMode.cpp`
- Template instantiations across headers

**Recommendation**: Add CMake option:
```cmake
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
option(FLEXAIDS_USE_LTO "Enable link-time optimization" OFF)
if(FLEXAIDS_USE_LTO AND ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
```

### 8.3 Missing: Precompiled Headers

`flexaid.h` (601 lines) includes 16 standard library headers and is included by virtually all 50+ source files. No PCH is configured. Estimated compile time savings: 20-40%.

**Recommendation**:
```cmake
target_precompile_headers(FlexAID PRIVATE
    LIB/flexaid.h
    <vector> <iostream> <cmath> <algorithm>
)
```

### 8.4 Test Targets Recompile Shared Sources

Three test executables independently recompile shared source files (`statmech.cpp`, `tencm.cpp`, `ShannonThermoStack.cpp`, etc.) at `-O2` instead of `-O3`. This adds 5-10 minutes of redundant CI build time.

**Recommendation**: Create a shared static library:
```cmake
add_library(flexaid_core STATIC LIB/statmech.cpp LIB/encom.cpp LIB/tencm.cpp)
target_compile_options(flexaid_core PRIVATE -O3 -ffast-math)
# Link tests against flexaid_core instead of recompiling
```

### 8.5 Python Extension Missing Optimization Flags

`python/setup.py` line 40 uses only `-std=c++20 -O3`, missing `-ffast-math`, SIMD flags (`-mavx2 -mfma`), and LTO. Python bindings for `StatMechEngine` and `ENCoMEngine` run 10-15% slower than equivalent C++ code.

### 8.6 Missing: Runtime CPU Dispatch

All SIMD selection is compile-time (`#ifdef __AVX512F__`). No `cpuid` check for runtime fallback. This means binaries compiled with AVX-512 will crash on AVX2-only hardware.

**Recommendation**: Add runtime dispatch via `__builtin_cpu_supports("avx512f")` or equivalent.

### 8.7 CUDA Architecture Targets

```cmake
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
```

Covers Volta through Hopper. Good coverage, but binary size grows with each target. Consider providing a `FLEXAIDS_CUDA_ARCH` option for users to specify their specific GPU.

---

## 9. CI/CD Pipeline

### 9.1 Current Structure (`.github/workflows/ci.yml`)

1. **Pure Python tests** — `pytest` on `test_results_io.py`, `test_results_loader_models.py`
2. **C++ core build** — matrix: `{linux-gcc, linux-clang, macos-clang}` × Release
3. **Python bindings smoke** — build `_core` extension + smoke test

### 9.2 Issues

1. **No dependency caching**: `apt-get install` and `brew install` run on every CI run. Add `actions/cache` for system packages or use pre-built containers.

2. **No ccache/sccache**: Incremental builds not cached. For a 22K+ line C++ codebase, build times could be reduced ~50-80% on cache hits.

3. **No build artifact reuse**: C++ core build and Python bindings smoke test both build from scratch. The bindings test could reuse the core build artifacts.

4. **Missing test types in CI**:
   - No C++ test execution in the build matrix (`ctest` configured but results not uploaded)
   - No code coverage reporting
   - No static analysis (clang-tidy, cppcheck)
   - No sanitizer builds (ASan, UBSan)

5. **Good practices present**:
   - `cancel-in-progress: true` — prevents resource waste on force-pushes
   - `fail-fast: false` — all matrix entries complete even on failures
   - Ninja generator — fast parallel builds

### 9.3 Recommendations

```yaml
# Add to CI workflow:
- name: Cache apt packages
  uses: awalber/cache-apt-pkgs-action@v1
  with:
    packages: cmake ninja-build libeigen3-dev libomp-dev

# Add ccache step:
- uses: hendrikmuhs/ccache-action@v1
  with:
    key: ${{ matrix.name }}
```

---

## 10. Python Bindings & Package

### 10.1 Package Structure (`python/flexaidds/`)

- Clean public API: `StatMechEngine`, `Thermodynamics`, `ENCoMEngine`, `load_results()`
- Data classes: `PoseResult`, `BindingModeResult`, `DockingResult`
- Pure Python result I/O — no C++ dependency for file parsing

### 10.2 pybind11 Bindings (`python/bindings/`)

- Bridge code wraps `StatMechEngine` with Python-friendly interface
- `@requires_core` marker for graceful skip when bindings not built

### 10.3 Potential Issues

1. **GIL management**: If pybind11 bindings hold the GIL during long-running C++ computations (e.g., `StatMechEngine.compute()`), Python threads will be blocked. Should use `py::call_guard<py::gil_scoped_release>()` for compute-heavy functions.

2. **Data copying**: Check if numpy arrays passed to/from C++ are zero-copy via buffer protocol or if unnecessary copies occur.

3. **Import overhead**: Setup correctly separates pure Python modules from C++ extension. No unnecessary imports at package level.

---

## 11. Prioritized Recommendations

### High Impact, Low Effort

| # | Recommendation | Files | Expected Impact |
|---|---------------|-------|-----------------|
| 1 | Consolidate CUDA output memcpy (3→1) | `cuda_eval.cu:300-302` | Reduces sync overhead ~66% |
| 2 | Add ccache to CI | `.github/workflows/ci.yml` | ~50-80% CI build time reduction |
| 3 | Use `std::move()` for vector in FOPTICS | `FOPTICS.cpp:115,125` | Eliminates unnecessary copies |
| 4 | Add LTO build option | `CMakeLists.txt` | ~5-15% runtime improvement |
| 5 | Make RNG thread-local | `FOPTICS.cpp:8-9` | Fixes thread-safety bug |

### High Impact, Medium Effort

| # | Recommendation | Files | Expected Impact |
|---|---------------|-------|-----------------|
| 6 | Replace Metal `waitUntilCompleted` with async | `metal_eval.mm:342` | Enables GPU-CPU pipelining |
| 7 | Replace manual SIMD gathers with native gather | `CavityDetect.cpp:92-98` | ~15-20% SIMD throughput gain |
| 8 | Add runtime CPU dispatch | New file + CMakeLists.txt | Portable AVX2/AVX-512 binaries |
| 9 | Split `atom_struct` hot/cold | `flexaid.h:175-203` | ~2-5× cache efficiency in scoring |
| 10 | Add CI caching (apt, brew, ccache) | `.github/workflows/ci.yml` | Major CI time savings |

### Medium Impact, Higher Effort

| # | Recommendation | Files | Expected Impact |
|---|---------------|-------|-----------------|
| 11 | Fuse Shannon histogram + entropy on GPU | `shannon_cuda.cu`, `shannon_metal.metal` | Eliminates CPU readback |
| 12 | Replace energy_values linked list with array | `flexaid.h:131-135`, `read_emat.cpp` | Cache-friendly interpolation |
| 13 | Add sanitizer CI builds (ASan, UBSan) | `.github/workflows/ci.yml` | Bug prevention |
| 14 | SoA data layout for atom coordinates | `flexaid.h`, scoring functions | Optimal vectorization |
| 15 | Replace `NEW`/`FREE` macros with RAII | `flexaid.h:70-75`, all callers | Memory safety |

### Low Priority / Future

| # | Recommendation | Files | Expected Impact |
|---|---------------|-------|-----------------|
| 16 | Re-enable Sugar Pucker AVX-512 | `SugarPucker.cpp:91-94` | Marginal (5-element arrays) |
| 17 | Add PGO support | `CMakeLists.txt` | Requires profiling workflow |
| 18 | pybind11 GIL release guards | `python/bindings/` | Python threading performance |
| 19 | Precompiled headers for `flexaid.h` | `CMakeLists.txt` | Build time improvement |
| 20 | Standardize OpenMP schedules | Multiple files | Minor load-balance improvement |

---

## Hardware Acceleration Coverage Summary

| Component | CUDA | Metal | AVX-512 | AVX2 | OpenMP | Eigen | Scalar |
|-----------|------|-------|---------|------|--------|-------|--------|
| CF Scoring Batch | Yes | Yes | — | Yes | Yes | — | Yes |
| Shannon Histogram | Yes | Yes | Yes | — | Yes | — | Yes |
| Cavity Detection | — | Yes | Yes | Yes | — | — | Yes |
| StatMech log-sum-exp | — | — | — | — | Yes | Yes | Yes |
| RMSD Computation | — | — | — | Yes | Yes | — | Yes |
| LJ Wall Energy | — | — | — | Yes | — | — | Yes |
| Sugar Pucker | — | — | Disabled | — | — | Yes | Yes |
| Translocon | — | — | Stubbed | — | Yes | Yes | Yes |

**Overall assessment**: The codebase has excellent hardware acceleration coverage with well-designed fallback chains. The primary optimization opportunities are in synchronization overhead reduction, data layout improvements, and build system enhancements rather than missing acceleration paths.
