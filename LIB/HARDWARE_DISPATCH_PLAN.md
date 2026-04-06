# Hardware Dispatch System: Comprehensive Review & Enhancement Plan

**Author**: Claude (code review & architectural plan)
**Date**: 2026-04-06
**Scope**: All hardware acceleration code in FlexAIDdS
**Status**: PLAN ONLY — no implementation yet

---

## Table of Contents

1. [Phase 1: Current State Review](#phase-1-current-state-review)
2. [Phase 2: Enhancement Plan](#phase-2-enhancement-plan)
3. [Architecture Diagram](#architecture-diagram)
4. [Class Hierarchy](#class-hierarchy)
5. [File-by-File Implementation Plan](#file-by-file-implementation-plan)
6. [Build System Changes](#build-system-changes)
7. [Test Matrix](#test-matrix)
8. [Migration Path](#migration-path)

---

## Phase 1: Current State Review

### 1.1 Inventory of All Hardware-Related Files

#### Dispatch & Detection Layer (6 files)

| File | Lines | State | Purpose |
|------|-------|-------|---------|
| `LIB/HardwareDispatch.h/cpp` | ~500 | **Functional** | Main dispatcher: Meyers singleton, per-kernel backend selection |
| `LIB/hardware_dispatch.h/cpp` | ~440 | **Functional** | Lower-level dispatch: Boltzmann, log-sum-exp, Shannon entropy |
| `LIB/hardware_detect.h/cpp` | ~220 | **Functional** | CPUID probing, GPU detection, capability caching |
| `LIB/GPUContextPool.h` | ~200 | **Functional** | Thread-safe GPU context singleton with mutex+CV |
| `LIB/GAContext.h/cpp` | ~80 | **Functional** | Per-instance GA state for re-entrant parallel execution |
| `LIB/ProcessLigand/ROCmDispatch.h/cpp` | ~350 | **Functional** | ROCm dispatch class + CPU-only stubs |

#### CUDA Backend (5 kernel files)

| File | Lines | State | Purpose |
|------|-------|-------|---------|
| `LIB/cuda_eval.cu/cuh` | ~340 | **Functional** | Chromosome fitness (CF+WAL+SAS) batch scoring |
| `LIB/ShannonThermoStack/shannon_cuda.cu/cuh` | ~100 | **Functional** | Histogram binning kernel |
| `LIB/tENCoM/tencm_cuda.cu/cuh` | ~250 | **Functional** | Contact discovery + Hessian assembly |
| `LIB/TurboQuant.cu` | ~230 | **Functional** | Vector quantize/dequantize |
| `LIB/gpu_fast_optics.cu` | ~250 | **Functional** | k-NN search for FOPTICS clustering |

#### Metal Backend (5 kernel files + 5 bridges)

| File | Lines | State | Purpose |
|------|-------|-------|---------|
| `LIB/metal_eval.h/mm` | ~380 | **Functional** | Chromosome fitness (CF+WAL+SAS) batch scoring |
| `LIB/CavityDetect/CavityDetect.metal` | ~165 | **Functional** | SURFNET probe-sphere generation |
| `LIB/CavityDetect/CavityDetectMetalBridge.h/mm` | ~160 | **Functional** | Obj-C++ bridge for cavity detection |
| `LIB/ShannonThermoStack/shannon_metal.metal` | ~130 | **Functional** | Histogram + Boltzmann + reduction kernels |
| `LIB/ShannonThermoStack/ShannonMetalBridge.h/mm` | ~310 | **Functional** | Exemplary: thread-safe singleton, CPU fallback |
| `LIB/tENCoM/tencm_metal.h/mm` | ~350 | **Functional** | Contact discovery + Hessian (embedded MSL) |
| `LIB/TurboQuant.metal` | ~160 | **Functional** | Batch quantize/dequantize |
| `LIB/TurboQuantMetalBridge.h/mm` | ~250 | **Functional** | Obj-C++ bridge for TurboQuant |

#### ROCm/HIP Backend (4 files)

| File | Lines | State | Purpose |
|------|-------|-------|---------|
| `LIB/hip_eval.hip` | ~680 | **Functional** | Chromosome fitness (gene pack + CF scoring) |
| `LIB/hip_eval.h` | ~60 | **Functional** | Public API header |
| `LIB/ProcessLigand/hip_eval.hip` | ~240 | **Functional** | Per-pose energy (LJ+Coulomb) |
| `LIB/rocm_detect.cpp` | ~180 | **Functional** | Runtime device detection |

#### SIMD / Vectorization (5 files)

| File | Lines | State | Purpose |
|------|-------|-------|---------|
| `LIB/simd_distance.h` | ~534 | **Production** | AVX-512/AVX2 distance, LJ-wall, dot3, Boltzmann |
| `LIB/soft_contact_matrix.h` | ~550 | **Production** | AVX-512/AVX2 gather-based contact scoring |
| `LIB/VoronoiCFBatch_SoA.h` | ~300 | **Functional** | Partial SoA vectorization (pre-screen only) |
| `LIB/TurboQuant.h` | ~1300 | **Production** | AVX-512/AVX2 quantization with Eigen QR |
| `LIB/VoronoiCFBatch.h` | ~250 | **Production** | Batch Voronoi CF with OpenMP |

#### CMake Configuration (3 files)

| File | Lines | Purpose |
|------|-------|---------|
| `CMakeLists.txt` (root) | ~1816 | All GPU/SIMD detection, compilation, linking |
| `LIB/CMakeLists.txt` | ~192 | flexaid_core OBJECT library source list |
| `cmake/MetalAcceleration.cmake` | ~212 | Metal-specific build helper |
| `cmake/CMakeLists_rocm_patch.cmake` | ~193 | ROCm CMake integration guide |

#### OpenMP Usage
- **57 pragma sites** across 20+ files
- Patterns: parallel for, thread-local accumulation, reductions, collapse(2), nowait
- Hybrid SIMD+OpenMP in `hardware_dispatch.cpp` and `ShannonThermoStack.cpp`

#### Eigen Usage
- **~25 files**, always-on (hard requirement)
- Critical: `SelfAdjointEigenSolver` for TENCoM, `ArrayXd` for partition functions
- Auto-vectorizes to AVX2/AVX-512 transparently

---

### 1.2 Current Backend Selection Logic

```
Runtime Priority Chain (per-kernel-type aware):

  Fitness Evaluation:    CUDA(100) > ROCm(90) > Metal(80) > CPU
  Shannon Entropy:       CUDA > Metal > AVX512+OMP > AVX512 > OMP > scalar
  Distance/RMSD:         AVX512 > AVX2 > OMP > scalar  (GPU overhead too high)
  Boltzmann Weights:     AVX512 > AVX2 > OMP > scalar
  Contact Discovery:     CUDA > Metal > AVX512 > AVX2 > scalar
  Hessian Assembly:      CUDA > Metal > Eigen+OMP > scalar
  Eigendecomposition:    Eigen SelfAdjointEigenSolver (CPU only, no GPU path)
  k-NN (FOPTICS):        CUDA > CPU  (no Metal/ROCm path)
  Cavity Detection:      Metal > CPU  (no CUDA/ROCm path)
  Vector Quantization:   CUDA > Metal > AVX512 > AVX2 > scalar
```

GPU dispatch thresholds (intelligent):
- Shannon entropy GPU: N > 500,000 samples (below this, CPU is faster)
- TENCoM GPU: N >= GPU_THRESHOLD (configurable)
- Metal eval: pop_size checked against max_pop

---

### 1.3 Critical Issues Found

#### P0 — Must Fix (Safety / Correctness)

| # | File | Issue | Impact |
|---|------|-------|--------|
| 1 | `tencm_cuda.cu:19-21` | **Static `s_initialised`, `s_available`, `s_stream` unprotected** — concurrent `init()` calls race | Data corruption, crash |
| 2 | `tencm_cuda.cu:148-179` | **`cudaMalloc`/`cudaMemcpy` calls NOT error-checked** — silent allocation failures | Silent wrong results |
| 3 | `tencm_cuda.cu:243` | **Division by `sqrtf(...)` without zero-check** — produces NaN | NaN propagation |
| 4 | `tencm_cuda.cu:40` | **`cudaSetDevice(0)` hardcoded** — ignores multi-GPU | Wrong device on multi-GPU |

#### P1 — Should Fix (Silent Failures / Data Loss)

| # | File | Issue | Impact |
|---|------|-------|--------|
| 5 | `cuda_eval.cu:290-297` | Pop/gene overflow prints to stderr, returns silently | No fitness scores, undetected |
| 6 | `cuda_eval.cu:282-286` | SAS > 512 atoms silently gives zero SAS energy | Wrong energetics |
| 7 | `metal_eval.mm:237-254` | Device creation / shader compile errors not propagated | Hard crash downstream |
| 8 | `metal_eval.mm:302-309` | `pop_size > max_pop` silently returns | No fitness scores |
| 9 | `CavityDetectMetalBridge.mm:132` | Sphere count silently clamped to kMaxSpheres | Lost cavity data |
| 10 | `shannon_cuda.cu:88-93` | Validation failure returns silently (zero-fill) | Wrong entropy |
| 11 | `TurboQuantMetalBridge.mm:144` | No check that `d <= maxTotalThreadsPerThreadgroup` | Silent GPU failure |
| 12 | `TurboQuant.cu:206` | No stream validation (null stream crashes) | CUDA crash |

#### P2 — Performance Issues

| # | File | Issue | Est. Impact |
|---|------|-------|-------------|
| 13 | `CavityDetect.metal:62-86` | Branch divergence in clash check + redundant burial computation | ~40-60% GPU waste |
| 14 | `tencm_cuda.cu:99-123` | O(N^2) scan for pair→(i,j) mapping on GPU | Poor GPU utilization |
| 15 | `tencm_cuda.cu:207-226` | Half threads idle (upper-triangle only), double atomicAdd | ~40-50% underuse |
| 16 | `tencm_metal.mm:64-90` | Same triangular load imbalance as CUDA variant | ~40-50% underuse |
| 17 | `metal_eval.mm:313-316` | Row-major strided double→float conversion | ~20% transfer overhead |
| 18 | `gpu_fast_optics.cu:136-194` | Thread-0 serial merge O(N * LOCAL_K_MAX) | Limits k-NN scale |

#### P3 — Missing Fallback Paths

| # | Component | Missing |
|---|-----------|---------|
| 19 | `metal_eval` | No CPU fallback if Metal unavailable |
| 20 | `cuda_eval` | No CPU fallback for gene evaluation |
| 21 | `gpu_fast_optics` | No Metal/ROCm path (CUDA only) |
| 22 | `CavityDetect` | No CUDA/ROCm path (Metal only) |
| 23 | `tencm_cuda` / `tencm_metal` | No CPU fallback in these files (caller must provide) |
| 24 | `TurboQuant` (Metal) | Stubs only, no inline CPU implementation |

#### P4 — Inconsistencies Across Backends

| # | Issue | Details |
|---|-------|---------|
| 25 | **Two dispatcher files** | `HardwareDispatch.h/cpp` AND `hardware_dispatch.h/cpp` — overlapping responsibility |
| 26 | **ARC flag mismatch** | Root CMakeLists uses `-fno-objc-arc`; MetalAcceleration.cmake uses `-fobjc-arc` |
| 27 | **Error handling inconsistency** | ShannonMetalBridge: exemplary CPU fallback; metal_eval: zero fallback |
| 28 | **Context management** | GPUContextPool manages CUDA/Metal; ROCm manages its own context |
| 29 | **Async dispatch** | No completion callbacks anywhere — all GPU calls are synchronous |
| 30 | **Feature parity** | k-NN only on CUDA; cavity detection only on Metal |

---

### 1.4 What Works Well (Do Not Change)

These components are production-grade and should be preserved as-is:

1. **`ShannonMetalBridge.mm`** — Exemplary: `std::once_flag` singleton, 4 kernels, CPU fallback for all, thread-safe. Reference implementation.
2. **`simd_distance.h`** — Clean compile-time dispatch, FMA, Newton-Raphson rsqrt, proper tail loops.
3. **`hardware_detect.cpp`** — Correct CPUID probing (leaf 7 for AVX-512), static cache, cross-platform.
4. **`GPUContextPool.h`** — Sophisticated mutex+CV with double-checked locking, ref counting, dimension-change serialization.
5. **`soft_contact_matrix.h`** — Proper AVX-512 gather intrinsics, `__mmask16` masking, Eigen fallback.
6. **OpenMP thread-local accumulation** pattern in `statmech.cpp`, `tencm.cpp`, `ShannonThermoStack.cpp`.
7. **Conditional parallelism** (`if(n > threshold)`) in ParallelCampaign, DatasetRunner, TransloconInsertion.
8. **Eigen integration** — Zero-copy `Map<>`, vectorized log/exp/select, auto-vectorization.

---

## Phase 2: Enhancement Plan

### 2.1 Design Goals

1. **Single entry point** for all hardware dispatch (merge the two dispatcher files)
2. **Runtime detection** with compile-time gating (current pattern — preserve it)
3. **Complete fallback chain**: CUDA/ROCm → Metal → AVX-512 → AVX2 → AVX → SSE4.2 → scalar
4. **Production error handling**: every GPU call checked, errors propagated (no silent failures)
5. **RAII for all GPU allocations**: no manual `cudaFree`/`hipFree` in caller code
6. **Context pooling extended to ROCm** (currently only CUDA/Metal)
7. **Async dispatch foundation**: optional completion callbacks for GPU kernels
8. **Testable**: every backend independently testable with mock/stub support

### 2.2 Non-Goals (Explicitly Out of Scope)

- GPU eigendecomposition (Eigen CPU is fast enough for M <= 50)
- Vulkan compute backend (too much surface area for too little gain)
- WebGPU/WASM GPU (the typescript/wasm/ build is orphaned and unrelated)
- Rewriting working kernel code (only fix bugs, not rewrite)

---

## Architecture Diagram

```
                         ┌─────────────────────────────┐
                         │       User Code              │
                         │  (gaboom, statmech, tencm,   │
                         │   ParallelDock, Campaign)     │
                         └──────────┬──────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   UnifiedHardwareDispatch      │
                    │   (Meyers singleton)           │
                    │                               │
                    │  detect() → HardwareInfo       │
                    │  best_backend(KernelType)      │
                    │  dispatch<KernelType>(args)     │
                    │  override_backend(Backend)      │
                    └──────────┬────────────────────┘
                               │
              ┌────────────────┼────────────────────┐
              │                │                    │
              ▼                ▼                    ▼
     ┌────────────┐   ┌────────────┐      ┌────────────┐
     │ GPUBackend │   │ SIMDBackend│      │ ScalarBack │
     │ Interface  │   │ Interface  │      │ end        │
     └─────┬──────┘   └─────┬──────┘      └────────────┘
           │                │
     ┌─────┼─────┐    ┌─────┼─────┐
     │     │     │    │     │     │
     ▼     ▼     ▼    ▼     ▼     ▼
   CUDA  ROCm  Metal AVX512 AVX2 SSE4.2
     │     │     │
     ▼     ▼     ▼
  ┌──────────────────────────────────────┐
  │        GPUContextPool                │
  │  (mutex + CV + ref-count per device) │
  │                                      │
  │  CUDAContext  ROCmContext  MetalCtx   │
  │  (RAII)       (RAII)      (RAII)     │
  └──────────────────────────────────────┘

  ┌──────────────────────────────────────┐
  │       Kernel Registry                │
  │                                      │
  │  KernelType        Backends          │
  │  ────────────      ─────────         │
  │  FITNESS_EVAL  →  CUDA,ROCm,Metal,CPU│
  │  SHANNON_HIST  →  CUDA,Metal,AVX,CPU │
  │  CONTACT_DISC  →  CUDA,Metal,AVX,CPU │
  │  HESSIAN_ASM   →  CUDA,Metal,Eigen   │
  │  KNN_SEARCH    →  CUDA,ROCm,CPU      │
  │  CAVITY_DET    →  Metal,CPU           │
  │  TURBO_QUANT   →  CUDA,Metal,AVX,CPU │
  │  LOG_SUM_EXP   →  AVX512,AVX2,OMP    │
  │  BOLTZMANN     →  AVX512,AVX2,OMP    │
  │  DISTANCE      →  AVX512,AVX2,OMP    │
  └──────────────────────────────────────┘
```

### Memory Management Model

```
  ┌─────────────────────────────────────┐
  │         GPUBuffer<T>                │
  │  (RAII wrapper, move-only)          │
  │                                     │
  │  - device_ptr_  (owning)            │
  │  - size_        (element count)     │
  │  - backend_     (CUDA/ROCm/Metal)   │
  │                                     │
  │  ctor(size, backend) → allocate     │
  │  dtor()              → free         │
  │  upload(host_ptr)    → H2D copy     │
  │  download(host_ptr)  → D2H copy     │
  │  raw()               → device_ptr   │
  └─────────────────────────────────────┘

  ┌─────────────────────────────────────┐
  │       GPUEvent (async support)      │
  │  (completion token, move-only)      │
  │                                     │
  │  - event_ (cudaEvent / MTLEvent)    │
  │  - is_complete() → bool             │
  │  - wait()        → block            │
  │  - on_complete(callback)            │
  └─────────────────────────────────────┘
```

---

## Class Hierarchy

```cpp
// ---- Enums ----

enum class Backend : uint8_t {
    AUTO    = 0,   // Let dispatcher choose
    SCALAR  = 1,
    SSE42   = 2,
    AVX     = 3,   // (reserved, not currently used)
    AVX2    = 4,
    AVX512  = 5,
    OPENMP  = 6,
    CUDA    = 7,
    ROCM    = 8,
    METAL   = 9
};

enum class KernelType : uint8_t {
    FITNESS_EVAL,     // Chromosome batch scoring
    SHANNON_HIST,     // Shannon entropy histogram
    CONTACT_DISC,     // Contact discovery (ENCoM/tENCoM)
    HESSIAN_ASM,      // Hessian matrix assembly
    KNN_SEARCH,       // k-NN for FOPTICS
    CAVITY_DET,       // SURFNET cavity detection
    TURBO_QUANT,      // Vector quantization
    LOG_SUM_EXP,      // Numerically stable reduction
    BOLTZMANN_BATCH,  // Boltzmann weight computation
    DISTANCE_BATCH,   // Pairwise distance computation
    RMSD              // RMSD calculation
};

// ---- Error Handling ----

enum class DispatchError : uint8_t {
    OK = 0,
    NO_BACKEND,        // No backend available for this kernel
    ALLOC_FAILED,      // GPU memory allocation failed
    LAUNCH_FAILED,     // Kernel launch failed
    SYNC_FAILED,       // Device synchronization failed
    INVALID_ARGS,      // Bad input dimensions / null pointers
    OVERFLOW,          // Buffer capacity exceeded
    DEVICE_LOST        // GPU device became unavailable
};

struct DispatchResult {
    DispatchError error = DispatchError::OK;
    Backend       used_backend = Backend::AUTO;
    double        elapsed_ms = 0.0;   // optional timing
    std::string   detail;             // error detail string (empty on OK)

    explicit operator bool() const { return error == DispatchError::OK; }
};

// ---- Core Classes ----

class HardwareInfo {
    // CPU
    bool has_sse42, has_avx2, has_fma, has_avx512f, has_avx512dq, has_avx512bw;
    bool has_avx512vnni;
    bool has_avx512;  // composite: f && dq && bw
    // GPU
    bool has_cuda;  int cuda_device_count; ...
    bool has_rocm;  int rocm_device_count; ...
    bool has_metal; ...
    // Parallelism
    bool has_openmp; int omp_max_threads;
    bool has_eigen;
    // Methods
    std::string summary() const;
};

class UnifiedHardwareDispatch {  // Meyers singleton
public:
    static UnifiedHardwareDispatch& instance();

    // Detection (idempotent, thread-safe)
    const HardwareInfo& detect();
    bool is_available(Backend b) const;

    // Backend selection
    Backend best_backend(KernelType kt) const;
    void set_override(Backend b);
    void clear_override();

    // ---- Dispatch entry points ----
    // Each returns DispatchResult instead of void/silent-fail

    DispatchResult compute_shannon_entropy(
        const double* values, int n, int num_bins,
        double& out_entropy, Backend b = Backend::AUTO);

    DispatchResult compute_boltzmann_batch(
        const double* energies, int n, double beta,
        double* out_weights, Backend b = Backend::AUTO);

    DispatchResult log_sum_exp(
        const double* values, int n,
        double& out_result, Backend b = Backend::AUTO);

    DispatchResult eval_fitness_batch(
        /* GA population args */,
        Backend b = Backend::AUTO);

    // ... other kernel dispatches ...

    // Context pool access (extended to ROCm)
    GPUContextPool& gpu_pool();

private:
    UnifiedHardwareDispatch();
    HardwareInfo info_;
    std::optional<Backend> override_;
    GPUContextPool pool_;
};

class GPUContextPool {  // Extended with ROCm
    // Existing CUDA + Metal context management
    // NEW: ROCm context with same mutex+CV+refcount pattern
    // NEW: GPUBuffer<T> factory for RAII allocations
};
```

---

## File-by-File Implementation Plan

### Phase A: Fix P0 Critical Issues (safety / correctness)

These fixes are small, surgical, and must happen first.

#### A1. `LIB/tENCoM/tencm_cuda.cu` — Thread safety + error checking

**Changes:**
1. Add `static std::mutex s_mtx;` and guard `init()` with `std::lock_guard`
2. Wrap ALL `cudaMalloc`/`cudaMemcpy` calls with `CUDA_CHECK()` macro
3. Add zero-check before `sqrtf()` division (line 243)
4. Replace `cudaSetDevice(0)` with device selection from GPUContextPool
5. Add `CUDA_CHECK` to `shutdown()` stream destruction

**Estimated size:** ~40 lines changed

#### A2. `LIB/cuda_eval.cu` — Error propagation

**Changes:**
1. Return `DispatchError` (or throw) from `cuda_eval_batch()` when pop_size/n_genes overflow
2. Log AND return error code for SAS atom overflow (>512)
3. Guard `cuda_eval_shutdown()` against double-free with a flag

**Estimated size:** ~25 lines changed

#### A3. `LIB/metal_eval.mm` — Error propagation + fallback

**Changes:**
1. Return `DispatchError` from `metal_eval_init()` if device/shader creation fails
2. Return `DispatchError` from `metal_eval_batch()` if pop_size > max_pop
3. Add CPU fallback function signature (implementation can delegate to `cffunction`/`spfunction`)

**Estimated size:** ~30 lines changed

### Phase B: Merge Dispatchers into UnifiedHardwareDispatch

Currently there are TWO dispatcher files with overlapping responsibility:
- `HardwareDispatch.h/cpp` — high-level (Shannon, Boltzmann, distance, RMSD)
- `hardware_dispatch.h/cpp` — lower-level (log-sum-exp, Boltzmann, AVX-512 kernels)

#### B1. Create `LIB/UnifiedHardwareDispatch.h` (NEW)

**Contents:**
- Merge `HardwareDispatch.h` and `hardware_dispatch.h` into single header
- Define `Backend`, `KernelType`, `DispatchError`, `DispatchResult` enums/structs
- Declare `UnifiedHardwareDispatch` class (Meyers singleton)
- Keep same internal dispatch logic, just unified interface
- Include `GPUContextPool.h` as member

**Note:** Preserve both existing headers as thin `#include` forwards for backward compat during migration.

#### B2. Create `LIB/UnifiedHardwareDispatch.cpp` (NEW)

**Contents:**
- Merge implementations from `HardwareDispatch.cpp` and `hardware_dispatch.cpp`
- All dispatch functions return `DispatchResult` instead of void/raw-value
- Wrap every GPU call in try/catch that converts to `DispatchError`
- Preserve all existing SIMD kernels (boltzmann_avx512, log_sum_exp_avx512, etc.)
- Add SSE4.2 fallback path for `log_sum_exp` and `boltzmann_batch` (currently jumps from AVX2 to scalar)

**Estimated size:** ~600 lines (mostly moved from existing files)

#### B3. Deprecate old headers

**Changes to `LIB/HardwareDispatch.h`:**
```cpp
#pragma once
#pragma message("HardwareDispatch.h is deprecated — use UnifiedHardwareDispatch.h")
#include "UnifiedHardwareDispatch.h"
using HardwareDispatcher = UnifiedHardwareDispatch; // alias
```

**Same for `LIB/hardware_dispatch.h`.**

### Phase C: Extend GPUContextPool for ROCm

#### C1. `LIB/GPUContextPool.h` — Add ROCm context management

**Changes:**
1. Add `rocm_mtx_`, `rocm_cv_`, `rocm_ref_count_`, `rocm_rebuilding_` (mirror CUDA pattern)
2. Add `acquire_rocm(n_atoms, n_types, init_fn)` / `release_rocm(handle)` methods
3. Destructor calls `hip_eval_shutdown()` if ROCm context exists
4. Guard all ROCm code with `#ifdef FLEXAIDS_USE_ROCM`

**Estimated size:** ~80 lines added

### Phase D: RAII GPU Memory Wrapper

#### D1. Create `LIB/GPUBuffer.h` (NEW)

```cpp
template<typename T>
class GPUBuffer {
    void* ptr_ = nullptr;
    size_t count_ = 0;
    Backend backend_ = Backend::SCALAR;

public:
    GPUBuffer() = default;
    GPUBuffer(size_t count, Backend backend);  // allocate
    ~GPUBuffer();                               // free
    GPUBuffer(GPUBuffer&&) noexcept;            // move
    GPUBuffer& operator=(GPUBuffer&&) noexcept;
    GPUBuffer(const GPUBuffer&) = delete;       // no copy

    DispatchError upload(const T* host_data, size_t count);
    DispatchError download(T* host_data, size_t count) const;
    T* raw() { return static_cast<T*>(ptr_); }
    size_t size() const { return count_; }
    explicit operator bool() const { return ptr_ != nullptr; }
};
```

**Implementation:** Switches on `backend_` for `cudaMalloc`/`hipMalloc`/`MTLBuffer`.
**Estimated size:** ~150 lines

#### D2. Migrate existing kernel launch code

For each kernel file (`cuda_eval.cu`, `tencm_cuda.cu`, etc.):
- Replace raw `cudaMalloc`/`cudaFree` with `GPUBuffer<T>` construction/destruction
- Errors automatically checked and propagated via RAII destructor
- **Do not change kernel code itself** — only the host-side launch wrapper

### Phase E: Runtime SIMD Detection Enhancement

#### E1. `LIB/hardware_detect.cpp` — Add SSE4.2 and AVX baseline detection

**Current:** Detects SSE4.2, AVX2, FMA, AVX-512(F/DQ/BW/VNNI)
**Add:** Detect plain AVX (CPUID leaf 1, bit 28 of ECX) — currently not checked
**Add:** Report composite flags: `has_avx` (for future AVX-only kernels if needed)

**Estimated size:** ~10 lines added

#### E2. `LIB/simd_distance.h` — Add SSE4.2 fallback functions

**Current:** AVX-512 → AVX2 → scalar. No SSE4.2 path.
**Add:** `distance2_1x4()` using `_mm_dp_ps` or `_mm_mul_ps + _mm_hadd_ps`
**Add:** `lj_wall_4x()` SSE4.2 variant

**Rationale:** Some HPC nodes have SSE4.2 but not AVX2 (older Xeon). Also useful as intermediate fallback.
**Estimated size:** ~100 lines added

### Phase F: Async Dispatch Foundation

#### F1. Create `LIB/GPUEvent.h` (NEW)

```cpp
class GPUEvent {
    // CUDA: cudaEvent_t
    // ROCm: hipEvent_t
    // Metal: id<MTLCommandBuffer> completion handler
    Backend backend_;
    void* event_ = nullptr;

public:
    GPUEvent(Backend b);
    ~GPUEvent();
    bool is_complete() const;
    void wait();
    void on_complete(std::function<void(DispatchResult)> callback);  // Metal only initially
};
```

**Rationale:** Foundation for future async dispatch. Initially only Metal benefits (native completion handlers). CUDA/ROCm can poll or use stream callbacks.
**Estimated size:** ~120 lines

#### F2. Add async variants to dispatch

For fitness evaluation (the most expensive kernel):
```cpp
DispatchResult eval_fitness_batch_async(
    /* args */, GPUEvent& completion, Backend b = Backend::AUTO);
```

**Scope:** Only for `eval_fitness_batch` initially. Other kernels remain synchronous.

### Phase G: Fill Feature Parity Gaps

#### G1. `LIB/gpu_fast_optics.cu` — Add Metal path

**Current:** CUDA only, `#ifdef FLEXAIDS_USE_CUDA` guard.
**Add:** Metal kernel for k-NN search. The algorithm is embarrassingly parallel — good Metal candidate.
**File:** `LIB/gpu_fast_optics_metal.metal` + `LIB/gpu_fast_optics_metal_bridge.mm`
**Estimated size:** ~200 lines

#### G2. `LIB/CavityDetect/` — Add CUDA path

**Current:** Metal only.
**Add:** CUDA kernel mirroring `CavityDetect.metal` logic.
**File:** `LIB/CavityDetect/CavityDetect.cu`
**Estimated size:** ~150 lines

#### G3. CPU fallback for `metal_eval` and `cuda_eval`

**Add:** `cpu_eval_batch()` in `LIB/cpu_eval.cpp` that calls existing `cffunction`/`spfunction`/`vcfunction`.
This is the fallback when no GPU is available. Currently the caller (gaboom.cpp) already has this path inline — extract it.
**Estimated size:** ~100 lines

---

## Build System Changes

### CMakeLists.txt Modifications

#### 1. Fix ARC flag inconsistency

```cmake
# BEFORE (root CMakeLists.txt): -fno-objc-arc
# BEFORE (MetalAcceleration.cmake): -fobjc-arc
# AFTER: Standardize on -fno-objc-arc everywhere (manual retain/release is safer for C++ interop)
```

#### 2. Add new source files to flexaid_core

```cmake
# In LIB/CMakeLists.txt, add:
set(FLEXAID_DISPATCH_SOURCES
    UnifiedHardwareDispatch.cpp
    GPUBuffer.h          # header-only template
    GPUEvent.h           # header-only (mostly)
    cpu_eval.cpp         # CPU fallback for fitness eval
)
list(APPEND FLEXAID_CORE_SOURCES ${FLEXAID_DISPATCH_SOURCES})
```

#### 3. Add ROCm context to GPUContextPool

```cmake
# When FLEXAIDS_USE_ROCM, add hip_eval.hip and ROCmDispatch.cpp to core library
if(FLEXAIDS_USE_ROCM)
    list(APPEND FLEXAID_CORE_SOURCES
        ProcessLigand/ROCmDispatch.cpp
        rocm_detect.cpp
    )
endif()
```

#### 4. Add feature-parity kernel files

```cmake
# New Metal k-NN kernel
if(FLEXAIDS_USE_METAL)
    list(APPEND METAL_SOURCES
        gpu_fast_optics_metal_bridge.mm
    )
    # Compile gpu_fast_optics_metal.metal → .metallib
endif()

# New CUDA cavity detection
if(FLEXAIDS_USE_CUDA)
    list(APPEND CUDA_SOURCES
        CavityDetect/CavityDetect.cu
    )
endif()
```

#### 5. Validate Metal SDK at configure time

```cmake
if(FLEXAIDS_USE_METAL)
    find_program(XCRUN xcrun)
    if(NOT XCRUN)
        message(FATAL_ERROR "xcrun not found — Metal shader compilation requires Xcode command-line tools")
    endif()
endif()
```

---

## Test Matrix

### Backend x Platform x Precision

```
                    Linux       Linux       macOS       macOS       Windows
                    x86_64      x86_64      arm64       x86_64      x86_64
                    (GCC)       (Clang)     (Clang)     (Clang)     (MSVC)
────────────────────────────────────────────────────────────────────────────
CUDA               ✓           ✓           ✗           ✓*          ✓
ROCm               ✓           ✓           ✗           ✗           ✗
Metal              ✗           ✗           ✓           ✓           ✗
AVX-512            ✓           ✓           ✗           ✓           ✓
AVX2               ✓           ✓           ✗           ✓           ✓
SSE4.2             ✓           ✓           ✗           ✓           ✓
NEON               ✗           ✗           auto        ✗           ✗
OpenMP             ✓           ✓           ✓           ✓           ✓
Eigen              ✓           ✓           ✓           ✓           ✓
Scalar             ✓           ✓           ✓           ✓           ✓

* CUDA on macOS x86_64 is deprecated by Apple but still functional
```

### Test Categories

#### Unit Tests (per-backend, in `tests/`)

| Test File | What It Tests |
|-----------|---------------|
| `test_hardware_detect.cpp` | CPUID probing, GPU detection, capability caching |
| `test_dispatch_backend.cpp` | Backend selection logic, override, fallback chain |
| `test_dispatch_error.cpp` | DispatchResult propagation, error codes |
| `test_gpu_buffer.cpp` | GPUBuffer RAII: alloc, upload, download, move, double-free safety |
| `test_gpu_context_pool.cpp` | Context reuse, dimension change serialization, ref counting |
| `test_shannon_dispatch.cpp` | Shannon entropy across all backends (compare against known values) |
| `test_boltzmann_dispatch.cpp` | Boltzmann weights: numerical accuracy vs reference |
| `test_lse_dispatch.cpp` | log-sum-exp: overflow/underflow edge cases |
| `test_fitness_dispatch.cpp` | Fitness eval: GPU vs CPU comparison (1e-5 tolerance) |
| `test_simd_distance.cpp` | AVX-512/AVX2/SSE/scalar distance: correctness + tail handling |

#### Integration Tests

| Test | What It Tests |
|------|---------------|
| `test_dispatch_fallback_chain.cpp` | Force unavailable backends, verify graceful fallback |
| `test_concurrent_dispatch.cpp` | Multiple threads dispatching simultaneously |
| `test_mixed_backend_campaign.cpp` | ParallelCampaign with different backends per region |

#### Benchmark Tests (in `tests/benchmarks/`)

| Benchmark | What It Measures |
|-----------|-----------------|
| `bench_dispatch_overhead.cpp` | Dispatch latency (ns) per backend for trivial kernel |
| `bench_shannon_backends.cpp` | Shannon entropy throughput: scalar → AVX2 → AVX512 → GPU |
| `bench_fitness_backends.cpp` | Fitness eval throughput: CPU → CUDA → ROCm → Metal |
| `bench_gpu_transfer.cpp` | H2D/D2H transfer bandwidth per GPU backend |

#### CI Matrix (`.github/workflows/ci.yml` additions)

```yaml
# Add to existing cxx_core_build matrix:
jobs:
  cxx_core_build:
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
            compiler: gcc
            gpu: none        # CPU-only (AVX2/OpenMP/Eigen)
          - os: ubuntu-latest
            compiler: clang
            gpu: none
          - os: macos-14      # Apple Silicon
            compiler: clang
            gpu: metal
          - os: macos-13      # Intel Mac
            compiler: clang
            gpu: none        # Metal optional
          # GPU CI requires self-hosted runners:
          # - os: self-hosted-cuda
          #   gpu: cuda
          # - os: self-hosted-rocm
          #   gpu: rocm
```

---

## Migration Path

### Step 1: Fix P0 Bugs (1 PR)

Fix the 4 critical issues in `tencm_cuda.cu` and error propagation in `cuda_eval.cu` / `metal_eval.mm`. These are surgical fixes to existing files — no new files, no API changes.

**Risk:** Low — fixes are additive (adding checks), not restructuring.

### Step 2: Create UnifiedHardwareDispatch (1 PR)

Create `UnifiedHardwareDispatch.h/cpp` by merging the two existing dispatchers. Create thin forwarding headers for backward compat. Update all callers to use new include (can be mechanical find-replace).

**Risk:** Medium — many callers change includes. Mitigated by forwarding headers.

### Step 3: Add GPUBuffer RAII Wrapper (1 PR)

Create `GPUBuffer.h`. Migrate one kernel file at a time (start with `shannon_cuda.cu` as it's smallest). Verify no behavioral change.

**Risk:** Low — each migration is isolated.

### Step 4: Extend GPUContextPool for ROCm (1 PR)

Add ROCm context management to GPUContextPool. This unifies the context lifecycle for all three GPU backends.

**Risk:** Low — additive change, ROCm code already exists but manages its own context.

### Step 5: Fill Feature Parity Gaps (2-3 PRs)

- PR 5a: CPU fallback for `metal_eval` and `cuda_eval`
- PR 5b: Metal k-NN kernel
- PR 5c: CUDA cavity detection kernel

**Risk:** Medium — new kernel code. Each PR is independent and can be done in parallel.

### Step 6: Async Foundation (1 PR)

Create `GPUEvent.h`. Add async variant of `eval_fitness_batch`. This is additive — existing sync API unchanged.

**Risk:** Low — purely additive, opt-in.

### Step 7: SSE4.2 Fallback Path (1 PR)

Add SSE4.2 kernels to `simd_distance.h`. Update dispatch chain.

**Risk:** Low — additive SIMD path with existing scalar fallback as safety net.

### Dependency Graph

```
Step 1 (P0 fixes)
  │
  ▼
Step 2 (Unified dispatch)
  │
  ├──▶ Step 3 (GPUBuffer)
  │      │
  │      ▼
  │    Step 4 (ROCm context pool)
  │
  ├──▶ Step 5a (CPU fallback)
  │
  ├──▶ Step 5b (Metal k-NN)  ◀── independent
  │
  ├──▶ Step 5c (CUDA cavity) ◀── independent
  │
  ├──▶ Step 6 (Async foundation)
  │
  └──▶ Step 7 (SSE4.2 path)   ◀── independent
```

Steps 5b, 5c, 6, and 7 are independent of each other and can proceed in parallel after Step 2.

---

## Summary of Changes

| Category | Files Modified | Files Created | Lines Changed (est.) |
|----------|---------------|---------------|---------------------|
| P0 fixes | 3 | 0 | ~95 |
| Unified dispatch | 2 deprecated, all callers | 2 | ~600 (mostly moved) |
| GPUBuffer | 5 kernel files | 1 | ~300 |
| ROCm pool | 1 | 0 | ~80 |
| CPU fallback | 1 (gaboom.cpp extract) | 1 | ~100 |
| Feature parity | 0 | 4-5 | ~550 |
| Async foundation | 0 | 1 | ~120 |
| SSE4.2 path | 1 | 0 | ~100 |
| **Total** | **~12** | **~10** | **~1,945** |

---

## Appendix: Files NOT Changing

The following are production-grade and require no modifications:

- `LIB/simd_distance.h` (except SSE4.2 additions)
- `LIB/soft_contact_matrix.h`
- `LIB/hardware_detect.h/cpp` (except minor AVX detection addition)
- `LIB/ShannonThermoStack/ShannonMetalBridge.h/mm` (reference implementation)
- `LIB/ShannonThermoStack/shannon_metal.metal`
- `LIB/GAContext.h/cpp`
- `LIB/DistributedBackend.h`, `ThreadBackend.h`, `MPIBackend.h`
- All Python code (`python/`)
- All test fixtures (`python/conftest.py`)
- `LIB/VoronoiCFBatch.h`
- All OpenMP pragma sites (no changes to parallelization strategy)
- All Eigen usage (no changes to linear algebra)
