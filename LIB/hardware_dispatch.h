// hardware_dispatch.h — Unified hardware dispatch layer for FlexAIDdS
//
// Provides a single entry point for batched chromosome evaluation that
// automatically selects the optimal backend at runtime:
//
//   1. CUDA GPU         (highest throughput, requires NVIDIA GPU)
//   2. Metal GPU        (Apple Silicon, unified memory, zero-copy)
//   3. AVX-512 + OpenMP (16-wide SIMD × N threads)
//   4. AVX-512          (16-wide SIMD, single thread)
//   5. AVX2 + OpenMP    (8-wide SIMD × N threads)
//   6. OpenMP scalar    (multi-threaded scalar)
//   7. Scalar           (baseline fallback)
//
// The dispatch layer also provides hardware-accelerated primitives for
// statistical mechanics computations (partition functions, Boltzmann
// weights, log-sum-exp reductions) with per-call telemetry.
//
// Apache-2.0 © 2026 Le Bonhomme Pharma
#pragma once

#include "hardware_detect.h"
#include <vector>
#include <string>
#include <cstdint>
#include <span>
#include <chrono>

// Forward declarations for GPU contexts (opaque handles)
#ifdef FLEXAIDS_USE_CUDA
struct CudaEvalCtx;
#endif
#ifdef FLEXAIDS_USE_METAL
struct MetalEvalCtx;
#endif

namespace flexaids {

// Backend selection priority (highest → lowest)
enum class HardwareBackend : uint8_t {
    CUDA    = 0,
    METAL   = 1,
    AVX512  = 2,
    AVX2    = 3,
    OPENMP  = 4,
    SCALAR  = 5,
};

// Human-readable backend name
const char* backend_name(HardwareBackend b) noexcept;

// Select the best available backend based on runtime hardware detection.
HardwareBackend select_backend();

// Select the best CPU-only backend (no GPU).
HardwareBackend select_cpu_backend();

// ─── Per-call telemetry ──────────────────────────────────────────────────────
// Lightweight performance counters returned from dispatched operations.

struct DispatchTelemetry {
    HardwareBackend backend;          // Actual backend used
    double          wall_time_ms;     // Wall-clock time in milliseconds
    int64_t         elements;         // Number of elements processed
    double          throughput_meps;  // Million elements per second

    std::string summary() const;
};

// ─── Batched Boltzmann weight computation ────────────────────────────────────
// Computes w[i] = exp(-beta * (E[i] - E_min)) for numerical stability.
// Dispatches to Metal GPU, AVX-512, AVX2+OpenMP, OpenMP, or scalar path.

struct BoltzmannBatchResult {
    std::vector<double> weights;    // unnormalised Boltzmann factors
    double              log_Z;      // ln(partition function) = ln(Σ w_i) + beta*E_min
    double              E_min;      // reference energy
    DispatchTelemetry   telemetry;  // performance counters
};

BoltzmannBatchResult compute_boltzmann_batch(
    std::span<const double> energies,
    double                  beta);

// ─── Batched partition function reduction ────────────────────────────────────
// Log-sum-exp with hardware dispatch (Metal/Eigen/AVX-512/OpenMP/scalar).
double log_sum_exp_dispatch(std::span<const double> values);

// ─── Dispatch report ─────────────────────────────────────────────────────────
// Returns a structured report of which backend was selected and why.
struct DispatchReport {
    HardwareBackend selected;
    std::string     reason;
    std::string     hw_summary;
};

DispatchReport get_dispatch_report();

}  // namespace flexaids
