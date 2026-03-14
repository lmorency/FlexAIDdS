// HardwareDispatch.h — Unified runtime hardware dispatch layer
//
// Provides runtime backend detection and selection for FlexAIDdS.
// Instead of compile-time-only #ifdef dispatch, this layer detects
// available hardware at startup and routes compute kernels to the
// optimal backend at runtime.
//
// Priority order (highest first):
//   1. CUDA  (NVIDIA GPU)
//   2. Metal (Apple GPU)
//   3. AVX-512 + OpenMP
//   4. AVX-512 scalar
//   5. AVX-2 + OpenMP
//   6. OpenMP scalar
//   7. Scalar fallback
//
// Apache-2.0 (c) 2026 Le Bonhomme Pharma / NRGlab
#pragma once

#include <string>
#include <vector>
#include <array>
#include <functional>
#include <chrono>
#include <cstdint>

namespace hw {

// ─── Backend enumeration ────────────────────────────────────────────────────

enum class Backend : uint8_t {
    SCALAR  = 0,   // Always available
    OPENMP  = 1,   // OpenMP thread parallelism
    AVX2    = 2,   // AVX2 + FMA SIMD (8 floats / 4 doubles)
    AVX512  = 3,   // AVX-512 SIMD (16 floats / 8 doubles)
    METAL   = 4,   // Apple Metal GPU
    CUDA    = 5,   // NVIDIA CUDA GPU
    AUTO    = 255   // Let dispatcher choose best available
};

// ─── Hardware capabilities (detected at runtime) ─────────────────────────────

struct HardwareInfo {
    // CPU
    std::string cpu_name;
    int         logical_cores    = 1;
    bool        has_avx2         = false;
    bool        has_fma          = false;
    bool        has_avx512f      = false;
    bool        has_avx512dq     = false;
    bool        has_avx512bw     = false;
    bool        has_openmp       = false;
    int         omp_max_threads  = 1;

    // GPU
    bool        has_cuda         = false;
    std::string cuda_device_name;
    int         cuda_sm_count    = 0;
    std::size_t cuda_global_mem  = 0;   // bytes

    bool        has_metal        = false;
    std::string metal_device_name;

    // Eigen
    bool        has_eigen        = false;
};

// ─── Dispatch capabilities per kernel type ───────────────────────────────────

enum class KernelType : uint8_t {
    SHANNON_ENTROPY   = 0,   // Shannon histogram binning
    FITNESS_EVAL      = 1,   // Chromosome CF evaluation
    DISTANCE_BATCH    = 2,   // Batched geometric distance
    BOLTZMANN_WEIGHTS = 3,   // Boltzmann weight computation
    PARTITION_FUNC    = 4,   // log-sum-exp partition function
};

// ─── Benchmark result for one backend/kernel combination ─────────────────────

struct BenchmarkResult {
    Backend     backend;
    KernelType  kernel;
    double      time_seconds;
    double      speedup;          // vs scalar baseline
    double      throughput;       // items/second
    std::string backend_name;
};

// ─── HardwareDispatcher (singleton) ──────────────────────────────────────────

class HardwareDispatcher {
public:
    // Singleton access (thread-safe via Meyers' singleton)
    static HardwareDispatcher& instance();

    // Detect hardware capabilities (called once at startup)
    void detect();

    // Get detected hardware info
    const HardwareInfo& info() const noexcept { return info_; }

    // Select best available backend for a given kernel type
    Backend best_backend(KernelType kernel = KernelType::SHANNON_ENTROPY) const;

    // Force a specific backend (for testing / user override)
    void set_override(Backend b) noexcept { override_ = b; }
    void clear_override() noexcept { override_ = Backend::AUTO; }
    Backend current_override() const noexcept { return override_; }

    // Check if a specific backend is available at runtime
    bool is_available(Backend b) const noexcept;

    // Human-readable name for a backend
    static const char* backend_name(Backend b) noexcept;

    // List all available backends (in priority order, best first)
    std::vector<Backend> available_backends() const;

    // Generate a human-readable report of detected hardware
    std::string hardware_report() const;

    // ─── Dispatched compute functions ────────────────────────────────────

    // Shannon entropy: dispatches to best available backend
    double compute_shannon_entropy(const std::vector<double>& values,
                                   int num_bins = 20,
                                   Backend backend = Backend::AUTO);

    // Boltzmann weights with runtime dispatch
    std::vector<double> compute_boltzmann_weights(
        const std::vector<double>& energies,
        double beta,
        Backend backend = Backend::AUTO);

    // Log-sum-exp with runtime dispatch (for partition functions)
    double log_sum_exp(const std::vector<double>& values,
                       Backend backend = Backend::AUTO);

    // Batched squared distance (float, for geometric primitives)
    void distance2_batch(const float* ax, const float* ay, const float* az,
                         float bx, float by, float bz,
                         float* out, int n,
                         Backend backend = Backend::AUTO);

    // RMSD computation with runtime dispatch
    float rmsd(const float* a_xyz, const float* b_xyz, int n_atoms,
               Backend backend = Backend::AUTO);

private:
    HardwareDispatcher() = default;

    HardwareInfo info_;
    bool         detected_  = false;
    Backend      override_  = Backend::AUTO;

    // Internal detection helpers
    void detect_cpu();
    void detect_gpu();
    void detect_libraries();

    // Internal dispatch implementations
    double shannon_scalar(const std::vector<double>& values, int num_bins);
    double shannon_openmp(const std::vector<double>& values, int num_bins);
    double shannon_avx512(const std::vector<double>& values, int num_bins);
    double shannon_avx512_omp(const std::vector<double>& values, int num_bins);

    double lse_scalar(const std::vector<double>& values);
    double lse_openmp(const std::vector<double>& values);
    double lse_avx512(const std::vector<double>& values);
    double lse_eigen(const std::vector<double>& values);

    float rmsd_scalar(const float* a, const float* b, int n);
    float rmsd_avx2(const float* a, const float* b, int n);
    float rmsd_avx512(const float* a, const float* b, int n);
    float rmsd_openmp(const float* a, const float* b, int n);
};

}  // namespace hw
