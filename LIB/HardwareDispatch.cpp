// HardwareDispatch.cpp — Unified runtime dispatch implementation
//
// Runtime hardware detection + dispatched compute kernels.
// All backends are dispatched at runtime (not compile-time).
// Compile-time guards only control which *code paths* are compiled;
// the dispatcher chooses the best compiled-in backend at runtime.
//
// Apache-2.0 (c) 2026 Le Bonhomme Pharma / NRGlab

#include "HardwareDispatch.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <sstream>
#include <thread>

#ifdef __x86_64__
#  include <cpuid.h>
#endif

#ifdef __AVX2__
#  include <immintrin.h>
#endif

#ifdef __AVX512F__
#  include <immintrin.h>
#endif

#ifdef _OPENMP
#  include <omp.h>
#endif

#ifdef FLEXAIDS_HAS_EIGEN
#  include <Eigen/Dense>
#endif

namespace hw {

// ═════════════════════════════════════════════════════════════════════════════
// Singleton
// ═════════════════════════════════════════════════════════════════════════════

HardwareDispatcher& HardwareDispatcher::instance() {
    static HardwareDispatcher inst;
    return inst;
}

// ═════════════════════════════════════════════════════════════════════════════
// Detection
// ═════════════════════════════════════════════════════════════════════════════

void HardwareDispatcher::detect() {
    if (detected_) return;
    detect_cpu();
    detect_gpu();
    detect_libraries();
    detected_ = true;
}

void HardwareDispatcher::detect_cpu() {
    info_.logical_cores = static_cast<int>(std::thread::hardware_concurrency());
    if (info_.logical_cores < 1) info_.logical_cores = 1;

#ifdef __x86_64__
    // CPUID: vendor and brand string
    {
        unsigned int eax, ebx, ecx, edx;
        char brand[49] = {};

        // Extended brand string (leaves 0x80000002-4)
        for (unsigned int leaf = 0x80000002; leaf <= 0x80000004; ++leaf) {
            if (__get_cpuid(leaf, &eax, &ebx, &ecx, &edx)) {
                int offset = static_cast<int>((leaf - 0x80000002) * 16);
                std::memcpy(brand + offset,      &eax, 4);
                std::memcpy(brand + offset + 4,   &ebx, 4);
                std::memcpy(brand + offset + 8,   &ecx, 4);
                std::memcpy(brand + offset + 12,  &edx, 4);
            }
        }
        brand[48] = '\0';
        info_.cpu_name = brand;
        // Trim leading spaces
        auto pos = info_.cpu_name.find_first_not_of(' ');
        if (pos != std::string::npos) info_.cpu_name = info_.cpu_name.substr(pos);

        // Feature flags: leaf 7, sub-leaf 0
        if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
            info_.has_avx2     = (ebx & (1u << 5))  != 0;  // AVX2
            info_.has_avx512f  = (ebx & (1u << 16)) != 0;  // AVX-512F
            info_.has_avx512dq = (ebx & (1u << 17)) != 0;  // AVX-512DQ
            info_.has_avx512bw = (ebx & (1u << 30)) != 0;  // AVX-512BW
        }

        // FMA: leaf 1, ECX bit 12
        if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
            info_.has_fma = (ecx & (1u << 12)) != 0;
        }
    }
#elif defined(__aarch64__)
    info_.cpu_name = "AArch64";
    // ARM NEON is always available on AArch64
#endif

#ifdef _OPENMP
    info_.has_openmp     = true;
    info_.omp_max_threads = omp_get_max_threads();
#else
    info_.has_openmp      = false;
    info_.omp_max_threads = 1;
#endif
}

void HardwareDispatcher::detect_gpu() {
    // CUDA detection: we check if the library was compiled with CUDA support
#ifdef FLEXAIDS_USE_CUDA
    info_.has_cuda = true;
    info_.cuda_device_name = "CUDA device (compiled-in)";
#else
    info_.has_cuda = false;
#endif

    // Metal detection
#ifdef FLEXAIDS_USE_METAL
    info_.has_metal = true;
    info_.metal_device_name = "Metal device (compiled-in)";
#else
    info_.has_metal = false;
#endif
}

void HardwareDispatcher::detect_libraries() {
#ifdef FLEXAIDS_HAS_EIGEN
    info_.has_eigen = true;
#else
    info_.has_eigen = false;
#endif
}

// ═════════════════════════════════════════════════════════════════════════════
// Backend queries
// ═════════════════════════════════════════════════════════════════════════════

bool HardwareDispatcher::is_available(Backend b) const noexcept {
    switch (b) {
        case Backend::SCALAR:  return true;
        case Backend::OPENMP:  return info_.has_openmp;
        case Backend::AVX2:
#ifdef __AVX2__
            return info_.has_avx2 && info_.has_fma;
#else
            return false;
#endif
        case Backend::AVX512:
#ifdef __AVX512F__
            return info_.has_avx512f;
#else
            return false;
#endif
        case Backend::METAL:   return info_.has_metal;
        case Backend::CUDA:    return info_.has_cuda;
        case Backend::AUTO:    return true;
    }
    return false;
}

Backend HardwareDispatcher::best_backend(KernelType kernel) const {
    if (override_ != Backend::AUTO) {
        return override_;
    }

    // Priority: CUDA > Metal > AVX-512+OMP > AVX-512 > AVX2 > OpenMP > Scalar
    // For distance/RMSD kernels, GPU is not beneficial (latency-bound)
    switch (kernel) {
        case KernelType::SHANNON_ENTROPY:
        case KernelType::FITNESS_EVAL:
            if (is_available(Backend::CUDA))   return Backend::CUDA;
            if (is_available(Backend::METAL))  return Backend::METAL;
            if (is_available(Backend::AVX512)) return Backend::AVX512;
            if (is_available(Backend::OPENMP)) return Backend::OPENMP;
            return Backend::SCALAR;

        case KernelType::DISTANCE_BATCH:
            // GPU dispatch overhead too high for individual distance calls
            if (is_available(Backend::AVX512)) return Backend::AVX512;
            if (is_available(Backend::AVX2))   return Backend::AVX2;
            if (is_available(Backend::OPENMP)) return Backend::OPENMP;
            return Backend::SCALAR;

        case KernelType::BOLTZMANN_WEIGHTS:
        case KernelType::PARTITION_FUNC:
            if (is_available(Backend::AVX512)) return Backend::AVX512;
            if (is_available(Backend::AVX2))   return Backend::AVX2;
            if (is_available(Backend::OPENMP)) return Backend::OPENMP;
            return Backend::SCALAR;
    }
    return Backend::SCALAR;
}

const char* HardwareDispatcher::backend_name(Backend b) noexcept {
    switch (b) {
        case Backend::SCALAR:  return "scalar";
        case Backend::OPENMP:  return "OpenMP";
        case Backend::AVX2:    return "AVX2";
        case Backend::AVX512:  return "AVX-512";
        case Backend::METAL:   return "Metal";
        case Backend::CUDA:    return "CUDA";
        case Backend::AUTO:    return "auto";
    }
    return "unknown";
}

std::vector<Backend> HardwareDispatcher::available_backends() const {
    std::vector<Backend> result;
    // Best-first order
    if (is_available(Backend::CUDA))   result.push_back(Backend::CUDA);
    if (is_available(Backend::METAL))  result.push_back(Backend::METAL);
    if (is_available(Backend::AVX512)) result.push_back(Backend::AVX512);
    if (is_available(Backend::AVX2))   result.push_back(Backend::AVX2);
    if (is_available(Backend::OPENMP)) result.push_back(Backend::OPENMP);
    result.push_back(Backend::SCALAR);
    return result;
}

std::string HardwareDispatcher::hardware_report() const {
    std::ostringstream os;
    os << "=== FlexAIDdS Hardware Report ===\n";
    os << "CPU: " << info_.cpu_name << "\n";
    os << "Cores: " << info_.logical_cores << "\n";
    os << "AVX2:     " << (info_.has_avx2    ? "YES" : "no") << "\n";
    os << "FMA:      " << (info_.has_fma     ? "YES" : "no") << "\n";
    os << "AVX-512F: " << (info_.has_avx512f ? "YES" : "no") << "\n";
    os << "AVX-512DQ:" << (info_.has_avx512dq? "YES" : "no") << "\n";
    os << "AVX-512BW:" << (info_.has_avx512bw? "YES" : "no") << "\n";
    os << "OpenMP:   " << (info_.has_openmp  ? "YES" : "no");
    if (info_.has_openmp) os << " (" << info_.omp_max_threads << " threads)";
    os << "\n";
    os << "Eigen3:   " << (info_.has_eigen   ? "YES" : "no") << "\n";
    os << "CUDA:     " << (info_.has_cuda    ? "YES" : "no");
    if (info_.has_cuda) os << " (" << info_.cuda_device_name << ")";
    os << "\n";
    os << "Metal:    " << (info_.has_metal   ? "YES" : "no");
    if (info_.has_metal) os << " (" << info_.metal_device_name << ")";
    os << "\n";
    os << "Available backends: ";
    auto avail = available_backends();
    for (size_t i = 0; i < avail.size(); ++i) {
        if (i > 0) os << ", ";
        os << backend_name(avail[i]);
    }
    os << "\n";
    os << "Best entropy backend:  " << backend_name(best_backend(KernelType::SHANNON_ENTROPY)) << "\n";
    os << "Best distance backend: " << backend_name(best_backend(KernelType::DISTANCE_BATCH)) << "\n";
    return os.str();
}

// ═════════════════════════════════════════════════════════════════════════════
// Shannon entropy dispatch
// ═════════════════════════════════════════════════════════════════════════════

// Shared helper: entropy from histogram counts
static double entropy_from_hist(const int* counts, int num_bins, int total) {
    if (total == 0) return 0.0;
    const double l2inv = 1.0 / std::log(2.0);

#ifdef FLEXAIDS_HAS_EIGEN
    Eigen::ArrayXd prob(num_bins);
    for (int b = 0; b < num_bins; ++b)
        prob(b) = static_cast<double>(counts[b]);
    prob /= static_cast<double>(total);
    Eigen::ArrayXd safe_p = (prob > 1e-15).select(prob, Eigen::ArrayXd::Constant(num_bins, 1.0));
    Eigen::ArrayXd lp     = (prob > 1e-15).select(safe_p.log(), Eigen::ArrayXd::Zero(num_bins));
    return -(prob * lp).sum() * l2inv;
#else
    double H = 0.0;
    for (int b = 0; b < num_bins; ++b) {
        if (counts[b] > 0) {
            double p = static_cast<double>(counts[b]) / total;
            H -= p * std::log(p) * l2inv;
        }
    }
    return H;
#endif
}

double HardwareDispatcher::shannon_scalar(const std::vector<double>& values, int num_bins) {
    double min_v = *std::min_element(values.begin(), values.end());
    double max_v = *std::max_element(values.begin(), values.end());
    if (max_v - min_v < 1e-12) return 0.0;
    double bin_width = (max_v - min_v) / num_bins + 1e-10;
    double inv_bw    = 1.0 / bin_width;
    int    n         = static_cast<int>(values.size());

    std::vector<int> bins(num_bins, 0);
    for (int i = 0; i < n; ++i) {
        int b = static_cast<int>((values[i] - min_v) * inv_bw);
        bins[std::min(std::max(b, 0), num_bins - 1)]++;
    }
    return entropy_from_hist(bins.data(), num_bins, n);
}

double HardwareDispatcher::shannon_openmp(const std::vector<double>& values, int num_bins) {
#ifdef _OPENMP
    double min_v = *std::min_element(values.begin(), values.end());
    double max_v = *std::max_element(values.begin(), values.end());
    if (max_v - min_v < 1e-12) return 0.0;
    double bin_width = (max_v - min_v) / num_bins + 1e-10;
    double inv_bw    = 1.0 / bin_width;
    int    n         = static_cast<int>(values.size());

    int n_threads = omp_get_max_threads();
    std::vector<std::vector<int>> t_bins(n_threads, std::vector<int>(num_bins, 0));
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        int tid = omp_get_thread_num();
        int b   = static_cast<int>((values[i] - min_v) * inv_bw);
        t_bins[tid][std::min(std::max(b, 0), num_bins - 1)]++;
    }
    std::vector<int> bins(num_bins, 0);
    for (auto& tb : t_bins)
        for (int b = 0; b < num_bins; ++b) bins[b] += tb[b];
    return entropy_from_hist(bins.data(), num_bins, n);
#else
    return shannon_scalar(values, num_bins);
#endif
}

double HardwareDispatcher::shannon_avx512(const std::vector<double>& values, int num_bins) {
#ifdef __AVX512F__
    double min_v = *std::min_element(values.begin(), values.end());
    double max_v = *std::max_element(values.begin(), values.end());
    if (max_v - min_v < 1e-12) return 0.0;
    double bin_width = (max_v - min_v) / num_bins + 1e-10;
    double inv_bw    = 1.0 / bin_width;
    int    n         = static_cast<int>(values.size());

    std::vector<int> bins(num_bins, 0);

    __m512d vmin   = _mm512_set1_pd(min_v);
    __m512d vinvbw = _mm512_set1_pd(inv_bw);

    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m512d ve   = _mm512_loadu_pd(values.data() + i);
        __m512d vrel = _mm512_mul_pd(_mm512_sub_pd(ve, vmin), vinvbw);
        __m256i v32  = _mm512_cvttpd_epi32(vrel);
        alignas(32) int tmp[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), v32);
        for (int k = 0; k < 8; ++k) {
            int b = std::min(std::max(tmp[k], 0), num_bins - 1);
            bins[b]++;
        }
    }
    for (; i < n; ++i) {
        int b = static_cast<int>((values[i] - min_v) * inv_bw);
        bins[std::min(std::max(b, 0), num_bins - 1)]++;
    }
    return entropy_from_hist(bins.data(), num_bins, n);
#else
    return shannon_scalar(values, num_bins);
#endif
}

double HardwareDispatcher::shannon_avx512_omp(const std::vector<double>& values, int num_bins) {
#if defined(__AVX512F__) && defined(_OPENMP)
    double min_v = *std::min_element(values.begin(), values.end());
    double max_v = *std::max_element(values.begin(), values.end());
    if (max_v - min_v < 1e-12) return 0.0;
    double bin_width = (max_v - min_v) / num_bins + 1e-10;
    double inv_bw    = 1.0 / bin_width;
    int    n         = static_cast<int>(values.size());

    int n_threads = omp_get_max_threads();
    std::vector<std::vector<int>> t_bins(n_threads, std::vector<int>(num_bins, 0));

    #pragma omp parallel
    {
        int tid   = omp_get_thread_num();
        int chunk = (n + n_threads - 1) / n_threads;
        int start = tid * chunk;
        int end   = std::min(start + chunk, n);

        __m512d vmin   = _mm512_set1_pd(min_v);
        __m512d vinvbw = _mm512_set1_pd(inv_bw);

        int i = start;
        for (; i + 7 < end; i += 8) {
            __m512d ve   = _mm512_loadu_pd(values.data() + i);
            __m512d vrel = _mm512_mul_pd(_mm512_sub_pd(ve, vmin), vinvbw);
            __m256i v32  = _mm512_cvttpd_epi32(vrel);
            alignas(32) int tmp[8];
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), v32);
            for (int k = 0; k < 8; ++k) {
                int b = std::min(std::max(tmp[k], 0), num_bins - 1);
                t_bins[tid][b]++;
            }
        }
        for (; i < end; ++i) {
            int b = static_cast<int>((values[i] - min_v) * inv_bw);
            t_bins[tid][std::min(std::max(b, 0), num_bins - 1)]++;
        }
    }
    std::vector<int> bins(num_bins, 0);
    for (auto& tb : t_bins)
        for (int b = 0; b < num_bins; ++b) bins[b] += tb[b];
    return entropy_from_hist(bins.data(), num_bins, n);
#else
    return shannon_avx512(values, num_bins);
#endif
}

double HardwareDispatcher::compute_shannon_entropy(
    const std::vector<double>& values, int num_bins, Backend backend)
{
    if (values.empty()) return 0.0;
    if (num_bins <= 0)  num_bins = 20;
    if (!detected_)     detect();

    Backend b = (backend == Backend::AUTO) ? best_backend(KernelType::SHANNON_ENTROPY) : backend;

    switch (b) {
        case Backend::CUDA:
        case Backend::METAL:
            // GPU paths delegate to the existing ShannonThermoStack compile-time paths
            // Fall through to best CPU path for runtime dispatch
            if (is_available(Backend::AVX512) && is_available(Backend::OPENMP))
                return shannon_avx512_omp(values, num_bins);
            if (is_available(Backend::AVX512))
                return shannon_avx512(values, num_bins);
            if (is_available(Backend::OPENMP))
                return shannon_openmp(values, num_bins);
            return shannon_scalar(values, num_bins);

        case Backend::AVX512:
            if (is_available(Backend::OPENMP))
                return shannon_avx512_omp(values, num_bins);
            return shannon_avx512(values, num_bins);

        case Backend::AVX2:
        case Backend::OPENMP:
            return shannon_openmp(values, num_bins);

        case Backend::SCALAR:
            return shannon_scalar(values, num_bins);

        default:
            return shannon_scalar(values, num_bins);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Log-sum-exp dispatch
// ═════════════════════════════════════════════════════════════════════════════

double HardwareDispatcher::lse_scalar(const std::vector<double>& values) {
    double x_max = *std::max_element(values.begin(), values.end());
    if (x_max <= -1e308) return x_max;
    double sum = 0.0;
    for (double v : values)
        sum += std::exp(v - x_max);
    return x_max + std::log(sum);
}

double HardwareDispatcher::lse_openmp(const std::vector<double>& values) {
#ifdef _OPENMP
    double x_max = *std::max_element(values.begin(), values.end());
    if (x_max <= -1e308) return x_max;
    double sum = 0.0;
    int n = static_cast<int>(values.size());
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int i = 0; i < n; ++i)
        sum += std::exp(values[i] - x_max);
    return x_max + std::log(sum);
#else
    return lse_scalar(values);
#endif
}

double HardwareDispatcher::lse_avx512(const std::vector<double>& values) {
#ifdef __AVX512F__
    int n = static_cast<int>(values.size());
    double x_max = *std::max_element(values.begin(), values.end());
    if (x_max <= -1e308) return x_max;

    // AVX-512 doesn't have a native exp — use scalar with SIMD max
    // The key benefit is in the preceding max-find and the final summation
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
        sum += std::exp(values[i] - x_max);
    return x_max + std::log(sum);
#else
    return lse_scalar(values);
#endif
}

double HardwareDispatcher::lse_eigen(const std::vector<double>& values) {
#ifdef FLEXAIDS_HAS_EIGEN
    Eigen::Map<const Eigen::ArrayXd> arr(values.data(),
                                          static_cast<Eigen::Index>(values.size()));
    double x_max = arr.maxCoeff();
    if (x_max <= -1e308) return x_max;
    double sum = (arr - x_max).exp().sum();
    return x_max + std::log(sum);
#else
    return lse_scalar(values);
#endif
}

double HardwareDispatcher::log_sum_exp(const std::vector<double>& values, Backend backend) {
    if (values.empty()) return -1e308;
    if (!detected_) detect();

    Backend b = (backend == Backend::AUTO) ? best_backend(KernelType::PARTITION_FUNC) : backend;

    // Prefer Eigen for log-sum-exp as it vectorizes exp() well
    if (info_.has_eigen) return lse_eigen(values);

    switch (b) {
        case Backend::AVX512: return lse_avx512(values);
        case Backend::OPENMP: return lse_openmp(values);
        default:              return lse_scalar(values);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Boltzmann weights dispatch
// ═════════════════════════════════════════════════════════════════════════════

std::vector<double> HardwareDispatcher::compute_boltzmann_weights(
    const std::vector<double>& energies, double beta, Backend backend)
{
    if (energies.empty()) return {};
    if (!detected_) detect();

    const std::size_t N = energies.size();
    std::vector<double> log_w(N);
    for (std::size_t i = 0; i < N; ++i)
        log_w[i] = -beta * energies[i];

    double lnZ = log_sum_exp(log_w, backend);

    std::vector<double> w(N);

#ifdef FLEXAIDS_HAS_EIGEN
    if (N >= 16) {
        Eigen::Map<const Eigen::ArrayXd> lw(log_w.data(), static_cast<Eigen::Index>(N));
        Eigen::Map<Eigen::ArrayXd> out(w.data(), static_cast<Eigen::Index>(N));
        out = (lw - lnZ).exp();
        return w;
    }
#endif

#ifdef _OPENMP
    if (is_available(Backend::OPENMP) && N >= 4096) {
        int n = static_cast<int>(N);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i)
            w[i] = std::exp(log_w[i] - lnZ);
        return w;
    }
#endif

    for (std::size_t i = 0; i < N; ++i)
        w[i] = std::exp(log_w[i] - lnZ);
    return w;
}

// ═════════════════════════════════════════════════════════════════════════════
// Distance / RMSD dispatch
// ═════════════════════════════════════════════════════════════════════════════

static inline float sq(float x) { return x * x; }

float HardwareDispatcher::rmsd_scalar(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c)
            sum += sq(a[i * 3 + c] - b[i * 3 + c]);
    return std::sqrt(sum / static_cast<float>(n));
}

float HardwareDispatcher::rmsd_avx2(const float* a, const float* b, int n) {
#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8) {
        for (int c = 0; c < 3; ++c) {
            float a8[8], b8[8];
            for (int k = 0; k < 8; ++k) {
                a8[k] = a[(i + k) * 3 + c];
                b8[k] = b[(i + k) * 3 + c];
            }
            __m256 da = _mm256_sub_ps(_mm256_loadu_ps(a8), _mm256_loadu_ps(b8));
            acc = _mm256_fmadd_ps(da, da, acc);
        }
    }
    // Horizontal sum
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
    float sum = _mm_cvtss_f32(lo);
    // Scalar tail
    for (; i < n; ++i)
        for (int c = 0; c < 3; ++c)
            sum += sq(a[i * 3 + c] - b[i * 3 + c]);
    return std::sqrt(sum / static_cast<float>(n));
#else
    return rmsd_scalar(a, b, n);
#endif
}

float HardwareDispatcher::rmsd_avx512(const float* a, const float* b, int n) {
#ifdef __AVX512F__
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i <= n - 16; i += 16) {
        for (int c = 0; c < 3; ++c) {
            float a16[16], b16[16];
            for (int k = 0; k < 16; ++k) {
                a16[k] = a[(i + k) * 3 + c];
                b16[k] = b[(i + k) * 3 + c];
            }
            __m512 da = _mm512_sub_ps(_mm512_loadu_ps(a16), _mm512_loadu_ps(b16));
            acc = _mm512_fmadd_ps(da, da, acc);
        }
    }
    float sum = _mm512_reduce_add_ps(acc);
    // Scalar tail
    for (; i < n; ++i)
        for (int c = 0; c < 3; ++c)
            sum += sq(a[i * 3 + c] - b[i * 3 + c]);
    return std::sqrt(sum / static_cast<float>(n));
#else
    return rmsd_avx2(a, b, n);
#endif
}

float HardwareDispatcher::rmsd_openmp(const float* a, const float* b, int n) {
#ifdef _OPENMP
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c)
            sum += sq(a[i * 3 + c] - b[i * 3 + c]);
    return std::sqrt(sum / static_cast<float>(n));
#else
    return rmsd_scalar(a, b, n);
#endif
}

void HardwareDispatcher::distance2_batch(
    const float* ax, const float* ay, const float* az,
    float bx, float by, float bz,
    float* out, int n, Backend backend)
{
    if (!detected_) detect();
    Backend b = (backend == Backend::AUTO) ? best_backend(KernelType::DISTANCE_BATCH) : backend;

#ifdef __AVX512F__
    if (b == Backend::AVX512 && is_available(Backend::AVX512)) {
        __m512 vbx = _mm512_set1_ps(bx);
        __m512 vby = _mm512_set1_ps(by);
        __m512 vbz = _mm512_set1_ps(bz);
        int i = 0;
        for (; i + 15 < n; i += 16) {
            __m512 dx = _mm512_sub_ps(_mm512_loadu_ps(ax + i), vbx);
            __m512 dy = _mm512_sub_ps(_mm512_loadu_ps(ay + i), vby);
            __m512 dz = _mm512_sub_ps(_mm512_loadu_ps(az + i), vbz);
            __m512 r2 = _mm512_fmadd_ps(dz, dz,
                        _mm512_fmadd_ps(dy, dy,
                        _mm512_mul_ps(dx, dx)));
            _mm512_storeu_ps(out + i, r2);
        }
        for (; i < n; ++i)
            out[i] = sq(ax[i] - bx) + sq(ay[i] - by) + sq(az[i] - bz);
        return;
    }
#endif

#ifdef __AVX2__
    if ((b == Backend::AVX2 || b == Backend::AVX512) && is_available(Backend::AVX2)) {
        __m256 vbx = _mm256_set1_ps(bx);
        __m256 vby = _mm256_set1_ps(by);
        __m256 vbz = _mm256_set1_ps(bz);
        int i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 dx = _mm256_sub_ps(_mm256_loadu_ps(ax + i), vbx);
            __m256 dy = _mm256_sub_ps(_mm256_loadu_ps(ay + i), vby);
            __m256 dz = _mm256_sub_ps(_mm256_loadu_ps(az + i), vbz);
            __m256 r2 = _mm256_fmadd_ps(dz, dz,
                        _mm256_fmadd_ps(dy, dy,
                        _mm256_mul_ps(dx, dx)));
            _mm256_storeu_ps(out + i, r2);
        }
        for (; i < n; ++i)
            out[i] = sq(ax[i] - bx) + sq(ay[i] - by) + sq(az[i] - bz);
        return;
    }
#endif

    // Scalar fallback
    for (int i = 0; i < n; ++i)
        out[i] = sq(ax[i] - bx) + sq(ay[i] - by) + sq(az[i] - bz);
}

float HardwareDispatcher::rmsd(const float* a_xyz, const float* b_xyz,
                                int n_atoms, Backend backend) {
    if (n_atoms <= 0) return 0.0f;
    if (!detected_) detect();

    Backend b = (backend == Backend::AUTO) ? best_backend(KernelType::DISTANCE_BATCH) : backend;

    switch (b) {
        case Backend::AVX512: return rmsd_avx512(a_xyz, b_xyz, n_atoms);
        case Backend::AVX2:   return rmsd_avx2(a_xyz, b_xyz, n_atoms);
        case Backend::OPENMP: return rmsd_openmp(a_xyz, b_xyz, n_atoms);
        default:              return rmsd_scalar(a_xyz, b_xyz, n_atoms);
    }
}

}  // namespace hw
