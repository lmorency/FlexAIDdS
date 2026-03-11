// hardware_dispatch.cpp — Unified hardware dispatch layer
//
// Runtime backend selection and accelerated primitives for:
//   – Boltzmann weight computation (partition function inner loop)
//   – Log-sum-exp reduction (numerical stability critical path)
//
// Dispatch priority: CUDA > Metal > AVX-512 > AVX2 > OpenMP > Scalar
//
// Apache-2.0 © 2026 Le Bonhomme Pharma
#include "hardware_dispatch.h"
#include "hardware_detect.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>

#ifdef _OPENMP
#  include <omp.h>
#endif

#ifdef FLEXAIDS_HAS_EIGEN
#  include <Eigen/Dense>
#endif

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#  include <immintrin.h>
#  define HAS_AVX512_RT 1
#else
#  define HAS_AVX512_RT 0
#endif

#if defined(__AVX2__)
#  include <immintrin.h>
#endif

namespace flexaids {

// ─── backend_name ────────────────────────────────────────────────────────────

const char* backend_name(HardwareBackend b) noexcept {
    switch (b) {
        case HardwareBackend::CUDA:   return "CUDA";
        case HardwareBackend::METAL:  return "Metal";
        case HardwareBackend::AVX512: return "AVX-512";
        case HardwareBackend::AVX2:   return "AVX2";
        case HardwareBackend::OPENMP: return "OpenMP";
        case HardwareBackend::SCALAR: return "scalar";
    }
    return "unknown";
}

// ─── select_backend ──────────────────────────────────────────────────────────

HardwareBackend select_backend() {
    const auto& hw = detect_hardware();

    // GPU backends take priority for large batches
    if (hw.has_cuda && hw.cuda_sm_major >= 7)
        return HardwareBackend::CUDA;

    if (hw.has_metal)
        return HardwareBackend::METAL;

    return select_cpu_backend();
}

HardwareBackend select_cpu_backend() {
    const auto& hw = detect_hardware();

    // AVX-512 with compile-time support
#if HAS_AVX512_RT
    if (hw.has_avx512)
        return HardwareBackend::AVX512;
#endif

    if (hw.has_avx2)
        return HardwareBackend::AVX2;

    if (hw.has_openmp && hw.openmp_max_threads > 1)
        return HardwareBackend::OPENMP;

    return HardwareBackend::SCALAR;
}

// ─── Boltzmann batch: scalar path ────────────────────────────────────────────

static BoltzmannBatchResult boltzmann_scalar(
    std::span<const double> energies, double beta)
{
    const int n = static_cast<int>(energies.size());
    double E_min = *std::min_element(energies.begin(), energies.end());

    std::vector<double> weights(n);
    for (int i = 0; i < n; ++i)
        weights[i] = std::exp(-beta * (energies[i] - E_min));

    double sum_w = std::accumulate(weights.begin(), weights.end(), 0.0);
    double log_Z = std::log(sum_w) - beta * E_min;

    return { std::move(weights), log_Z, E_min };
}

// ─── Boltzmann batch: OpenMP path ────────────────────────────────────────────

#ifdef _OPENMP
static BoltzmannBatchResult boltzmann_openmp(
    std::span<const double> energies, double beta)
{
    const int n = static_cast<int>(energies.size());

    // Parallel min
    double E_min = std::numeric_limits<double>::max();
    #pragma omp parallel for reduction(min:E_min) schedule(static)
    for (int i = 0; i < n; ++i)
        E_min = std::min(E_min, energies[i]);

    std::vector<double> weights(n);
    double sum_w = 0.0;

    #pragma omp parallel for reduction(+:sum_w) schedule(static)
    for (int i = 0; i < n; ++i) {
        weights[i] = std::exp(-beta * (energies[i] - E_min));
        sum_w += weights[i];
    }

    double log_Z = std::log(sum_w) - beta * E_min;
    return { std::move(weights), log_Z, E_min };
}
#endif

// ─── Boltzmann batch: AVX-512 path ───────────────────────────────────────────

#if HAS_AVX512_RT
static BoltzmannBatchResult boltzmann_avx512(
    std::span<const double> energies, double beta)
{
    const int n = static_cast<int>(energies.size());

    // Find E_min with AVX-512 reduction
    double E_min = std::numeric_limits<double>::max();
    {
        __m512d vmin = _mm512_set1_pd(E_min);
        int i = 0;
        for (; i + 7 < n; i += 8) {
            __m512d ve = _mm512_loadu_pd(energies.data() + i);
            vmin = _mm512_min_pd(vmin, ve);
        }
        E_min = _mm512_reduce_min_pd(vmin);
        for (; i < n; ++i)
            E_min = std::min(E_min, energies[i]);
    }

    // Compute weights = exp(-beta * (E - E_min))
    std::vector<double> weights(n);
    __m512d vneg_beta = _mm512_set1_pd(-beta);
    __m512d vEmin     = _mm512_set1_pd(E_min);
    double sum_w = 0.0;

    int i = 0;

#ifdef _OPENMP
    // AVX-512 + OpenMP combined path
    #pragma omp parallel reduction(+:sum_w)
    {
        int tid = omp_get_thread_num();
        int nt  = omp_get_num_threads();
        int chunk = (n + nt - 1) / nt;
        int start = tid * chunk;
        int end   = std::min(start + chunk, n);

        int j = start;
        for (; j + 7 < end; j += 8) {
            __m512d ve  = _mm512_loadu_pd(energies.data() + j);
            __m512d arg = _mm512_mul_pd(_mm512_sub_pd(ve, vEmin), vneg_beta);

            // exp via scalar fallback (portable, no SVML dependency)
            alignas(64) double tmp[8];
            _mm512_storeu_pd(tmp, arg);
            for (int k = 0; k < 8; ++k) tmp[k] = std::exp(tmp[k]);
            __m512d vw = _mm512_loadu_pd(tmp);
            _mm512_storeu_pd(weights.data() + j, vw);
            sum_w += _mm512_reduce_add_pd(vw);
        }
        for (; j < end; ++j) {
            weights[j] = std::exp(-beta * (energies[j] - E_min));
            sum_w += weights[j];
        }
    }
#else
    for (; i + 7 < n; i += 8) {
        __m512d ve  = _mm512_loadu_pd(energies.data() + i);
        __m512d arg = _mm512_mul_pd(_mm512_sub_pd(ve, vEmin), vneg_beta);

        alignas(64) double tmp[8];
        _mm512_storeu_pd(tmp, arg);
        for (int k = 0; k < 8; ++k) tmp[k] = std::exp(tmp[k]);
        __m512d vw = _mm512_loadu_pd(tmp);
        _mm512_storeu_pd(weights.data() + i, vw);
        sum_w += _mm512_reduce_add_pd(vw);
    }
    for (; i < n; ++i) {
        weights[i] = std::exp(-beta * (energies[i] - E_min));
        sum_w += weights[i];
    }
#endif

    double log_Z = std::log(sum_w) - beta * E_min;
    return { std::move(weights), log_Z, E_min };
}
#endif  // HAS_AVX512_RT

// ─── Boltzmann batch: Eigen path ─────────────────────────────────────────────

#ifdef FLEXAIDS_HAS_EIGEN
static BoltzmannBatchResult boltzmann_eigen(
    std::span<const double> energies, double beta)
{
    const int n = static_cast<int>(energies.size());
    Eigen::Map<const Eigen::ArrayXd> E(energies.data(), n);

    double E_min = E.minCoeff();
    Eigen::ArrayXd w = (-beta * (E - E_min)).exp();
    double sum_w = w.sum();
    double log_Z = std::log(sum_w) - beta * E_min;

    std::vector<double> weights(n);
    Eigen::Map<Eigen::ArrayXd>(weights.data(), n) = w;
    return { std::move(weights), log_Z, E_min };
}
#endif

// ─── compute_boltzmann_batch ─────────────────────────────────────────────────

BoltzmannBatchResult compute_boltzmann_batch(
    std::span<const double> energies,
    double                  beta)
{
    if (energies.empty())
        return { {}, 0.0, 0.0 };

    HardwareBackend cpu = select_cpu_backend();

#if HAS_AVX512_RT
    if (cpu == HardwareBackend::AVX512)
        return boltzmann_avx512(energies, beta);
#endif

#ifdef FLEXAIDS_HAS_EIGEN
    // Eigen vectorisation for AVX2/OpenMP paths (auto-vectorised)
    return boltzmann_eigen(energies, beta);
#endif

#ifdef _OPENMP
    if (cpu == HardwareBackend::OPENMP || cpu == HardwareBackend::AVX2)
        return boltzmann_openmp(energies, beta);
#endif

    return boltzmann_scalar(energies, beta);
}

// ─── log_sum_exp_dispatch ────────────────────────────────────────────────────

double log_sum_exp_dispatch(std::span<const double> values) {
    if (values.empty())
        return -std::numeric_limits<double>::infinity();

    const int n = static_cast<int>(values.size());

    // Find max for numerical stability
    double x_max = *std::max_element(values.begin(), values.end());
    if (!std::isfinite(x_max))
        return x_max;

#if HAS_AVX512_RT
    {
        double sum = 0.0;
        int i = 0;
        for (; i + 7 < n; i += 8) {
            __m512d vx    = _mm512_loadu_pd(values.data() + i);
            __m512d vdiff = _mm512_sub_pd(vx, _mm512_set1_pd(x_max));

            alignas(64) double tmp[8];
            _mm512_storeu_pd(tmp, vdiff);
            for (int k = 0; k < 8; ++k) tmp[k] = std::exp(tmp[k]);
            sum += _mm512_reduce_add_pd(_mm512_loadu_pd(tmp));
        }
        for (; i < n; ++i)
            sum += std::exp(values[i] - x_max);
        return x_max + std::log(sum);
    }
#endif

#ifdef FLEXAIDS_HAS_EIGEN
    {
        Eigen::Map<const Eigen::ArrayXd> x(values.data(), n);
        return x_max + std::log((x - x_max).exp().sum());
    }
#endif

#ifdef _OPENMP
    {
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum) schedule(static)
        for (int i = 0; i < n; ++i)
            sum += std::exp(values[i] - x_max);
        return x_max + std::log(sum);
    }
#endif

    // Scalar fallback
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
        sum += std::exp(values[i] - x_max);
    return x_max + std::log(sum);
}

// ─── get_dispatch_report ─────────────────────────────────────────────────────

DispatchReport get_dispatch_report() {
    const auto& hw = detect_hardware();
    HardwareBackend sel = select_backend();

    std::string reason;
    switch (sel) {
        case HardwareBackend::CUDA:
            reason = "CUDA GPU detected (" + hw.cuda_device_name + ", " + hw.cuda_arch + ")";
            break;
        case HardwareBackend::METAL:
            reason = "Metal GPU detected (" + hw.metal_gpu_name + ")";
            break;
        case HardwareBackend::AVX512:
            reason = "AVX-512 F+DQ+BW detected on CPU";
            break;
        case HardwareBackend::AVX2:
            reason = "AVX2+FMA detected on CPU";
            break;
        case HardwareBackend::OPENMP:
            reason = "OpenMP with " + std::to_string(hw.openmp_max_threads) + " threads";
            break;
        case HardwareBackend::SCALAR:
            reason = "No acceleration available; using scalar baseline";
            break;
    }

    return { sel, reason, hw.summary() };
}

}  // namespace flexaids
