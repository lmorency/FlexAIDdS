// ShannonThermoStack.cpp — multi-path implementation
//
// Hardware dispatch priority (runtime):
//   1. CUDA GPU   (FLEXAIDS_USE_CUDA)
//   2. Metal GPU  (FLEXAIDS_HAS_METAL_SHANNON, Apple Silicon)
//   3. AVX-512    (__AVX512F__)  — 8 doubles/cycle histogram binning
//   4. OpenMP     (_OPENMP)
//   5. Scalar     (always available)
//
// Eigen is used for vectorised log() / probability array ops on all CPU paths.
#include "ShannonThermoStack.h"

#ifdef FLEXAIDS_HAS_METAL_SHANNON
#  include "ShannonMetalBridge.h"
#endif

#ifdef FLEXAIDS_USE_CUDA
#  include "shannon_cuda.cuh"
static ShannonCudaCtx s_cuda_ctx;
static bool           s_cuda_ready = false;
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

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace shannon_thermo {

// ─── entropy from bin counts (Eigen-vectorised) ───────────────────────────────
static double entropy_from_counts(const int* counts, int num_bins, int total) {
    if (total == 0) return 0.0;
    const double l2inv = 1.0 / std::log(2.0);

#ifdef FLEXAIDS_HAS_EIGEN
    Eigen::ArrayXd prob(num_bins);
    for (int b = 0; b < num_bins; ++b)
        prob(b) = static_cast<double>(counts[b]);
    prob /= static_cast<double>(total);
    // Mask zeros before log to avoid -inf; Eigen evaluates log vectorised
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

// ─── AVX-512 private histogram ────────────────────────────────────────────────
#ifdef __AVX512F__
static void histogram_avx512(const double* values, int n,
                               double min_v, double inv_bw, int num_bins,
                               std::vector<int>& priv)
{
    int i = 0;
    __m512d vmin   = _mm512_set1_pd(min_v);
    __m512d vinvbw = _mm512_set1_pd(inv_bw);

    for (; i + 7 < n; i += 8) {
        __m512d ve   = _mm512_loadu_pd(values + i);
        __m512d vrel = _mm512_fmadd_pd(ve, vinvbw,
                           _mm512_mul_pd(_mm512_set1_pd(-min_v), vinvbw));
        // Convert 8 doubles → 8 int32 (truncate)
        __m256i v32 = _mm512_cvttpd_epi32(vrel);
        alignas(32) int tmp[8];
        _mm256_storeu_si256((__m256i*)tmp, v32);
        for (int k = 0; k < 8; ++k) {
            int b = std::min(std::max(tmp[k], 0), num_bins - 1);
            priv[b]++;
        }
    }
    for (; i < n; ++i) {
        int b = static_cast<int>((values[i] - min_v) * inv_bw);
        b = std::min(std::max(b, 0), num_bins - 1);
        priv[b]++;
    }
}
#endif // __AVX512F__

// ─── compute_shannon_entropy ─────────────────────────────────────────────────

double compute_shannon_entropy(const std::vector<double>& values, int num_bins) {
    if (values.empty()) return 0.0;
    if (num_bins <= 0)  num_bins = DEFAULT_HIST_BINS;

    double min_v = *std::min_element(values.begin(), values.end());
    double max_v = *std::max_element(values.begin(), values.end());
    if (max_v - min_v < 1e-12) return 0.0;
    double bin_width = (max_v - min_v) / num_bins + 1e-10;
    double inv_bw    = 1.0 / bin_width;
    int    n         = static_cast<int>(values.size());

    std::vector<int> bins(num_bins, 0);

// ── 1. CUDA ───────────────────────────────────────────────────────────────────
#ifdef FLEXAIDS_USE_CUDA
    {
        if (!s_cuda_ready) {
            shannon_cuda_init(s_cuda_ctx, 1 << 20, num_bins);
            s_cuda_ready = true;
        }
        if (n <= s_cuda_ctx.capacity) {
            shannon_cuda_histogram(s_cuda_ctx, values.data(), n,
                                   min_v, bin_width, bins.data());
            return entropy_from_counts(bins.data(), num_bins, n);
        }
    }
#endif

// ── 2. Metal ──────────────────────────────────────────────────────────────────
#ifdef FLEXAIDS_HAS_METAL_SHANNON
    return ShannonMetalBridge::compute_shannon_entropy_metal(values, num_bins);
#endif

// ── 3. AVX-512 (+ optional OpenMP for multi-threaded private histograms) ──────
#ifdef __AVX512F__
    {
#  ifdef _OPENMP
        int n_threads = omp_get_max_threads();
        std::vector<std::vector<int>> t_bins(n_threads, std::vector<int>(num_bins, 0));
        #pragma omp parallel
        {
            int tid    = omp_get_thread_num();
            int chunk  = (n + n_threads - 1) / n_threads;
            int start  = tid * chunk;
            int end    = std::min(start + chunk, n);
            if (start < end)
                histogram_avx512(values.data() + start, end - start,
                                 min_v, inv_bw, num_bins, t_bins[tid]);
        }
        for (auto& tb : t_bins)
            for (int b = 0; b < num_bins; ++b) bins[b] += tb[b];
#  else
        histogram_avx512(values.data(), n, min_v, inv_bw, num_bins, bins);
#  endif
    }

// ── 4. OpenMP scalar ──────────────────────────────────────────────────────────
#elif defined(_OPENMP)
    {
        int n_threads = omp_get_max_threads();
        std::vector<std::vector<int>> t_bins(n_threads, std::vector<int>(num_bins, 0));
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            int tid = omp_get_thread_num();
            int b   = static_cast<int>((values[i] - min_v) * inv_bw);
            t_bins[tid][std::min(std::max(b, 0), num_bins - 1)]++;
        }
        for (auto& tb : t_bins)
            for (int b = 0; b < num_bins; ++b) bins[b] += tb[b];
    }

// ── 5. Scalar ─────────────────────────────────────────────────────────────────
#else
    for (int i = 0; i < n; ++i) {
        int b = static_cast<int>((values[i] - min_v) * inv_bw);
        bins[std::min(std::max(b, 0), num_bins - 1)]++;
    }
#endif

    return entropy_from_counts(bins.data(), num_bins, n);
}

double compute_shannon_entropy_discrete(const std::vector<int>& counts) {
    int total = std::accumulate(counts.begin(), counts.end(), 0);
    return entropy_from_counts(counts.data(), static_cast<int>(counts.size()), total);
}

// ─── compute_torsional_vibrational_entropy (Eigen-vectorised) ────────────────

double compute_torsional_vibrational_entropy(
    const std::vector<tencm::NormalMode>& modes,
    double temperature_K)
{
    if (modes.empty()) return 0.0;
    const double kT = kB_kcal * temperature_K;

#ifdef FLEXAIDS_HAS_EIGEN
    // Collect valid eigenvalues into Eigen array, then vectorise
    std::vector<double> ev_buf;
    ev_buf.reserve(modes.size());
    for (size_t m = 6; m < modes.size(); ++m)
        if (modes[m].eigenvalue > 1e-6) ev_buf.push_back(modes[m].eigenvalue);
    if (ev_buf.empty()) return 0.0;

    Eigen::Map<Eigen::ArrayXd> evals(ev_buf.data(), (int)ev_buf.size());
    Eigen::ArrayXd ln_arg = kT / evals; // element-wise
    // S_mode = kB*(1 + ln(kBT/ω)) for modes where ln_arg > 1e-6
    Eigen::ArrayXd mask = (ln_arg > 1e-6).cast<double>();
    return kB_kcal * (mask * (1.0 + ln_arg.log())).sum();
#else
    double S = 0.0;
    for (size_t m = 6; m < modes.size(); ++m) {
        if (modes[m].eigenvalue < 1e-6) continue;
        double la = kT / modes[m].eigenvalue;
        if (la > 1e-6) S += kB_kcal * (1.0 + std::log(la));
    }
    return S;
#endif
}

// ─── run_shannon_thermo_stack ────────────────────────────────────────────────

FullThermoResult run_shannon_thermo_stack(
    const statmech::StatMechEngine& stat_engine,
    const tencm::TorsionalENM&      tencm_model,
    double                          base_deltaG,
    double                          temperature_K)
{
    auto weights = stat_engine.boltzmann_weights();
    std::vector<double> log_weights;
    log_weights.reserve(weights.size());
    for (double w : weights)
        if (w > 0.0) log_weights.push_back(-std::log(w));

    double S_conf_bits  = compute_shannon_entropy(log_weights, DEFAULT_HIST_BINS);
    double S_vib        = tencm_model.is_built()
                          ? compute_torsional_vibrational_entropy(tencm_model.modes(), temperature_K)
                          : 0.0;
    double S_conf_phys  = S_conf_bits * kB_kcal;
    // Standard additive entropy: S_total = S_conf + S_vib
    // (quasi-harmonic approximation; no nonlinear coupling)
    double total_S      = S_conf_phys + S_vib;
    double S_contrib    = -temperature_K * total_S;
    double final_dG     = base_deltaG + S_contrib;

    const char* hw =
#if defined(FLEXAIDS_USE_CUDA)
        "CUDA";
#elif defined(FLEXAIDS_HAS_METAL_SHANNON)
        "Metal";
#elif defined(__AVX512F__)
        "AVX-512";
#elif defined(_OPENMP)
        "OpenMP";
#else
        "scalar";
#endif

    std::string report =
        std::string("ShannonThermoStack[") + hw +
#ifdef FLEXAIDS_HAS_EIGEN
        "+Eigen"
#endif
        "]: S_conf=" + std::to_string(S_conf_bits) +
        " bits, S_vib=" + std::to_string(S_vib) +
        " kcal/mol/K, ΔG=" + std::to_string(final_dG) + " kcal/mol";

    return { final_dG, S_conf_bits, S_vib, S_contrib, report };
}

} // namespace shannon_thermo
