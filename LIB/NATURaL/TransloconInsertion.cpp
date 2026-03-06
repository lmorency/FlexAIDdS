// TransloconInsertion.cpp — Sec translocon lateral gating implementation
//
// Hardware dispatch (compile-time, priority order):
//   1. AVX-512  (_mm512_*) — 8 double-precision scores/cycle
//   2. AVX2     (_mm256_*) — 4 double-precision scores/cycle
//   3. Eigen    — vectorised position-weight dot product
//   4. Scalar   — portable fallback
//
// OpenMP parallel scan across windows (enabled when FLEXAIDS_USE_OPENMP).
#include "TransloconInsertion.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#ifdef FLEXAIDS_HAS_EIGEN
#  include <Eigen/Dense>
#endif

#if defined(_OPENMP)
#  include <omp.h>
#endif

namespace translocon {

// ─── constructor ──────────────────────────────────────────────────────────────
TransloconInsertion::TransloconInsertion(double temperature_K,
                                          double insertion_threshold,
                                          int    tunnel_length_aa)
    : T_K_(temperature_K),
      kT_(0.001987206 * temperature_K),
      threshold_(insertion_threshold),
      tunnel_len_(tunnel_length_aa)
{}

// ─── score_window_scalar ──────────────────────────────────────────────────────
double TransloconInsertion::score_window_scalar(const char* seq, int len) const noexcept {
    double dG = 0.0;
    for (int i = 0; i < len; ++i) {
        int c = static_cast<unsigned char>(seq[i]);
        double contrib = HESSA_SCALE[c] * position_weight(i, len);
        dG += contrib;
    }
    // Helix-dipole correction: favorable for hydrophobic core
    dG += HELIX_DIPOLE_CORR * static_cast<double>(len);
    return dG;
}

// ─── score_window_eigen ───────────────────────────────────────────────────────
double TransloconInsertion::score_window_eigen(const char* seq, int len) const noexcept {
#ifdef FLEXAIDS_HAS_EIGEN
    Eigen::ArrayXd hessa(len), weights(len);
    for (int i = 0; i < len; ++i) {
        int c = static_cast<unsigned char>(seq[i]);
        hessa(i)  = HESSA_SCALE[c];
        weights(i) = position_weight(i, len);
    }
    double dG = (hessa * weights).sum();
    dG += HELIX_DIPOLE_CORR * static_cast<double>(len);
    return dG;
#else
    return score_window_scalar(seq, len);
#endif
}

// ─── score_window_avx2 ────────────────────────────────────────────────────────
double TransloconInsertion::score_window_avx2(const char* seq, int len) const noexcept {
#ifdef __AVX2__
    // Process 4 positions at a time with AVX2 (double precision)
    __m256d acc = _mm256_setzero_pd();
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        // Load 4 Hessa values
        __m256d h = _mm256_set_pd(
            HESSA_SCALE[static_cast<unsigned char>(seq[i+3])],
            HESSA_SCALE[static_cast<unsigned char>(seq[i+2])],
            HESSA_SCALE[static_cast<unsigned char>(seq[i+1])],
            HESSA_SCALE[static_cast<unsigned char>(seq[i+0])]);
        // Load 4 position weights
        __m256d w = _mm256_set_pd(
            position_weight(i+3, len),
            position_weight(i+2, len),
            position_weight(i+1, len),
            position_weight(i+0, len));
        // FMA: acc += h * w
        acc = _mm256_fmadd_pd(h, w, acc);
    }
    // Horizontal sum
    __m128d lo  = _mm256_castpd256_pd128(acc);
    __m128d hi  = _mm256_extractf128_pd(acc, 1);
    __m128d sum = _mm_add_pd(lo, hi);
    sum = _mm_hadd_pd(sum, sum);
    double dG;
    _mm_store_sd(&dG, sum);
    // Scalar tail
    for (; i < len; ++i) {
        int c = static_cast<unsigned char>(seq[i]);
        dG += HESSA_SCALE[c] * position_weight(i, len);
    }
    dG += HELIX_DIPOLE_CORR * static_cast<double>(len);
    return dG;
#else
    return score_window_eigen(seq, len);
#endif
}

// ─── score_window_avx512 ──────────────────────────────────────────────────────
double TransloconInsertion::score_window_avx512(const char* seq, int len) const noexcept {
#ifdef FLEXAIDS_USE_AVX512
    // Process 8 positions at a time with AVX-512 (double precision)
    __m512d acc = _mm512_setzero_pd();
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        __m512d h = _mm512_set_pd(
            HESSA_SCALE[static_cast<unsigned char>(seq[i+7])],
            HESSA_SCALE[static_cast<unsigned char>(seq[i+6])],
            HESSA_SCALE[static_cast<unsigned char>(seq[i+5])],
            HESSA_SCALE[static_cast<unsigned char>(seq[i+4])],
            HESSA_SCALE[static_cast<unsigned char>(seq[i+3])],
            HESSA_SCALE[static_cast<unsigned char>(seq[i+2])],
            HESSA_SCALE[static_cast<unsigned char>(seq[i+1])],
            HESSA_SCALE[static_cast<unsigned char>(seq[i+0])]);
        __m512d w = _mm512_set_pd(
            position_weight(i+7, len),
            position_weight(i+6, len),
            position_weight(i+5, len),
            position_weight(i+4, len),
            position_weight(i+3, len),
            position_weight(i+2, len),
            position_weight(i+1, len),
            position_weight(i+0, len));
        acc = _mm512_fmadd_pd(h, w, acc);
    }
    double dG = _mm512_reduce_add_pd(acc);
    for (; i < len; ++i) {
        int c = static_cast<unsigned char>(seq[i]);
        dG += HESSA_SCALE[c] * position_weight(i, len);
    }
    dG += HELIX_DIPOLE_CORR * static_cast<double>(len);
    return dG;
#else
    return score_window_avx2(seq, len);
#endif
}

// ─── score_window (dispatch) ──────────────────────────────────────────────────
double TransloconInsertion::score_window(const std::string& sequence,
                                          int                start_res,
                                          int                len) const
{
    if (start_res < 0 || start_res >= static_cast<int>(sequence.size()))
        return 0.0;
    int actual_len = std::min(len, static_cast<int>(sequence.size()) - start_res);
    if (actual_len <= 0) return 0.0;

    const char* ptr = sequence.data() + start_res;

#ifdef FLEXAIDS_USE_AVX512
    return score_window_avx512(ptr, actual_len);
#elif defined(__AVX2__)
    return score_window_avx2(ptr, actual_len);
#elif defined(FLEXAIDS_HAS_EIGEN)
    return score_window_eigen(ptr, actual_len);
#else
    return score_window_scalar(ptr, actual_len);
#endif
}

// ─── check_window ─────────────────────────────────────────────────────────────
TMWindow TransloconInsertion::check_window(const std::string& sequence,
                                            int                start_res) const
{
    int len = std::min(TM_WINDOW_LEN,
                       static_cast<int>(sequence.size()) - start_res);
    double dG = score_window(sequence, start_res, len);
    double p  = 1.0 / (1.0 + std::exp(dG / kT_));
    return TMWindow{start_res, len, dG, p, (p >= threshold_)};
}

// ─── scan ─────────────────────────────────────────────────────────────────────
std::vector<TMWindow> TransloconInsertion::scan(const std::string& sequence) const {
    int n = static_cast<int>(sequence.size());
    if (n < TM_WINDOW_LEN) return {};

    int n_windows = n - TM_WINDOW_LEN + 1;
    std::vector<TMWindow> windows(n_windows);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(n_windows > 64)
#endif
    for (int w = 0; w < n_windows; ++w) {
        int len = std::min(TM_WINDOW_LEN, n - w);
        double dG = score_window(sequence, w, len);
        double p  = 1.0 / (1.0 + std::exp(dG / kT_));
        windows[w] = TMWindow{w, len, dG, p, (p >= threshold_)};
    }

    return windows;
}

} // namespace translocon
