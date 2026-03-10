// simd_distance.h — AVX2-vectorised geometric primitives for FlexAIDdS
// Requires -mavx2 -mfma (GCC/Clang) or /arch:AVX2 (MSVC)
#pragma once

#include <cmath>
#include <cstdint>
#include <array>

#ifdef _MSC_VER
#  define FLEXAIDS_RESTRICT __restrict
#else
#  define FLEXAIDS_RESTRICT __restrict__
#endif

#if defined(__AVX2__)
#  include <immintrin.h>
#  define FLEXAIDS_HAS_AVX2 1
#else
#  define FLEXAIDS_HAS_AVX2 0
#endif

namespace simd {

// ─── scalar fallbacks ────────────────────────────────────────────────────────

inline float sq(float x) noexcept { return x * x; }

inline float distance2_scalar(const float* FLEXAIDS_RESTRICT a,
                               const float* FLEXAIDS_RESTRICT b) noexcept {
    return sq(a[0]-b[0]) + sq(a[1]-b[1]) + sq(a[2]-b[2]);
}

inline float distance_scalar(const float* a, const float* b) noexcept {
    return std::sqrt(distance2_scalar(a, b));
}

// Cross product c = a × b (3-D)
inline void cross3(const float* a, const float* b, float* c) noexcept {
    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];
}

inline float dot3(const float* a, const float* b) noexcept {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

inline void normalize3(float* v) noexcept {
    float inv_len = 1.0f / std::sqrt(dot3(v, v));
    v[0] *= inv_len;  v[1] *= inv_len;  v[2] *= inv_len;
}

// ─── AVX2 implementations ────────────────────────────────────────────────────

#if FLEXAIDS_HAS_AVX2

// Horizontal sum of an __m256 register
inline float hsum256_ps(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
    return _mm_cvtss_f32(lo);
}

// Squared distances between one point B and 8 points A stored in SOA layout.
//   ax[8], ay[8], az[8] – x/y/z of 8 A atoms
//   bx, by, bz           – coordinates of atom B
//   out[8]               – results
inline void distance2_1x8(const float* FLEXAIDS_RESTRICT ax,
                           const float* FLEXAIDS_RESTRICT ay,
                           const float* FLEXAIDS_RESTRICT az,
                           float bx, float by, float bz,
                           float* FLEXAIDS_RESTRICT out) noexcept {
    __m256 vbx = _mm256_set1_ps(bx);
    __m256 vby = _mm256_set1_ps(by);
    __m256 vbz = _mm256_set1_ps(bz);
    __m256 dx  = _mm256_sub_ps(_mm256_loadu_ps(ax), vbx);
    __m256 dy  = _mm256_sub_ps(_mm256_loadu_ps(ay), vby);
    __m256 dz  = _mm256_sub_ps(_mm256_loadu_ps(az), vbz);
    __m256 r2  = _mm256_fmadd_ps(dz, dz,
                 _mm256_fmadd_ps(dy, dy,
                 _mm256_mul_ps(dx, dx)));
    _mm256_storeu_ps(out, r2);
}

// Sum of squared distances (for RMSD pre-accumulation) over N atoms.
// a_xyz: N×3 interleaved, b_xyz: N×3 interleaved
// Returns Σ |a_i - b_i|²
inline float sum_sq_distances(const float* FLEXAIDS_RESTRICT a_xyz,
                               const float* FLEXAIDS_RESTRICT b_xyz,
                               int N) noexcept {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= N - 8; i += 8) {
        // Load 8 xyz triplets for a and b in AOS layout (24 floats each)
        // We handle 1 component at a time: x, y, z separately
        // Gather with stride-3 – use scalar here for correctness; the inner
        // loop is over atoms so the hot path is the per-component subtraction.
        for (int c = 0; c < 3; ++c) {
            // Gather 8 values at stride 3
            float a8[8], b8[8];
            for (int k = 0; k < 8; ++k) {
                a8[k] = a_xyz[(i+k)*3 + c];
                b8[k] = b_xyz[(i+k)*3 + c];
            }
            __m256 da = _mm256_sub_ps(_mm256_loadu_ps(a8), _mm256_loadu_ps(b8));
            acc = _mm256_fmadd_ps(da, da, acc);
        }
    }
    float sum = hsum256_ps(acc);
    for (; i < N; ++i) {
        for (int c = 0; c < 3; ++c)
            sum += sq(a_xyz[i*3+c] - b_xyz[i*3+c]);
    }
    return sum;
}

// Lennard-Jones r^-12 wall energy for 8 distances simultaneously.
//   r2[8]  – squared distances (must NOT be zero)
//   rAB12  – (permeability * r_AB)^12 precomputed
//   k_wall – wall constant
// Writes Ewall[8]; caller is responsible for masking < clash_distance.
inline void lj_wall_8x(const float* FLEXAIDS_RESTRICT r2,
                        float inv_rAB12,
                        float k_wall,
                        float* FLEXAIDS_RESTRICT Ewall) noexcept {
    __m256 vr2    = _mm256_loadu_ps(r2);
    // Approximate inv_r2 via Newton step on _mm256_rcp_ps
    __m256 inv_r2 = _mm256_rcp_ps(vr2);
    // One Newton-Raphson refinement: inv_r2 ≈ inv_r2*(2 - vr2*inv_r2)
    inv_r2 = _mm256_mul_ps(inv_r2,
              _mm256_sub_ps(_mm256_set1_ps(2.0f),
              _mm256_mul_ps(vr2, inv_r2)));
    __m256 inv_r4  = _mm256_mul_ps(inv_r2, inv_r2);
    __m256 inv_r6  = _mm256_mul_ps(inv_r4, inv_r2);
    __m256 inv_r12 = _mm256_mul_ps(inv_r6, inv_r6);
    __m256 vKWALL  = _mm256_set1_ps(k_wall);
    __m256 vrAB12  = _mm256_set1_ps(inv_rAB12);
    __m256 e = _mm256_mul_ps(vKWALL, _mm256_sub_ps(inv_r12, vrAB12));
    _mm256_storeu_ps(Ewall, e);
}

// Batched dot products: result[i] = dot(a[i], b[i]) for i in [0,N)
// a, b are Nx3 in interleaved layout. N must be multiple of 8 or padded.
inline void dot3_batch(const float* FLEXAIDS_RESTRICT a,
                       const float* FLEXAIDS_RESTRICT b,
                       float* FLEXAIDS_RESTRICT out, int N) noexcept {
    __m256 acc = _mm256_setzero_ps();
    // Separate into SOA on-the-fly (each component processed independently)
    int i = 0;
    for (; i <= N - 8; i += 8) {
        __m256 s = _mm256_setzero_ps();
        for (int c = 0; c < 3; ++c) {
            float a8[8], b8[8];
            for (int k = 0; k < 8; ++k) {
                a8[k] = a[(i+k)*3+c];
                b8[k] = b[(i+k)*3+c];
            }
            s = _mm256_fmadd_ps(_mm256_loadu_ps(a8),
                                _mm256_loadu_ps(b8), s);
        }
        _mm256_storeu_ps(out + i, s);
    }
    for (; i < N; ++i) {
        out[i] = a[i*3]*b[i*3] + a[i*3+1]*b[i*3+1] + a[i*3+2]*b[i*3+2];
    }
    (void)acc;
}

#else  // ─── scalar fallback versions ─────────────────────────────────────────

inline void distance2_1x8(const float* ax, const float* ay, const float* az,
                           float bx, float by, float bz,
                           float* out) noexcept {
    for (int k = 0; k < 8; ++k)
        out[k] = sq(ax[k]-bx) + sq(ay[k]-by) + sq(az[k]-bz);
}

inline float sum_sq_distances(const float* a, const float* b, int N) noexcept {
    float s = 0;
    for (int i = 0; i < N; ++i)
        for (int c = 0; c < 3; ++c)
            s += sq(a[i*3+c] - b[i*3+c]);
    return s;
}

inline void lj_wall_8x(const float* r2, float inv_rAB12, float k_wall,
                        float* Ewall) noexcept {
    for (int k = 0; k < 8; ++k) {
        float inv_r6 = 1.0f / (r2[k]*r2[k]*r2[k]);
        float inv_r12 = inv_r6 * inv_r6;
        Ewall[k] = k_wall * (inv_r12 - inv_rAB12);
    }
}

inline void dot3_batch(const float* a, const float* b, float* out, int N) noexcept {
    for (int i = 0; i < N; ++i)
        out[i] = a[i*3]*b[i*3] + a[i*3+1]*b[i*3+1] + a[i*3+2]*b[i*3+2];
}

#endif  // FLEXAIDS_HAS_AVX2

// ─── dispatch helper: compile-time dispatch to best available ────────────────

// RMSD between two coordinate arrays (N atoms, interleaved xyz)
inline float rmsd(const float* a, const float* b, int N) noexcept {
    return std::sqrt(sum_sq_distances(a, b, N) / static_cast<float>(N));
}

}  // namespace simd
