// simd_distance.h — AVX2/AVX-512 vectorised geometric primitives for FlexAIDdS
// Requires -mavx2 -mfma (GCC/Clang) or /arch:AVX2 (MSVC) for AVX2 path.
// Requires -mavx512f -mavx512dq -mavx512bw for AVX-512 path.
#pragma once

#include <cmath>
#include <cstdint>
#include <array>

#ifdef _MSC_VER
#  define FLEXAIDS_RESTRICT __restrict
#else
#  define FLEXAIDS_RESTRICT __restrict__
#endif

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#  include <immintrin.h>
#  define FLEXAIDS_HAS_AVX512 1
#  define FLEXAIDS_HAS_AVX2   1       // AVX-512 implies AVX2
#elif defined(__AVX2__)
#  include <immintrin.h>
#  define FLEXAIDS_HAS_AVX512 0
#  define FLEXAIDS_HAS_AVX2   1
#else
#  define FLEXAIDS_HAS_AVX512 0
#  define FLEXAIDS_HAS_AVX2   0
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

// ─── AVX-512 implementations ─────────────────────────────────────────────────

#if FLEXAIDS_HAS_AVX512

// Horizontal sum of a __m512 register (16 floats → 1 float)
inline float hsum512_ps(__m512 v) noexcept {
    return _mm512_reduce_add_ps(v);
}

// Squared distances between one point B and 16 points A stored in SOA layout.
//   ax[16], ay[16], az[16] – x/y/z of 16 A atoms
//   bx, by, bz              – coordinates of atom B
//   out[16]                 – results
inline void distance2_1x16(const float* FLEXAIDS_RESTRICT ax,
                            const float* FLEXAIDS_RESTRICT ay,
                            const float* FLEXAIDS_RESTRICT az,
                            float bx, float by, float bz,
                            float* FLEXAIDS_RESTRICT out) noexcept {
    __m512 vbx = _mm512_set1_ps(bx);
    __m512 vby = _mm512_set1_ps(by);
    __m512 vbz = _mm512_set1_ps(bz);
    __m512 dx  = _mm512_sub_ps(_mm512_loadu_ps(ax), vbx);
    __m512 dy  = _mm512_sub_ps(_mm512_loadu_ps(ay), vby);
    __m512 dz  = _mm512_sub_ps(_mm512_loadu_ps(az), vbz);
    __m512 r2  = _mm512_fmadd_ps(dz, dz,
                 _mm512_fmadd_ps(dy, dy,
                 _mm512_mul_ps(dx, dx)));
    _mm512_storeu_ps(out, r2);
}

// Also provide the 8-wide version using AVX-512 (uses 256-bit subset)
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

// Sum of squared distances over N atoms (AOS interleaved xyz), 16-wide.
// Returns Σ |a_i - b_i|²
inline float sum_sq_distances(const float* FLEXAIDS_RESTRICT a_xyz,
                               const float* FLEXAIDS_RESTRICT b_xyz,
                               int N) noexcept {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i <= N - 16; i += 16) {
        for (int c = 0; c < 3; ++c) {
            float a16[16], b16[16];
            for (int k = 0; k < 16; ++k) {
                a16[k] = a_xyz[(i+k)*3 + c];
                b16[k] = b_xyz[(i+k)*3 + c];
            }
            __m512 da = _mm512_sub_ps(_mm512_loadu_ps(a16), _mm512_loadu_ps(b16));
            acc = _mm512_fmadd_ps(da, da, acc);
        }
    }
    float sum = hsum512_ps(acc);
    // Scalar tail
    for (; i < N; ++i)
        for (int c = 0; c < 3; ++c)
            sum += sq(a_xyz[i*3+c] - b_xyz[i*3+c]);
    return sum;
}

// Lennard-Jones r^-12 wall energy for 16 distances simultaneously.
//   r2[16]  – squared distances (must NOT be zero)
//   inv_rAB12 – (permeability * r_AB)^12 precomputed
//   k_wall  – wall constant
inline void lj_wall_16x(const float* FLEXAIDS_RESTRICT r2,
                         float inv_rAB12,
                         float k_wall,
                         float* FLEXAIDS_RESTRICT Ewall) noexcept {
    __m512 vr2     = _mm512_loadu_ps(r2);
    __m512 inv_r2  = _mm512_rcp14_ps(vr2);
    // Newton-Raphson refinement: inv_r2 *= 2 - vr2 * inv_r2
    inv_r2 = _mm512_mul_ps(inv_r2,
              _mm512_sub_ps(_mm512_set1_ps(2.0f),
              _mm512_mul_ps(vr2, inv_r2)));
    __m512 inv_r4  = _mm512_mul_ps(inv_r2, inv_r2);
    __m512 inv_r6  = _mm512_mul_ps(inv_r4, inv_r2);
    __m512 inv_r12 = _mm512_mul_ps(inv_r6, inv_r6);
    __m512 e = _mm512_mul_ps(_mm512_set1_ps(k_wall),
               _mm512_sub_ps(inv_r12, _mm512_set1_ps(inv_rAB12)));
    _mm512_storeu_ps(Ewall, e);
}

// 8-wide LJ wall (AVX2-compatible intrinsics available via AVX-512 superset)
inline void lj_wall_8x(const float* FLEXAIDS_RESTRICT r2,
                        float inv_rAB12,
                        float k_wall,
                        float* FLEXAIDS_RESTRICT Ewall) noexcept {
    __m256 vr2    = _mm256_loadu_ps(r2);
    __m256 inv_r2 = _mm256_rcp_ps(vr2);
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

// Batched dot products: result[i] = dot(a[i], b[i]), 16-wide
inline void dot3_batch(const float* FLEXAIDS_RESTRICT a,
                       const float* FLEXAIDS_RESTRICT b,
                       float* FLEXAIDS_RESTRICT out, int N) noexcept {
    int i = 0;
    for (; i <= N - 16; i += 16) {
        __m512 s = _mm512_setzero_ps();
        for (int c = 0; c < 3; ++c) {
            float a16[16], b16[16];
            for (int k = 0; k < 16; ++k) {
                a16[k] = a[(i+k)*3+c];
                b16[k] = b[(i+k)*3+c];
            }
            s = _mm512_fmadd_ps(_mm512_loadu_ps(a16),
                                _mm512_loadu_ps(b16), s);
        }
        _mm512_storeu_ps(out + i, s);
    }
    for (; i < N; ++i) {
        out[i] = a[i*3]*b[i*3] + a[i*3+1]*b[i*3+1] + a[i*3+2]*b[i*3+2];
    }
}

// Horizontal sum for AVX2 subset (still needed for some callers)
inline float hsum256_ps(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
    return _mm_cvtss_f32(lo);
}

// ─── AVX2 implementations ────────────────────────────────────────────────────

#elif FLEXAIDS_HAS_AVX2

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

// ─── AVX-512 implementations (16-wide float, 8-wide double) ─────────────────

#if FLEXAIDS_HAS_AVX512

// Horizontal sum of an __m512 register (16 floats → 1 float)
inline float hsum512_ps(__m512 v) noexcept {
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    __m256 s  = _mm256_add_ps(lo, hi);
    return hsum256_ps(s);
}

// Horizontal sum of an __m512d register (8 doubles → 1 double)
inline double hsum512_pd(__m512d v) noexcept {
    __m256d lo = _mm512_castpd512_pd256(v);
    __m256d hi = _mm512_extractf64x4_pd(v, 1);
    __m256d s  = _mm256_add_pd(lo, hi);
    // reduce 4→2→1
    __m128d s_lo = _mm256_castpd256_pd128(s);
    __m128d s_hi = _mm256_extractf128_pd(s, 1);
    s_lo = _mm_add_pd(s_lo, s_hi);
    s_lo = _mm_add_sd(s_lo, _mm_unpackhi_pd(s_lo, s_lo));
    return _mm_cvtsd_f64(s_lo);
}

// Squared distances between one point B and 16 points A stored in SOA layout.
//   ax[16], ay[16], az[16] – x/y/z of 16 A atoms
//   bx, by, bz             – coordinates of atom B
//   out[16]                – results
inline void distance2_1x16(const float* FLEXAIDS_RESTRICT ax,
                            const float* FLEXAIDS_RESTRICT ay,
                            const float* FLEXAIDS_RESTRICT az,
                            float bx, float by, float bz,
                            float* FLEXAIDS_RESTRICT out) noexcept {
    __m512 vbx = _mm512_set1_ps(bx);
    __m512 vby = _mm512_set1_ps(by);
    __m512 vbz = _mm512_set1_ps(bz);
    __m512 dx  = _mm512_sub_ps(_mm512_loadu_ps(ax), vbx);
    __m512 dy  = _mm512_sub_ps(_mm512_loadu_ps(ay), vby);
    __m512 dz  = _mm512_sub_ps(_mm512_loadu_ps(az), vbz);
    __m512 r2  = _mm512_fmadd_ps(dz, dz,
                 _mm512_fmadd_ps(dy, dy,
                 _mm512_mul_ps(dx, dx)));
    _mm512_storeu_ps(out, r2);
}

// Sum of squared distances over N atoms (AVX-512, 16-wide).
// a_xyz: N×3 interleaved, b_xyz: N×3 interleaved
inline float sum_sq_distances_avx512(const float* FLEXAIDS_RESTRICT a_xyz,
                                      const float* FLEXAIDS_RESTRICT b_xyz,
                                      int N) noexcept {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i <= N - 16; i += 16) {
        for (int c = 0; c < 3; ++c) {
            float a16[16], b16[16];
            for (int k = 0; k < 16; ++k) {
                a16[k] = a_xyz[(i+k)*3 + c];
                b16[k] = b_xyz[(i+k)*3 + c];
            }
            __m512 da = _mm512_sub_ps(_mm512_loadu_ps(a16), _mm512_loadu_ps(b16));
            acc = _mm512_fmadd_ps(da, da, acc);
        }
    }
    float sum = _mm512_reduce_add_ps(acc);
    // Tail with AVX2
    for (; i < N; ++i) {
        for (int c = 0; c < 3; ++c)
            sum += sq(a_xyz[i*3+c] - b_xyz[i*3+c]);
    }
    return sum;
}

// Lennard-Jones r^-12 wall energy for 16 distances simultaneously (AVX-512).
inline void lj_wall_16x(const float* FLEXAIDS_RESTRICT r2,
                         float inv_rAB12,
                         float k_wall,
                         float* FLEXAIDS_RESTRICT Ewall) noexcept {
    __m512 vr2    = _mm512_loadu_ps(r2);
    // AVX-512 has full-precision reciprocal with _mm512_rcp14_ps + Newton step
    __m512 inv_r2 = _mm512_rcp14_ps(vr2);
    inv_r2 = _mm512_mul_ps(inv_r2,
              _mm512_sub_ps(_mm512_set1_ps(2.0f),
              _mm512_mul_ps(vr2, inv_r2)));
    __m512 inv_r4  = _mm512_mul_ps(inv_r2, inv_r2);
    __m512 inv_r6  = _mm512_mul_ps(inv_r4, inv_r2);
    __m512 inv_r12 = _mm512_mul_ps(inv_r6, inv_r6);
    __m512 vKWALL  = _mm512_set1_ps(k_wall);
    __m512 vrAB12  = _mm512_set1_ps(inv_rAB12);
    __m512 e = _mm512_mul_ps(vKWALL, _mm512_sub_ps(inv_r12, vrAB12));
    _mm512_storeu_ps(Ewall, e);
}

// Batched dot products for 16 vectors at a time (AVX-512).
inline void dot3_batch_avx512(const float* FLEXAIDS_RESTRICT a,
                               const float* FLEXAIDS_RESTRICT b,
                               float* FLEXAIDS_RESTRICT out, int N) noexcept {
    int i = 0;
    for (; i <= N - 16; i += 16) {
        __m512 s = _mm512_setzero_ps();
        for (int c = 0; c < 3; ++c) {
            float a16[16], b16[16];
            for (int k = 0; k < 16; ++k) {
                a16[k] = a[(i+k)*3+c];
                b16[k] = b[(i+k)*3+c];
            }
            s = _mm512_fmadd_ps(_mm512_loadu_ps(a16),
                                _mm512_loadu_ps(b16), s);
        }
        _mm512_storeu_ps(out + i, s);
    }
    // Tail: use AVX2 path for remaining
    for (; i < N; ++i) {
        out[i] = a[i*3]*b[i*3] + a[i*3+1]*b[i*3+1] + a[i*3+2]*b[i*3+2];
    }
}

// Boltzmann factor batch: compute exp(-beta * E[i]) for 8 doubles (AVX-512d).
// Uses fast approximation via polynomial; caller can use exact std::exp fallback
// for validation.
inline void boltzmann_batch_8d(const double* FLEXAIDS_RESTRICT energies,
                                double beta,
                                double* FLEXAIDS_RESTRICT weights,
                                int N) noexcept {
    // Process 8 doubles at a time using 512-bit double lanes
    int i = 0;
    __m512d vneg_beta = _mm512_set1_pd(-beta);
    for (; i + 7 < N; i += 8) {
        __m512d ve = _mm512_loadu_pd(energies + i);
        __m512d arg = _mm512_mul_pd(ve, vneg_beta);
        // Use SVML _mm512_exp_pd if available, else scalar fallback
#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
        __m512d result = _mm512_exp_pd(arg);
        _mm512_storeu_pd(weights + i, result);
#else
        // Scalar fallback for exp — still benefits from 512-bit load/store
        alignas(64) double tmp[8];
        _mm512_storeu_pd(tmp, arg);
        for (int k = 0; k < 8; ++k) tmp[k] = std::exp(tmp[k]);
        _mm512_storeu_pd(weights + i, _mm512_loadu_pd(tmp));
#endif
    }
    for (; i < N; ++i)
        weights[i] = std::exp(-beta * energies[i]);
}

#endif  // FLEXAIDS_HAS_AVX512

// ─────────────────────────────────────────────────────────────────────────────

#else  // ─── scalar fallback versions ─────────────────────────────────────────

inline void distance2_1x16(const float* ax, const float* ay, const float* az,
                            float bx, float by, float bz,
                            float* out) noexcept {
    for (int k = 0; k < 16; ++k)
        out[k] = sq(ax[k]-bx) + sq(ay[k]-by) + sq(az[k]-bz);
}

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

inline void lj_wall_16x(const float* r2, float inv_rAB12, float k_wall,
                          float* Ewall) noexcept {
    for (int k = 0; k < 16; ++k) {
        float inv_r6 = 1.0f / (r2[k]*r2[k]*r2[k]);
        float inv_r12 = inv_r6 * inv_r6;
        Ewall[k] = k_wall * (inv_r12 - inv_rAB12);
    }
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

#endif  // FLEXAIDS_HAS_AVX512 / FLEXAIDS_HAS_AVX2

// ─── dispatch helper: compile-time dispatch to best available ────────────────

// RMSD between two coordinate arrays (N atoms, interleaved xyz)
inline float rmsd(const float* a, const float* b, int N) noexcept {
    return std::sqrt(sum_sq_distances(a, b, N) / static_cast<float>(N));
}

}  // namespace simd
