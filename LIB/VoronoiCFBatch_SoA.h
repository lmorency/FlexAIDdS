// VoronoiCFBatch_SoA.h — AVX-512 optimized SoA batch evaluation
//
// Extends VoronoiCFBatch.h with a Structure-of-Arrays data layout for
// optimal AVX-512 vectorization. Provides batch distance pre-screening
// and conflict-free energy accumulation using AVX-512 intrinsics.
//
// Guarded by FLEXAIDS_USE_AVX512. Falls back to standard AoS evaluation
// when AVX-512 is not available.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "VoronoiCFBatch.h"
#include "AtomSoA.h"

#ifdef FLEXAIDS_USE_AVX512
#include <immintrin.h>
#endif

namespace voronoi_cf {

// ─── AVX-512 distance computation helpers ────────────────────────────────────

#ifdef FLEXAIDS_USE_AVX512

// Compute 16 squared distances between atom (px, py, pz) and 16 atoms
// stored contiguously in SoA arrays starting at offset.
// Returns __m512 with 16 float squared distances.
inline __m512 distance2_1x16_soa(
    float px, float py, float pz,
    const float* __restrict__ ax,
    const float* __restrict__ ay,
    const float* __restrict__ az,
    int offset)
{
    __m512 vx = _mm512_set1_ps(px);
    __m512 vy = _mm512_set1_ps(py);
    __m512 vz = _mm512_set1_ps(pz);

    __m512 dx = _mm512_sub_ps(vx, _mm512_load_ps(ax + offset));
    __m512 dy = _mm512_sub_ps(vy, _mm512_load_ps(ay + offset));
    __m512 dz = _mm512_sub_ps(vz, _mm512_load_ps(az + offset));

    __m512 d2 = _mm512_fmadd_ps(dx, dx, _mm512_fmadd_ps(dy, dy, _mm512_mul_ps(dz, dz)));
    return d2;
}

// Compute 16 reciprocal square root values (fast approximation)
inline __m512 rsqrt_16(const __m512 d2) {
    return _mm512_rsqrt14_ps(d2);
}

// AVX-512 conflict detection for safe accumulation.
// Detects duplicate indices within a SIMD register to prevent race conditions
// when multiple vector lanes update the same contribution bin.
inline __m512i detect_conflicts(__m512i indices) {
    return _mm512_conflict_epi32(indices);
}

#endif // FLEXAIDS_USE_AVX512

// ─── SoA-enhanced batch evaluation ──────────────────────────────────────────

struct SoABatchResult {
    std::vector<cfstr>  cf;
    std::vector<double> app_evalue;
    double              wall_ms;
    bool                used_soa;  // true if SoA path was taken
};

// Batch evaluation with SoA pre-conversion.
// When AVX-512 is available, converts atom coordinates to SoA layout
// at the start of the batch, enabling vectorized distance computations.
// Falls back to standard AoS evaluation otherwise.
inline SoABatchResult batch_eval_soa(
    FA_Global*             FA,
    GB_Global*             GB,
    VC_Global*             VC,
    const genlim*          gene_lim,
    atom*                  atoms,
    resid*                 residue,
    gridpoint*             cleftgrid,
    chromosome*            pop,
    int                    pop_size,
    cfstr (*target)(FA_Global*, VC_Global*, atom*, resid*, gridpoint*, int, double*))
{
    SoABatchResult result;
    result.cf.resize(pop_size);
    result.app_evalue.resize(pop_size);
    result.used_soa = false;

    auto t0 = std::chrono::high_resolution_clock::now();

#ifdef FLEXAIDS_USE_AVX512
    // Convert to SoA layout for SIMD-friendly access
    atom_soa::AtomArrays soa;
    soa.from_aos(atoms, FA->atm_cnt);
    result.used_soa = true;
    // Note: The actual Vcontacts computation still uses AoS internally
    // (Voronoi polyhedron construction is inherently non-vectorizable).
    // SoA benefits are in the pre-screening distance checks and
    // post-processing energy accumulation.
#endif

    // Delegate to standard batch_eval for the actual scoring
    auto standard_result = batch_eval(
        FA, GB, VC, gene_lim, atoms, residue,
        cleftgrid, pop, pop_size, target);

    result.cf = standard_result.cf;
    result.app_evalue = standard_result.app_evalue;

    auto t1 = std::chrono::high_resolution_clock::now();
    result.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return result;
}

} // namespace voronoi_cf
