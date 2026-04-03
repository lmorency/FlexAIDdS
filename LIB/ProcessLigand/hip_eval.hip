// hip_eval.hip — ROCm/HIP GPU kernels for FlexAIDdS energy evaluation
//
// Copyright 2026 Le Bonhomme Pharma
// SPDX-License-Identifier: Apache-2.0
//
// Source-compatible with CUDA: compile with hipcc or nvcc.
//   hipcc -O3 --amdgpu-target=gfx1100 -c hip_eval.hip -o hip_eval.o
//   nvcc  -O3 -arch=sm_80              -c hip_eval.hip -o hip_eval.o  (rename to .cu)
//
// Kernels:
//   hip_eval_kernel — per-pose energy evaluation (LJ 6-12 + electrostatics)
//   hip_eval_launch — C launcher called from ROCmDispatch.cpp
//
// Data layout:
//   d_coords   : float[n_atoms * n_poses * 3]  — all poses packed, atoms first
//   d_types    : uint8_t[n_atoms]               — atom type (shared across poses)
//   d_energies : float[n_poses]                 — output, one energy per pose
//
// Each GPU thread handles one pose. Atom-pair interactions are iterated in a
// sequential inner loop. This is the simplest correct implementation; a future
// optimisation pass can add shared-memory tiling.

#ifdef __HIP_PLATFORM_HCC__
#  include <hip/hip_runtime.h>
#else
// Compile as CUDA if hipcc not detected
#  include <cuda_runtime.h>
// HIP → CUDA symbol aliases
#  define hipLaunchKernelGGL(f, g, b, s, stream, ...) \
       f<<<g, b, s, stream>>>(__VA_ARGS__)
#  define __device__ __device__
#  define __global__ __global__
#  define __shared__ __shared__
#  define hipDeviceSynchronize cudaDeviceSynchronize
#endif

#include <cstdint>
#include <cfloat>
#include <cmath>

// ---------------------------------------------------------------------------
// Type descriptor (mirrors FlexAID 40-type system)
// We only need vdW radii and ε for the LJ potential.
// Values sourced from FlexAID's NRG parameter tables.
// ---------------------------------------------------------------------------

// LJ well depth ε (kcal/mol) indexed by SYBYL FlexAID type (0-30)
__device__ __constant__ float d_lj_eps[31] = {
    0.086f,  // 0  C.1
    0.086f,  // 1  C.3
    0.086f,  // 2  C.2
    0.086f,  // 3  C.ar
    0.170f,  // 4  N.3
    0.170f,  // 5  N.2
    0.170f,  // 6  N.ar
    0.170f,  // 7  N.am
    0.170f,  // 8  N.pl3
    0.170f,  // 9  N.4
    0.210f,  // 10 O.3
    0.210f,  // 11 O.2
    0.210f,  // 12 O.co2
    0.061f,  // 13 F
    0.265f,  // 14 Cl
    0.320f,  // 15 Br
    0.200f,  // 16 S.3
    0.200f,  // 17 S.2
    0.200f,  // 18 S.O
    0.200f,  // 19 S.O2
    0.200f,  // 20 P.3
    0.400f,  // 21 I
    0.020f,  // 22 H
    0.000f,  // 23-29 unused
    0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f,
    0.010f,  // 30 Fe
};

// vdW radius R_min/2 (Å) indexed by SYBYL FlexAID type
__device__ __constant__ float d_lj_rmin2[31] = {
    1.900f,  // 0  C.1
    2.000f,  // 1  C.3
    2.000f,  // 2  C.2
    1.850f,  // 3  C.ar
    1.750f,  // 4  N.3
    1.750f,  // 5  N.2
    1.750f,  // 6  N.ar
    1.750f,  // 7  N.am
    1.750f,  // 8  N.pl3
    1.750f,  // 9  N.4
    1.600f,  // 10 O.3
    1.600f,  // 11 O.2
    1.600f,  // 12 O.co2
    1.470f,  // 13 F
    1.770f,  // 14 Cl
    1.850f,  // 15 Br
    1.900f,  // 16 S.3
    1.900f,  // 17 S.2
    1.900f,  // 18 S.O
    1.900f,  // 19 S.O2
    2.100f,  // 20 P.3
    2.100f,  // 21 I
    1.200f,  // 22 H
    0.000f,  // 23-29 unused
    0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f,
    1.300f,  // 30 Fe
};

// Coulomb constant k_e / (4πε₀) in kcal·Å/(mol·e²) with ε_r = 4.0
// k = 332.0636 / 4.0 = 83.015
__device__ static constexpr float COULOMB_CONST = 83.015f;

// LJ cutoff
__device__ static constexpr float LJ_CUTOFF = 10.0f;
__device__ static constexpr float LJ_CUTOFF2 = LJ_CUTOFF * LJ_CUTOFF;

// ---------------------------------------------------------------------------
// LJ 12-6 energy between two atoms
// E_LJ = ε[(Rmin/r)^12 - 2(Rmin/r)^6]
// ---------------------------------------------------------------------------

__device__ inline float lj_energy(float r2, float eps, float rmin) {
    if (r2 > LJ_CUTOFF2) return 0.0f;
    if (r2 < 0.25f) r2 = 0.25f; // clash guard (0.5 Å min distance)
    float r6  = (rmin * rmin * rmin * rmin * rmin * rmin) / (r2 * r2 * r2);
    return eps * (r6 * r6 - 2.0f * r6);
}

// ---------------------------------------------------------------------------
// Per-pose energy evaluation kernel
// Thread index = pose index. Each thread iterates all atom pairs.
// ---------------------------------------------------------------------------

__global__ void hip_eval_kernel(const float*   d_coords,   // [n_atoms*n_poses*3]
                                 const uint8_t* d_types,    // [n_atoms]
                                 const float*   d_charges,  // [n_atoms] partial charges
                                 float*         d_energies, // [n_poses] output
                                 int            n_poses,
                                 int            n_atoms) {
    int pose = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
             + static_cast<int>(threadIdx.x);
    if (pose >= n_poses) return;

    // Coordinate base for this pose
    const float* coords = d_coords + pose * n_atoms * 3;

    float energy = 0.0f;

    for (int i = 0; i < n_atoms - 1; ++i) {
        float xi = coords[i * 3 + 0];
        float yi = coords[i * 3 + 1];
        float zi = coords[i * 3 + 2];
        uint8_t ti = d_types[i] & 0x3Fu; // base type (bits 0-5)
        float   qi = d_charges[i];
        float   ei = d_lj_eps[ti < 31 ? ti : 0];
        float   ri = d_lj_rmin2[ti < 31 ? ti : 0];

        for (int j = i + 1; j < n_atoms; ++j) {
            float dx = xi - coords[j * 3 + 0];
            float dy = yi - coords[j * 3 + 1];
            float dz = zi - coords[j * 3 + 2];
            float r2 = dx*dx + dy*dy + dz*dz;

            uint8_t tj = d_types[j] & 0x3Fu;
            float   qj = d_charges[j];
            float   ej = d_lj_eps[tj < 31 ? tj : 0];
            float   rj = d_lj_rmin2[tj < 31 ? tj : 0];

            // Combined LJ parameters (Lorentz-Berthelot mixing)
            float eps_ij  = sqrtf(ei * ej);
            float rmin_ij = ri + rj; // R_min = R_min/2_i + R_min/2_j

            energy += lj_energy(r2, eps_ij, rmin_ij);

            // Coulomb
            if (r2 < LJ_CUTOFF2 && r2 > 0.0f) {
                float r_inv = rsqrtf(r2);
                energy += COULOMB_CONST * qi * qj * r_inv;
            }
        }
    }

    d_energies[pose] = energy;
}

// ---------------------------------------------------------------------------
// C launcher (called from ROCmDispatch.cpp)
// ---------------------------------------------------------------------------

extern "C"
void hip_eval_launch(const float*   d_coords,
                     const uint8_t* d_types,
                     float*         d_energies,
                     int            n_poses,
                     int            n_atoms,
                     int            block_size) {
    // Allocate and upload a zero charge array if not provided by the caller.
    // The full API passes charges separately; for this stub we use zeroed charges.
    // In production, the caller should pass d_charges directly.
    float* d_charges = nullptr;

#ifdef __HIP_PLATFORM_HCC__
    hipMalloc(&d_charges, n_atoms * sizeof(float));
    hipMemset(d_charges, 0, n_atoms * sizeof(float));
#else
    cudaMalloc(&d_charges, n_atoms * sizeof(float));
    cudaMemset(d_charges, 0, n_atoms * sizeof(float));
#endif

    int grid_size = (n_poses + block_size - 1) / block_size;

    hipLaunchKernelGGL(hip_eval_kernel,
                       dim3(grid_size), dim3(block_size), 0, 0,
                       d_coords, d_types, d_charges, d_energies,
                       n_poses, n_atoms);

#ifdef __HIP_PLATFORM_HCC__
    hipFree(d_charges);
#else
    cudaFree(d_charges);
#endif
}

// ---------------------------------------------------------------------------
// Extended launcher with explicit charge array (preferred production API)
// ---------------------------------------------------------------------------

extern "C"
void hip_eval_launch_with_charges(const float*   d_coords,
                                   const uint8_t* d_types,
                                   const float*   d_charges,
                                   float*         d_energies,
                                   int            n_poses,
                                   int            n_atoms,
                                   int            block_size) {
    int grid_size = (n_poses + block_size - 1) / block_size;
    hipLaunchKernelGGL(hip_eval_kernel,
                       dim3(grid_size), dim3(block_size), 0, 0,
                       d_coords, d_types, d_charges, d_energies,
                       n_poses, n_atoms);
}
