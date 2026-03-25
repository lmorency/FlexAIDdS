// tencm_cuda.cuh — CUDA interface for TENCoM Hessian assembly
//
// GPU-accelerated contact discovery and Hessian assembly for large proteins.
// Falls back to CPU when N_residues < GPU_THRESHOLD.
//
// Build requirement: CUDA toolkit, -DFLEXAIDS_USE_CUDA=ON
#pragma once

#ifdef FLEXAIDS_USE_CUDA

#include <cstddef>

namespace tencm { namespace cuda {

// Minimum residue count to justify GPU overhead (PCIe transfer cost)
inline constexpr int GPU_THRESHOLD = 256;

// One-time device context initialisation.
// Returns true if a CUDA-capable device is available.
bool init();
void shutdown();
bool is_available();

// GPU-accelerated contact discovery.
//   ca_xyz: [N x 3] Cα coordinates (row-major, host memory)
//   contacts_ij: output pairs (i,j) with r² ≤ rc²  (caller pre-allocates upper-bound N*(N-1)/2)
//   contacts_k:  output spring constants per pair
//   contacts_r0: output equilibrium distances
//   Returns: number of contacts found
int build_contacts_gpu(const float* ca_xyz, int N,
                       float cutoff, float k0,
                       int* contacts_ij,     // [max_contacts x 2]
                       float* contacts_k,
                       float* contacts_r0);

// GPU-accelerated Hessian assembly.
//   ca_xyz:      [N x 3] Cα coordinates
//   contacts_ij: [C x 2] contact pairs (i,j)
//   contacts_k:  [C] spring constants
//   M:           number of bonds (N-1)
//   C:           number of contacts
//   H_out:       [M x M] output Hessian (host, row-major double)
void assemble_hessian_gpu(const float* ca_xyz, int N,
                          const int* contacts_ij,
                          const float* contacts_k,
                          int M, int C,
                          double* H_out);

}}  // namespace tencm::cuda

#endif  // FLEXAIDS_USE_CUDA
