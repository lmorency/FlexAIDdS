// tencm_cpu_fallback.h — CPU fallback for TENCoM contact discovery & Hessian assembly
//
// Provides the same API as tencm_cuda.cuh / tencm_metal.h but runs entirely
// on the CPU using Eigen + OpenMP.  Used when no GPU backend is available.
//
// Apache-2.0 © 2026 Le Bonhomme Pharma
#pragma once

#include <cstddef>

namespace tencm { namespace cpu_fallback {

// CPU-accelerated contact discovery.
//   ca_xyz:         [N x 3] Cα coordinates (row-major, float)
//   cutoff:         distance cutoff (Å)
//   k0:             base spring constant
//   contacts_ij:    output pairs (i,j)  [caller pre-allocates upper-bound]
//   contacts_k:     output spring constants per pair
//   contacts_r0:    output equilibrium distances
//   Returns: number of contacts found, or -1 on error
int build_contacts_cpu(const float* ca_xyz, int N,
                       float cutoff, float k0,
                       int*   contacts_ij,
                       float* contacts_k,
                       float* contacts_r0);

// CPU-accelerated Hessian assembly with Eigen + OpenMP.
//   ca_xyz:      [N x 3] Cα coordinates
//   contacts_ij: [C x 2] contact pairs (i,j)
//   contacts_k:  [C] spring constants
//   M:           number of bonds (N-1)
//   C:           number of contacts
//   H_out:       [M x M] output Hessian (row-major double, zeroed by caller)
void assemble_hessian_cpu(const float* ca_xyz, int N,
                          const int*   contacts_ij,
                          const float* contacts_k,
                          int M, int C,
                          double* H_out);

}}  // namespace tencm::cpu_fallback
