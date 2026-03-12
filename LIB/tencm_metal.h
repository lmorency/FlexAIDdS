// tencm_metal.h — C++ interface to Apple Metal TENCoM GPU kernels
//
// Provides GPU-accelerated contact discovery and Hessian assembly for
// TENCoM on Apple Silicon / Metal-capable Macs.
//
// Implementation is in tencm_metal.mm (Objective-C++); only compiled on
// APPLE targets with -DFLEXAIDS_USE_METAL=ON.
// Callers guard behind #ifdef FLEXAIDS_HAS_METAL_TENCM.
#pragma once

#ifdef FLEXAIDS_HAS_METAL_TENCM

#include <vector>

namespace tencm { namespace metal {

// Minimum residue count to justify GPU overhead
inline constexpr int GPU_THRESHOLD = 256;

// One-time Metal device + pipeline setup.
// Returns true if a Metal-capable GPU is available.
bool init();
void shutdown();
bool is_available();

// GPU-accelerated contact discovery.
//   ca_xyz: [N x 3] Cα coordinates (row-major, float)
//   Returns: number of contacts; populates output vectors.
int build_contacts_gpu(const float* ca_xyz, int N,
                       float cutoff, float k0,
                       std::vector<int>&   contacts_ij,
                       std::vector<float>& contacts_k,
                       std::vector<float>& contacts_r0);

// GPU-accelerated Hessian assembly.
//   ca_xyz:      [N x 3] Cα coordinates
//   contacts_ij: [C x 2] contact pairs
//   contacts_k:  [C] spring constants
//   M:           number of bonds (N-1)
//   C:           number of contacts
//   H_out:       [M x M] output Hessian (row-major double)
void assemble_hessian_gpu(const float* ca_xyz, int N,
                          const int* contacts_ij,
                          const float* contacts_k,
                          int M, int C,
                          double* H_out);

}}  // namespace tencm::metal

#endif  // FLEXAIDS_HAS_METAL_TENCM
