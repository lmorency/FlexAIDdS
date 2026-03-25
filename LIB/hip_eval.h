// Copyright 2026 Le Bonhomme Pharma
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// ============================================================================
// hip_eval.h
//
// Public API for the FlexAIDS ROCm/HIP chromosome-evaluation backend.
//
// Mirror of cuda_eval.cuh; all symbols live in the `hip_eval` namespace.
// Include this header wherever GPU-accelerated population evaluation is
// dispatched (typically from the HardwareDispatch evaluation router).
//
// Compilation guard: the entire header and its implementation (hip_eval.hip)
// are compiled only when FLEXAIDS_USE_ROCM is defined by the build system.
// ============================================================================

#pragma once

#ifdef FLEXAIDS_USE_ROCM

#include <hip/hip_runtime.h>

#include <cstddef>
#include <string>

// Forward-include only the minimal FlexAIDS types needed here.
// Adjust the relative path if the project layout differs.
#include "../LIB/flexaid.h"

namespace hip_eval {

// ============================================================================
// DeviceInfo
// ============================================================================

/// Snapshot of relevant AMD GPU properties for the evaluation backend.
struct DeviceInfo {
    std::string name;            ///< Human-readable device name (prop.name)
    int         compute_units;   ///< Streaming multiprocessor / CU count
    std::size_t global_mem;      ///< Total global memory in bytes
    int         gcn_arch;        ///< Numeric GCN arch code, e.g. 908 (gfx908)
    int         wavefront_size;  ///< 64 on GCN/CDNA; may be 32 on RDNA wave32 mode
    int         max_threads_per_cu; ///< Max threads per CU (from prop.maxThreadsPerMultiProcessor)
    int         shared_mem_per_cu;  ///< Shared (LDS) bytes per CU
};

/// Query device properties without initialising the evaluation backend.
/// Safe to call before init().
DeviceInfo get_device_info(int device_id = 0);

// ============================================================================
// Lifecycle
// ============================================================================

/// Initialise the HIP evaluation backend on the specified device.
///
/// Must be called once before any batch_evaluate() or upload_emat() calls.
/// Allocates persistent device buffers sized for the maximum population.
///
/// @param device_id  HIP device index (0 = first AMD GPU).
/// @param max_pop    Maximum population size to pre-allocate for.  Passing 0
///                   uses a built-in default (FLEXAIDS_HIP_DEFAULT_MAX_POP).
void init(int device_id = 0, int max_pop = 0);

/// Release all device memory and HIP resources.
/// After shutdown(), init() must be called again before any evaluation.
void shutdown();

// ============================================================================
// Energy-matrix upload
// ============================================================================

/// Upload a pre-sampled energy matrix to device constant / global memory.
///
/// @param emat_sampled  Row-major float array of shape [n_types × n_samples].
/// @param n_types       Number of atom/residue types.
/// @param n_samples     Number of distance samples per type pair.
///
/// This is called once per receptor-ligand pair setup and is idempotent.
/// Subsequent calls replace the previously uploaded matrix.
void upload_emat(const float* emat_sampled, int n_types, int n_samples);

// ============================================================================
// Batch chromosome evaluation
// ============================================================================

/// Evaluate a full population of chromosomes on the AMD GPU.
///
/// Mirrors cuda_eval::batch_evaluate exactly so the HardwareDispatch router
/// can call both backends through a uniform interface.
///
/// @param h_genes        Host pointer: row-major [pop_size × num_genes] doubles.
///                       Each row is one chromosome's gene vector.
/// @param h_com          Host output: complementarity score per chromosome [pop_size].
/// @param h_wal          Host output: wall-clash penalty per chromosome [pop_size].
/// @param h_sas          Host output: solvent-accessible-surface term [pop_size].
/// @param pop_size       Number of chromosomes (population size).
/// @param num_genes      Length of each chromosome's gene vector.
/// @param emat_sampled   Energy matrix (may be nullptr if upload_emat was called).
///                       If non-null, triggers an implicit upload_emat() call.
/// @param n_types        Number of atom/residue types in the energy matrix.
/// @param n_samples      Distance samples per type pair in the energy matrix.
///
/// The function is synchronous from the host's perspective: it returns only
/// after all results have been copied back to h_com, h_wal, and h_sas.
void batch_evaluate(
    const double* h_genes,
    double*       h_com,
    double*       h_wal,
    double*       h_sas,
    int           pop_size,
    int           num_genes,
    const float*  emat_sampled,
    int           n_types,
    int           n_samples
);

// ============================================================================
// Low-level kernel launchers (exposed for testing / profiling)
// ============================================================================

/// Launch the complementarity-function scoring kernel directly.
/// Requires device buffers to have been set up by a preceding init() call.
///
/// @param d_genes     Device pointer: [pop_size × num_genes] float32.
/// @param d_com       Device output: complementarity [pop_size].
/// @param d_wal       Device output: wall penalty [pop_size].
/// @param d_sas       Device output: SAS term [pop_size].
/// @param pop_size    Population size.
/// @param num_genes   Genes per chromosome.
/// @param stream      HIP stream (hipStreamDefault = 0 for the default stream).
void launch_cf_score_kernel(
    const float* d_genes,
    float*       d_com,
    float*       d_wal,
    float*       d_sas,
    int          pop_size,
    int          num_genes,
    hipStream_t  stream = hipStreamDefault
);

/// Launch the gene-packing kernel (double → packed float on-device).
///
/// @param d_genes_f64  Device pointer: unpacked double genes [pop_size × num_genes].
/// @param d_genes_f32  Device output: packed float genes [pop_size × num_genes].
/// @param total        Total elements (pop_size × num_genes).
/// @param stream       HIP stream.
void launch_pack_genes_kernel(
    const double* d_genes_f64,
    float*        d_genes_f32,
    int           total,
    hipStream_t   stream = hipStreamDefault
);

}  // namespace hip_eval

#endif  // FLEXAIDS_USE_ROCM
