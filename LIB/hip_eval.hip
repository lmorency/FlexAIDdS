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
// hip_eval.hip
//
// AMD ROCm/HIP chromosome-evaluation kernels for FlexAIDS.
//
// Compile with:
//   hipcc -std=c++20 -O3 --offload-arch=gfx908,gfx90a,gfx942 \
//         -DFLEXAIDS_USE_ROCM hip_eval.hip -o hip_eval.o
//
// Or via CMake with the patch in CMakeLists_rocm_patch.cmake.
//
// ── Architecture notes ──────────────────────────────────────────────────────
//  • AMD GCN/CDNA wavefront = 64 threads (vs. 32 on NVIDIA).
//  • All warp-width parameters use the runtime variable `warpSize` so that
//    this file compiles correctly on both AMD (warpSize=64) and NVIDIA/HIP
//    (warpSize=32) targets.  Never hardcode 32 or 64.
//  • LDS (local data store) == __shared__ memory; usage and sync are identical.
//  • __syncthreads() / __syncwarp() semantics are preserved by HIP.
//  • hipLaunchKernelGGL is the portable macro; <<<>>> syntax also works.
//  • For shuffle reductions on AMD, use __shfl_down() (no sync mask needed
//    pre-gfx10) or the HIP-provided __shfl_down_sync() where available.
// ============================================================================

#ifdef FLEXAIDS_USE_ROCM

#include "hip_eval.h"

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Build-time tunables
// ---------------------------------------------------------------------------

/// Default threads per block.  Must be a multiple of the wavefront size.
/// 256 works for both wavefront-64 (AMD) and wavefront-32 (NVIDIA/HIP).
#ifndef FLEXAIDS_HIP_BLOCK_SIZE
#   define FLEXAIDS_HIP_BLOCK_SIZE 256
#endif

/// Pre-allocated maximum population size.  Larger values are supported by
/// re-allocating at runtime.
#ifndef FLEXAIDS_HIP_DEFAULT_MAX_POP
#   define FLEXAIDS_HIP_DEFAULT_MAX_POP 4096
#endif

/// Maximum energy-matrix flat size in constant memory (floats).
/// 128 types × 1024 samples = 131,072 floats = 512 KiB.
#ifndef FLEXAIDS_HIP_EMAT_MAX
#   define FLEXAIDS_HIP_EMAT_MAX (128 * 1024)
#endif

// ---------------------------------------------------------------------------
// Error-checking macro
// ---------------------------------------------------------------------------

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t _e = (call);                                                \
        if (_e != hipSuccess) {                                                \
            char _msg[256];                                                    \
            std::snprintf(_msg, sizeof(_msg),                                  \
                "[FlexAIDS hip_eval] %s:%d  %s  → %s",                       \
                __FILE__, __LINE__, #call, hipGetErrorString(_e));             \
            throw std::runtime_error(_msg);                                    \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Constant memory – energy matrix
// ---------------------------------------------------------------------------

/// The energy matrix lives in constant memory for read-only, broadcast access.
/// On AMD CDNA, constant memory is cached in the scalar data cache.
__constant__ float c_emat[FLEXAIDS_HIP_EMAT_MAX];

/// Metadata for the constant-memory energy matrix.
__constant__ int   c_n_types;    ///< Number of atom/residue types
__constant__ int   c_n_samples;  ///< Distance samples per type pair

// ---------------------------------------------------------------------------
// Device helper: energy-matrix lookup
// ---------------------------------------------------------------------------

/// Bilinear (in distance index space) lookup into the constant energy matrix.
/// @param type_a  Atom type index for the first atom  [0, n_types)
/// @param type_b  Atom type index for the second atom [0, n_types)
/// @param dist_idx Pre-computed integer distance sample index
/// @returns       Interaction energy (float)
__device__ __forceinline__
float emat_lookup(int type_a, int type_b, int dist_idx) {
    // Symmetric: use upper triangle.
    if (type_a > type_b) { int tmp = type_a; type_a = type_b; type_b = tmp; }
    int row = type_a * c_n_types + type_b;
    int idx = row * c_n_samples + dist_idx;
    return c_emat[idx];
}

// ---------------------------------------------------------------------------
// Device helper: wavefront-width reduction (warp reduce sum)
// ---------------------------------------------------------------------------

/// Full-wavefront (or warp) reduction using shuffle-down.
/// Works correctly for both wavefront=64 (AMD) and warpSize=32 (NVIDIA).
__device__ __forceinline__
float warp_reduce_sum(float val) {
    // Unroll for all possible offsets up to warpSize/2.
    // The compiler will evaluate warpSize at compile time for the target.
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
#if defined(__HIP_PLATFORM_AMD__)
        val += __shfl_down(val, offset);
#else
        // NVIDIA / HIP-on-NVIDIA path: use the sync-mask variant.
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
#endif
    }
    return val;
}

/// Block-wide reduction using shared memory + warp_reduce_sum.
/// All threads must call this; result is valid only in thread 0.
__device__ __forceinline__
float block_reduce_sum(float val, float* smem) {
    int lane    = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    val = warp_reduce_sum(val);

    if (lane == 0) smem[warp_id] = val;
    __syncthreads();

    // Only the first warp finishes the reduction.
    int n_warps = blockDim.x / warpSize;
    val = (threadIdx.x < n_warps) ? smem[threadIdx.x] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum(val);

    return val;
}

// ---------------------------------------------------------------------------
// Kernel: gene packing  (double → float, in-place type conversion)
// ---------------------------------------------------------------------------

/// Convert a flat array of double-precision genes to single precision on the
/// device.  Each thread handles one element.
///
/// The FlexAIDS GA stores chromosomes as doubles on the host but the GPU
/// kernels operate in float32 for throughput; this kernel amortises the
/// conversion cost by doing it once on-device rather than on the host.
///
/// @param d_in   Input:  [total] double values (pop_size × num_genes)
/// @param d_out  Output: [total] float values
/// @param total  Total number of elements
__global__
void hip_pack_genes_kernel(const double* __restrict__ d_in,
                           float*        __restrict__ d_out,
                           int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        d_out[idx] = static_cast<float>(d_in[idx]);
    }
}

// ---------------------------------------------------------------------------
// Kernel: complementarity-function (CF) score
// ---------------------------------------------------------------------------

/// Compute ic2cf-style complementarity and penalty scores for one population.
///
/// Grid layout:
///   gridDim.x  = pop_size   (one block per chromosome)
///   blockDim.x = FLEXAIDS_HIP_BLOCK_SIZE
///
/// Each block evaluates one chromosome and writes three floats to the output:
///   d_com[blockIdx.x]  – complementarity score
///   d_wal[blockIdx.x]  – wall-clash penalty
///   d_sas[blockIdx.x]  – solvent-accessible-surface term
///
/// @param d_genes_f32  Packed float genes [pop_size × num_genes]
/// @param d_com        Output: COM score [pop_size]
/// @param d_wal        Output: wall penalty [pop_size]
/// @param d_sas        Output: SAS term [pop_size]
/// @param pop_size     Number of chromosomes
/// @param num_genes    Genes per chromosome
__global__
void hip_cf_score_kernel(const float* __restrict__ d_genes_f32,
                         float*       __restrict__ d_com,
                         float*       __restrict__ d_wal,
                         float*       __restrict__ d_sas,
                         int pop_size,
                         int num_genes)
{
    // ── Shared memory layout ────────────────────────────────────────────────
    // We need one float per warp for the block reduction, plus space for the
    // gene vector of this chromosome.
    extern __shared__ float smem[];
    // smem[0 .. num_genes-1]              : gene vector for this chromosome
    // smem[num_genes .. num_genes+nw-1]   : warp-reduction scratch (nw warps)

    int chrom = blockIdx.x;
    if (chrom >= pop_size) return;

    const int tid     = threadIdx.x;
    const int n_warps = blockDim.x / warpSize;

    float* s_genes = smem;                    // [num_genes]
    float* s_warp  = smem + num_genes;        // [n_warps]  — reduction scratch

    // ── Load gene vector into shared memory ─────────────────────────────────
    const float* gene_row = d_genes_f32 + chrom * num_genes;
    for (int g = tid; g < num_genes; g += blockDim.x) {
        s_genes[g] = gene_row[g];
    }
    __syncthreads();

    // ── Complementarity accumulation ────────────────────────────────────────
    // Each thread accumulates a partial sum over its assigned gene-pair
    // interactions.  The outer loop over pairs is distributed across threads.
    //
    // In the real FlexAIDS kernel this is a double nested loop over atom
    // contacts derived from the gene encoding (translational DOFs → receptor-
    // ligand atom distances → energy table lookup).  Here we implement the
    // structural skeleton; the inner physics is expressed as the emat_lookup
    // call pattern that mirrors the CUDA version.

    float local_com = 0.0f;
    float local_wal = 0.0f;
    float local_sas = 0.0f;

    // Decode rigid-body translational genes (first 3 DOFs): Tx, Ty, Tz.
    float tx = (num_genes > 0) ? s_genes[0] : 0.0f;
    float ty = (num_genes > 1) ? s_genes[1] : 0.0f;
    float tz = (num_genes > 2) ? s_genes[2] : 0.0f;

    // Decode rotational gene (quaternion encoded as angle + axis, genes 3..6).
    // The exact decoding mirrors ic2cf in the CPU backend.
    float rot_angle = (num_genes > 3) ? s_genes[3] : 0.0f;

    // Remaining genes are torsional DOFs (one per flexible bond).
    // Iterate over ligand atoms × receptor atoms per thread stripe.
    // n_contacts estimation: (num_genes - 7) contacts × n_types interactions.
    int n_flex = (num_genes > 7) ? (num_genes - 7) : 0;

    for (int contact = tid; contact < (n_flex * c_n_types); contact += blockDim.x) {
        int flex_idx = contact / c_n_types;
        int type_b   = contact % c_n_types;

        // Gene value encodes torsional displacement; map to sample index.
        float gene_val = (7 + flex_idx < num_genes) ? s_genes[7 + flex_idx] : 0.0f;

        // Translate gene value [−π, π] → distance sample index [0, n_samples).
        float norm_val  = (gene_val + 3.14159265f) / (2.0f * 3.14159265f);
        int   dist_idx  = __float2int_rn(norm_val * static_cast<float>(c_n_samples - 1));
        dist_idx = max(0, min(dist_idx, c_n_samples - 1));

        // Evaluate pairwise energy using a representative type for atom A
        // (in practice, atom type is decoded from the gene bit-packing; here
        // we use flex_idx % n_types as a structurally equivalent placeholder).
        int type_a = flex_idx % c_n_types;
        float energy = emat_lookup(type_a, type_b, dist_idx);

        // Partition energy into complementarity, wall, and SAS components.
        // Convention mirrors cuda_eval.cu:
        //   energy < -0.5   → favourable contact → COM
        //   energy > +1.0   → steric clash       → WAL
        //   otherwise       → surface term        → SAS
        if (energy < -0.5f) {
            local_com += energy;          // negative = attractive
        } else if (energy > 1.0f) {
            local_wal += energy;          // positive = clash
        } else {
            local_sas += energy;
        }
    }

    // Add a rigid-body translation penalty term (keeps ligand in binding site).
    // Each thread computes its contribution to the site-distance penalty.
    if (tid == 0) {
        float dist2 = tx * tx + ty * ty + tz * tz;
        float site_penalty = 0.01f * dist2;
        local_wal += site_penalty;

        // Rotational entropy term (small penalty proportional to rotation angle).
        local_sas += 0.005f * rot_angle * rot_angle;
    }

    // ── Block reduction for each score component ────────────────────────────
    // Reuse s_warp for successive reductions (serialised with __syncthreads).

    float com_total = block_reduce_sum(local_com, s_warp);
    __syncthreads();
    float wal_total = block_reduce_sum(local_wal, s_warp);
    __syncthreads();
    float sas_total = block_reduce_sum(local_sas, s_warp);

    // ── Write results ───────────────────────────────────────────────────────
    if (tid == 0) {
        d_com[chrom] = com_total;
        d_wal[chrom] = wal_total;
        d_sas[chrom] = sas_total;
    }
}

// ---------------------------------------------------------------------------
// Kernel: batch chromosome evaluation (full pipeline)
// ---------------------------------------------------------------------------

/// End-to-end evaluation kernel that combines gene decoding and CF scoring
/// in a single pass, avoiding intermediate memory round-trips.
///
/// Grid layout:
///   gridDim.x  = ceil(pop_size / FLEXAIDS_HIP_BLOCK_SIZE)
///   blockDim.x = FLEXAIDS_HIP_BLOCK_SIZE
///
/// Each thread handles one chromosome independently (no shared-memory
/// reduction needed; results written directly).  Use this kernel when
/// pop_size >> num_genes (wide populations, short chromosomes).
///
/// @param d_genes_f32  [pop_size × num_genes] float genes (device)
/// @param d_com        [pop_size] output complementarity (device)
/// @param d_wal        [pop_size] output wall penalty (device)
/// @param d_sas        [pop_size] output SAS term (device)
/// @param pop_size     Population size
/// @param num_genes    Genes per chromosome
__global__
void hip_batch_eval_kernel(const float* __restrict__ d_genes_f32,
                           float*       __restrict__ d_com,
                           float*       __restrict__ d_wal,
                           float*       __restrict__ d_sas,
                           int pop_size,
                           int num_genes)
{
    int chrom = blockIdx.x * blockDim.x + threadIdx.x;
    if (chrom >= pop_size) return;

    const float* g = d_genes_f32 + chrom * num_genes;

    // ── Decode rigid-body DOFs ──────────────────────────────────────────────
    float tx        = (num_genes > 0) ? g[0] : 0.0f;
    float ty        = (num_genes > 1) ? g[1] : 0.0f;
    float tz        = (num_genes > 2) ? g[2] : 0.0f;
    float rot_angle = (num_genes > 3) ? g[3] : 0.0f;
    // Rotation axis (genes 4,5,6) omitted from energy calc (affects geometry
    // setup in a full implementation; atom positions precomputed externally).

    // ── Complementarity / contact scoring loop ─────────────────────────────
    float com = 0.0f;
    float wal = 0.0f;
    float sas = 0.0f;

    int n_flex = (num_genes > 7) ? (num_genes - 7) : 0;

    for (int fi = 0; fi < n_flex; ++fi) {
        float gene_val = g[7 + fi];

        // Map gene → distance sample index.
        float norm_val = (gene_val + 3.14159265f) / (2.0f * 3.14159265f);
        int   dist_idx = __float2int_rn(norm_val * static_cast<float>(c_n_samples - 1));
        dist_idx = max(0, min(dist_idx, c_n_samples - 1));

        // Sum over type interactions for this flexible DOF.
        for (int t = 0; t < c_n_types; ++t) {
            int type_a = fi % c_n_types;
            float energy = emat_lookup(type_a, t, dist_idx);

            if (energy < -0.5f)      com += energy;
            else if (energy > 1.0f)  wal += energy;
            else                     sas += energy;
        }
    }

    // Rigid-body distance + rotation penalty.
    float dist2 = tx * tx + ty * ty + tz * tz;
    wal += 0.01f * dist2;
    sas += 0.005f * rot_angle * rot_angle;

    // ── Write ───────────────────────────────────────────────────────────────
    d_com[chrom] = com;
    d_wal[chrom] = wal;
    d_sas[chrom] = sas;
}

// ===========================================================================
// Host-side implementation — namespace hip_eval
// ===========================================================================

namespace hip_eval {

// ---------------------------------------------------------------------------
// Module-level state
// ---------------------------------------------------------------------------

namespace {

struct EvalState {
    bool   initialised  = false;
    int    device_id    = 0;
    int    max_pop      = 0;
    int    num_genes    = 0;  // set on first batch_evaluate call

    // Device buffers
    double* d_genes_f64 = nullptr;  // [max_pop × max_genes] double
    float*  d_genes_f32 = nullptr;  // [max_pop × max_genes] float
    float*  d_com       = nullptr;  // [max_pop] float
    float*  d_wal       = nullptr;  // [max_pop] float
    float*  d_sas       = nullptr;  // [max_pop] float

    // Staged host buffer for output (pinned for async copy performance).
    float*  h_com_pinned = nullptr;
    float*  h_wal_pinned = nullptr;
    float*  h_sas_pinned = nullptr;
};

EvalState g_state;

/// (Re)allocate device gene buffers if pop_size or num_genes grew.
void maybe_realloc(int pop_size, int num_genes) {
    bool need_realloc = (pop_size  > g_state.max_pop)  ||
                        (num_genes > g_state.num_genes) ||
                        (g_state.d_genes_f32 == nullptr);

    if (!need_realloc) return;

    // Free existing buffers.
    if (g_state.d_genes_f64) HIP_CHECK(hipFree(g_state.d_genes_f64));
    if (g_state.d_genes_f32) HIP_CHECK(hipFree(g_state.d_genes_f32));
    if (g_state.d_com)       HIP_CHECK(hipFree(g_state.d_com));
    if (g_state.d_wal)       HIP_CHECK(hipFree(g_state.d_wal));
    if (g_state.d_sas)       HIP_CHECK(hipFree(g_state.d_sas));
    if (g_state.h_com_pinned) HIP_CHECK(hipHostFree(g_state.h_com_pinned));
    if (g_state.h_wal_pinned) HIP_CHECK(hipHostFree(g_state.h_wal_pinned));
    if (g_state.h_sas_pinned) HIP_CHECK(hipHostFree(g_state.h_sas_pinned));

    int new_max_pop   = (pop_size  > g_state.max_pop)  ? pop_size  : g_state.max_pop;
    int new_max_genes = (num_genes > g_state.num_genes) ? num_genes : g_state.num_genes;

    std::size_t gene_bytes  = static_cast<std::size_t>(new_max_pop) * new_max_genes;
    std::size_t score_bytes = static_cast<std::size_t>(new_max_pop);

    HIP_CHECK(hipMalloc(&g_state.d_genes_f64, gene_bytes  * sizeof(double)));
    HIP_CHECK(hipMalloc(&g_state.d_genes_f32, gene_bytes  * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_state.d_com,        score_bytes * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_state.d_wal,        score_bytes * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_state.d_sas,        score_bytes * sizeof(float)));

    // Pinned host output buffers for fast DtoH transfer.
    HIP_CHECK(hipHostMalloc(&g_state.h_com_pinned, score_bytes * sizeof(float), 0));
    HIP_CHECK(hipHostMalloc(&g_state.h_wal_pinned, score_bytes * sizeof(float), 0));
    HIP_CHECK(hipHostMalloc(&g_state.h_sas_pinned, score_bytes * sizeof(float), 0));

    g_state.max_pop   = new_max_pop;
    g_state.num_genes = new_max_genes;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// DeviceInfo
// ---------------------------------------------------------------------------

DeviceInfo get_device_info(int device_id) {
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    DeviceInfo info;
    info.name              = prop.name;
    info.compute_units     = prop.multiProcessorCount;
    info.global_mem        = prop.totalGlobalMem;
    info.gcn_arch          = prop.gcnArch;
    info.wavefront_size    = prop.warpSize;
    info.max_threads_per_cu = prop.maxThreadsPerMultiProcessor;
    info.shared_mem_per_cu  = static_cast<int>(prop.sharedMemPerMultiprocessor);
    return info;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void init(int device_id, int max_pop) {
    if (g_state.initialised) {
        // Idempotent: if re-initialising on the same device, no-op.
        if (device_id == g_state.device_id) return;
        shutdown();
    }

    HIP_CHECK(hipSetDevice(device_id));
    g_state.device_id   = device_id;
    g_state.initialised = true;

    int effective_max_pop = (max_pop > 0) ? max_pop : FLEXAIDS_HIP_DEFAULT_MAX_POP;
    g_state.max_pop = 0;  // Force allocation in maybe_realloc.
    maybe_realloc(effective_max_pop, 64);  // Reserve for 64-gene chromosomes initially.
}

void shutdown() {
    if (!g_state.initialised) return;

    HIP_CHECK(hipSetDevice(g_state.device_id));

    if (g_state.d_genes_f64) { hipFree(g_state.d_genes_f64); g_state.d_genes_f64 = nullptr; }
    if (g_state.d_genes_f32) { hipFree(g_state.d_genes_f32); g_state.d_genes_f32 = nullptr; }
    if (g_state.d_com)       { hipFree(g_state.d_com);       g_state.d_com = nullptr; }
    if (g_state.d_wal)       { hipFree(g_state.d_wal);       g_state.d_wal = nullptr; }
    if (g_state.d_sas)       { hipFree(g_state.d_sas);       g_state.d_sas = nullptr; }
    if (g_state.h_com_pinned){ hipHostFree(g_state.h_com_pinned); g_state.h_com_pinned = nullptr; }
    if (g_state.h_wal_pinned){ hipHostFree(g_state.h_wal_pinned); g_state.h_wal_pinned = nullptr; }
    if (g_state.h_sas_pinned){ hipHostFree(g_state.h_sas_pinned); g_state.h_sas_pinned = nullptr; }

    g_state.initialised = false;
    g_state.max_pop     = 0;
    g_state.num_genes   = 0;
}

// ---------------------------------------------------------------------------
// Energy-matrix upload
// ---------------------------------------------------------------------------

void upload_emat(const float* emat_sampled, int n_types, int n_samples) {
    assert(g_state.initialised && "hip_eval::init() must be called before upload_emat()");

    std::size_t flat_size = static_cast<std::size_t>(n_types) * n_types * n_samples;
    if (flat_size > FLEXAIDS_HIP_EMAT_MAX) {
        throw std::runtime_error(
            "[hip_eval] Energy matrix too large for constant memory. "
            "Increase FLEXAIDS_HIP_EMAT_MAX or switch to global memory.");
    }

    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_emat),      emat_sampled, flat_size * sizeof(float)));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_n_types),   &n_types,     sizeof(int)));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_n_samples), &n_samples,   sizeof(int)));
}

// ---------------------------------------------------------------------------
// Low-level kernel launchers
// ---------------------------------------------------------------------------

void launch_pack_genes_kernel(const double* d_genes_f64,
                              float*        d_genes_f32,
                              int           total,
                              hipStream_t   stream)
{
    if (total <= 0) return;
    int block = FLEXAIDS_HIP_BLOCK_SIZE;
    int grid  = (total + block - 1) / block;
    hipLaunchKernelGGL(hip_pack_genes_kernel, grid, block, 0, stream,
                       d_genes_f64, d_genes_f32, total);
    HIP_CHECK(hipGetLastError());
}

void launch_cf_score_kernel(const float* d_genes_f32,
                            float*       d_com,
                            float*       d_wal,
                            float*       d_sas,
                            int          pop_size,
                            int          num_genes,
                            hipStream_t  stream)
{
    if (pop_size <= 0) return;

    int block = FLEXAIDS_HIP_BLOCK_SIZE;
    int grid  = pop_size;  // one block per chromosome

    // Shared memory: gene array + warp-reduction scratch.
    // n_warps = block / warpSize (runtime); we use block/64 as a conservative
    // upper bound since actual warpSize ≤ 64 on all current hardware.
    int smem_genes  = num_genes * static_cast<int>(sizeof(float));
    int smem_warps  = (block / 32) * static_cast<int>(sizeof(float)); // ≥ actual n_warps
    int smem_total  = smem_genes + smem_warps;

    hipLaunchKernelGGL(hip_cf_score_kernel, grid, block, smem_total, stream,
                       d_genes_f32, d_com, d_wal, d_sas, pop_size, num_genes);
    HIP_CHECK(hipGetLastError());
}

// ---------------------------------------------------------------------------
// Batch evaluate (main entry point)
// ---------------------------------------------------------------------------

void batch_evaluate(const double* h_genes,
                    double*       h_com,
                    double*       h_wal,
                    double*       h_sas,
                    int           pop_size,
                    int           num_genes,
                    const float*  emat_sampled,
                    int           n_types,
                    int           n_samples)
{
    assert(g_state.initialised && "hip_eval::init() must be called before batch_evaluate()");
    assert(h_genes && h_com && h_wal && h_sas);
    assert(pop_size > 0 && num_genes > 0);

    // ── 0. Upload energy matrix if provided ─────────────────────────────────
    if (emat_sampled) {
        upload_emat(emat_sampled, n_types, n_samples);
    }

    // ── 1. Ensure device buffers are large enough ────────────────────────────
    maybe_realloc(pop_size, num_genes);

    std::size_t gene_count = static_cast<std::size_t>(pop_size) * num_genes;
    std::size_t pop_count  = static_cast<std::size_t>(pop_size);

    // ── 2. Copy genes host → device (double, async) ───────────────────────────
    // Using hipStreamDefault with async transfers on pinned memory allows
    // overlap between DMA and kernel execution on separate hardware engines.
    HIP_CHECK(hipMemcpyAsync(g_state.d_genes_f64,
                             h_genes,
                             gene_count * sizeof(double),
                             hipMemcpyHostToDevice, hipStreamDefault));

    // ── 3. Pack: double → float on device ───────────────────────────────────
    launch_pack_genes_kernel(g_state.d_genes_f64, g_state.d_genes_f32,
                             static_cast<int>(gene_count),
                             hipStreamDefault);

    // ── 4. Choose kernel based on problem shape ──────────────────────────────
    // For small populations (< 4 × block_size), the per-chromosome block
    // approach (hip_cf_score_kernel) gives better occupancy by assigning an
    // entire wavefront-group to one chromosome.
    // For large populations, the flat batch kernel is more efficient.
    const int BATCH_THRESHOLD = 4 * FLEXAIDS_HIP_BLOCK_SIZE;

    if (pop_size < BATCH_THRESHOLD) {
        launch_cf_score_kernel(g_state.d_genes_f32,
                               g_state.d_com, g_state.d_wal, g_state.d_sas,
                               pop_size, num_genes, hipStreamDefault);
    } else {
        int block = FLEXAIDS_HIP_BLOCK_SIZE;
        int grid  = (pop_size + block - 1) / block;
        hipLaunchKernelGGL(hip_batch_eval_kernel, grid, block, 0, hipStreamDefault,
                           g_state.d_genes_f32,
                           g_state.d_com, g_state.d_wal, g_state.d_sas,
                           pop_size, num_genes);
        HIP_CHECK(hipGetLastError());
    }

    // ── 5. Copy results device → pinned host (float, async) ──────────────────
    HIP_CHECK(hipMemcpyAsync(g_state.h_com_pinned, g_state.d_com,
                             pop_count * sizeof(float), hipMemcpyDeviceToHost,
                             hipStreamDefault));
    HIP_CHECK(hipMemcpyAsync(g_state.h_wal_pinned, g_state.d_wal,
                             pop_count * sizeof(float), hipMemcpyDeviceToHost,
                             hipStreamDefault));
    HIP_CHECK(hipMemcpyAsync(g_state.h_sas_pinned, g_state.d_sas,
                             pop_count * sizeof(float), hipMemcpyDeviceToHost,
                             hipStreamDefault));

    // Synchronise stream before reading pinned memory.
    HIP_CHECK(hipStreamSynchronize(hipStreamDefault));

    // ── 6. Widen float → double and write to caller's buffers ───────────────
    for (int i = 0; i < pop_size; ++i) {
        h_com[i] = static_cast<double>(g_state.h_com_pinned[i]);
        h_wal[i] = static_cast<double>(g_state.h_wal_pinned[i]);
        h_sas[i] = static_cast<double>(g_state.h_sas_pinned[i]);
    }
}

}  // namespace hip_eval

#endif  // FLEXAIDS_USE_ROCM
