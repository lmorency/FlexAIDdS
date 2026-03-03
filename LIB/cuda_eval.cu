// cuda_eval.cu — CUDA batched chromosome evaluation kernels
//
// Architecture:
//   Grid:  pop_size blocks  (one chromosome per block)
//   Block: 256 threads      (cooperative reduction over atom pairs)
//
// Each block:
//   1. Loads its chromosome's gene vector into shared memory
//   2. Threads cooperatively compute pairwise contact areas / distances
//      for the ligand-optimised residues against all protein atoms
//   3. Accumulates complementarity function (CF) per atom pair
//   4. Block-level warp-shuffle reduction → single CF value per chromosome
//
// This is a simplified "scoring-only" GPU path: it assumes Cartesian
// coordinates have already been built on the host (or in a prior kernel)
// and evaluates pairwise distance-based CF contributions on device.

#ifdef FLEXAIDS_USE_CUDA

#include "cuda_eval.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ─── error checking macro ────────────────────────────────────────────────────
#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                cudaGetErrorString(err));                               \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while(0)

// ─── device constants ────────────────────────────────────────────────────────
static constexpr int BLOCK_SIZE = 256;

// ─── context structure ───────────────────────────────────────────────────────
struct CudaEvalCtx {
    // Device arrays
    float*  d_atom_xyz;      // [n_atoms × 3]
    int*    d_atom_type;     // [n_atoms]
    float*  d_atom_radius;   // [n_atoms]
    float*  d_emat;          // [n_types × n_types]
    double* d_genes;         // [max_pop × max_genes]
    double* d_cf_out;        // [max_pop]

    int n_atoms;
    int n_types;
    int max_pop;

    // Ligand atom range and VdW permeability (from FA_Global)
    int   lig_first;  // 0-based index of first ligand atom
    int   lig_last;   // 0-based index of last ligand atom
    float perm;       // van-der-Waals permeability
};

// ─── pairwise CF kernel ──────────────────────────────────────────────────────
//
// Simplified scoring: for each chromosome c, compute
//   CF(c) = Σ_{i<j} E(type_i, type_j) * contact_weight(r_ij)
// where contact_weight is a soft switching function around
// the sum of atomic radii.
//
// The gene vector encodes a rigid-body transform (tx,ty,tz,rx,ry,rz)
// applied to the ligand atoms.  For this kernel we assume the host has
// already built the transformed coordinates and uploaded them as
// per-chromosome coordinate offsets in d_genes[c * n_genes + 0..5].

__device__ float soft_contact(float r2, float rsum, float perm) {
    // Smooth switching function: 1 at r=rsum, decays to 0 at r >> rsum
    float rcut2 = rsum * rsum * perm * perm;
    if (r2 > rcut2 * 4.0f) return 0.0f;     // beyond cutoff
    float ratio2 = r2 / rcut2;
    if (ratio2 < 1.0f) {
        // Clash region: LJ-like wall  → return large positive (penalty)
        float inv_r6 = 1.0f / (ratio2 * ratio2 * ratio2);
        return 1e6f * (inv_r6 - 1.0f);
    }
    // Attractive region: Gaussian-like decay
    return expf(-2.0f * (ratio2 - 1.0f));
}

__global__ void kernel_eval_cf(
    const float*  __restrict__ atom_xyz,     // [N × 3]
    const int*    __restrict__ atom_type,    // [N]
    const float*  __restrict__ atom_radius,  // [N]
    const float*  __restrict__ emat,         // [T × T]
    const double* __restrict__ genes,        // [pop × n_genes]
    double*       __restrict__ cf_out,       // [pop]
    int N,          // n_atoms
    int T,          // n_types
    int n_genes,
    int lig_first,  // first ligand atom index
    int lig_last,   // last ligand atom index
    float perm)     // permeability
{
    int chrom_id = blockIdx.x;
    int tid      = threadIdx.x;

    // Each chromosome's rigid-body offset (simplified: 3 translations)
    __shared__ float tx, ty, tz;
    if (tid == 0) {
        tx = static_cast<float>(genes[chrom_id * n_genes + 0]);
        ty = static_cast<float>(genes[chrom_id * n_genes + 1]);
        tz = static_cast<float>(genes[chrom_id * n_genes + 2]);
    }
    __syncthreads();

    // Number of ligand atoms
    int n_lig = lig_last - lig_first + 1;
    // Total protein-ligand pairs to evaluate
    int n_pairs = n_lig * (N - n_lig);

    float local_cf = 0.0f;

    // Stride through atom pairs
    for (int p = tid; p < n_pairs; p += BLOCK_SIZE) {
        int lig_idx = p / (N - n_lig);      // which ligand atom (relative)
        int pro_rel = p % (N - n_lig);       // which protein atom (relative)

        int ai = lig_first + lig_idx;        // absolute ligand atom index
        int aj = (pro_rel < lig_first) ? pro_rel : (pro_rel + n_lig);  // skip ligand range

        // Ligand atom position with translation applied
        float lx = atom_xyz[ai * 3 + 0] + tx;
        float ly = atom_xyz[ai * 3 + 1] + ty;
        float lz = atom_xyz[ai * 3 + 2] + tz;

        float dx = lx - atom_xyz[aj * 3 + 0];
        float dy = ly - atom_xyz[aj * 3 + 1];
        float dz = lz - atom_xyz[aj * 3 + 2];
        float r2 = dx*dx + dy*dy + dz*dz;

        float rsum = atom_radius[ai] + atom_radius[aj];
        float cw = soft_contact(r2, rsum, perm);

        // Energy-matrix lookup
        int ti = atom_type[ai];
        int tj = atom_type[aj];
        float e_ij = emat[ti * T + tj];

        local_cf += e_ij * cw;
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        local_cf += __shfl_down_sync(0xFFFFFFFF, local_cf, offset);

    // Shared-memory reduction across warps
    __shared__ float warp_sums[BLOCK_SIZE / 32];
    int lane   = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) warp_sums[warp_id] = local_cf;
    __syncthreads();

    // First warp reduces the warp sums
    if (warp_id == 0) {
        float val = (tid < (BLOCK_SIZE / 32)) ? warp_sums[tid] : 0.0f;
        for (int offset = (BLOCK_SIZE / 32) / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (tid == 0) cf_out[chrom_id] = static_cast<double>(val);
    }
}

// ─── host API ────────────────────────────────────────────────────────────────

CudaEvalCtx* cuda_eval_init(int   n_atoms,
                             int   n_types,
                             int   max_pop,
                             int   lig_first,
                             int   lig_last,
                             float perm,
                             const float* h_atom_xyz,
                             const int*   h_atom_type,
                             const float* h_atom_radius,
                             const float* h_emat)
{
    CudaEvalCtx* ctx = new CudaEvalCtx;
    ctx->n_atoms   = n_atoms;
    ctx->n_types   = n_types;
    ctx->max_pop   = max_pop;
    ctx->lig_first = lig_first;
    ctx->lig_last  = lig_last;
    ctx->perm      = perm;

    size_t xyz_bytes    = static_cast<size_t>(n_atoms) * 3 * sizeof(float);
    size_t type_bytes   = static_cast<size_t>(n_atoms) * sizeof(int);
    size_t radius_bytes = static_cast<size_t>(n_atoms) * sizeof(float);
    size_t emat_bytes   = static_cast<size_t>(n_types) * n_types * sizeof(float);
    size_t gene_bytes   = static_cast<size_t>(max_pop) * 256 * sizeof(double); // generous gene buffer
    size_t cf_bytes     = static_cast<size_t>(max_pop) * sizeof(double);

    CUDA_CHECK(cudaMalloc(&ctx->d_atom_xyz,    xyz_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_atom_type,   type_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_atom_radius, radius_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_emat,        emat_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_genes,       gene_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_cf_out,      cf_bytes));

    CUDA_CHECK(cudaMemcpy(ctx->d_atom_xyz,    h_atom_xyz,    xyz_bytes,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_atom_type,   h_atom_type,   type_bytes,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_atom_radius, h_atom_radius, radius_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_emat,        h_emat,        emat_bytes,   cudaMemcpyHostToDevice));

    return ctx;
}

void cuda_eval_batch(CudaEvalCtx* ctx,
                     int pop_size,
                     int n_genes,
                     const double* h_genes,
                     double*       h_cf_out)
{
    size_t gene_bytes = static_cast<size_t>(pop_size) * n_genes * sizeof(double);
    size_t cf_bytes   = static_cast<size_t>(pop_size) * sizeof(double);

    CUDA_CHECK(cudaMemcpy(ctx->d_genes, h_genes, gene_bytes, cudaMemcpyHostToDevice));

    // Launch: one block per chromosome, BLOCK_SIZE threads per block
    int lig_first = ctx->lig_first;
    int lig_last  = ctx->lig_last;
    float perm    = ctx->perm;

    kernel_eval_cf<<<pop_size, BLOCK_SIZE>>>(
        ctx->d_atom_xyz,
        ctx->d_atom_type,
        ctx->d_atom_radius,
        ctx->d_emat,
        ctx->d_genes,
        ctx->d_cf_out,
        ctx->n_atoms,
        ctx->n_types,
        n_genes,
        lig_first,
        lig_last,
        perm);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_cf_out, ctx->d_cf_out, cf_bytes, cudaMemcpyDeviceToHost));
}

void cuda_eval_shutdown(CudaEvalCtx* ctx) {
    if (!ctx) return;
    cudaFree(ctx->d_atom_xyz);
    cudaFree(ctx->d_atom_type);
    cudaFree(ctx->d_atom_radius);
    cudaFree(ctx->d_emat);
    cudaFree(ctx->d_genes);
    cudaFree(ctx->d_cf_out);
    delete ctx;
}

#endif  // FLEXAIDS_USE_CUDA
