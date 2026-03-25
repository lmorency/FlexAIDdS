// cuda_eval.cu — Full-fidelity CUDA batched chromosome evaluation kernels
//
// Architecture:
//   Grid:  pop_size threadblocks  (one chromosome per block)
//   Block: 256 threads            (cooperative reduction over atom pairs)
//
// Each block:
//   1. Loads its chromosome's first-three gene values (tx,ty,tz) into shared.
//   2. Initialises per-ligand SAS counters in shared memory.
//   3. Threads cooperatively evaluate all ligand-protein atom pairs:
//        a. Normalised contact area via linear switching function.
//        b. COM: energy-matrix lookup via sampled density-function table.
//        c. WAL: KWALL × (r⁻¹² − (perm·rAB)⁻¹²) for clashing pairs.
//        d. SAS: atomicAdd into shared per-ligand SAS counter.
//   4. SAS energy contribution computed per ligand atom using the
//      solvent column (T-1) of the energy matrix.
//   5. Warp-shuffle + shared-memory reduction → three CF scalars per chromosome.

#ifdef FLEXAIDS_USE_CUDA

#include "cuda_eval.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ─── error-checking macro ─────────────────────────────────────────────────────
#define CUDA_CHECK(call) do {                                             \
    cudaError_t _e = (call);                                              \
    if (_e != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d  %s\n", __FILE__, __LINE__,    \
                cudaGetErrorString(_e));                                  \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
} while (0)

// ─── constants ────────────────────────────────────────────────────────────────
static constexpr int   BLOCK_SIZE     = 256;
static constexpr int   N_EMAT_SAMPLES = 128;   // must match CUDA_EMAT_SAMPLES in .cuh
static constexpr float Rw             = 1.4f;  // water probe radius (Å)
static constexpr float KWALL_F        = 1.0e6f;

// Maximum ligand atoms handled in shared-memory SAS accumulator.
// Ligands with more atoms fall back to zero SAS contribution.
static constexpr int MAX_LIG_SAS = 256;

// ─── context ─────────────────────────────────────────────────────────────────
struct CudaEvalCtx {
    float*  d_atom_xyz;      // [n_atoms × 3]
    int*    d_atom_type;     // [n_atoms]
    float*  d_atom_radius;   // [n_atoms]
    float*  d_emat_sampled;  // [n_types × n_types × N_EMAT_SAMPLES]
    double* d_genes;         // [max_pop × max_genes]
    double* d_com_out;       // [max_pop]
    double* d_wal_out;       // [max_pop]
    double* d_sas_out;       // [max_pop]

    int   n_atoms;
    int   n_types;
    int   max_pop;
    int   max_genes;
    int   lig_first;
    int   lig_last;
    float perm;
};

// ─── device helper: interpolated energy-matrix lookup ────────────────────────
__device__ __forceinline__ float gpu_get_yval(
        const float* __restrict__ emat_sampled,
        int t1, int t2, int T, float rel_area)
{
    int   base = (t1 * T + t2) * N_EMAT_SAMPLES;
    rel_area   = fmaxf(0.0f, fminf(1.0f, rel_area));
    float kf   = rel_area * (N_EMAT_SAMPLES - 1.0f);
    int   k0   = (int)kf;
    int   k1   = min(k0 + 1, N_EMAT_SAMPLES - 1);
    float frac = kf - (float)k0;
    return emat_sampled[base + k0] * (1.0f - frac)
         + emat_sampled[base + k1] * frac;
}

// ─── full-fidelity CF kernel ──────────────────────────────────────────────────
__global__ void kernel_eval_cf_full(
    const float*  __restrict__ atom_xyz,       // [N × 3]
    const int*    __restrict__ atom_type,      // [N]
    const float*  __restrict__ atom_radius,    // [N]
    const float*  __restrict__ emat_sampled,   // [T × T × N_EMAT_SAMPLES]
    const double* __restrict__ genes,          // [pop × n_genes]
    double*       __restrict__ com_out,        // [pop]
    double*       __restrict__ wal_out,        // [pop]
    double*       __restrict__ sas_out,        // [pop]
    int N, int T, int n_genes,
    int lig_first, int lig_last, float perm)
{
    const int chrom_id = blockIdx.x;
    const int tid      = threadIdx.x;

    // Translation from genes (first three genes encode tx, ty, tz).
    __shared__ float tx, ty, tz;
    if (tid == 0) {
        tx = (float)genes[chrom_id * n_genes + 0];
        ty = (float)genes[chrom_id * n_genes + 1];
        tz = (float)genes[chrom_id * n_genes + 2];
    }

    // Per-ligand SAS accumulator (shared memory, initialised to 4π(rA+Rw)²).
    __shared__ float lig_sas[MAX_LIG_SAS];
    const int n_lig = lig_last - lig_first + 1;
    const int n_pro = N - n_lig;
    for (int la = tid; la < n_lig && la < MAX_LIG_SAS; la += BLOCK_SIZE) {
        float rwa     = atom_radius[lig_first + la] + Rw;
        lig_sas[la]   = 4.0f * 3.141592653589793f * rwa * rwa;
    }
    __syncthreads();

    // ── pair loop ────────────────────────────────────────────────────────────
    const int n_pairs = n_lig * n_pro;
    float local_com = 0.0f, local_wal = 0.0f;

    for (int pr = tid; pr < n_pairs; pr += BLOCK_SIZE) {
        const int li      = pr / n_pro;
        const int pro_rel = pr % n_pro;
        const int ai      = lig_first + li;
        const int aj      = (pro_rel < lig_first) ? pro_rel : (pro_rel + n_lig);

        // Ligand atom position with translation applied.
        const float lx = atom_xyz[ai * 3 + 0] + tx;
        const float ly = atom_xyz[ai * 3 + 1] + ty;
        const float lz = atom_xyz[ai * 3 + 2] + tz;

        const float dx = lx - atom_xyz[aj * 3 + 0];
        const float dy = ly - atom_xyz[aj * 3 + 1];
        const float dz = lz - atom_xyz[aj * 3 + 2];
        const float r  = sqrtf(dx*dx + dy*dy + dz*dz + 1e-10f);

        const float rA    = atom_radius[ai];
        const float rB    = atom_radius[aj];
        const float rsum  = rA + rB;
        const float rwa_A = rA + Rw;
        const float surf_A = 4.0f * 3.141592653589793f * rwa_A * rwa_A;
        const float outer_r = rsum + 2.0f * Rw;  // rA + rB + 2·Rw

        // Normalised contact area: linear from 1 at r=rsum to 0 at r=outer_r.
        float rel_area = 0.0f;
        if      (r < rsum)    rel_area = 1.0f;
        else if (r < outer_r) rel_area = 1.0f - (r - rsum) / (outer_r - rsum);

        // Subtract contact area from this ligand atom's SAS counter.
        if (rel_area > 0.0f && li < MAX_LIG_SAS) {
            atomicAdd(&lig_sas[li], -rel_area * surf_A);
        }

        // COM: energy-matrix interpolation scaled by normalised contact area.
        const int   ti   = atom_type[ai];
        const int   tj   = atom_type[aj];
        const float yval = gpu_get_yval(emat_sampled, ti, tj, T, rel_area);
        local_com += yval * rel_area;

        // WAL: repulsive wall energy when r < perm × (rA+rB).
        const float clash_r = perm * rsum;
        if (r < clash_r) {
            const float inv_r12  = 1.0f / powf(r,       12.0f);
            const float inv_cr12 = 1.0f / powf(clash_r, 12.0f);
            local_wal += KWALL_F * (inv_r12 - inv_cr12);
        }
    }
    // Pair loop done; ensure all atomicAdds to lig_sas are visible.
    __syncthreads();

    // ── warp-level reduction for COM and WAL ─────────────────────────────────
    for (int off = warpSize / 2; off > 0; off >>= 1) {
        local_com += __shfl_down_sync(0xFFFFFFFF, local_com, off);
        local_wal += __shfl_down_sync(0xFFFFFFFF, local_wal, off);
    }
    __shared__ float warp_com[BLOCK_SIZE / 32];
    __shared__ float warp_wal[BLOCK_SIZE / 32];
    const int lane = tid % 32, wid = tid / 32;
    if (lane == 0) { warp_com[wid] = local_com; warp_wal[wid] = local_wal; }
    __syncthreads();

    // ── SAS contribution (per ligand atom) ───────────────────────────────────
    float local_sas = 0.0f;
    if (n_lig <= MAX_LIG_SAS) {
        for (int la = tid; la < n_lig; la += BLOCK_SIZE) {
            const float sas_rem  = fmaxf(0.0f, lig_sas[la]);
            const float rwa_la   = atom_radius[lig_first + la] + Rw;
            const float surf_la  = 4.0f * 3.141592653589793f * rwa_la * rwa_la;
            const float sas_norm = sas_rem / surf_la;
            const int   ti_la    = atom_type[lig_first + la];
            // Solvent interaction: last column of the energy matrix (index T-1).
            const float yval_sas = gpu_get_yval(emat_sampled, ti_la, T - 1, T, sas_norm);
            local_sas += yval_sas * sas_norm;
        }
    }
    for (int off = warpSize / 2; off > 0; off >>= 1)
        local_sas += __shfl_down_sync(0xFFFFFFFF, local_sas, off);
    __shared__ float warp_sas[BLOCK_SIZE / 32];
    if (lane == 0) warp_sas[wid] = local_sas;
    __syncthreads();

    // ── final cross-warp reduction (warp 0 only) ─────────────────────────────
    if (wid == 0) {
        const int nwarps = BLOCK_SIZE / 32;
        float vcom = (lane < nwarps) ? warp_com[lane] : 0.0f;
        float vwal = (lane < nwarps) ? warp_wal[lane] : 0.0f;
        float vsas = (lane < nwarps) ? warp_sas[lane] : 0.0f;
        for (int off = nwarps / 2; off > 0; off >>= 1) {
            vcom += __shfl_down_sync(0xFFFFFFFF, vcom, off);
            vwal += __shfl_down_sync(0xFFFFFFFF, vwal, off);
            vsas += __shfl_down_sync(0xFFFFFFFF, vsas, off);
        }
        if (lane == 0) {
            com_out[chrom_id] = (double)vcom;
            wal_out[chrom_id] = (double)vwal;
            sas_out[chrom_id] = (double)vsas;
        }
    }
}

// ─── host API ────────────────────────────────────────────────────────────────

CudaEvalCtx* cuda_eval_init(int   n_atoms,
                             int   n_types,
                             int   max_pop,
                             int   max_genes,
                             int   lig_first,
                             int   lig_last,
                             float perm,
                             const float* h_atom_xyz,
                             const int*   h_atom_type,
                             const float* h_atom_radius,
                             const float* h_emat_sampled)
{
    CudaEvalCtx* ctx = new CudaEvalCtx;
    ctx->n_atoms   = n_atoms;
    ctx->n_types   = n_types;
    ctx->max_pop   = max_pop;
    ctx->max_genes = max_genes;
    ctx->lig_first = lig_first;
    ctx->lig_last  = lig_last;
    ctx->perm      = perm;

    const size_t xyz_bytes  = (size_t)n_atoms * 3          * sizeof(float);
    const size_t type_bytes = (size_t)n_atoms               * sizeof(int);
    const size_t rad_bytes  = (size_t)n_atoms               * sizeof(float);
    const size_t em_bytes   = (size_t)n_types * n_types * N_EMAT_SAMPLES * sizeof(float);
    const size_t gene_bytes = (size_t)max_pop * max_genes   * sizeof(double);
    const size_t cf_bytes   = (size_t)max_pop               * sizeof(double);

    CUDA_CHECK(cudaMalloc(&ctx->d_atom_xyz,     xyz_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_atom_type,    type_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_atom_radius,  rad_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_emat_sampled, em_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_genes,        gene_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_com_out,      cf_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_wal_out,      cf_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_sas_out,      cf_bytes));

    CUDA_CHECK(cudaMemcpy(ctx->d_atom_xyz,     h_atom_xyz,     xyz_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_atom_type,    h_atom_type,    type_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_atom_radius,  h_atom_radius,  rad_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_emat_sampled, h_emat_sampled, em_bytes,   cudaMemcpyHostToDevice));

    return ctx;
}

void cuda_eval_batch(CudaEvalCtx*  ctx,
                     int           pop_size,
                     int           n_genes,
                     const double* h_genes,
                     double*       h_com_out,
                     double*       h_wal_out,
                     double*       h_sas_out)
{
    // Validate against allocated buffer sizes.
    if (pop_size > ctx->max_pop) {
        fprintf(stderr, "cuda_eval: pop_size %d exceeds max_pop %d\n", pop_size, ctx->max_pop);
        return;
    }
    if (n_genes > ctx->max_genes) {
        fprintf(stderr, "cuda_eval: n_genes %d exceeds max_genes %d\n", n_genes, ctx->max_genes);
        return;
    }

    const size_t gene_bytes = (size_t)pop_size * n_genes * sizeof(double);
    const size_t cf_bytes   = (size_t)pop_size            * sizeof(double);

    CUDA_CHECK(cudaMemcpy(ctx->d_genes, h_genes, gene_bytes, cudaMemcpyHostToDevice));

    kernel_eval_cf_full<<<pop_size, BLOCK_SIZE>>>(
        ctx->d_atom_xyz,
        ctx->d_atom_type,
        ctx->d_atom_radius,
        ctx->d_emat_sampled,
        ctx->d_genes,
        ctx->d_com_out,
        ctx->d_wal_out,
        ctx->d_sas_out,
        ctx->n_atoms,
        ctx->n_types,
        n_genes,
        ctx->lig_first,
        ctx->lig_last,
        ctx->perm);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_com_out, ctx->d_com_out, cf_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_wal_out, ctx->d_wal_out, cf_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sas_out, ctx->d_sas_out, cf_bytes, cudaMemcpyDeviceToHost));
}

void cuda_eval_shutdown(CudaEvalCtx* ctx)
{
    if (!ctx) return;
    cudaFree(ctx->d_atom_xyz);
    cudaFree(ctx->d_atom_type);
    cudaFree(ctx->d_atom_radius);
    cudaFree(ctx->d_emat_sampled);
    cudaFree(ctx->d_genes);
    cudaFree(ctx->d_com_out);
    cudaFree(ctx->d_wal_out);
    cudaFree(ctx->d_sas_out);
    delete ctx;
}

#endif  // FLEXAIDS_USE_CUDA
