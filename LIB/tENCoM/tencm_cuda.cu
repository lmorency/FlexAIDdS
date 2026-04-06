// tencm_cuda.cu — CUDA kernels for TENCoM contact discovery & Hessian assembly
//
// Two kernels:
//   1. contact_discovery_kernel: O(N²) all-pairs with warp-level reductions
//   2. hessian_assembly_kernel:  one block per contact, accumulates H_kl via atomicAdd
//
// Designed for proteins with N > 256 residues where GPU pays off.
#ifdef FLEXAIDS_USE_CUDA

#include "tencm_cuda.cuh"
#include "../gpu_buffer.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

namespace tencm { namespace cuda {

// ─── Device state ───────────────────────────────────────────────────────────

static bool       s_initialised = false;
static bool       s_available   = false;
static cudaStream_t s_stream    = nullptr;

bool init() {
    if (s_initialised) return s_available;
    s_initialised = true;

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        s_available = false;
        return false;
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 3) {
        s_available = false;
        return false;
    }

    cudaSetDevice(0);
    cudaStreamCreate(&s_stream);
    s_available = true;
    return true;
}

void shutdown() {
    if (s_stream) { cudaStreamDestroy(s_stream); s_stream = nullptr; }
    s_initialised = false;
    s_available = false;
}

bool is_available() { return s_available; }

// ─── Helpers ────────────────────────────────────────────────────────────────

// Cross product on device
__device__ void d_cross3(const float* a, const float* b, float* c) {
    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];
}

// Torsional Jacobian: J_k(atom_i) = axis_k × (r_i - pivot_k) if i > k, else 0
__device__ void d_jac(const float* ca_xyz, int N,
                      const float* bond_axis, const float* bond_pivot,
                      int bond_k, int atom_i,
                      float* out) {
    if (atom_i <= bond_k) {
        out[0] = out[1] = out[2] = 0.0f;
        return;
    }
    float d[3] = {
        ca_xyz[atom_i*3+0] - bond_pivot[bond_k*3+0],
        ca_xyz[atom_i*3+1] - bond_pivot[bond_k*3+1],
        ca_xyz[atom_i*3+2] - bond_pivot[bond_k*3+2]
    };
    const float* axis = &bond_axis[bond_k*3];
    d_cross3(axis, d, out);
}

// ─── Contact discovery kernel ───────────────────────────────────────────────
// Grid: one thread per (i,j) pair in upper triangle (j > i+1)
// Uses atomic counter for compact contact list

__global__ void contact_discovery_kernel(
    const float* __restrict__ ca_xyz, int N,
    float cutoff2, float cutoff, float k0,
    int* __restrict__ contacts_ij,
    float* __restrict__ contacts_k,
    float* __restrict__ contacts_r0,
    int* __restrict__ contact_count,
    int max_contacts)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Map linear tid to (i, j) pair in upper triangle (j >= i+2)
    // Total pairs: Σ_{i=0}^{N-2} (N - i - 2) for j >= i+2
    // Use sequential scan (GPU has plenty of threads)
    int pair = tid;
    for (int i = 0; i < N - 1; ++i) {
        int row_len = N - i - 2;  // number of j's for this i (j = i+2..N-1)
        if (pair < row_len) {
            int j = i + 2 + pair;
            float dx = ca_xyz[i*3+0] - ca_xyz[j*3+0];
            float dy = ca_xyz[i*3+1] - ca_xyz[j*3+1];
            float dz = ca_xyz[i*3+2] - ca_xyz[j*3+2];
            float r2 = dx*dx + dy*dy + dz*dz;

            if (r2 <= cutoff2) {
                int slot = atomicAdd(contact_count, 1);
                if (slot < max_contacts) {
                    float r0 = sqrtf(r2);
                    float ratio = cutoff / r0;
                    float r3 = ratio * ratio * ratio;
                    contacts_ij[slot*2+0] = i;
                    contacts_ij[slot*2+1] = j;
                    contacts_k[slot]  = k0 * (r3 * r3);
                    contacts_r0[slot] = r0;
                }
            }
            return;
        }
        pair -= row_len;
    }
}

int build_contacts_gpu(const float* ca_xyz, int N,
                       float cutoff, float k0,
                       int* contacts_ij_out,
                       float* contacts_k_out,
                       float* contacts_r0_out)
{
    if (!s_available || N < GPU_THRESHOLD) return -1;

    // Total upper-triangle pairs (j >= i+2)
    long long total_pairs = 0;
    for (int i = 0; i < N-1; ++i)
        total_pairs += (N - i - 2);

    int max_contacts = static_cast<int>(total_pairs);  // worst case

    // Device allocations (RAII — freed automatically on scope exit or exception)
    GPUBuffer<float> d_ca(N * 3, GPUBackend::CUDA);
    GPUBuffer<int>   d_contacts_ij(max_contacts * 2, GPUBackend::CUDA);
    GPUBuffer<float> d_contacts_k(max_contacts, GPUBackend::CUDA);
    GPUBuffer<float> d_contacts_r0(max_contacts, GPUBackend::CUDA);
    GPUBuffer<int>   d_count(1, GPUBackend::CUDA);

    cudaMemcpyAsync(d_ca.data(), ca_xyz, N*3*sizeof(float), cudaMemcpyHostToDevice, s_stream);
    cudaMemsetAsync(d_count.data(), 0, sizeof(int), s_stream);

    int block_size = 256;
    int grid_size = (static_cast<int>(total_pairs) + block_size - 1) / block_size;

    contact_discovery_kernel<<<grid_size, block_size, 0, s_stream>>>(
        d_ca.data(), N, cutoff*cutoff, cutoff, k0,
        d_contacts_ij.data(), d_contacts_k.data(), d_contacts_r0.data(),
        d_count.data(), max_contacts);

    int h_count = 0;
    cudaMemcpyAsync(&h_count, d_count.data(), sizeof(int), cudaMemcpyDeviceToHost, s_stream);
    cudaStreamSynchronize(s_stream);

    if (h_count > 0 && h_count <= max_contacts) {
        d_contacts_ij.download(contacts_ij_out, h_count * 2);
        d_contacts_k.download(contacts_k_out, h_count);
        d_contacts_r0.download(contacts_r0_out, h_count);
    }

    return h_count;
}

// ─── Hessian assembly kernel ────────────────────────────────────────────────
// One block per contact; threads within block handle (k, l) pairs
// Uses atomicAdd on global memory H[k*M + l]

__global__ void hessian_assembly_kernel(
    const float* __restrict__ ca_xyz, int N,
    const float* __restrict__ bond_axis,   // [M x 3]
    const float* __restrict__ bond_pivot,  // [M x 3]
    const int*   __restrict__ contacts_ij, // [C x 2]
    const float* __restrict__ contacts_k,  // [C]
    int M,
    double* __restrict__ H_out)            // [M x M]
{
    int ci = blockIdx.x;  // one block per contact
    int ci_i = contacts_ij[ci*2+0];
    int ci_j = contacts_ij[ci*2+1];
    float kij = contacts_k[ci];

    // Each thread handles one (k, l) pair with l >= k
    // Thread layout: linearize upper triangle
    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    for (int idx = tid; idx < M * M; idx += total_threads) {
        int k = idx / M;
        int l = idx % M;
        if (l < k) continue;  // only upper triangle + diagonal

        float jki[3], jkj[3], jli[3], jlj[3];
        d_jac(ca_xyz, N, bond_axis, bond_pivot, k, ci_i, jki);
        d_jac(ca_xyz, N, bond_axis, bond_pivot, k, ci_j, jkj);
        d_jac(ca_xyz, N, bond_axis, bond_pivot, l, ci_i, jli);
        d_jac(ca_xyz, N, bond_axis, bond_pivot, l, ci_j, jlj);

        float djk[3] = { jki[0]-jkj[0], jki[1]-jkj[1], jki[2]-jkj[2] };
        float djl[3] = { jli[0]-jlj[0], jli[1]-jlj[1], jli[2]-jlj[2] };

        double contrib = static_cast<double>(kij) *
            static_cast<double>(djk[0]*djl[0] + djk[1]*djl[1] + djk[2]*djl[2]);

        atomicAdd(&H_out[k * M + l], contrib);
        if (l != k) atomicAdd(&H_out[l * M + k], contrib);
    }
}

void assemble_hessian_gpu(const float* ca_xyz, int N,
                          const int* contacts_ij,
                          const float* contacts_k,
                          int M, int C,
                          double* H_out)
{
    if (!s_available || C == 0) return;

    // Build bond axes and pivots on host
    std::vector<float> bond_axis(M * 3), bond_pivot(M * 3);
    for (int k = 0; k < M; ++k) {
        float ax = ca_xyz[(k+1)*3+0] - ca_xyz[k*3+0];
        float ay = ca_xyz[(k+1)*3+1] - ca_xyz[k*3+1];
        float az = ca_xyz[(k+1)*3+2] - ca_xyz[k*3+2];
        float inv = 1.0f / sqrtf(ax*ax + ay*ay + az*az);
        bond_axis[k*3+0] = ax*inv; bond_axis[k*3+1] = ay*inv; bond_axis[k*3+2] = az*inv;
        bond_pivot[k*3+0] = 0.5f*(ca_xyz[k*3+0] + ca_xyz[(k+1)*3+0]);
        bond_pivot[k*3+1] = 0.5f*(ca_xyz[k*3+1] + ca_xyz[(k+1)*3+1]);
        bond_pivot[k*3+2] = 0.5f*(ca_xyz[k*3+2] + ca_xyz[(k+1)*3+2]);
    }

    // Device allocations (RAII — freed automatically on scope exit or exception)
    GPUBuffer<float>  d_ca(N * 3, GPUBackend::CUDA);
    GPUBuffer<float>  d_axis(M * 3, GPUBackend::CUDA);
    GPUBuffer<float>  d_pivot(M * 3, GPUBackend::CUDA);
    GPUBuffer<int>    d_contacts_ij(C * 2, GPUBackend::CUDA);
    GPUBuffer<float>  d_contacts_k(C, GPUBackend::CUDA);
    GPUBuffer<double> d_H(M * M, GPUBackend::CUDA);

    d_H.zero();
    cudaMemcpyAsync(d_ca.data(), ca_xyz, N*3*sizeof(float), cudaMemcpyHostToDevice, s_stream);
    cudaMemcpyAsync(d_axis.data(), bond_axis.data(), M*3*sizeof(float), cudaMemcpyHostToDevice, s_stream);
    cudaMemcpyAsync(d_pivot.data(), bond_pivot.data(), M*3*sizeof(float), cudaMemcpyHostToDevice, s_stream);
    cudaMemcpyAsync(d_contacts_ij.data(), contacts_ij, C*2*sizeof(int), cudaMemcpyHostToDevice, s_stream);
    cudaMemcpyAsync(d_contacts_k.data(), contacts_k, C*sizeof(float), cudaMemcpyHostToDevice, s_stream);

    // Launch: one block per contact, 256 threads per block
    int block_size = 256;
    hessian_assembly_kernel<<<C, block_size, 0, s_stream>>>(
        d_ca.data(), N, d_axis.data(), d_pivot.data(),
        d_contacts_ij.data(), d_contacts_k.data(), M, d_H.data());

    cudaMemcpyAsync(H_out, d_H.data(), M*M*sizeof(double), cudaMemcpyDeviceToHost, s_stream);
    cudaStreamSynchronize(s_stream);
}

}}  // namespace tencm::cuda

#endif  // FLEXAIDS_USE_CUDA
