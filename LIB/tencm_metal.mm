// tencm_metal.mm — Apple Metal GPU kernels for TENCoM
//
// Two compute pipelines:
//   1. contact_discovery: O(N²) all-pairs distance check (threadgroup-parallel)
//   2. hessian_assembly:  per-contact H_kl accumulation with atomic float add
//
// Metal Shading Language kernel is embedded as a string and compiled at runtime.
// Follows the same pattern as ShannonMetalBridge.mm and CavityDetectMetalBridge.mm.

#ifdef FLEXAIDS_HAS_METAL_TENCM

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "tencm_metal.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace tencm { namespace metal {

// ─── Embedded MSL kernel source ─────────────────────────────────────────────

static const char* s_msl_source = R"MSL(
#include <metal_stdlib>
using namespace metal;

// Cross product
static float3 cross3(float3 a, float3 b) {
    return float3(a.y*b.z - a.z*b.y,
                  a.z*b.x - a.x*b.z,
                  a.x*b.y - a.y*b.x);
}

// Torsional Jacobian
static float3 jac_fn(device const float* ca_xyz,
                      device const float* bond_axis,
                      device const float* bond_pivot,
                      int bond_k, int atom_i) {
    if (atom_i <= bond_k) return float3(0.0f);
    float3 ri = float3(ca_xyz[atom_i*3+0], ca_xyz[atom_i*3+1], ca_xyz[atom_i*3+2]);
    float3 pk = float3(bond_pivot[bond_k*3+0], bond_pivot[bond_k*3+1], bond_pivot[bond_k*3+2]);
    float3 ek = float3(bond_axis[bond_k*3+0], bond_axis[bond_k*3+1], bond_axis[bond_k*3+2]);
    return cross3(ek, ri - pk);
}

// Contact discovery kernel
// Grid: one thread per (i,j) pair; uses atomic counter for compaction
kernel void contact_discovery(
    device const float*  ca_xyz       [[buffer(0)]],
    device       int*    contacts_ij  [[buffer(1)]],
    device       float*  contacts_k   [[buffer(2)]],
    device       float*  contacts_r0  [[buffer(3)]],
    device atomic_uint*  contact_cnt  [[buffer(4)]],
    constant     int&    N            [[buffer(5)]],
    constant     float&  cutoff2      [[buffer(6)]],
    constant     float&  cutoff       [[buffer(7)]],
    constant     float&  k0           [[buffer(8)]],
    constant     int&    max_contacts [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    // Map linear tid to (i, j) pair
    int pair = int(tid);
    for (int i = 0; i < N - 1; ++i) {
        int row_len = N - i - 2;
        if (pair < row_len) {
            int j = i + 2 + pair;
            float dx = ca_xyz[i*3+0] - ca_xyz[j*3+0];
            float dy = ca_xyz[i*3+1] - ca_xyz[j*3+1];
            float dz = ca_xyz[i*3+2] - ca_xyz[j*3+2];
            float r2 = dx*dx + dy*dy + dz*dz;

            if (r2 <= cutoff2) {
                uint slot = atomic_fetch_add_explicit(contact_cnt, 1u, memory_order_relaxed);
                if (int(slot) < max_contacts) {
                    float r0 = sqrt(r2);
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

// Hessian assembly kernel
// Grid: one threadgroup per contact; threads handle (k,l) pairs
kernel void hessian_assembly(
    device const float*  ca_xyz       [[buffer(0)]],
    device const float*  bond_axis    [[buffer(1)]],
    device const float*  bond_pivot   [[buffer(2)]],
    device const int*    contacts_ij  [[buffer(3)]],
    device const float*  contacts_k_buf [[buffer(4)]],
    device       float*  H_out        [[buffer(5)]],  // float atomics (Metal 2)
    constant     int&    M            [[buffer(6)]],
    constant     int&    N            [[buffer(7)]],
    uint ci  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]])
{
    int ci_i = contacts_ij[ci*2+0];
    int ci_j = contacts_ij[ci*2+1];
    float kij = contacts_k_buf[ci];

    for (uint idx = tid; idx < uint(M * M); idx += tcount) {
        int k = int(idx) / M;
        int l = int(idx) % M;
        if (l < k) continue;

        float3 jki = jac_fn(ca_xyz, bond_axis, bond_pivot, k, ci_i);
        float3 jkj = jac_fn(ca_xyz, bond_axis, bond_pivot, k, ci_j);
        float3 jli = jac_fn(ca_xyz, bond_axis, bond_pivot, l, ci_i);
        float3 jlj = jac_fn(ca_xyz, bond_axis, bond_pivot, l, ci_j);

        float3 djk = jki - jkj;
        float3 djl = jli - jlj;
        float contrib = kij * dot(djk, djl);

        // CAS-loop atomic float add (Metal 2 compatible)
        // H_out is float (not double) for Metal atomic compatibility
        device atomic_uint* addr_kl = (device atomic_uint*)&H_out[k*M+l];
        uint expected = atomic_load_explicit(addr_kl, memory_order_relaxed);
        while (true) {
            float current = as_type<float>(expected);
            float desired = current + contrib;
            if (atomic_compare_exchange_weak_explicit(addr_kl, &expected,
                    as_type<uint>(desired), memory_order_relaxed, memory_order_relaxed))
                break;
        }

        if (l != k) {
            device atomic_uint* addr_lk = (device atomic_uint*)&H_out[l*M+k];
            expected = atomic_load_explicit(addr_lk, memory_order_relaxed);
            while (true) {
                float current = as_type<float>(expected);
                float desired = current + contrib;
                if (atomic_compare_exchange_weak_explicit(addr_lk, &expected,
                        as_type<uint>(desired), memory_order_relaxed, memory_order_relaxed))
                    break;
            }
        }
    }
}
)MSL";

// ─── Device state ───────────────────────────────────────────────────────────

static id<MTLDevice>              s_device   = nil;
static id<MTLCommandQueue>        s_queue    = nil;
static id<MTLComputePipelineState> s_contact_pipeline = nil;
static id<MTLComputePipelineState> s_hessian_pipeline = nil;
static bool s_initialised = false;
static bool s_available   = false;

bool init() {
    if (s_initialised) return s_available;
    s_initialised = true;

    @autoreleasepool {
        s_device = MTLCreateSystemDefaultDevice();
        if (!s_device) {
            s_available = false;
            return false;
        }

        s_queue = [s_device newCommandQueue];

        // Compile MSL source
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:s_msl_source];
        id<MTLLibrary> library = [s_device newLibraryWithSource:source options:nil error:&error];
        if (!library) {
            NSLog(@"TENCoM Metal: shader compilation failed: %@", error);
            s_available = false;
            return false;
        }

        id<MTLFunction> contactFn = [library newFunctionWithName:@"contact_discovery"];
        id<MTLFunction> hessianFn = [library newFunctionWithName:@"hessian_assembly"];

        s_contact_pipeline = [s_device newComputePipelineStateWithFunction:contactFn error:&error];
        s_hessian_pipeline = [s_device newComputePipelineStateWithFunction:hessianFn error:&error];

        if (!s_contact_pipeline || !s_hessian_pipeline) {
            NSLog(@"TENCoM Metal: pipeline creation failed: %@", error);
            s_available = false;
            return false;
        }

        s_available = true;
    }
    return true;
}

void shutdown() {
    s_contact_pipeline = nil;
    s_hessian_pipeline = nil;
    s_queue  = nil;
    s_device = nil;
    s_initialised = false;
    s_available = false;
}

bool is_available() { return s_available; }

// ─── Contact discovery ──────────────────────────────────────────────────────

int build_contacts_gpu(const float* ca_xyz, int N,
                       float cutoff, float k0,
                       std::vector<int>&   contacts_ij,
                       std::vector<float>& contacts_k,
                       std::vector<float>& contacts_r0)
{
    if (!s_available || N < GPU_THRESHOLD) return -1;

    @autoreleasepool {
        // Total pairs
        int total_pairs = 0;
        for (int i = 0; i < N-1; ++i) total_pairs += (N - i - 2);
        int max_contacts = total_pairs;

        float cutoff2 = cutoff * cutoff;

        // Create Metal buffers
        id<MTLBuffer> buf_ca   = [s_device newBufferWithBytes:ca_xyz
                                   length:N*3*sizeof(float)
                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_ij   = [s_device newBufferWithLength:max_contacts*2*sizeof(int)
                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_k    = [s_device newBufferWithLength:max_contacts*sizeof(float)
                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_r0   = [s_device newBufferWithLength:max_contacts*sizeof(float)
                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_cnt  = [s_device newBufferWithLength:sizeof(uint32_t)
                                   options:MTLResourceStorageModeShared];
        memset(buf_cnt.contents, 0, sizeof(uint32_t));

        id<MTLCommandBuffer> cmd = [s_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:s_contact_pipeline];
        [enc setBuffer:buf_ca  offset:0 atIndex:0];
        [enc setBuffer:buf_ij  offset:0 atIndex:1];
        [enc setBuffer:buf_k   offset:0 atIndex:2];
        [enc setBuffer:buf_r0  offset:0 atIndex:3];
        [enc setBuffer:buf_cnt offset:0 atIndex:4];
        [enc setBytes:&N           length:sizeof(int)   atIndex:5];
        [enc setBytes:&cutoff2     length:sizeof(float) atIndex:6];
        [enc setBytes:&cutoff      length:sizeof(float) atIndex:7];
        [enc setBytes:&k0          length:sizeof(float) atIndex:8];
        [enc setBytes:&max_contacts length:sizeof(int)  atIndex:9];

        NSUInteger tgSize = s_contact_pipeline.maxTotalThreadsPerThreadgroup;
        if (tgSize > 256) tgSize = 256;
        MTLSize gridSize = MTLSizeMake(total_pairs, 1, 1);
        MTLSize tgSizeMTL = MTLSizeMake(tgSize, 1, 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSizeMTL];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        uint32_t count = *static_cast<uint32_t*>(buf_cnt.contents);
        int h_count = static_cast<int>(count);

        if (h_count > 0 && h_count <= max_contacts) {
            contacts_ij.resize(h_count * 2);
            contacts_k.resize(h_count);
            contacts_r0.resize(h_count);
            memcpy(contacts_ij.data(), buf_ij.contents, h_count*2*sizeof(int));
            memcpy(contacts_k.data(), buf_k.contents, h_count*sizeof(float));
            memcpy(contacts_r0.data(), buf_r0.contents, h_count*sizeof(float));
        }

        return h_count;
    }
}

// ─── Hessian assembly ───────────────────────────────────────────────────────

void assemble_hessian_gpu(const float* ca_xyz, int N,
                          const int* contacts_ij,
                          const float* contacts_k,
                          int M, int C,
                          double* H_out)
{
    if (!s_available || C == 0) return;

    @autoreleasepool {
        // Build bond axes and pivots
        std::vector<float> bond_axis(M * 3), bond_pivot(M * 3);
        for (int k = 0; k < M; ++k) {
            float ax = ca_xyz[(k+1)*3+0] - ca_xyz[k*3+0];
            float ay = ca_xyz[(k+1)*3+1] - ca_xyz[k*3+1];
            float az = ca_xyz[(k+1)*3+2] - ca_xyz[k*3+2];
            float inv = 1.0f / std::sqrt(ax*ax + ay*ay + az*az);
            bond_axis[k*3+0] = ax*inv; bond_axis[k*3+1] = ay*inv; bond_axis[k*3+2] = az*inv;
            bond_pivot[k*3+0] = 0.5f*(ca_xyz[k*3+0] + ca_xyz[(k+1)*3+0]);
            bond_pivot[k*3+1] = 0.5f*(ca_xyz[k*3+1] + ca_xyz[(k+1)*3+1]);
            bond_pivot[k*3+2] = 0.5f*(ca_xyz[k*3+2] + ca_xyz[(k+1)*3+2]);
        }

        // H_out in float for Metal atomics (convert back to double after)
        std::vector<float> H_float(M * M, 0.0f);

        id<MTLBuffer> buf_ca    = [s_device newBufferWithBytes:ca_xyz length:N*3*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_axis  = [s_device newBufferWithBytes:bond_axis.data() length:M*3*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_pivot = [s_device newBufferWithBytes:bond_pivot.data() length:M*3*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_ij    = [s_device newBufferWithBytes:contacts_ij length:C*2*sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_k     = [s_device newBufferWithBytes:contacts_k length:C*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_H     = [s_device newBufferWithBytes:H_float.data() length:M*M*sizeof(float) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmd = [s_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:s_hessian_pipeline];
        [enc setBuffer:buf_ca    offset:0 atIndex:0];
        [enc setBuffer:buf_axis  offset:0 atIndex:1];
        [enc setBuffer:buf_pivot offset:0 atIndex:2];
        [enc setBuffer:buf_ij    offset:0 atIndex:3];
        [enc setBuffer:buf_k     offset:0 atIndex:4];
        [enc setBuffer:buf_H     offset:0 atIndex:5];
        [enc setBytes:&M length:sizeof(int) atIndex:6];
        [enc setBytes:&N length:sizeof(int) atIndex:7];

        NSUInteger tgSize = s_hessian_pipeline.maxTotalThreadsPerThreadgroup;
        if (tgSize > 256) tgSize = 256;
        MTLSize gridSize = MTLSizeMake(C, 1, 1);
        MTLSize tgSizeMTL = MTLSizeMake(tgSize, 1, 1);
        [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSizeMTL];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        // Copy float → double
        const float* result = static_cast<const float*>(buf_H.contents);
        for (int idx = 0; idx < M*M; ++idx)
            H_out[idx] = static_cast<double>(result[idx]);
    }
}

}}  // namespace tencm::metal

#endif  // FLEXAIDS_HAS_METAL_TENCM
