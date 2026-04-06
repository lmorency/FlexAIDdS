// TurboQuantMetalBridge.mm — Objective-C++ dispatch for TurboQuant.metal
// Bridges C++ TurboQuantMSE batch operations to the Metal GPU kernel.
// Follows the CavityDetectMetalBridge.mm pattern.
//
// Copyright 2026 Le Bonhomme Pharma. Licensed under Apache-2.0.

#include "TurboQuantMetalBridge.h"

#ifdef FLEXAIDS_USE_METAL

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstring>
#include <cstdio>

// GPU-side parameter struct (must match TurboQuant.metal QuantizeParams)
struct TQMetalParams {
    uint32_t N;
    uint32_t d;
    uint32_t bit_width;
    uint32_t num_boundaries;
    uint32_t num_centroids;
};

// ─── Metal state management (lazy-init, reusable) ───────────────────────────

namespace {

struct TQMetalState {
    id<MTLDevice>              device       = nil;
    id<MTLCommandQueue>        queue        = nil;
    id<MTLLibrary>             library      = nil;
    id<MTLComputePipelineState> quantize_pso = nil;
    id<MTLComputePipelineState> dequant_pso  = nil;
    bool                       initialized  = false;

    bool init() {
        if (initialized) return (device != nil);

        initialized = true;
        device = MTLCreateSystemDefaultDevice();
        if (!device) return false;

        // Load compiled metallib (path injected by CMake)
        NSError* err = nil;
        NSString* metallib_path = @TURBOQUANT_METALLIB_PATH;
        library = [device newLibraryWithURL:[NSURL fileURLWithPath:metallib_path]
                                       error:&err];
        if (!library) {
            NSLog(@"[TurboQuant] metallib not found at %@: %@",
                  metallib_path, err.localizedDescription);
            return false;
        }

        id<MTLFunction> fn_quant = [library newFunctionWithName:@"turboquant_quantize"];
        id<MTLFunction> fn_deq   = [library newFunctionWithName:@"turboquant_dequantize"];
        if (!fn_quant || !fn_deq) return false;

        quantize_pso = [device newComputePipelineStateWithFunction:fn_quant error:&err];
        if (!quantize_pso) return false;

        dequant_pso = [device newComputePipelineStateWithFunction:fn_deq error:&err];
        if (!dequant_pso) return false;

        queue = [device newCommandQueue];
        return (queue != nil);
    }
};

TQMetalState& metal_state() {
    static TQMetalState s;
    return s;
}

} // anonymous namespace

// =============================================================================
// BATCH QUANTIZE
// =============================================================================

bool turboquant_metal_batch_quantize(
    const float*  pi_data,
    const float*  input_data,
    uint8_t*      out_indices,
    float*        out_norms,
    const float*  boundaries,
    int           num_boundaries,
    int           N,
    int           d,
    int           bit_width)
{
    if (N <= 0 || d <= 0) return false;

    auto& st = metal_state();
    if (!st.init()) return false;

    id<MTLDevice> dev = st.device;

    // ── Create buffers ──────────────────────────────────────────────────
    const size_t pi_bytes   = static_cast<size_t>(d) * d * sizeof(float);
    const size_t in_bytes   = static_cast<size_t>(N) * d * sizeof(float);
    const size_t idx_bytes  = static_cast<size_t>(N) * d * sizeof(uint8_t);
    const size_t norm_bytes = static_cast<size_t>(N) * sizeof(float);
    const size_t bd_bytes   = static_cast<size_t>(num_boundaries) * sizeof(float);

    id<MTLBuffer> buf_pi  = [dev newBufferWithBytes:pi_data
                                              length:pi_bytes
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_in  = [dev newBufferWithBytes:input_data
                                              length:in_bytes
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_idx = [dev newBufferWithLength:idx_bytes
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_norm = [dev newBufferWithLength:norm_bytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_bd  = [dev newBufferWithBytes:boundaries
                                              length:bd_bytes
                                             options:MTLResourceStorageModeShared];

    TQMetalParams params;
    params.N              = static_cast<uint32_t>(N);
    params.d              = static_cast<uint32_t>(d);
    params.bit_width      = static_cast<uint32_t>(bit_width);
    params.num_boundaries = static_cast<uint32_t>(num_boundaries);
    params.num_centroids  = static_cast<uint32_t>(1 << bit_width);

    id<MTLBuffer> buf_params = [dev newBufferWithBytes:&params
                                                 length:sizeof(TQMetalParams)
                                                options:MTLResourceStorageModeShared];

    // ── Dispatch ────────────────────────────────────────────────────────
    id<MTLCommandBuffer> cmd = [st.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:st.quantize_pso];
    [enc setBuffer:buf_pi    offset:0 atIndex:0];
    [enc setBuffer:buf_in    offset:0 atIndex:1];
    [enc setBuffer:buf_idx   offset:0 atIndex:2];
    [enc setBuffer:buf_norm  offset:0 atIndex:3];
    [enc setBuffer:buf_bd    offset:0 atIndex:4];
    [enc setBuffer:buf_params offset:0 atIndex:5];

    // One threadgroup per vector, d threads per threadgroup
    NSUInteger tpg = static_cast<NSUInteger>(d);
    NSUInteger maxTpg = st.quantize_pso.maxTotalThreadsPerThreadgroup;
    if (tpg > maxTpg) {
        fprintf(stderr, "[TurboQuant] d=%d exceeds maxTotalThreadsPerThreadgroup=%lu, clamping\n",
                d, (unsigned long)maxTpg);
        tpg = maxTpg;
    }
    MTLSize threadgroupSize = MTLSizeMake(tpg, 1, 1);
    MTLSize gridSize        = MTLSizeMake(static_cast<NSUInteger>(N), 1, 1);

    [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if (cmd.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "[TurboQuant] Metal command buffer error: %s\n",
                cmd.error ? [[cmd.error localizedDescription] UTF8String] : "unknown");
        return false;
    }

    // ── Read back results ───────────────────────────────────────────────
    std::memcpy(out_indices, buf_idx.contents, idx_bytes);
    std::memcpy(out_norms,   buf_norm.contents, norm_bytes);

    return true;
}

// =============================================================================
// BATCH DEQUANTIZE
// =============================================================================

bool turboquant_metal_batch_dequantize(
    const float*   pit_data,
    const uint8_t* indices,
    const float*   centroids,
    float*         out_data,
    int            N,
    int            d,
    int            bit_width)
{
    if (N <= 0 || d <= 0) return false;

    auto& st = metal_state();
    if (!st.init()) return false;

    id<MTLDevice> dev = st.device;

    const size_t pit_bytes  = static_cast<size_t>(d) * d * sizeof(float);
    const size_t idx_bytes  = static_cast<size_t>(N) * d * sizeof(uint8_t);
    const int    k          = 1 << bit_width;
    const size_t cent_bytes = static_cast<size_t>(k) * sizeof(float);
    const size_t out_bytes  = static_cast<size_t>(N) * d * sizeof(float);

    id<MTLBuffer> buf_pit  = [dev newBufferWithBytes:pit_data
                                               length:pit_bytes
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_idx  = [dev newBufferWithBytes:indices
                                               length:idx_bytes
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_cent = [dev newBufferWithBytes:centroids
                                               length:cent_bytes
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_out  = [dev newBufferWithLength:out_bytes
                                               options:MTLResourceStorageModeShared];

    TQMetalParams params;
    params.N              = static_cast<uint32_t>(N);
    params.d              = static_cast<uint32_t>(d);
    params.bit_width      = static_cast<uint32_t>(bit_width);
    params.num_boundaries = static_cast<uint32_t>(k - 1);
    params.num_centroids  = static_cast<uint32_t>(k);

    id<MTLBuffer> buf_params = [dev newBufferWithBytes:&params
                                                 length:sizeof(TQMetalParams)
                                                options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmd = [st.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:st.dequant_pso];
    [enc setBuffer:buf_pit    offset:0 atIndex:0];
    [enc setBuffer:buf_idx    offset:0 atIndex:1];
    [enc setBuffer:buf_cent   offset:0 atIndex:2];
    [enc setBuffer:buf_out    offset:0 atIndex:3];
    [enc setBuffer:buf_params offset:0 atIndex:4];

    NSUInteger tpg = static_cast<NSUInteger>(d);
    NSUInteger maxTpg = st.dequant_pso.maxTotalThreadsPerThreadgroup;
    if (tpg > maxTpg) {
        fprintf(stderr, "[TurboQuant] dequant d=%d exceeds maxTotalThreadsPerThreadgroup=%lu, clamping\n",
                d, (unsigned long)maxTpg);
        tpg = maxTpg;
    }
    MTLSize threadgroupSize = MTLSizeMake(tpg, 1, 1);
    MTLSize gridSize        = MTLSizeMake(static_cast<NSUInteger>(N), 1, 1);

    [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if (cmd.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "[TurboQuant] dequant Metal command buffer error: %s\n",
                cmd.error ? [[cmd.error localizedDescription] UTF8String] : "unknown");
        return false;
    }

    std::memcpy(out_data, buf_out.contents, out_bytes);

    return true;
}

#else // FLEXAIDS_USE_METAL not defined

bool turboquant_metal_batch_quantize(
    const float*, const float*, uint8_t*, float*,
    const float*, int, int, int, int)
{
    return false;  // CPU path handles everything
}

bool turboquant_metal_batch_dequantize(
    const float*, const uint8_t*, const float*, float*,
    int, int, int)
{
    return false;
}

#endif // FLEXAIDS_USE_METAL
