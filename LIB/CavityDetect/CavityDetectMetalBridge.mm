// CavityDetectMetalBridge.mm — Objective-C++ dispatch for CavityDetect.metal
// Bridges C++ CavityDetector::detect() to the Metal GPU kernel on Apple Silicon.
// Apache-2.0 © 2026 Le Bonhomme Pharma

#include "CavityDetectMetalBridge.h"

#ifdef FLEXAIDS_USE_METAL

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <vector>
#include <cstring>

// GPU-side struct layout (must match CavityDetect.metal)
struct GPUAtom {
    float pos[3];
    float radius;
};

struct GPUSphere {
    float center[3];
    float radius;
    int   cleft_id;
    int   _pad;
};

struct DetectParams {
    unsigned int n_atoms;
    float        min_radius;
    float        max_radius;
    float        kwall;
};

// Maximum output spheres per detect() call
static constexpr int kMaxSpheres = 65536;

bool cavity_detect_metal_dispatch(
    const cavity_detect::MetalAtom* atoms,
    int n_atoms,
    float min_radius,
    float max_radius,
    std::vector<cavity_detect::MetalSphereResult>& out_spheres)
{
    out_spheres.clear();
    if (n_atoms < 2) return false;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return false;

    // ── Load .metallib (path injected by CMake at build time) ──────────────
    NSString* metallib_path = @CAVITY_METALLIB_PATH;
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithURL:[NSURL fileURLWithPath:metallib_path]
                                             error:&err];
    if (!lib) {
        // Fallback: compile from embedded source (dev builds without metallib)
        NSLog(@"[CavityDetect] metallib not found at %@, using inline compile: %@",
              metallib_path, err.localizedDescription);
        return false;  // CPU path will handle it
    }

    id<MTLFunction> fn = [lib newFunctionWithName:@"generate_probes"];
    if (!fn) return false;

    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pipeline) return false;

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) return false;

    // ── Buffers ─────────────────────────────────────────────────────────────

    // Atom buffer
    std::vector<GPUAtom> gpu_atoms(static_cast<std::size_t>(n_atoms));
    for (int i = 0; i < n_atoms; ++i) {
        gpu_atoms[static_cast<std::size_t>(i)].pos[0] = atoms[i].pos[0];
        gpu_atoms[static_cast<std::size_t>(i)].pos[1] = atoms[i].pos[1];
        gpu_atoms[static_cast<std::size_t>(i)].pos[2] = atoms[i].pos[2];
        gpu_atoms[static_cast<std::size_t>(i)].radius = atoms[i].radius;
    }
    id<MTLBuffer> atomBuf = [device newBufferWithBytes:gpu_atoms.data()
                                                length:gpu_atoms.size() * sizeof(GPUAtom)
                                               options:MTLResourceStorageModeShared];

    // Sphere output buffer (pre-allocated, max kMaxSpheres)
    id<MTLBuffer> sphereBuf = [device newBufferWithLength:kMaxSpheres * sizeof(GPUSphere)
                                                  options:MTLResourceStorageModeShared];
    std::memset(sphereBuf.contents, 0, kMaxSpheres * sizeof(GPUSphere));

    // Atomic counter (starts at 0)
    int zero = 0;
    id<MTLBuffer> countBuf = [device newBufferWithBytes:&zero
                                                 length:sizeof(int)
                                                options:MTLResourceStorageModeShared];

    // Parameters
    DetectParams params{
        static_cast<unsigned int>(n_atoms),
        min_radius,
        max_radius,
        0.0f,  // kwall (hard rejection)
    };
    id<MTLBuffer> paramBuf = [device newBufferWithBytes:&params
                                                 length:sizeof(DetectParams)
                                                options:MTLResourceStorageModeShared];

    // ── Dispatch ─────────────────────────────────────────────────────────────
    // Total pairs = N*(N-1)/2
    const long long N   = n_atoms;
    const long long n_pairs = N * (N - 1) / 2;

    id<MTLCommandBuffer>      cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:pipeline];
    [enc setBuffer:atomBuf   offset:0 atIndex:0];
    [enc setBuffer:sphereBuf offset:0 atIndex:1];
    [enc setBuffer:countBuf  offset:0 atIndex:2];
    [enc setBuffer:paramBuf  offset:0 atIndex:3];

    const NSUInteger tpg = pipeline.maxTotalThreadsPerThreadgroup;
    MTLSize grid      = MTLSizeMake(static_cast<NSUInteger>(n_pairs), 1, 1);
    MTLSize threadgrp = MTLSizeMake(tpg, 1, 1);

    [enc dispatchThreads:grid threadsPerThreadgroup:threadgrp];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    // ── Read back results ────────────────────────────────────────────────────
    int n_out = *static_cast<int*>(countBuf.contents);
    if (n_out > kMaxSpheres) n_out = kMaxSpheres;

    const GPUSphere* raw = static_cast<const GPUSphere*>(sphereBuf.contents);
    out_spheres.reserve(static_cast<std::size_t>(n_out));
    for (int s = 0; s < n_out; ++s) {
        cavity_detect::MetalSphereResult r;
        r.center[0] = raw[s].center[0];
        r.center[1] = raw[s].center[1];
        r.center[2] = raw[s].center[2];
        r.radius    = raw[s].radius;
        out_spheres.push_back(r);
    }

    return true;
}

#else // FLEXAIDS_USE_METAL not defined

bool cavity_detect_metal_dispatch(
    const cavity_detect::MetalAtom*,
    int,
    float, float,
    std::vector<cavity_detect::MetalSphereResult>&)
{
    return false;  // CPU paths handle everything
}

#endif // FLEXAIDS_USE_METAL
