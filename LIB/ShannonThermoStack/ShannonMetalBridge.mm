// ShannonMetalBridge.mm — Objective-C++ bridge to Metal GPU Shannon kernel
//
// Compiled only on APPLE targets with FLEXAIDS_HAS_METAL_SHANNON defined.
// Dispatches a parallel histogram kernel on the GPU, then computes Shannon
// entropy on the CPU from the resulting bin counts.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "ShannonMetalBridge.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace ShannonMetalBridge {

// CPU fallback Shannon entropy from bin counts
static double cpu_shannon_from_bins(const std::vector<int>& bins) {
    int total = 0;
    for (int c : bins) total += c;
    if (total == 0) return 0.0;

    double H = 0.0;
    for (int c : bins) {
        if (c > 0) {
            double p = static_cast<double>(c) / total;
            H -= p * std::log(p);
        }
    }
    return H;
}

double compute_shannon_entropy_metal(const std::vector<double>& energies,
                                     int num_bins)
{
    if (energies.empty()) return 0.0;
    if (num_bins <= 0)    num_bins = 20;

    // ── attempt Metal device ──────────────────────────────────────────────────
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        // No Metal device — CPU fallback
        double min_v = *std::min_element(energies.begin(), energies.end());
        double max_v = *std::max_element(energies.begin(), energies.end());
        if (max_v - min_v < 1e-12) return 0.0;
        double bw = (max_v - min_v) / num_bins + 1e-10;
        std::vector<int> bins(num_bins, 0);
        for (double e : energies) {
            int b = std::min(std::max((int)((e - min_v) / bw), 0), num_bins - 1);
            bins[b]++;
        }
        return cpu_shannon_from_bins(bins);
    }

    // ── compile the Metal library from default library (shannon_metal.metal) ──
    NSError* err = nil;
    id<MTLLibrary> library = [device newDefaultLibrary];
    if (!library) {
        // Library not embedded (non-macOS build system); CPU fallback
        double min_v = *std::min_element(energies.begin(), energies.end());
        double max_v = *std::max_element(energies.begin(), energies.end());
        if (max_v - min_v < 1e-12) return 0.0;
        double bw = (max_v - min_v) / num_bins + 1e-10;
        std::vector<int> bins(num_bins, 0);
        for (double e : energies) {
            int b = std::min(std::max((int)((e - min_v) / bw), 0), num_bins - 1);
            bins[b]++;
        }
        return cpu_shannon_from_bins(bins);
    }

    id<MTLFunction>             function = [library newFunctionWithName:@"shannon_histogram"];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&err];
    if (!pipeline || err) return 0.0;

    id<MTLCommandQueue> queue = [device newCommandQueue];

    // ── compute binning parameters ────────────────────────────────────────────
    NSUInteger n      = energies.size();
    double     min_v  = *std::min_element(energies.begin(), energies.end());
    double     max_v  = *std::max_element(energies.begin(), energies.end());
    double     bw     = (max_v - min_v) / num_bins + 1e-10;

    // ── buffers ───────────────────────────────────────────────────────────────
    id<MTLBuffer> energy_buf = [device newBufferWithBytes:energies.data()
                                                   length:n * sizeof(double)
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> bin_buf    = [device newBufferWithLength:num_bins * sizeof(int)
                                                   options:MTLResourceStorageModeShared];
    memset(bin_buf.contents, 0, num_bins * sizeof(int));

    // ── encode and dispatch ───────────────────────────────────────────────────
    id<MTLCommandBuffer>        cmd_buf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc    = [cmd_buf computeCommandEncoder];

    [enc setComputePipelineState:pipeline];
    [enc setBuffer:energy_buf offset:0 atIndex:0];
    [enc setBuffer:bin_buf    offset:0 atIndex:1];
    [enc setBytes:&n          length:sizeof(NSUInteger) atIndex:2];
    [enc setBytes:&num_bins   length:sizeof(int)        atIndex:3];
    [enc setBytes:&min_v      length:sizeof(double)     atIndex:4];
    [enc setBytes:&bw         length:sizeof(double)     atIndex:5];

    MTLSize tpg = MTLSizeMake(256, 1, 1);
    MTLSize ng  = MTLSizeMake((n + 255) / 256, 1, 1);
    [enc dispatchThreadgroups:ng threadsPerThreadgroup:tpg];
    [enc endEncoding];
    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];

    // ── read back bin counts and compute entropy on CPU ───────────────────────
    int* bin_data = static_cast<int*>(bin_buf.contents);
    std::vector<int> bins(bin_data, bin_data + num_bins);
    return cpu_shannon_from_bins(bins);
}

} // namespace ShannonMetalBridge
