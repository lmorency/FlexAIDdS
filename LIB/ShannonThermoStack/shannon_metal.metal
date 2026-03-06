// shannon_metal.metal — GPU histogram kernel for Shannon entropy computation
//
// Performance design:
//   – 256 threads per threadgroup (optimal for Apple Silicon SIMD width)
//   – threadgroup-local histogram (reduces global atomic contention ~32×)
//   – Final merge via atomic_fetch_add to global bin buffer
//   – Supports arbitrary n (out-of-bounds threads are no-ops)
#include <metal_stdlib>
using namespace metal;

kernel void shannon_histogram(
    device const double*     energies  [[buffer(0)]],
    device atomic_int*       bins      [[buffer(1)]],
    constant uint&           n         [[buffer(2)]],
    constant int&            num_bins  [[buffer(3)]],
    constant double&         min_v     [[buffer(4)]],
    constant double&         bin_width [[buffer(5)]],
    uint                     gid       [[thread_position_in_grid]],
    uint                     lid       [[thread_position_in_threadgroup]],
    uint                     tg_size   [[threads_per_threadgroup]])
{
    // Threadgroup-local histogram to reduce global atomic pressure
    threadgroup atomic_int local_bins[256]; // max num_bins = 256
    if (lid < (uint)num_bins)
        atomic_store_explicit(&local_bins[lid], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bin this thread's element
    if (gid < n) {
        double e = energies[gid];
        int b = (int)((e - min_v) / bin_width);
        b = clamp(b, 0, num_bins - 1);
        atomic_fetch_add_explicit(&local_bins[b], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Merge threadgroup histogram to global (only threads 0..num_bins-1)
    if (lid < (uint)num_bins) {
        int local_count = atomic_load_explicit(&local_bins[lid], memory_order_relaxed);
        if (local_count > 0)
            atomic_fetch_add_explicit(&bins[lid], local_count, memory_order_relaxed);
    }
}
