// CavityDetect.metal — GPU probe-sphere generation (Metal 3, Apple Silicon)
// Each thread processes one atom-pair (i, j): midpoint probe + KWALL clash rejection.
// LP's Get_Cleft logic — native, parallel, zero GPL contact.
// Apache-2.0 © 2026 Le Bonhomme Pharma

#include <metal_stdlib>
using namespace metal;

// ─── data types (must match CavityDetectMetalBridge.mm) ─────────────────────

struct GPUAtom {
    float3 pos;
    float  radius;
};

struct GPUSphere {
    float3 center;
    float  radius;
    int    cleft_id;  // set to 0 here; clustered on CPU
    int    _pad;
};

struct DetectParams {
    uint  n_atoms;
    float min_radius;
    float max_radius;
    float kwall;      // clash softening (currently 0 = hard rejection)
};

// ─── probe generation kernel ────────────────────────────────────────────────

kernel void generate_probes(
    device const GPUAtom*  atoms         [[ buffer(0) ]],
    device       GPUSphere* spheres      [[ buffer(1) ]],
    device atomic_int*      sphere_count [[ buffer(2) ]],
    constant DetectParams&  params       [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]])
{
    // Map linear thread index to unique atom pair (i, j), i < j.
    // Total threads dispatched = N*(N-1)/2 rounded up.
    const uint N = params.n_atoms;

    // Inverse triangular-number mapping: find i,j from gid
    // Using the formula: i = floor((2N - 1 - sqrt((2N-1)^2 - 8*gid)) / 2)
    uint i = uint(floor((2.0f * float(N) - 1.0f
                         - sqrt((2.0f * float(N) - 1.0f) * (2.0f * float(N) - 1.0f)
                                - 8.0f * float(gid)))
                        * 0.5f));
    uint j = gid - (i * (2u * N - i - 1u)) / 2u + i + 1u;

    if (i >= N || j >= N || i >= j) return;

    const float3 pi = atoms[i].pos;
    const float3 pj = atoms[j].pos;
    const float  ri = atoms[i].radius;
    const float  rj = atoms[j].radius;

    float3 mid = (pi + pj) * 0.5f;
    float  d   = distance(pi, pj);

    // Initial probe radius: half the surface-to-surface gap
    float r = d * 0.5f - 0.5f * (ri + rj);
    if (r < params.min_radius) return;
    if (r > params.max_radius) r = params.max_radius;

    // SURFNET clash rejection: shrink r to fit between all other atoms.
    // Inner loop over all atoms — this is the hot path that Metal parallelises.
    for (uint k = 0u; k < N; ++k) {
        if (k == i || k == j) continue;
        float gap = distance(mid, atoms[k].pos) - atoms[k].radius;
        if (gap < r) r = gap;
    }

    if (r < params.min_radius) return;
    if (r > params.max_radius) r = params.max_radius;

    // Atomically claim an output slot (cap at 65536 spheres per run)
    int idx = atomic_fetch_add_explicit(sphere_count, 1, memory_order_relaxed);
    if (idx < 65536) {
        GPUSphere s;
        s.center   = mid;
        s.radius   = r;
        s.cleft_id = 0;
        s._pad     = 0;
        spheres[idx] = s;
    }
}
