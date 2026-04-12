// cpu_fast_optics.cpp — CPU fallback for k-nearest-neighbour (FastOPTICS)
//
// Uses OpenMP to parallelise per-query-point k-NN search.
// Produces results equivalent to gpu_fast_optics.cu.
//
// Apache-2.0 © 2026 Le Bonhomme Pharma

#include "cpu_fast_optics.h"
#include <algorithm>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

void cpu_foptics_knn(
    const std::vector<std::pair<chromosome*, std::vector<float>>>& points,
    int k, int nDim,
    std::vector<std::vector<int>>& out_neighbors,
    std::vector<std::vector<float>>& out_distances)
{
    const int N = static_cast<int>(points.size());
    if (N == 0 || k <= 0) return;

    // Flatten points into row-major array for cache-friendly access
    std::vector<float> flat(static_cast<size_t>(N) * nDim, 0.0f);
    for (int i = 0; i < N; ++i) {
        const auto& coords = points[i].second;
        for (int d = 0; d < nDim; ++d) {
            flat[static_cast<size_t>(i) * nDim + d] =
                (d < static_cast<int>(coords.size())) ? coords[d] : 0.0f;
        }
    }

    out_neighbors.resize(N);
    out_distances.resize(N);

    // Per-query k-NN: O(N*k) per query point using partial sort
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 4)
#endif
    for (int q = 0; q < N; ++q) {
        const float* qp = &flat[static_cast<size_t>(q) * nDim];

        // Compute distances to all other points
        struct DistIdx { float dist; int idx; };
        std::vector<DistIdx> candidates;
        candidates.reserve(N - 1);

        for (int p = 0; p < N; ++p) {
            if (p == q) continue;
            const float* pp = &flat[static_cast<size_t>(p) * nDim];
            float dist2 = 0.0f;
            for (int d = 0; d < nDim; ++d) {
                float diff = qp[d] - pp[d];
                dist2 += diff * diff;
            }
            candidates.push_back({std::sqrt(dist2), p});
        }

        // Partial sort to get k nearest
        const int actual_k = std::min(k, static_cast<int>(candidates.size()));
        std::partial_sort(candidates.begin(),
                          candidates.begin() + actual_k,
                          candidates.end(),
                          [](const DistIdx& a, const DistIdx& b) {
                              return a.dist < b.dist;
                          });

        out_neighbors[q].resize(actual_k);
        out_distances[q].resize(actual_k);
        for (int i = 0; i < actual_k; ++i) {
            out_neighbors[q][i] = candidates[i].idx;
            out_distances[q][i] = candidates[i].dist;
        }
    }
}
