// cpu_fast_optics.h — CPU fallback for k-nearest-neighbour search (FastOPTICS)
//
// Provides the same API as gpu_fast_optics.cu but runs on the CPU with
// OpenMP parallelism.  Used when no GPU backend (CUDA) is available.
//
// Apache-2.0 © 2026 Le Bonhomme Pharma
#pragma once

#include "FOPTICS.h"
#include <vector>
#include <utility>

// CPU k-NN search for FastOPTICS clustering.
//
// points:        vector of (chromosome*, coordinates) pairs
// k:             number of nearest neighbours to find
// nDim:          dimensionality of each point
// out_neighbors: [N][variable] neighbour indices  (resized and filled)
// out_distances: [N][k] neighbour distances        (resized and filled)
void cpu_foptics_knn(
    const std::vector<std::pair<chromosome*, std::vector<float>>>& points,
    int k, int nDim,
    std::vector<std::vector<int>>& out_neighbors,
    std::vector<std::vector<float>>& out_distances);
