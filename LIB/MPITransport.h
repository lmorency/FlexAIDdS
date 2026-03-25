// MPITransport.h — MPI communication layer for distributed parallel docking
//
// Conditional compilation: only active when FLEXAIDS_USE_MPI is defined.
// When MPI is unavailable, provides no-op stubs.
#pragma once

#include "SharedPosePool.h"
#include "ParallelDock.h"
#include <vector>

class MPITransport {
public:
    static void init(int* argc, char*** argv);
    static void finalize();
    static int rank();
    static int world_size();

    // Broadcast grid regions from rank 0 to all ranks
    static void broadcast_regions(std::vector<GridRegion>& regions);

    // Allgather shared pose pool across all ranks
    static void exchange_poses(SharedPosePool& local_pool);

    // Gather region results from all ranks to rank 0
    static std::vector<RegionResult> gather_results(
        const std::vector<RegionResult>& local_results,
        int total_regions);
};
