/// @file MPITransport.h
/// @brief MPI communication layer for distributed parallel docking.
///
/// Conditional compilation: only active when FLEXAIDS_USE_MPI is defined.
/// When MPI is unavailable, provides no-op stubs.
///
/// Supports generic exchange of any type satisfying the MPIMergeable concept
/// (serialize() + deserialize_merge()), eliminating type-specific boilerplate.
///
/// Copyright 2026 Le Bonhomme Pharma
/// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "SharedPosePool.h"
#include "ParallelDock.h"
#include "MPIMergeable.h"
#include <vector>

class MPITransport {
public:
    static void init(int* argc, char*** argv);
    static void finalize();
    static int rank();
    static int world_size();

    /// Broadcast grid regions from rank 0 to all ranks.
    static void broadcast_regions(std::vector<GridRegion>& regions);

    /// Allgather shared pose pool across all ranks (legacy, type-specific).
    static void exchange_poses(SharedPosePool& local_pool);

    /// Generic Allgather exchange for any MPIMergeable type.
    ///
    /// Each rank serializes its local accumulator, performs an MPI_Allgatherv,
    /// then deserialize_merges all remote buffers into the local accumulator.
    /// No-op when MPI is unavailable (single-rank stub).
    ///
    /// @tparam T Any type satisfying transport::MPIMergeable.
    /// @param local The local accumulator to exchange.
    template <transport::MPIMergeable T>
    static void exchange(T& local);

    /// Gather region results from all ranks to rank 0.
    static std::vector<RegionResult> gather_results(
        const std::vector<RegionResult>& local_results,
        int total_regions);
};

// ── Template implementation ────────────────────────────────────────────

#ifdef FLEXAIDS_USE_MPI
#include <mpi.h>

template <transport::MPIMergeable T>
void MPITransport::exchange(T& local) {
    auto local_buf = local.serialize();
    int local_size = static_cast<int>(local_buf.size());

    int world = world_size();
    std::vector<int> sizes(static_cast<size_t>(world));
    MPI_Allgather(&local_size, 1, MPI_INT,
                  sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> displs(static_cast<size_t>(world));
    int total = 0;
    for (int i = 0; i < world; ++i) {
        displs[static_cast<size_t>(i)] = total;
        total += sizes[static_cast<size_t>(i)];
    }

    std::vector<char> all_buf(static_cast<size_t>(total));
    MPI_Allgatherv(local_buf.data(), local_size, MPI_CHAR,
                   all_buf.data(), sizes.data(), displs.data(), MPI_CHAR,
                   MPI_COMM_WORLD);

    int my_rank = rank();
    for (int i = 0; i < world; ++i) {
        if (i == my_rank) continue;
        local.deserialize_merge(
            all_buf.data() + displs[static_cast<size_t>(i)],
            static_cast<size_t>(sizes[static_cast<size_t>(i)]));
    }
}

#else
// No-MPI stub: exchange is a no-op on a single rank.
template <transport::MPIMergeable T>
void MPITransport::exchange(T& /*local*/) {}
#endif
