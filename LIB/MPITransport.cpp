// MPITransport.cpp — MPI communication for distributed parallel docking
//
// When FLEXAIDS_USE_MPI is not defined, provides single-rank stubs.

#include "MPITransport.h"
#include <cstdio>
#include <cstring>

#ifdef FLEXAIDS_USE_MPI
#include <mpi.h>

void MPITransport::init(int* argc, char*** argv) {
    MPI_Init(argc, argv);
}

void MPITransport::finalize() {
    MPI_Finalize();
}

int MPITransport::rank() {
    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
}

int MPITransport::world_size() {
    int s;
    MPI_Comm_size(MPI_COMM_WORLD, &s);
    return s;
}

void MPITransport::broadcast_regions(std::vector<GridRegion>& regions) {
    int n = (int)regions.size();
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank() != 0) regions.resize(n);

    // Broadcast each region's data
    for (int i = 0; i < n; i++) {
        MPI_Bcast(&regions[i].region_id, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(regions[i].center, 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&regions[i].radius, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&regions[i].num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int nidx = (int)regions[i].grid_indices.size();
        MPI_Bcast(&nidx, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank() != 0) regions[i].grid_indices.resize(nidx);
        MPI_Bcast(regions[i].grid_indices.data(), nidx, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void MPITransport::exchange_poses(SharedPosePool& local_pool) {
    auto local_buf = local_pool.serialize();
    int local_size = (int)local_buf.size();

    // Gather sizes
    int world = world_size();
    std::vector<int> sizes(world);
    MPI_Allgather(&local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Compute displacements
    std::vector<int> displs(world);
    int total = 0;
    for (int i = 0; i < world; i++) {
        displs[i] = total;
        total += sizes[i];
    }

    // Allgatherv
    std::vector<char> all_buf(total);
    MPI_Allgatherv(local_buf.data(), local_size, MPI_CHAR,
                   all_buf.data(), sizes.data(), displs.data(), MPI_CHAR,
                   MPI_COMM_WORLD);

    // Merge remote pools into local
    int my_rank = rank();
    for (int i = 0; i < world; i++) {
        if (i == my_rank) continue;
        local_pool.deserialize_merge(all_buf.data() + displs[i], sizes[i]);
    }
}

std::vector<RegionResult> MPITransport::gather_results(
    const std::vector<RegionResult>& local_results,
    int total_regions)
{
    // Simplified: serialize energies and gather
    // Each rank sends its results as serialized energy arrays
    int my_rank = rank();
    int world = world_size();

    // For each local result, serialize
    int n_local = (int)local_results.size();

    // Gather counts
    std::vector<int> counts(world);
    MPI_Allgather(&n_local, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // For simplicity, gather full RegionResult data via serialized doubles
    // In production, this would use a custom MPI datatype
    std::vector<RegionResult> all_results(total_regions);

    // Copy local results
    for (const auto& r : local_results) {
        if (r.region_id >= 0 && r.region_id < total_regions)
            all_results[r.region_id] = r;
    }

    // Broadcast each rank's results to all
    // (simplified all-to-all — production would use targeted sends)
    for (int r = 0; r < total_regions; r++) {
        int owner = r % world;  // round-robin assignment
        int n_energies = (int)all_results[r].energies.size();
        MPI_Bcast(&n_energies, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (n_energies > 0) {
            if (my_rank != owner) all_results[r].energies.resize(n_energies);
            MPI_Bcast(all_results[r].energies.data(), n_energies, MPI_DOUBLE,
                      owner, MPI_COMM_WORLD);
            if (my_rank != owner) all_results[r].multiplicities.resize(n_energies, 1);
        }
        MPI_Bcast(&all_results[r].best_energy, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        MPI_Bcast(all_results[r].best_coor, 3, MPI_FLOAT, owner, MPI_COMM_WORLD);
    }

    return all_results;
}

#else
// ── No-MPI stubs ────────────────────────────────────────────────────────────

void MPITransport::init(int*, char***) {}
void MPITransport::finalize() {}
int  MPITransport::rank() { return 0; }
int  MPITransport::world_size() { return 1; }
void MPITransport::broadcast_regions(std::vector<GridRegion>&) {}
void MPITransport::exchange_poses(SharedPosePool&) {}

std::vector<RegionResult> MPITransport::gather_results(
    const std::vector<RegionResult>& local_results, int)
{
    return local_results;
}

#endif // FLEXAIDS_USE_MPI
