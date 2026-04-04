// MPIBackend.h — MPI-based distributed docking backend
//
// Implements the distributed::Backend interface for MPI cluster environments.
// Only compiled when FLEXAIDS_USE_MPI is defined.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef FLEXAIDS_USE_MPI

#include <mpi.h>
#include <string>
#include <vector>
#include <cstring>

namespace distributed {

class MPIBackend : public Backend {
public:
    void init(int* argc, char*** argv) override {
        MPI_Init(argc, argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    }

    void finalize() override {
        MPI_Finalize();
    }

    int rank() const override { return rank_; }
    int world_size() const override { return world_size_; }

    void broadcast_config(std::string& config) override {
        int len = static_cast<int>(config.size());
        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank_ != 0) config.resize(len);
        MPI_Bcast(config.data(), len, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    std::vector<WorkResult> gather_results(
        const std::vector<WorkResult>& local_results) override {
        // Serialize local results: count + (region_id, energy, file_path_len, file_path)
        int local_count = static_cast<int>(local_results.size());
        std::vector<int> all_counts(world_size_);
        MPI_Gather(&local_count, 1, MPI_INT,
                   all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        // For simplicity, gather energies and region IDs as doubles/ints
        std::vector<double> local_energies(local_count);
        std::vector<int> local_ids(local_count);
        for (int i = 0; i < local_count; ++i) {
            local_energies[i] = local_results[i].best_energy;
            local_ids[i] = local_results[i].region_id;
        }

        // Gather on rank 0
        std::vector<WorkResult> all_results;
        if (rank_ == 0) {
            int total = 0;
            std::vector<int> displs(world_size_);
            for (int r = 0; r < world_size_; ++r) {
                displs[r] = total;
                total += all_counts[r];
            }

            std::vector<double> all_energies(total);
            std::vector<int> all_ids(total);
            MPI_Gatherv(local_energies.data(), local_count, MPI_DOUBLE,
                        all_energies.data(), all_counts.data(), displs.data(),
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(local_ids.data(), local_count, MPI_INT,
                        all_ids.data(), all_counts.data(), displs.data(),
                        MPI_INT, 0, MPI_COMM_WORLD);

            all_results.resize(total);
            for (int i = 0; i < total; ++i) {
                all_results[i].region_id = all_ids[i];
                all_results[i].best_energy = all_energies[i];
            }
        } else {
            MPI_Gatherv(local_energies.data(), local_count, MPI_DOUBLE,
                        nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(local_ids.data(), local_count, MPI_INT,
                        nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
        }

        return all_results;
    }

    std::string name() const override { return "mpi"; }

private:
    int rank_ = 0;
    int world_size_ = 1;
};

} // namespace distributed

#endif // FLEXAIDS_USE_MPI
