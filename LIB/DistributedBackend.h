// DistributedBackend.h — Abstract interface for distributed docking backends
//
// Decouples the docking engine from specific distributed computing
// implementations (Apple Fleet, MPI, single-process threads).
//
// Concrete implementations:
//   ThreadBackend.h  — Single-process OpenMP (default)
//   MPIBackend.h     — MPI cluster backend (requires FLEXAIDS_USE_MPI)
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <memory>

namespace distributed {

// Result from a single distributed work unit
struct WorkResult {
    int region_id;
    double best_energy;
    std::string result_file;  // path to output PDB
};

// Abstract interface for distributed computing backends
class Backend {
public:
    virtual ~Backend() = default;

    // Initialize the backend (e.g., MPI_Init)
    virtual void init(int* argc, char*** argv) = 0;

    // Finalize the backend (e.g., MPI_Finalize)
    virtual void finalize() = 0;

    // Return the rank (process ID) of this worker
    virtual int rank() const = 0;

    // Return the total number of workers
    virtual int world_size() const = 0;

    // Broadcast a configuration string from rank 0 to all workers
    virtual void broadcast_config(std::string& config) = 0;

    // Gather results from all workers to rank 0
    virtual std::vector<WorkResult> gather_results(
        const std::vector<WorkResult>& local_results) = 0;

    // Human-readable backend name
    virtual std::string name() const = 0;
};

// Factory function to create a backend by name
// Valid names: "thread" (default), "mpi"
inline std::unique_ptr<Backend> create_backend(const std::string& type);

} // namespace distributed

// ─── ThreadBackend: single-process fallback ─────────────────────────────────

#include "ThreadBackend.h"

#ifdef FLEXAIDS_USE_MPI
#include "MPIBackend.h"
#endif

namespace distributed {

inline std::unique_ptr<Backend> create_backend(const std::string& type) {
#ifdef FLEXAIDS_USE_MPI
    if (type == "mpi") {
        return std::make_unique<MPIBackend>();
    }
#endif
    // Default: single-process thread backend
    return std::make_unique<ThreadBackend>();
}

} // namespace distributed
