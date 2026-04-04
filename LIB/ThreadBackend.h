// ThreadBackend.h — Single-process distributed backend (default)
//
// Implements the distributed::Backend interface for local execution.
// All operations are no-ops or trivial (rank=0, world_size=1).
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

namespace distributed {

// Forward-declare WorkResult and Backend from DistributedBackend.h
// (included via DistributedBackend.h before this file)

class ThreadBackend : public Backend {
public:
    void init(int* /*argc*/, char*** /*argv*/) override {}
    void finalize() override {}

    int rank() const override { return 0; }
    int world_size() const override { return 1; }

    void broadcast_config(std::string& /*config*/) override {
        // No-op: single process, config is already local
    }

    std::vector<WorkResult> gather_results(
        const std::vector<WorkResult>& local_results) override {
        return local_results; // single process: results are already gathered
    }

    std::string name() const override { return "thread"; }
};

} // namespace distributed
