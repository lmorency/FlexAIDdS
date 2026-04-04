// test_distributed_backend.cpp — Unit tests for distributed backend abstraction
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "DistributedBackend.h"

TEST(DistributedBackendTest, ThreadBackendDefaults) {
    auto backend = distributed::create_backend("thread");
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->name(), "thread");
    EXPECT_EQ(backend->rank(), 0);
    EXPECT_EQ(backend->world_size(), 1);
}

TEST(DistributedBackendTest, ThreadBackendInitFinalize) {
    auto backend = distributed::create_backend("thread");
    int argc = 0;
    char** argv = nullptr;
    // Should be no-ops, not crash
    backend->init(&argc, &argv);
    backend->finalize();
}

TEST(DistributedBackendTest, ThreadBackendBroadcast) {
    auto backend = distributed::create_backend("thread");
    std::string config = "{\"test\": true}";
    // Should be a no-op (single process)
    backend->broadcast_config(config);
    EXPECT_EQ(config, "{\"test\": true}");
}

TEST(DistributedBackendTest, ThreadBackendGather) {
    auto backend = distributed::create_backend("thread");
    std::vector<distributed::WorkResult> local = {
        {0, -42.5, "result_0.pdb"},
        {1, -38.2, "result_1.pdb"},
    };

    auto gathered = backend->gather_results(local);
    ASSERT_EQ(gathered.size(), 2u);
    EXPECT_EQ(gathered[0].region_id, 0);
    EXPECT_DOUBLE_EQ(gathered[0].best_energy, -42.5);
    EXPECT_EQ(gathered[1].result_file, "result_1.pdb");
}

TEST(DistributedBackendTest, DefaultBackendIsThread) {
    auto backend = distributed::create_backend("unknown");
    EXPECT_EQ(backend->name(), "thread");
}

#ifdef FLEXAIDS_USE_MPI
TEST(DistributedBackendTest, MPIBackendCreation) {
    auto backend = distributed::create_backend("mpi");
    EXPECT_EQ(backend->name(), "mpi");
}
#endif
