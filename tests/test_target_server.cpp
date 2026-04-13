// test_target_server.cpp — Integration tests for TargetServer
//
// Tests session management, grand partition function accumulation,
// and knowledge base updates through the TargetServer interface.
//
// Copyright 2026 Le Bonhomme Pharma
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "TargetServer.h"

#include <cstring>
#include <vector>
#include <thread>
#include <cmath>

using namespace target;

// ════════════════════════════════════════════════════════════════════════
// Helper: create synthetic FA_Global + atoms for validation tests
// ════════════════════════════════════════════════════════════════════════

static FA_Global make_test_fa(int atm_cnt = 50, int res_cnt = 50, int ntypes = 10)
{
    FA_Global fa{};
    fa.atm_cnt_real = atm_cnt;
    fa.atm_cnt = atm_cnt;
    fa.res_cnt = res_cnt;
    fa.ntypes = ntypes;
    fa.multi_model = false;
    fa.n_models = 1;
    return fa;
}

static std::vector<atom> make_atoms(int count)
{
    std::vector<atom> atoms(count);
    for (int i = 0; i < count; ++i) {
        std::memset(&atoms[i], 0, sizeof(atom));
        atoms[i].coor[0] = i * 3.8f;
        atoms[i].coor[1] = 0.0f;
        atoms[i].coor[2] = 0.0f;
        atoms[i].type = 1;
        atoms[i].ofres = i; // link atom to residue i
        std::strncpy(atoms[i].name, " CA ", 4);
        atoms[i].name[4] = '\0';
    }
    return atoms;
}

static std::vector<resid> make_residues(int count)
{
    std::vector<resid> residues(count);
    for (int i = 0; i < count; ++i) {
        std::memset(&residues[i], 0, sizeof(resid));
        std::strncpy(residues[i].name, "ALA", 3);
        residues[i].chn = 'A';
        residues[i].number = i + 1;
    }
    return residues;
}

// ════════════════════════════════════════════════════════════════════════
// Construction
// ════════════════════════════════════════════════════════════════════════

TEST(TargetServer, DefaultConstruction) {
    TargetServer server;
    EXPECT_NEAR(server.temperature(), 300.0, 1e-10);
    EXPECT_EQ(server.n_models(), 1);
    EXPECT_EQ(server.completed_sessions(), 0);
}

TEST(TargetServer, CustomConfig) {
    TargetConfig cfg;
    cfg.temperature_K = 310.0;
    cfg.n_models = 5;
    TargetServer server(cfg);
    EXPECT_NEAR(server.temperature(), 310.0, 1e-10);
    EXPECT_EQ(server.n_models(), 5);
}

// ════════════════════════════════════════════════════════════════════════
// Validation delegation
// ════════════════════════════════════════════════════════════════════════

TEST(TargetServer, ValidateGoodTarget) {
    TargetServer server;
    auto atoms = make_atoms(50);
    auto residues = make_residues(50);
    FA_Global fa = make_test_fa();

    auto result = server.validate(&fa, atoms.data(), residues.data(), 100);
    EXPECT_TRUE(result.valid);
    EXPECT_TRUE(result.errors.empty());
}

TEST(TargetServer, ValidateBadTarget) {
    TargetServer server;
    auto result = server.validate(nullptr, nullptr, nullptr, 0);
    EXPECT_FALSE(result.valid);
}

// ════════════════════════════════════════════════════════════════════════
// Session management
// ════════════════════════════════════════════════════════════════════════

TEST(TargetServer, CreateSession) {
    TargetServer server;
    auto s1 = server.create_session("aspirin");
    auto s2 = server.create_session("ibuprofen");

    EXPECT_EQ(s1.session_id, 0);
    EXPECT_EQ(s2.session_id, 1);
    EXPECT_EQ(s1.ligand_name, "aspirin");
    EXPECT_EQ(s2.ligand_name, "ibuprofen");
    EXPECT_FALSE(s1.completed);
    EXPECT_FALSE(s2.completed);
}

TEST(TargetServer, RegisterResult) {
    TargetServer server;
    auto session = server.create_session("aspirin");

    // Simulate completed docking
    session.completed = true;
    session.log_Z = 10.0;
    session.n_poses = 100;
    session.best_energy = -8.5;
    session.best_center[0] = 1.0f;
    session.best_center[1] = 2.0f;
    session.best_center[2] = 3.0f;

    server.register_result(session);

    EXPECT_EQ(server.completed_sessions(), 1);
    EXPECT_TRUE(server.grand_partition().has_ligand("aspirin"));
}

TEST(TargetServer, SkipIncompleteSession) {
    TargetServer server;
    auto session = server.create_session("aspirin");
    session.completed = false;

    server.register_result(session);  // should be ignored
    EXPECT_EQ(server.completed_sessions(), 0);
    EXPECT_FALSE(server.grand_partition().has_ligand("aspirin"));
}

// ════════════════════════════════════════════════════════════════════════
// Grand partition function through TargetServer
// ════════════════════════════════════════════════════════════════════════

TEST(TargetServer, CompetitiveBinding) {
    TargetServer server;

    // Register 3 ligands with different affinities
    auto dock = [&](const std::string& name, double log_Z) {
        auto s = server.create_session(name);
        s.completed = true;
        s.log_Z = log_Z;
        s.n_poses = 50;
        server.register_result(s);
    };

    dock("strong", 20.0);
    dock("medium", 10.0);
    dock("weak",    2.0);

    // Ranking
    auto ranks = server.rank_ligands();
    ASSERT_EQ(ranks.size(), 3u);
    EXPECT_EQ(ranks[0].name, "strong");
    EXPECT_EQ(ranks[1].name, "medium");
    EXPECT_EQ(ranks[2].name, "weak");

    // Selectivity
    EXPECT_NEAR(server.selectivity_ratio("strong", "weak"),
                std::exp(20.0 - 2.0), std::exp(18.0) * 1e-10);

    // Probabilities sum to 1
    double sum = server.binding_probability("strong")
               + server.binding_probability("medium")
               + server.binding_probability("weak")
               + server.grand_partition().empty_probability();
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

// ════════════════════════════════════════════════════════════════════════
// Re-docking (update existing ligand)
// ════════════════════════════════════════════════════════════════════════

TEST(TargetServer, RedockingOverwrite) {
    TargetServer server;

    // First docking
    auto s1 = server.create_session("aspirin");
    s1.completed = true;
    s1.log_Z = 5.0;
    server.register_result(s1);

    double dG_first = server.grand_partition().free_energy("aspirin");

    // Re-dock with a better estimate (overwrite, not merge)
    auto s2 = server.create_session("aspirin");
    s2.completed = true;
    s2.log_Z = 8.0;  // improved estimate
    server.register_result(s2);

    // ΔG should reflect the overwrite value, not a merge
    double dG_after = server.grand_partition().free_energy("aspirin");
    EXPECT_NEAR(dG_after, -statmech::kB_kcal * 300.0 * 8.0, 1e-10);

    // Re-docking with worse estimate should give less favorable ΔG
    auto s3 = server.create_session("aspirin");
    s3.completed = true;
    s3.log_Z = 3.0;
    server.register_result(s3);
    double dG_worse = server.grand_partition().free_energy("aspirin");
    EXPECT_GT(dG_worse, dG_after);  // less favorable
}

// ════════════════════════════════════════════════════════════════════════
// Knowledge base accumulation
// ════════════════════════════════════════════════════════════════════════

TEST(TargetServer, ConformerPriors) {
    TargetConfig cfg;
    cfg.n_models = 3;
    TargetServer server(cfg);

    auto s1 = server.create_session("lig1");
    s1.completed = true;
    s1.log_Z = 5.0;
    s1.conformer_populations = {0.7, 0.2, 0.1};
    server.register_result(s1);

    auto s2 = server.create_session("lig2");
    s2.completed = true;
    s2.log_Z = 4.0;
    s2.conformer_populations = {0.6, 0.3, 0.1};
    server.register_result(s2);

    auto priors = server.conformer_priors();
    ASSERT_EQ(priors.size(), 3u);

    // Conformer 0 should have highest posterior (0.7 + 0.6 + prior)
    EXPECT_GT(priors[0], priors[1]);
    EXPECT_GT(priors[1], priors[2]);

    // Should sum to 1
    double sum = priors[0] + priors[1] + priors[2];
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

TEST(TargetServer, BindingCenterAccumulation) {
    TargetServer server;

    auto s = server.create_session("lig1");
    s.completed = true;
    s.log_Z = 5.0;
    s.best_center[0] = 10.0f;
    s.best_center[1] = 20.0f;
    s.best_center[2] = 30.0f;
    s.best_energy = -7.5;
    server.register_result(s);

    auto hits = server.knowledge_base().all_hits();
    ASSERT_EQ(hits.size(), 1u);
    EXPECT_NEAR(hits[0].center[0], 10.0f, 1e-5);
    EXPECT_NEAR(hits[0].energy, -7.5, 1e-10);
    EXPECT_EQ(hits[0].ligand_name, "lig1");
}

// ════════════════════════════════════════════════════════════════════════
// Concurrent session registration (basic thread safety)
// ════════════════════════════════════════════════════════════════════════

TEST(TargetServer, ConcurrentRegistration) {
    TargetServer server;
    const int N = 50;

    auto worker = [&](int id) {
        auto s = server.create_session("lig_" + std::to_string(id));
        s.completed = true;
        s.log_Z = static_cast<double>(id);
        s.n_poses = 10;
        server.register_result(s);
    };

    std::vector<std::thread> threads;
    threads.reserve(N);
    for (int i = 0; i < N; ++i) {
        threads.emplace_back(worker, i);
    }
    for (auto& t : threads) t.join();

    EXPECT_EQ(server.completed_sessions(), N);
    EXPECT_EQ(server.grand_partition().num_ligands(), N);

    // Probabilities should sum to 1
    auto ranks = server.rank_ligands();
    double sum = server.grand_partition().empty_probability();
    for (const auto& r : ranks) sum += r.p_bound;
    EXPECT_NEAR(sum, 1.0, 1e-8);
}
