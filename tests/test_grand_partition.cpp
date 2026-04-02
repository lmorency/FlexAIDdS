// test_grand_partition.cpp — Unit tests for GrandPartitionFunction
//
// Verifies log-space arithmetic, binding probabilities, selectivity
// ratios, and ranking against analytical hand calculations.
//
// Copyright 2026 Le Bonhomme Pharma
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "GrandPartitionFunction.h"
#include "statmech.h"

#include <cmath>
#include <numbers>

using namespace target;

static constexpr double kT_300 = statmech::kB_kcal * 300.0;  // ~0.596 kcal/mol

// ════════════════════════════════════════════════════════════════════════
// Basic construction
// ════════════════════════════════════════════════════════════════════════

TEST(GrandPartition, EmptyXi) {
    GrandPartitionFunction gpf(300.0);
    EXPECT_EQ(gpf.num_ligands(), 0);
    // Ξ = 1 (empty site only) → ln Ξ = 0
    EXPECT_NEAR(gpf.log_Xi(), 0.0, 1e-12);
    // p(empty) = 1/1 = 1
    EXPECT_NEAR(gpf.empty_probability(), 1.0, 1e-12);
}

TEST(GrandPartition, InvalidTemperature) {
    EXPECT_THROW(GrandPartitionFunction(0.0), std::invalid_argument);
    EXPECT_THROW(GrandPartitionFunction(-100.0), std::invalid_argument);
}

// ════════════════════════════════════════════════════════════════════════
// Single ligand
// ════════════════════════════════════════════════════════════════════════

TEST(GrandPartition, SingleLigand) {
    GrandPartitionFunction gpf(300.0);

    // Z_A = e^10 (arbitrary large partition function)
    double log_Z_A = 10.0;
    gpf.add_ligand("ligandA", log_Z_A);

    EXPECT_EQ(gpf.num_ligands(), 1);
    EXPECT_TRUE(gpf.has_ligand("ligandA"));

    // Ξ = 1 + Z_A = 1 + e^10
    double expected_log_xi = std::log(1.0 + std::exp(10.0));
    EXPECT_NEAR(gpf.log_Xi(), expected_log_xi, 1e-10);

    // p(A) = Z_A / Ξ = e^10 / (1 + e^10) ≈ 0.999955
    double expected_pA = std::exp(10.0) / (1.0 + std::exp(10.0));
    EXPECT_NEAR(gpf.binding_probability("ligandA"), expected_pA, 1e-8);

    // p(empty) = 1/Ξ ≈ 0.000045
    EXPECT_NEAR(gpf.empty_probability(), 1.0 - expected_pA, 1e-8);

    // ΔG = -kT * ln(Z_A) = -kT * 10
    EXPECT_NEAR(gpf.delta_G("ligandA"), -kT_300 * 10.0, 1e-10);
}

// ════════════════════════════════════════════════════════════════════════
// Two ligands — competitive binding
// ════════════════════════════════════════════════════════════════════════

TEST(GrandPartition, TwoLigandsCompetitive) {
    GrandPartitionFunction gpf(300.0);

    // Z_A = e^10, Z_B = e^8
    gpf.add_ligand("A", 10.0);
    gpf.add_ligand("B",  8.0);

    // Ξ = 1 + e^10 + e^8
    double xi = 1.0 + std::exp(10.0) + std::exp(8.0);
    EXPECT_NEAR(std::exp(gpf.log_Xi()), xi, xi * 1e-10);

    // selectivity A/B = Z_A/Z_B = e^(10-8) = e^2 ≈ 7.389
    EXPECT_NEAR(gpf.selectivity("A", "B"), std::exp(2.0), 1e-10);
    EXPECT_NEAR(gpf.selectivity("B", "A"), std::exp(-2.0), 1e-10);

    // Probabilities must sum to 1
    double sum = gpf.binding_probability("A")
               + gpf.binding_probability("B")
               + gpf.empty_probability();
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

// ════════════════════════════════════════════════════════════════════════
// Equal ligands
// ════════════════════════════════════════════════════════════════════════

TEST(GrandPartition, EqualLigands) {
    GrandPartitionFunction gpf(300.0);

    // Three identical ligands with Z = e^5
    gpf.add_ligand("X", 5.0);
    gpf.add_ligand("Y", 5.0);
    gpf.add_ligand("Z", 5.0);

    // All should have equal binding probability
    double pX = gpf.binding_probability("X");
    double pY = gpf.binding_probability("Y");
    double pZ = gpf.binding_probability("Z");
    EXPECT_NEAR(pX, pY, 1e-12);
    EXPECT_NEAR(pY, pZ, 1e-12);

    // Selectivity between equal ligands = 1.0
    EXPECT_NEAR(gpf.selectivity("X", "Y"), 1.0, 1e-12);

    // All ΔG should be equal
    EXPECT_NEAR(gpf.delta_G("X"), gpf.delta_G("Y"), 1e-12);
}

// ════════════════════════════════════════════════════════════════════════
// Ranking
// ════════════════════════════════════════════════════════════════════════

TEST(GrandPartition, Ranking) {
    GrandPartitionFunction gpf(300.0);

    gpf.add_ligand("weak",    2.0);
    gpf.add_ligand("strong", 20.0);
    gpf.add_ligand("medium", 10.0);

    auto ranks = gpf.rank();
    ASSERT_EQ(ranks.size(), 3u);

    // Sorted by ΔG ascending (most negative = strongest binder first)
    EXPECT_EQ(ranks[0].name, "strong");
    EXPECT_EQ(ranks[1].name, "medium");
    EXPECT_EQ(ranks[2].name, "weak");

    // Check ΔG values
    EXPECT_NEAR(ranks[0].dG, -kT_300 * 20.0, 1e-10);
    EXPECT_NEAR(ranks[1].dG, -kT_300 * 10.0, 1e-10);
    EXPECT_NEAR(ranks[2].dG, -kT_300 *  2.0, 1e-10);

    // Probabilities must sum to < 1 (leaving room for empty)
    double psum = 0;
    for (const auto& r : ranks) psum += r.p_bound;
    EXPECT_LT(psum, 1.0);
    EXPECT_NEAR(psum + gpf.empty_probability(), 1.0, 1e-10);
}

// ════════════════════════════════════════════════════════════════════════
// Update (re-docking merge)
// ════════════════════════════════════════════════════════════════════════

TEST(GrandPartition, UpdateMerge) {
    GrandPartitionFunction gpf(300.0);

    // Initial: Z_A = e^5
    gpf.add_ligand("A", 5.0);
    double dG_before = gpf.delta_G("A");

    // Re-dock: additional Z_new = e^5 → Z_merged = 2·e^5
    gpf.update_ligand("A", 5.0);

    // ln(Z_merged) = ln(2·e^5) = 5 + ln(2)
    double expected_dG = -kT_300 * (5.0 + std::log(2.0));
    EXPECT_NEAR(gpf.delta_G("A"), expected_dG, 1e-10);

    // ΔG should be more negative (more favorable) after merging
    EXPECT_LT(gpf.delta_G("A"), dG_before);
}

// ════════════════════════════════════════════════════════════════════════
// Remove ligand
// ════════════════════════════════════════════════════════════════════════

TEST(GrandPartition, RemoveLigand) {
    GrandPartitionFunction gpf(300.0);
    gpf.add_ligand("A", 10.0);
    gpf.add_ligand("B",  5.0);
    EXPECT_EQ(gpf.num_ligands(), 2);

    gpf.remove_ligand("A");
    EXPECT_EQ(gpf.num_ligands(), 1);
    EXPECT_FALSE(gpf.has_ligand("A"));
    EXPECT_TRUE(gpf.has_ligand("B"));
}

// ════════════════════════════════════════════════════════════════════════
// Error handling
// ════════════════════════════════════════════════════════════════════════

TEST(GrandPartition, DuplicateAddThrows) {
    GrandPartitionFunction gpf(300.0);
    gpf.add_ligand("A", 5.0);
    EXPECT_THROW(gpf.add_ligand("A", 10.0), std::invalid_argument);
}

TEST(GrandPartition, QueryMissingThrows) {
    GrandPartitionFunction gpf(300.0);
    EXPECT_THROW(gpf.binding_probability("X"), std::invalid_argument);
    EXPECT_THROW(gpf.delta_G("X"), std::invalid_argument);
    EXPECT_THROW(gpf.update_ligand("X", 5.0), std::invalid_argument);
    EXPECT_THROW(gpf.remove_ligand("X"), std::invalid_argument);
}

// ════════════════════════════════════════════════════════════════════════
// Numerical stability — extreme values
// ════════════════════════════════════════════════════════════════════════

TEST(GrandPartition, ExtremeLogZ) {
    GrandPartitionFunction gpf(300.0);

    // Very large Z (ln Z = 500)
    gpf.add_ligand("huge", 500.0);
    // Very small Z (ln Z = -500)
    gpf.add_ligand("tiny", -500.0);

    // Ξ ≈ e^500 (dominated by "huge")
    EXPECT_NEAR(gpf.log_Xi(), 500.0, 1.0);  // within 1.0 of 500

    // p(huge) ≈ 1.0
    EXPECT_NEAR(gpf.binding_probability("huge"), 1.0, 1e-6);
    // p(tiny) ≈ 0.0
    EXPECT_NEAR(gpf.binding_probability("tiny"), 0.0, 1e-6);

    // No NaN or Inf
    EXPECT_TRUE(std::isfinite(gpf.log_Xi()));
    EXPECT_TRUE(std::isfinite(gpf.binding_probability("huge")));
    EXPECT_TRUE(std::isfinite(gpf.binding_probability("tiny")));
    EXPECT_TRUE(std::isfinite(gpf.empty_probability()));
}

// ════════════════════════════════════════════════════════════════════════
// StatMechEngine integration
// ════════════════════════════════════════════════════════════════════════

TEST(GrandPartition, FromStatMechEngine) {
    GrandPartitionFunction gpf(300.0);

    statmech::StatMechEngine engine(300.0);
    engine.add_sample(-5.0, 1);   // E = -5 kcal/mol
    engine.add_sample(-3.0, 2);   // E = -3 kcal/mol (2 counts)
    engine.add_sample(-1.0, 1);

    gpf.add_ligand("from_engine", engine);
    EXPECT_TRUE(gpf.has_ligand("from_engine"));

    auto thermo = engine.compute();
    EXPECT_NEAR(gpf.delta_G("from_engine"), thermo.free_energy, 1e-10);
}
