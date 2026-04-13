// tests/test_ligand_ring_flex.cpp
// Unit tests for the LigandRingFlex unified ring flexibility interface
// Covers: RingFlexGenes, randomise, mutate, crossover, compute_ring_entropy
// Apache-2.0 © 2026 Le Bonhomme Pharma

#include <gtest/gtest.h>
#include "../LIB/LigandRingFlex/LigandRingFlex.h"
#include "../LIB/LigandRingFlex/RingConformerLibrary.h"
#include "../LIB/LigandRingFlex/SugarPucker.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

using namespace ligand_ring_flex;
using namespace ring_conformer;

// ===========================================================================
// Helpers
// ===========================================================================

static RingFlexGenes make_six_ring_genes(int n_six, int n_five = 0, int n_sugar = 0) {
    RingFlexGenes g;
    g.conformer_indices.assign(n_six, 0);
    g.five_conformer_indices.assign(n_five, 0);
    g.sugar_phases.assign(n_sugar, 0.0f);
    g.sugar_types.assign(n_sugar, sugar_pucker::SugarType::FURANOSE);
    g.sugar_ring_indices.resize(n_sugar);
    return g;
}

// ===========================================================================
// RingFlexGenes — construction
// ===========================================================================

TEST(RingFlexGenes, DefaultConstructionIsEmpty) {
    RingFlexGenes g;
    EXPECT_TRUE(g.conformer_indices.empty());
    EXPECT_TRUE(g.five_conformer_indices.empty());
    EXPECT_TRUE(g.sugar_phases.empty());
    EXPECT_TRUE(g.sugar_types.empty());
    EXPECT_TRUE(g.sugar_ring_indices.empty());
}

TEST(RingFlexGenes, ManualConstruction) {
    auto g = make_six_ring_genes(3, 1, 2);
    EXPECT_EQ(g.conformer_indices.size(), 3u);
    EXPECT_EQ(g.five_conformer_indices.size(), 1u);
    EXPECT_EQ(g.sugar_phases.size(), 2u);
    EXPECT_EQ(g.sugar_types.size(), 2u);
}

// ===========================================================================
// randomise
// ===========================================================================

TEST(LigandRingFlex, RandomiseChangesConformerIndices) {
    auto g = make_six_ring_genes(4);
    // All start at 0; after randomise at least some should differ from 0
    // (run several times for statistical robustness)
    bool changed = false;
    for (int trial = 0; trial < 20 && !changed; ++trial) {
        randomise(g);
        for (uint8_t idx : g.conformer_indices)
            if (idx != 0) { changed = true; break; }
    }
    EXPECT_TRUE(changed);
}

TEST(LigandRingFlex, RandomiseSugarPhaseInRange) {
    auto g = make_six_ring_genes(0, 0, 3);
    randomise(g);
    for (float phase : g.sugar_phases) {
        EXPECT_GE(phase, 0.0f);
        EXPECT_LT(phase, 360.0f);
    }
}

TEST(LigandRingFlex, RandomisePreservesVectorLengths) {
    auto g = make_six_ring_genes(3, 2, 1);
    randomise(g);
    EXPECT_EQ(g.conformer_indices.size(), 3u);
    EXPECT_EQ(g.five_conformer_indices.size(), 2u);
    EXPECT_EQ(g.sugar_phases.size(), 1u);
}

TEST(LigandRingFlex, RandomiseEmptyGenesIsNoOp) {
    RingFlexGenes g;
    EXPECT_NO_THROW(randomise(g));
}

// ===========================================================================
// mutate
// ===========================================================================

TEST(LigandRingFlex, MutateDoesNotChangeLengths) {
    auto g = make_six_ring_genes(4, 2, 1);
    randomise(g);
    size_t n_six   = g.conformer_indices.size();
    size_t n_five  = g.five_conformer_indices.size();
    size_t n_sugar = g.sugar_phases.size();

    mutate(g, 1.0, 1.0); // force mutation on every gene
    EXPECT_EQ(g.conformer_indices.size(), n_six);
    EXPECT_EQ(g.five_conformer_indices.size(), n_five);
    EXPECT_EQ(g.sugar_phases.size(), n_sugar);
}

TEST(LigandRingFlex, MutateProbZeroChangesNothing) {
    auto g = make_six_ring_genes(4, 2, 2);
    randomise(g);
    auto copy = g;

    mutate(g, 0.0, 0.0);
    EXPECT_EQ(g.conformer_indices, copy.conformer_indices);
    EXPECT_EQ(g.five_conformer_indices, copy.five_conformer_indices);
    EXPECT_EQ(g.sugar_phases, copy.sugar_phases);
}

TEST(LigandRingFlex, MutateProbOneSometimesChanges) {
    auto g = make_six_ring_genes(8);
    randomise(g);
    auto original = g.conformer_indices;
    bool changed = false;
    for (int trial = 0; trial < 20 && !changed; ++trial) {
        auto gtest = g;
        mutate(gtest, 1.0, 0.0);
        if (gtest.conformer_indices != original) changed = true;
    }
    EXPECT_TRUE(changed);
}

TEST(LigandRingFlex, MutateSugarPhaseStaysInRange) {
    auto g = make_six_ring_genes(0, 0, 4);
    randomise(g);
    for (int trial = 0; trial < 10; ++trial) {
        mutate(g, 0.0, 1.0);
        for (float phase : g.sugar_phases) {
            EXPECT_GE(phase, 0.0f);
            EXPECT_LT(phase, 360.0f);
        }
    }
}

TEST(LigandRingFlex, MutateEmptyGenesIsNoOp) {
    RingFlexGenes g;
    EXPECT_NO_THROW(mutate(g, 1.0, 1.0));
}

// ===========================================================================
// crossover
// ===========================================================================

TEST(LigandRingFlex, CrossoverPreservesLengths) {
    auto a = make_six_ring_genes(6, 2, 1);
    auto b = make_six_ring_genes(6, 2, 1);
    randomise(a);
    randomise(b);

    crossover(a, b);

    EXPECT_EQ(a.conformer_indices.size(), 6u);
    EXPECT_EQ(b.conformer_indices.size(), 6u);
    EXPECT_EQ(a.five_conformer_indices.size(), 2u);
    EXPECT_EQ(b.five_conformer_indices.size(), 2u);
    EXPECT_EQ(a.sugar_phases.size(), 1u);
    EXPECT_EQ(b.sugar_phases.size(), 1u);
}

TEST(LigandRingFlex, CrossoverMixesGenes) {
    auto a = make_six_ring_genes(8);
    auto b = make_six_ring_genes(8);
    for (auto& v : a.conformer_indices) v = 0;
    for (auto& v : b.conformer_indices) v = 5;

    auto a_before = a.conformer_indices;
    auto b_before = b.conformer_indices;

    // After crossover the two children should collectively contain both 0s and 5s
    crossover(a, b);

    bool a_has_both = false, b_has_both = false;
    bool a_saw_5 = std::any_of(a.conformer_indices.begin(), a.conformer_indices.end(),
                               [](uint8_t v){ return v == 5; });
    bool b_saw_0 = std::any_of(b.conformer_indices.begin(), b.conformer_indices.end(),
                               [](uint8_t v){ return v == 0; });
    // At least one parent should have received genetic material from the other
    EXPECT_TRUE(a_saw_5 || b_saw_0);
}

TEST(LigandRingFlex, CrossoverEmptyGenesIsNoOp) {
    RingFlexGenes a, b;
    EXPECT_NO_THROW(crossover(a, b));
}

// ===========================================================================
// compute_ring_entropy
// ===========================================================================

TEST(LigandRingFlex, EntropyEmptyPopulationIsZero) {
    std::vector<RingFlexGenes> pop;
    double s = compute_ring_entropy(pop);
    EXPECT_NEAR(s, 0.0, 1e-10);
}

TEST(LigandRingFlex, EntropyUniformPopulationIsZero) {
    // All individuals have the same conformer → no diversity → S ≈ 0
    auto gene = make_six_ring_genes(3);
    for (auto& v : gene.conformer_indices) v = 2;
    std::vector<RingFlexGenes> pop(10, gene);
    double s = compute_ring_entropy(pop);
    EXPECT_NEAR(s, 0.0, 1e-9);
}

TEST(LigandRingFlex, EntropyDiversePopulationIsPositive) {
    // Population with maximally diverse conformer indices
    std::vector<RingFlexGenes> pop;
    for (uint8_t c = 0; c < 6; ++c) {
        auto g = make_six_ring_genes(1);
        g.conformer_indices[0] = c;
        pop.push_back(g);
    }
    double s = compute_ring_entropy(pop);
    EXPECT_GT(s, 0.0);
}

TEST(LigandRingFlex, EntropyIncreaseWithDiversity) {
    // Two individuals same → entropy_2 < entropy_6 (6 different conformers)
    auto g = make_six_ring_genes(1);
    std::vector<RingFlexGenes> pop_same;
    for (int i = 0; i < 6; ++i) { g.conformer_indices[0] = 0; pop_same.push_back(g); }
    double s_same = compute_ring_entropy(pop_same);

    std::vector<RingFlexGenes> pop_diverse;
    for (uint8_t c = 0; c < 6; ++c) {
        g.conformer_indices[0] = c;
        pop_diverse.push_back(g);
    }
    double s_diverse = compute_ring_entropy(pop_diverse);

    EXPECT_LT(s_same, s_diverse);
}

TEST(LigandRingFlex, EntropySingleIndividualIsZero) {
    std::vector<RingFlexGenes> pop = { make_six_ring_genes(3) };
    double s = compute_ring_entropy(pop);
    EXPECT_NEAR(s, 0.0, 1e-9);
}

// ===========================================================================
// MAIN
// ===========================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
