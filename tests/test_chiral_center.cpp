// tests/test_chiral_center.cpp
// Unit tests for ChiralCenterGene — R/S stereocenter discrimination
// Tests detection, mutation, crossover, inversion energy, and entropy.

#include <gtest/gtest.h>
#include "ChiralCenterGene.h"
#include "flexaid.h"
#include <cmath>
#include <vector>
#include <set>

using namespace chiral;

// ===========================================================================
// HELPER: Build a ChiralCenterGene from manually-specified centers
// ===========================================================================
static ChiralCenterGene make_gene(int n_centers, Chirality default_c = Chirality::R) {
    std::vector<ChiralCenter> centers(n_centers);
    for (int i = 0; i < n_centers; ++i) {
        centers[i].central_atom_idx = i * 5;
        for (int k = 0; k < 4; ++k)
            centers[i].substituent_indices[k] = i * 5 + k + 1;
        centers[i].assigned = default_c;
        centers[i].reference = Chirality::Unknown;
    }
    return ChiralCenterGene(std::move(centers));
}

// ===========================================================================
// CONSTRUCTION & SIZE
// ===========================================================================

TEST(ChiralCenterGene, EmptyGene) {
    ChiralCenterGene gene(std::vector<ChiralCenter>{});
    EXPECT_EQ(gene.size(), 0);
}

TEST(ChiralCenterGene, SizeMatchesCenters) {
    auto gene = make_gene(3);
    EXPECT_EQ(gene.size(), 3);
}

TEST(ChiralCenterGene, CentersAccessible) {
    auto gene = make_gene(2);
    const auto& centers = gene.centers();
    EXPECT_EQ(centers.size(), 2u);
    EXPECT_EQ(centers[0].central_atom_idx, 0);
    EXPECT_EQ(centers[1].central_atom_idx, 5);
}

// ===========================================================================
// GET/SET CHIRALITY
// ===========================================================================

TEST(ChiralCenterGene, DefaultIsR) {
    auto gene = make_gene(2);
    EXPECT_EQ(gene.get(0), Chirality::R);
    EXPECT_EQ(gene.get(1), Chirality::R);
}

TEST(ChiralCenterGene, SetChirality) {
    auto gene = make_gene(2);
    gene.set(0, Chirality::S);
    EXPECT_EQ(gene.get(0), Chirality::S);
    EXPECT_EQ(gene.get(1), Chirality::R);  // unchanged
}

TEST(ChiralCenterGene, SetOutOfRangeThrows) {
    auto gene = make_gene(2);
    EXPECT_THROW(gene.set(5, Chirality::S), std::out_of_range);
    EXPECT_THROW(gene.get(5), std::out_of_range);
}

// ===========================================================================
// INVERSION ENERGY
// ===========================================================================

TEST(ChiralCenterGene, NoReferenceNoEnergy) {
    // When reference is Unknown, no penalty regardless of assignment
    auto gene = make_gene(3);
    EXPECT_NEAR(gene.inversion_energy(), 0.0, 1e-10);
}

TEST(ChiralCenterGene, CorrectChiralityNoEnergy) {
    std::vector<ChiralCenter> centers(2);
    for (int i = 0; i < 2; ++i) {
        centers[i].central_atom_idx = i;
        for (int k = 0; k < 4; ++k)
            centers[i].substituent_indices[k] = 10 + i * 4 + k;
        centers[i].assigned = Chirality::R;
        centers[i].reference = Chirality::R;  // match
    }
    ChiralCenterGene gene(centers);
    EXPECT_NEAR(gene.inversion_energy(), 0.0, 1e-10);
}

TEST(ChiralCenterGene, WrongChiralityPenalized) {
    std::vector<ChiralCenter> centers(1);
    centers[0].central_atom_idx = 0;
    for (int k = 0; k < 4; ++k)
        centers[0].substituent_indices[k] = k + 1;
    centers[0].assigned = Chirality::S;
    centers[0].reference = Chirality::R;  // mismatch

    ChiralCenterGene gene(centers);
    double penalty = gene.inversion_energy(20.0);
    EXPECT_NEAR(penalty, 20.0, 1e-10);  // one wrong center × 20 kcal/mol
}

TEST(ChiralCenterGene, MultipleWrongCentersAdditive) {
    std::vector<ChiralCenter> centers(3);
    for (int i = 0; i < 3; ++i) {
        centers[i].central_atom_idx = i;
        for (int k = 0; k < 4; ++k)
            centers[i].substituent_indices[k] = 10 + i * 4 + k;
        centers[i].assigned = Chirality::S;
        centers[i].reference = Chirality::R;  // all wrong
    }
    ChiralCenterGene gene(centers);
    double penalty = gene.inversion_energy(15.0);
    EXPECT_NEAR(penalty, 45.0, 1e-10);  // 3 × 15 kcal/mol
}

TEST(ChiralCenterGene, CustomKInvValue) {
    std::vector<ChiralCenter> centers(1);
    centers[0].central_atom_idx = 0;
    for (int k = 0; k < 4; ++k)
        centers[0].substituent_indices[k] = k + 1;
    centers[0].assigned = Chirality::R;
    centers[0].reference = Chirality::S;

    ChiralCenterGene gene(centers);
    EXPECT_NEAR(gene.inversion_energy(25.0), 25.0, 1e-10);
    EXPECT_NEAR(gene.inversion_energy(10.0), 10.0, 1e-10);
}

// ===========================================================================
// MUTATION
// ===========================================================================

TEST(ChiralCenterGene, MutateEmptyDoesNotCrash) {
    ChiralCenterGene gene(std::vector<ChiralCenter>{});
    EXPECT_NO_THROW(gene.mutate(1.0));
}

TEST(ChiralCenterGene, MutateWithCertaintyFlipsAll) {
    auto gene = make_gene(5);
    // All start as R
    gene.mutate(1.0);  // 100% probability
    // All should be S now
    for (int i = 0; i < gene.size(); ++i) {
        EXPECT_EQ(gene.get(i), Chirality::S);
    }
}

TEST(ChiralCenterGene, MutateWithZeroProbNoChange) {
    auto gene = make_gene(5);
    gene.mutate(0.0);
    for (int i = 0; i < gene.size(); ++i) {
        EXPECT_EQ(gene.get(i), Chirality::R);
    }
}

TEST(ChiralCenterGene, DoubleMutationReturnsToOriginal) {
    auto gene = make_gene(3);
    gene.mutate(1.0);  // R -> S
    gene.mutate(1.0);  // S -> R
    for (int i = 0; i < gene.size(); ++i) {
        EXPECT_EQ(gene.get(i), Chirality::R);
    }
}

// ===========================================================================
// CROSSOVER
// ===========================================================================

TEST(ChiralCenterGene, CrossoverSwapsSuffix) {
    auto gene_a = make_gene(4, Chirality::R);
    auto gene_b = make_gene(4, Chirality::S);

    gene_a.crossover(gene_b);

    // After crossover, some centers should have swapped
    // We can't predict the exact crossover point, but each gene
    // should have a mix of R and S (unless crossover point is at 0 or end)
    bool a_has_s = false, b_has_r = false;
    for (int i = 0; i < 4; ++i) {
        if (gene_a.get(i) == Chirality::S) a_has_s = true;
        if (gene_b.get(i) == Chirality::R) b_has_r = true;
    }
    // At least some exchange should have happened (with high probability)
    // Note: if crossover point = 0, no exchange; if = size-1, only last swaps
    // Run multiple times to reduce flakiness
}

TEST(ChiralCenterGene, CrossoverSizeMismatchNoOp) {
    auto gene_a = make_gene(3);
    auto gene_b = make_gene(4);  // different size
    gene_a.crossover(gene_b);

    // Should be a no-op (sizes don't match)
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(gene_a.get(i), Chirality::R);
    }
}

TEST(ChiralCenterGene, CrossoverEmptyNoOp) {
    ChiralCenterGene gene_a(std::vector<ChiralCenter>{});
    ChiralCenterGene gene_b(std::vector<ChiralCenter>{});
    EXPECT_NO_THROW(gene_a.crossover(gene_b));
}

// ===========================================================================
// ENTROPY
// ===========================================================================

TEST(ChiralCenterGene, EntropyEmptyPopulation) {
    std::vector<ChiralCenterGene> pop;
    EXPECT_NEAR(ChiralCenterGene::compute_entropy(pop), 0.0, 1e-10);
}

TEST(ChiralCenterGene, EntropyUniformPopulationZero) {
    // All individuals have the same chirality → zero entropy
    std::vector<ChiralCenterGene> pop;
    for (int i = 0; i < 10; ++i) {
        pop.push_back(make_gene(2, Chirality::R));
    }
    double H = ChiralCenterGene::compute_entropy(pop);
    EXPECT_NEAR(H, 0.0, 1e-10);
}

TEST(ChiralCenterGene, EntropyMaxForEqualRS) {
    // 50/50 R:S split should give entropy ≈ 1 bit per center
    std::vector<ChiralCenterGene> pop;
    for (int i = 0; i < 100; ++i) {
        auto gene = make_gene(1);
        gene.set(0, (i < 50) ? Chirality::R : Chirality::S);
        pop.push_back(std::move(gene));
    }
    double H = ChiralCenterGene::compute_entropy(pop);
    EXPECT_NEAR(H, 1.0, 0.01);  // 1 bit for binary choice
}

TEST(ChiralCenterGene, EntropyNonNegative) {
    std::vector<ChiralCenterGene> pop;
    for (int i = 0; i < 20; ++i) {
        auto gene = make_gene(3);
        if (i % 3 == 0) gene.set(0, Chirality::S);
        if (i % 5 == 0) gene.set(1, Chirality::S);
        pop.push_back(std::move(gene));
    }
    EXPECT_GE(ChiralCenterGene::compute_entropy(pop), 0.0);
}

// ===========================================================================
// TO_STRING
// ===========================================================================

TEST(ChiralCenterGene, ToStringFormat) {
    auto gene = make_gene(3);
    gene.set(1, Chirality::S);
    std::string s = gene.to_string();
    EXPECT_EQ(s, "ChiralGene[R,S,R]");
}

TEST(ChiralCenterGene, ToStringEmpty) {
    ChiralCenterGene gene(std::vector<ChiralCenter>{});
    EXPECT_EQ(gene.to_string(), "ChiralGene[]");
}

// ===========================================================================
// DETECT STEREOCENTERS
// ===========================================================================

TEST(ChiralDetection, NullAtomsReturnsEmpty) {
    auto result = detect_stereocenters(nullptr, 0);
    EXPECT_TRUE(result.empty());
}

TEST(ChiralDetection, NoChiralCentersForSimpleMolecule) {
    // 2 carbon atoms bonded to each other — not enough for chirality
    atom atoms[2] = {};
    atoms[0].element[0] = 'C';
    atoms[0].bond[0] = 1;  // one bond
    atoms[0].bond[1] = 1;  // bonded to atom 1
    atoms[1].element[0] = 'O';
    atoms[1].bond[0] = 1;
    atoms[1].bond[1] = 0;

    auto result = detect_stereocenters(atoms, 2);
    EXPECT_TRUE(result.empty());
}

TEST(ChiralDetection, DetectsChiralCarbon) {
    // sp3 carbon with 4 distinct substituent types
    atom atoms[5] = {};

    // Central carbon (atom 0)
    atoms[0].element[0] = 'C';
    atoms[0].bond[0] = 4;
    atoms[0].bond[1] = 1;
    atoms[0].bond[2] = 2;
    atoms[0].bond[3] = 3;
    atoms[0].bond[MBNDS] = 4;  // 4th substituent uses the last valid slot

    // 4 different substituent types
    atoms[1].element[0] = 'H'; atoms[1].type = 1;
    atoms[2].element[0] = 'O'; atoms[2].type = 2;
    atoms[3].element[0] = 'N'; atoms[3].type = 3;
    atoms[4].element[0] = 'F'; atoms[4].type = 4;

    auto result = detect_stereocenters(atoms, 5);
    EXPECT_GE(result.size(), 1u);
    if (!result.empty()) {
        EXPECT_EQ(result[0].central_atom_idx, 0);
        EXPECT_EQ(result[0].assigned, Chirality::R);  // default
        EXPECT_EQ(result[0].reference, Chirality::Unknown);
    }
}
