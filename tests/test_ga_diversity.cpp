// test_ga_diversity.cpp — Unit tests for GA population diversity monitoring
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include <cstring>
#include "gaboom.h"
#include "ga_diversity.h"

class GADiversityTest : public ::testing::Test {
protected:
    static constexpr int NUM_CHROM = 100;
    static constexpr int NUM_GENES = 5;

    chromosome pop[NUM_CHROM];
    gene gene_storage[NUM_CHROM][NUM_GENES];
    genlim gene_lim[NUM_GENES];

    void SetUp() override {
        for (int c = 0; c < NUM_CHROM; ++c) {
            memset(&pop[c], 0, sizeof(chromosome));
            pop[c].genes = gene_storage[c];
        }
        for (int g = 0; g < NUM_GENES; ++g) {
            gene_lim[g].min = 0.0;
            gene_lim[g].max = 10.0;
        }
    }

    void make_uniform_population(std::mt19937& rng) {
        std::uniform_real_distribution<double> dist(0.0, 10.0);
        for (int c = 0; c < NUM_CHROM; ++c) {
            for (int g = 0; g < NUM_GENES; ++g) {
                pop[c].genes[g].to_ic = dist(rng);
            }
            pop[c].fitnes = static_cast<double>(NUM_CHROM - c);
        }
    }

    void make_converged_population() {
        // All chromosomes have nearly identical gene values
        for (int c = 0; c < NUM_CHROM; ++c) {
            for (int g = 0; g < NUM_GENES; ++g) {
                pop[c].genes[g].to_ic = 5.0 + 0.001 * (c % 3);
            }
            pop[c].fitnes = static_cast<double>(NUM_CHROM - c);
        }
    }
};

TEST_F(GADiversityTest, UniformPopulationHighEntropy) {
    std::mt19937 rng(42);
    make_uniform_population(rng);

    auto metrics = ga_diversity::compute_diversity(
        pop, NUM_CHROM, NUM_GENES, gene_lim, 0.3, 20);

    // Uniform distribution should have high entropy (close to 1.0)
    EXPECT_GT(metrics.allele_entropy, 0.5);
    EXPECT_FALSE(metrics.collapse_detected);
}

TEST_F(GADiversityTest, ConvergedPopulationLowEntropy) {
    make_converged_population();

    auto metrics = ga_diversity::compute_diversity(
        pop, NUM_CHROM, NUM_GENES, gene_lim, 0.3, 20);

    // Converged population should have very low entropy
    EXPECT_LT(metrics.allele_entropy, 0.3);
    EXPECT_TRUE(metrics.collapse_detected);
}

TEST_F(GADiversityTest, CatastrophicMutationChangesGenes) {
    make_converged_population();
    std::mt19937 rng(42);

    // Record original values
    double original_sum = 0.0;
    for (int c = 0; c < NUM_CHROM; ++c)
        for (int g = 0; g < NUM_GENES; ++g)
            original_sum += pop[c].genes[g].to_ic;

    ga_diversity::catastrophic_mutation(
        pop, NUM_CHROM, NUM_GENES, gene_lim, 0.2, rng);

    // After mutation, bottom 20% should have different genes
    double new_sum = 0.0;
    for (int c = 0; c < NUM_CHROM; ++c)
        for (int g = 0; g < NUM_GENES; ++g)
            new_sum += pop[c].genes[g].to_ic;

    EXPECT_NE(new_sum, original_sum);
}

TEST_F(GADiversityTest, CatastrophicMutationRespectsGeneLimits) {
    make_converged_population();
    std::mt19937 rng(42);

    ga_diversity::catastrophic_mutation(
        pop, NUM_CHROM, NUM_GENES, gene_lim, 0.5, rng);

    for (int c = 0; c < NUM_CHROM; ++c) {
        for (int g = 0; g < NUM_GENES; ++g) {
            EXPECT_GE(pop[c].genes[g].to_ic, gene_lim[g].min);
            EXPECT_LE(pop[c].genes[g].to_ic, gene_lim[g].max);
        }
    }
}

TEST_F(GADiversityTest, EmptyPopulation) {
    auto metrics = ga_diversity::compute_diversity(
        pop, 0, NUM_GENES, gene_lim, 0.3, 20);
    EXPECT_TRUE(metrics.collapse_detected);
}

TEST_F(GADiversityTest, ThresholdBoundary) {
    std::mt19937 rng(42);
    make_uniform_population(rng);

    // With a very high threshold, everything is "collapsed"
    auto metrics_high = ga_diversity::compute_diversity(
        pop, NUM_CHROM, NUM_GENES, gene_lim, 0.99, 20);
    EXPECT_TRUE(metrics_high.collapse_detected);

    // With a very low threshold, nothing is "collapsed"
    auto metrics_low = ga_diversity::compute_diversity(
        pop, NUM_CHROM, NUM_GENES, gene_lim, 0.01, 20);
    EXPECT_FALSE(metrics_low.collapse_detected);
}
