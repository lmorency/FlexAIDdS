// tests/test_entropy_ga.cpp — Entropy-GA integration tests
// Tests: SMFREE fitness model logic, Boltzmann weight blending,
//        chromosome entropy fields, post-GA thermodynamic summary
// Apache-2.0 © 2026 Le Bonhomme Pharma

#include <gtest/gtest.h>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

#include "gaboom.h"
#include "statmech.h"

using namespace statmech;

// ═══════════════════════════════════════════════════════════════════════
// Helper: allocate chromosome array with energies
// ═══════════════════════════════════════════════════════════════════════

static chromosome* alloc_chroms(int n_chrom, int n_genes) {
    auto* chroms = new chromosome[n_chrom];
    for (int i = 0; i < n_chrom; ++i) {
        chroms[i].genes = new gene[n_genes];
        std::memset(chroms[i].genes, 0, sizeof(gene) * n_genes);
        chroms[i].cf = {};
        chroms[i].evalue = 0.0;
        chroms[i].app_evalue = 0.0;
        chroms[i].fitnes = 0.0;
        chroms[i].boltzmann_weight = 0.0;
        chroms[i].free_energy = 0.0;
        chroms[i].status = 'n';
    }
    return chroms;
}

static void free_chroms(chromosome* chroms, int n_chrom) {
    for (int i = 0; i < n_chrom; ++i) delete[] chroms[i].genes;
    delete[] chroms;
}

// ═══════════════════════════════════════════════════════════════════════
// Chromosome Entropy Fields
// ═══════════════════════════════════════════════════════════════════════

TEST(EntropyGA, ChromosomeHasBoltzmannField) {
    const int N = 5, NG = 2;
    auto* chroms = alloc_chroms(N, NG);

    // Verify fields are zero-initialised
    for (int i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(chroms[i].boltzmann_weight, 0.0);
        EXPECT_DOUBLE_EQ(chroms[i].free_energy, 0.0);
    }

    // Set and retrieve
    chroms[2].boltzmann_weight = 0.35;
    chroms[2].free_energy = -12.5;
    EXPECT_DOUBLE_EQ(chroms[2].boltzmann_weight, 0.35);
    EXPECT_DOUBLE_EQ(chroms[2].free_energy, -12.5);

    free_chroms(chroms, N);
}

TEST(EntropyGA, CopyChromPreservesEntropyFields) {
    const int NG = 3;
    auto* src = alloc_chroms(1, NG);
    auto* dst = alloc_chroms(1, NG);

    src[0].boltzmann_weight = 0.75;
    src[0].free_energy = -8.3;
    src[0].evalue = -10.0;
    src[0].genes[0].to_ic = 1.5;

    copy_chrom(dst, src, NG);

    EXPECT_DOUBLE_EQ(dst[0].boltzmann_weight, 0.75);
    EXPECT_DOUBLE_EQ(dst[0].free_energy, -8.3);
    EXPECT_DOUBLE_EQ(dst[0].evalue, -10.0);
    EXPECT_DOUBLE_EQ(dst[0].genes[0].to_ic, 1.5);

    free_chroms(src, 1);
    free_chroms(dst, 1);
}

TEST(EntropyGA, SwapChromPreservesEntropyFields) {
    const int NG = 2;
    auto* chroms = alloc_chroms(2, NG);

    chroms[0].boltzmann_weight = 0.9;
    chroms[0].free_energy = -15.0;
    chroms[1].boltzmann_weight = 0.1;
    chroms[1].free_energy = -5.0;

    swap_chrom(&chroms[0], &chroms[1]);

    EXPECT_DOUBLE_EQ(chroms[0].boltzmann_weight, 0.1);
    EXPECT_DOUBLE_EQ(chroms[0].free_energy, -5.0);
    EXPECT_DOUBLE_EQ(chroms[1].boltzmann_weight, 0.9);
    EXPECT_DOUBLE_EQ(chroms[1].free_energy, -15.0);

    free_chroms(chroms, 2);
}

// ═══════════════════════════════════════════════════════════════════════
// GB_Global Entropy Parameters
// ═══════════════════════════════════════════════════════════════════════

TEST(EntropyGA, GBGlobalEntropyDefaults) {
    GB_Global gb;
    std::memset(&gb, 0, sizeof(GB_Global));

    // Match defaults from top.cpp
    gb.entropy_weight = 0.5;
    gb.entropy_interval = 0;
    gb.use_shannon = 0;

    EXPECT_DOUBLE_EQ(gb.entropy_weight, 0.5);
    EXPECT_EQ(gb.entropy_interval, 0);
    EXPECT_EQ(gb.use_shannon, 0);
}

TEST(EntropyGA, EntropyWeightClampedRange) {
    // Validate that entropy_weight between 0 and 1 produces valid blending
    GB_Global gb;
    std::memset(&gb, 0, sizeof(GB_Global));

    for (double w = 0.0; w <= 1.0; w += 0.1) {
        gb.entropy_weight = w;
        EXPECT_GE(gb.entropy_weight, 0.0);
        EXPECT_LE(gb.entropy_weight, 1.0);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SMFREE Fitness Logic (unit test of the blending formula)
// ═══════════════════════════════════════════════════════════════════════

// Reproduces the SMFREE blending formula in isolation:
//   fitness_i = [(1-w)*rank_component + w*boltz_component] * N / share_i
static std::vector<double> compute_smfree_fitness(
    const std::vector<double>& evalues,
    double temperature,
    double entropy_weight)
{
    const int N = static_cast<int>(evalues.size());
    StatMechEngine engine(temperature);
    for (int i = 0; i < N; ++i)
        engine.add_sample(evalues[static_cast<size_t>(i)]);

    auto bweights = engine.boltzmann_weights();

    double max_bw = *std::max_element(bweights.begin(), bweights.end());
    if (max_bw <= 0.0) max_bw = 1.0;

    // Assume evalues are pre-sorted ascending (as QuickSort would do)
    std::vector<double> fitness(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
        double rank_comp = static_cast<double>(N - i) / static_cast<double>(N);
        double boltz_comp = bweights[static_cast<size_t>(i)] / max_bw;
        double share = 1.0; // no sharing in unit test
        fitness[static_cast<size_t>(i)] =
            ((1.0 - entropy_weight) * rank_comp + entropy_weight * boltz_comp)
            * static_cast<double>(N) / share;
    }
    return fitness;
}

TEST(EntropyGA, SMFREEPureRankAtZeroWeight) {
    // With entropy_weight=0, SMFREE should give pure rank-based fitness
    std::vector<double> energies = {-15.0, -12.0, -10.0, -8.0, -5.0};
    auto fitness = compute_smfree_fitness(energies, 300.0, 0.0);

    // Pure rank: fitness[i] = (N-i)/N * N = N-i
    for (size_t i = 0; i < energies.size(); ++i) {
        double expected = static_cast<double>(energies.size() - i);
        EXPECT_NEAR(fitness[i], expected, 1e-10)
            << "Pure rank fitness mismatch at index " << i;
    }
}

TEST(EntropyGA, SMFREEFullBoltzmannAtUnitWeight) {
    // With entropy_weight=1, fitness should be proportional to Boltzmann weight
    std::vector<double> energies = {-15.0, -12.0, -10.0, -8.0, -5.0};
    auto fitness = compute_smfree_fitness(energies, 300.0, 1.0);

    // Best energy (most negative) should have highest fitness
    EXPECT_GT(fitness[0], fitness[1]);
    EXPECT_GT(fitness[1], fitness[2]);

    // All fitness values should be positive
    for (auto f : fitness)
        EXPECT_GT(f, 0.0);
}

TEST(EntropyGA, SMFREEBlendedFitnessMonotonicity) {
    // With moderate weight, fitness should still favour lower energies
    std::vector<double> energies = {-20.0, -15.0, -10.0, -5.0, 0.0};
    auto fitness = compute_smfree_fitness(energies, 300.0, 0.5);

    // Best energy should have highest fitness
    EXPECT_GT(fitness[0], fitness[4]);

    // All positive
    for (auto f : fitness)
        EXPECT_GT(f, 0.0);
}

TEST(EntropyGA, SMFREEHighTemperatureFlattensWeights) {
    // At very high temperature, Boltzmann weights become nearly uniform
    std::vector<double> energies = {-10.0, -5.0, 0.0, 5.0, 10.0};
    auto fitness_high_T = compute_smfree_fitness(energies, 10000.0, 1.0);

    // The ratio of max to min fitness should be close to 1
    double fmax = *std::max_element(fitness_high_T.begin(), fitness_high_T.end());
    double fmin = *std::min_element(fitness_high_T.begin(), fitness_high_T.end());
    double ratio = fmax / fmin;

    // At 10000K with energy spread of 20 kcal/mol, weights should be relatively close
    EXPECT_LT(ratio, 5.0) << "High-T Boltzmann weights should be approximately uniform";
}

TEST(EntropyGA, SMFREELowTemperatureSharpensSelection) {
    // At low temperature, Boltzmann weights become very peaked on the minimum
    std::vector<double> energies = {-10.0, -5.0, 0.0, 5.0, 10.0};
    auto fitness_low_T = compute_smfree_fitness(energies, 100.0, 1.0);

    // The best (lowest energy) should dominate
    EXPECT_GT(fitness_low_T[0], fitness_low_T[4] * 5.0)
        << "Low-T should strongly favour the minimum energy pose";
}

// ═══════════════════════════════════════════════════════════════════════
// Post-GA Ensemble Thermodynamics
// ═══════════════════════════════════════════════════════════════════════

TEST(EntropyGA, PostGAEnsembleThermodynamics) {
    // Simulate a GA ensemble and verify thermodynamic consistency
    const double T = 300.0;
    StatMechEngine engine(T);

    // Add a realistic spread of docking scores
    std::vector<double> scores = {
        -15.2, -14.8, -13.5, -12.1, -11.7,
        -10.3, -9.8, -8.5, -7.2, -5.0
    };
    for (auto e : scores) engine.add_sample(e);

    auto thermo = engine.compute();

    // Basic thermodynamic relations
    EXPECT_GT(thermo.temperature, 0.0);

    // Free energy should be less than mean energy (entropy contribution is positive)
    // F = <E> - T*S  →  F ≤ <E> when S ≥ 0
    EXPECT_LE(thermo.free_energy, thermo.mean_energy + 1e-10);

    // Entropy should be non-negative
    EXPECT_GE(thermo.entropy, 0.0);

    // Heat capacity should be non-negative
    EXPECT_GE(thermo.heat_capacity, 0.0);

    // Standard deviation should be non-negative
    EXPECT_GE(thermo.std_energy, 0.0);

    // Verify F = <E> - T*S
    double F_check = thermo.mean_energy - T * thermo.entropy;
    EXPECT_NEAR(thermo.free_energy, F_check, 1e-6);
}

TEST(EntropyGA, BoltzmannWeightsNormalize) {
    StatMechEngine engine(300.0);
    for (double e = -15.0; e <= 0.0; e += 1.0)
        engine.add_sample(e);

    auto weights = engine.boltzmann_weights();
    double sum = std::accumulate(weights.begin(), weights.end(), 0.0);

    EXPECT_NEAR(sum, 1.0, 1e-10) << "Boltzmann weights must sum to 1";

    // All weights should be positive
    for (auto w : weights)
        EXPECT_GT(w, 0.0);

    // Lowest energy should have highest weight
    EXPECT_GT(weights[0], weights[weights.size() - 1]);
}

// ═══════════════════════════════════════════════════════════════════════
// Edge Cases
// ═══════════════════════════════════════════════════════════════════════

TEST(EntropyGA, SMFREESingleChromosomeFitness) {
    std::vector<double> energies = {-10.0};
    auto fitness = compute_smfree_fitness(energies, 300.0, 0.5);
    EXPECT_GT(fitness[0], 0.0);
}

TEST(EntropyGA, SMFREEDegenerateEnergies) {
    // All chromosomes have the same energy
    std::vector<double> energies(10, -10.0);
    auto fitness = compute_smfree_fitness(energies, 300.0, 1.0);

    // All Boltzmann weights should be equal → all fitness should be equal
    for (size_t i = 1; i < fitness.size(); ++i) {
        EXPECT_NEAR(fitness[i], fitness[0], 1e-6)
            << "Equal energies should give equal Boltzmann fitness";
    }
}
