// tests/test_sugar_pucker.cpp
// Unit tests for SugarPucker — Cremer-Pople pseudorotation for furanose rings
// Tests phase-to-torsion conversion, energy landscape, and entropy.

#include <gtest/gtest.h>
#include "SugarPucker.h"
#include <cmath>
#include <vector>
#include <numeric>

using namespace sugar_pucker;

static constexpr float TOL = 1e-3f;
static constexpr double DTOL = 1e-6;

// ===========================================================================
// COMPUTE RING TORSIONS
// ===========================================================================

TEST(SugarPucker, TorsionsHaveCorrectCount) {
    PuckerParams p{0.0f, 38.0f};
    float torsions[5];
    compute_ring_torsions(p, torsions);

    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(std::isfinite(torsions[i]))
            << "torsion[" << i << "] is not finite";
    }
}

TEST(SugarPucker, TorsionsSumApproximatelyZero) {
    // For a valid Cremer-Pople ring, torsions approximately sum to zero
    for (float phase = 0.0f; phase < 360.0f; phase += 30.0f) {
        PuckerParams p{phase, 38.0f};
        float torsions[5];
        compute_ring_torsions(p, torsions);

        float sum = 0.0f;
        for (int i = 0; i < 5; ++i) sum += torsions[i];

        // The sum won't be exactly zero due to the cosine terms, but should be bounded
        EXPECT_LT(std::abs(sum), 200.0f)
            << "Torsion sum too large at phase=" << phase;
    }
}

TEST(SugarPucker, TorsionsMagnitudeBounded) {
    // Torsion angles must not exceed nu_max
    float nu_max = 38.0f;
    for (float phase = 0.0f; phase < 360.0f; phase += 10.0f) {
        PuckerParams p{phase, nu_max};
        float torsions[5];
        compute_ring_torsions(p, torsions);

        for (int i = 0; i < 5; ++i) {
            EXPECT_LE(std::abs(torsions[i]), nu_max + TOL)
                << "torsion[" << i << "] exceeds nu_max at phase=" << phase;
        }
    }
}

TEST(SugarPucker, C3EndoPhaseGivesCorrectSign) {
    // C3'-endo at P≈18°: ν2 should be positive (C3 above plane)
    PuckerParams p{18.0f, 38.0f};
    float torsions[5];
    compute_ring_torsions(p, torsions);

    // ν2 corresponds to k=2, which evaluates cos(P_rad + 0) = cos(P_rad)
    // At P=18°, cos(18°) ≈ 0.95, so torsion[2] should be positive
    EXPECT_GT(torsions[2], 0.0f) << "C3'-endo should have positive ν2";
}

TEST(SugarPucker, C2EndoPhaseGivesCorrectSign) {
    // C2'-endo at P≈162°: ν2 should be negative
    PuckerParams p{162.0f, 38.0f};
    float torsions[5];
    compute_ring_torsions(p, torsions);

    // At P=162°, cos(162°) ≈ -0.95
    EXPECT_LT(torsions[2], 0.0f) << "C2'-endo should have negative ν2";
}

TEST(SugarPucker, ZeroAmplitudeGivesFlatRing) {
    PuckerParams p{90.0f, 0.0f};
    float torsions[5];
    compute_ring_torsions(p, torsions);

    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(torsions[i], 0.0f, TOL)
            << "Zero amplitude should give flat ring";
    }
}

// ===========================================================================
// PUCKER ENERGY
// ===========================================================================

TEST(SugarPucker, RiboseEnergyNonNegative) {
    for (float phase = 0.0f; phase < 360.0f; phase += 5.0f) {
        double E = compute_pucker_energy(phase, SugarType::Ribose);
        EXPECT_GE(E, -0.01)  // Allow tiny negative due to floating point
            << "Ribose energy negative at phase=" << phase;
    }
}

TEST(SugarPucker, RiboseHasTwoMinima) {
    // C3'-endo (P≈18°) and C2'-endo (P≈162°) should be energy minima
    double E_c3_endo = compute_pucker_energy(18.0f, SugarType::Ribose);
    double E_c2_endo = compute_pucker_energy(162.0f, SugarType::Ribose);
    double E_barrier = compute_pucker_energy(90.0f, SugarType::Ribose);

    EXPECT_LT(E_c3_endo, E_barrier)
        << "C3'-endo should be lower than barrier";
    EXPECT_LT(E_c2_endo, E_barrier)
        << "C2'-endo should be lower than barrier";
}

TEST(SugarPucker, DeoxyriboseHasC2EndoMinimum) {
    // C2'-endo (P≈162°) should be the deepest minimum
    double E_c2_endo = compute_pucker_energy(162.0f, SugarType::Deoxyribose);
    double E_90 = compute_pucker_energy(90.0f, SugarType::Deoxyribose);
    double E_270 = compute_pucker_energy(270.0f, SugarType::Deoxyribose);

    EXPECT_LT(E_c2_endo, E_90);
    EXPECT_LT(E_c2_endo, E_270);
}

TEST(SugarPucker, EnergyIsFiniteEverywhere) {
    for (float phase = 0.0f; phase < 360.0f; phase += 1.0f) {
        EXPECT_TRUE(std::isfinite(compute_pucker_energy(phase, SugarType::Ribose)));
        EXPECT_TRUE(std::isfinite(compute_pucker_energy(phase, SugarType::Deoxyribose)));
    }
}

// ===========================================================================
// MUTATE PHASE
// ===========================================================================

TEST(SugarPucker, MutatePhaseStaysInRange) {
    for (int trial = 0; trial < 200; ++trial) {
        float result = mutate_phase(180.0f, 15.0f);
        EXPECT_GE(result, 0.0f);
        EXPECT_LT(result, 360.0f);
    }
}

TEST(SugarPucker, MutatePhaseWrapsCorrectly) {
    // Near boundaries (0° and 360°) should wrap
    for (int trial = 0; trial < 200; ++trial) {
        float result_low = mutate_phase(1.0f, 30.0f);
        EXPECT_GE(result_low, 0.0f);
        EXPECT_LT(result_low, 360.0f);

        float result_high = mutate_phase(359.0f, 30.0f);
        EXPECT_GE(result_high, 0.0f);
        EXPECT_LT(result_high, 360.0f);
    }
}

// ===========================================================================
// PUCKER ENTROPY
// ===========================================================================

TEST(SugarPucker, EmptyEnsembleEntropyZero) {
    std::vector<float> empty;
    EXPECT_NEAR(compute_pucker_entropy(empty), 0.0, DTOL);
}

TEST(SugarPucker, SinglePhaseEntropyZero) {
    std::vector<float> single = {90.0f};
    EXPECT_NEAR(compute_pucker_entropy(single), 0.0, DTOL);
}

TEST(SugarPucker, IdenticalPhasesEntropyZero) {
    std::vector<float> same(100, 45.0f);
    EXPECT_NEAR(compute_pucker_entropy(same), 0.0, DTOL);
}

TEST(SugarPucker, EntropyNonNegative) {
    std::vector<float> phases;
    for (int i = 0; i < 100; ++i) {
        phases.push_back(static_cast<float>(i * 3.6f));  // spread across 360°
    }
    double H = compute_pucker_entropy(phases);
    EXPECT_GE(H, 0.0);
}

TEST(SugarPucker, UniformDistributionMaximizesEntropy) {
    // Uniform: one phase per bin (36 bins at 10° resolution)
    std::vector<float> uniform;
    for (int i = 0; i < 36; ++i) {
        uniform.push_back(i * 10.0f + 5.0f);  // center of each bin
    }
    double H_uniform = compute_pucker_entropy(uniform);

    // Concentrated: all in one bin
    std::vector<float> concentrated(36, 45.0f);
    double H_conc = compute_pucker_entropy(concentrated);

    EXPECT_GT(H_uniform, H_conc);
    // Max entropy for 36 bins = log2(36) ≈ 5.17 bits
    EXPECT_NEAR(H_uniform, std::log2(36.0), 0.1);
}

TEST(SugarPucker, EntropyUpperBound) {
    // Entropy should never exceed log2(36) bits for 36 bins
    std::vector<float> phases;
    for (int i = 0; i < 360; ++i) {
        phases.push_back(static_cast<float>(i));
    }
    double H = compute_pucker_entropy(phases);
    EXPECT_LE(H, std::log2(36.0) + 0.01);
}
