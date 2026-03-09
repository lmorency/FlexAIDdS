// tests/test_binding_mode_statmech.cpp
// Unit tests for BindingMode ↔ StatMechEngine integration
// Part of FlexAIDΔS Phase 1 implementation roadmap
// Apache-2.0 © 2026 Le Bonhomme Pharma

#include <gtest/gtest.h>
#include "../LIB/BindingMode.h"
#include "../LIB/statmech.h"
#include "../LIB/gaboom.h"
#include <cmath>
#include <vector>
#include <chrono>

// ===========================================================================
// TEST FIXTURES
// ===========================================================================

class BindingModeStatMechTest : public ::testing::Test {
protected:
    // Mock structures for testing
    FA_Global* mock_fa;
    GB_Global* mock_gb;
    VC_Global* mock_vc;
    chromosome* mock_chroms;
    genlim* mock_gene_lim;
    atom* mock_atoms;
    resid* mock_residue;
    gridpoint* mock_cleftgrid;
    
    BindingPopulation* test_population;
    
    const double TEST_TEMPERATURE = 300.0;  // Kelvin
    const double EPSILON = 1e-6;  // Numerical tolerance
    
    void SetUp() override {
        // Initialize minimal mock structures
        mock_fa = new FA_Global();
        mock_fa->temperature = TEST_TEMPERATURE;
        mock_fa->num_atoms = 10;
        mock_fa->num_residues = 2;
        
        mock_gb = new GB_Global();
        mock_gb->num_genes = 6;  // 3 translation + 3 rotation
        
        mock_vc = new VC_Global();
        
        mock_chroms = new chromosome[5];  // 5 test chromosomes
        mock_gene_lim = new genlim[mock_gb->num_genes];
        mock_atoms = new atom[mock_fa->num_atoms];
        mock_residue = new resid[mock_fa->num_residues];
        mock_cleftgrid = new gridpoint[100];
        
        // Create test population
        test_population = new BindingPopulation(
            mock_fa, mock_gb, mock_vc,
            mock_chroms, mock_gene_lim,
            mock_atoms, mock_residue, mock_cleftgrid,
            5  // n_chrom
        );
    }
    
    void TearDown() override {
        delete test_population;
        delete[] mock_cleftgrid;
        delete[] mock_residue;
        delete[] mock_atoms;
        delete[] mock_gene_lim;
        delete[] mock_chroms;
        delete mock_vc;
        delete mock_gb;
        delete mock_fa;
    }
    
    // Helper: Create mock pose with specific CF
    Pose create_mock_pose(double cf_value, int index) {
        std::vector<float> empty_vec;
        Pose p(&mock_chroms[index], index, 0, 0.0, TEST_TEMPERATURE, empty_vec);
        p.CF = cf_value;
        return p;
    }
};

// ===========================================================================
// CORE FUNCTIONALITY TESTS
// ===========================================================================

TEST_F(BindingModeStatMechTest, LazyEngineRebuild) {
    BindingMode mode(test_population);
    
    // Add poses with known CF values
    for (int i = 0; i < 3; ++i) {
        Pose p = create_mock_pose(-10.0 - i * 2.0, i);
        mode.add_Pose(p);
    }
    
    // First call should build engine
    EXPECT_FALSE(mode.thermo_cache_valid_);  // Initially invalid
    auto thermo1 = mode.get_thermodynamics();
    EXPECT_TRUE(mode.thermo_cache_valid_);   // Now valid
    
    // Second call should reuse cache
    auto thermo2 = mode.get_thermodynamics();
    EXPECT_EQ(thermo1.free_energy, thermo2.free_energy);
    
    // Adding pose should invalidate cache
    Pose new_pose = create_mock_pose(-8.0, 3);
    mode.add_Pose(new_pose);
    EXPECT_FALSE(mode.thermo_cache_valid_);
}

TEST_F(BindingModeStatMechTest, ConsistencyWithLegacy) {
    BindingMode mode(test_population);
    
    // Add diverse CF distribution
    std::vector<double> cf_values = {-15.0, -12.0, -10.0, -8.0, -6.0};
    for (size_t i = 0; i < cf_values.size(); ++i) {
        Pose p = create_mock_pose(cf_values[i], i);
        mode.add_Pose(p);
    }
    
    // Legacy and new API should give identical results
    double legacy_energy = mode.compute_energy();
    double new_energy = mode.get_free_energy();
    EXPECT_NEAR(legacy_energy, new_energy, EPSILON);
    
    double legacy_enthalpy = mode.compute_enthalpy();
    auto thermo = mode.get_thermodynamics();
    EXPECT_NEAR(legacy_enthalpy, thermo.mean_energy, EPSILON);
    
    double legacy_entropy = mode.compute_entropy();
    EXPECT_NEAR(legacy_entropy, thermo.entropy, EPSILON);
}

TEST_F(BindingModeStatMechTest, BoltzmannWeightsNormalization) {
    BindingMode mode(test_population);
    
    // Add poses with known CF distribution
    std::vector<double> cf_values = {-20.0, -15.0, -10.0, -5.0};
    for (size_t i = 0; i < cf_values.size(); ++i) {
        Pose p = create_mock_pose(cf_values[i], i);
        mode.add_Pose(p);
    }
    
    auto weights = mode.get_boltzmann_weights();
    
    // Sum of weights should be 1.0 (within numerical tolerance)
    double sum = 0.0;
    for (double w : weights) {
        sum += w;
        EXPECT_GE(w, 0.0);  // All weights non-negative
    }
    EXPECT_NEAR(sum, 1.0, EPSILON);
    
    // Lowest CF should have highest weight
    double min_cf = *std::min_element(cf_values.begin(), cf_values.end());
    size_t min_idx = std::distance(cf_values.begin(), 
                                   std::find(cf_values.begin(), cf_values.end(), min_cf));
    double max_weight = *std::max_element(weights.begin(), weights.end());
    EXPECT_NEAR(weights[min_idx], max_weight, EPSILON);
}

TEST_F(BindingModeStatMechTest, EntropyBehavior) {
    BindingMode mode_sharp(test_population);
    BindingMode mode_broad(test_population);
    
    // Sharp distribution: all CFs similar
    for (int i = 0; i < 5; ++i) {
        Pose p = create_mock_pose(-10.0 - 0.1 * i, i);
        mode_sharp.add_Pose(p);
    }
    
    // Broad distribution: CFs widely spread
    std::vector<double> broad_cfs = {-20.0, -15.0, -10.0, -5.0, 0.0};
    for (size_t i = 0; i < broad_cfs.size(); ++i) {
        Pose p = create_mock_pose(broad_cfs[i], i);
        mode_broad.add_Pose(p);
    }
    
    double sharp_entropy = mode_sharp.compute_entropy();
    double broad_entropy = mode_broad.compute_entropy();
    
    // Broad distribution should have higher entropy
    EXPECT_GT(broad_entropy, sharp_entropy);
}

// ===========================================================================
// BINDINGPOPULATION TESTS
// ===========================================================================

TEST_F(BindingModeStatMechTest, DeltaGCalculation) {
    BindingMode mode1(test_population);
    BindingMode mode2(test_population);
    
    // Mode 1: lower energy (more favorable)
    for (int i = 0; i < 3; ++i) {
        Pose p = create_mock_pose(-15.0 - i, i);
        mode1.add_Pose(p);
    }
    
    // Mode 2: higher energy (less favorable)
    for (int i = 0; i < 3; ++i) {
        Pose p = create_mock_pose(-10.0 - i, i);
        mode2.add_Pose(p);
    }
    
    double delta_g = test_population->compute_delta_G(mode1, mode2);
    
    // ΔG should be positive (mode2 higher energy than mode1)
    EXPECT_GT(delta_g, 0.0);
}

TEST_F(BindingModeStatMechTest, GlobalEnsemble) {
    // Create two binding modes
    BindingMode mode1(test_population);
    for (int i = 0; i < 3; ++i) {
        Pose p = create_mock_pose(-15.0 - i, i);
        mode1.add_Pose(p);
    }
    
    BindingMode mode2(test_population);
    for (int i = 0; i < 2; ++i) {
        Pose p = create_mock_pose(-10.0 - i, i);
        mode2.add_Pose(p);
    }
    
    test_population->add_BindingMode(mode1);
    test_population->add_BindingMode(mode2);
    
    // Get global ensemble
    auto global_engine = test_population->get_global_ensemble();
    auto global_thermo = global_engine.get_thermodynamics();
    
    // Global ensemble should aggregate all 5 poses
    EXPECT_GT(global_engine.get_partition_function(), 0.0);
    EXPECT_LT(global_thermo.free_energy, 0.0);  // Should be negative (favorable)
}

// ===========================================================================
// CACHE INVALIDATION TESTS
// ===========================================================================

TEST_F(BindingModeStatMechTest, CacheInvalidationOnClear) {
    BindingMode mode(test_population);
    
    for (int i = 0; i < 3; ++i) {
        Pose p = create_mock_pose(-10.0 - i, i);
        mode.add_Pose(p);
    }
    
    // Build cache
    mode.get_thermodynamics();
    EXPECT_TRUE(mode.thermo_cache_valid_);
    
    // Clear should invalidate
    mode.clear_Poses();
    EXPECT_FALSE(mode.thermo_cache_valid_);
    EXPECT_EQ(mode.get_BindingMode_size(), 0);
}

TEST_F(BindingModeStatMechTest, MultipleRebuilds) {
    BindingMode mode(test_population);
    
    // First batch of poses
    for (int i = 0; i < 2; ++i) {
        Pose p = create_mock_pose(-15.0 - i, i);
        mode.add_Pose(p);
    }
    double energy1 = mode.get_free_energy();
    
    // Add more poses (invalidates cache)
    for (int i = 2; i < 4; ++i) {
        Pose p = create_mock_pose(-15.0 - i, i);
        mode.add_Pose(p);
    }
    double energy2 = mode.get_free_energy();
    
    // Energy should change (more states in ensemble)
    EXPECT_NE(energy1, energy2);
}

// ===========================================================================
// EDGE CASES
// ===========================================================================

TEST_F(BindingModeStatMechTest, EmptyMode) {
    BindingMode mode(test_population);
    
    // Empty mode should handle gracefully
    EXPECT_EQ(mode.get_BindingMode_size(), 0);
    
    // Calling thermodynamics on empty mode should not crash
    auto thermo = mode.get_thermodynamics();
    // Behavior on empty ensemble is implementation-defined
}

TEST_F(BindingModeStatMechTest, SinglePoseMode) {
    BindingMode mode(test_population);
    
    Pose p = create_mock_pose(-12.0, 0);
    mode.add_Pose(p);
    
    auto thermo = mode.get_thermodynamics();
    
    // Single state: entropy should be zero (no uncertainty)
    EXPECT_NEAR(thermo.entropy, 0.0, EPSILON);
    // Mean energy should equal the single CF value
    EXPECT_NEAR(thermo.mean_energy, -12.0, EPSILON);
}

TEST_F(BindingModeStatMechTest, HighTemperatureBehavior) {
    // Test at very high temperature (should flatten Boltzmann distribution)
    mock_fa->temperature = 1000.0;  // Very high T
    BindingPopulation* hot_population = new BindingPopulation(
        mock_fa, mock_gb, mock_vc,
        mock_chroms, mock_gene_lim,
        mock_atoms, mock_residue, mock_cleftgrid, 5
    );
    
    BindingMode mode(hot_population);
    
    std::vector<double> cf_values = {-20.0, -10.0, 0.0};
    for (size_t i = 0; i < cf_values.size(); ++i) {
        Pose p = create_mock_pose(cf_values[i], i);
        mode.add_Pose(p);
    }
    
    auto weights = mode.get_boltzmann_weights();
    
    // At high T, weights should be more uniform
    double max_weight = *std::max_element(weights.begin(), weights.end());
    double min_weight = *std::min_element(weights.begin(), weights.end());
    double weight_ratio = max_weight / min_weight;
    
    // Ratio should be smaller at high T than at low T
    EXPECT_LT(weight_ratio, 10.0);  // Reasonable threshold for T=1000K
    
    delete hot_population;
}

// ===========================================================================
// PERFORMANCE TESTS (Optional, disabled in CI)
// ===========================================================================

TEST_F(BindingModeStatMechTest, DISABLED_CachePerformance) {
    // Verifies that caching provides performance benefit
    // Disabled by default (use --gtest_also_run_disabled_tests to run)
    
    BindingMode mode(test_population);
    
    // Large number of poses
    for (int i = 0; i < 1000; ++i) {
        Pose p = create_mock_pose(-20.0 + i * 0.01, i % 5);
        mode.add_Pose(p);
    }
    
    // Time first call (rebuilds engine)
    auto start1 = std::chrono::high_resolution_clock::now();
    mode.get_thermodynamics();
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    
    // Time second call (uses cache)
    auto start2 = std::chrono::high_resolution_clock::now();
    mode.get_thermodynamics();
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    
    // Cached call should be significantly faster
    EXPECT_LT(duration2.count(), duration1.count() / 10);
}

// ===========================================================================
// MAIN
// ===========================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
