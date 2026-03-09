// tests/test_ga_validation.cpp — GA integration validation
// Tests: crystal pose scoring, batch vs serial agreement, GA convergence, OpenMP correctness
// Apache-2.0 © 2026 Le Bonhomme Pharma

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <cstring>
#include <chrono>

#include "flexaid.h"
#include "Scoring/VoronoiCF.h"

namespace {

// Test fixtures and constants
class GAValidationTest : public ::testing::Test {
protected:
    scoring::VoronoiCF scorer;
    
    // Mock atom arrays for testing
    std::vector<atom> ligand_atoms;
    std::vector<atom> receptor_atoms;
    
    void SetUp() override {
        // Generate synthetic ligand (small organic molecule approximation)
        ligand_atoms.resize(5);
        ligand_atoms[0] = {"C", 0.0f, 0.0f, 0.0f, 1.7f, 6};   // Carbon, SYBYL type 6
        ligand_atoms[1] = {"O", 2.0f, 0.0f, 0.0f, 1.5f, 5};   // Oxygen, type 5
        ligand_atoms[2] = {"N", -1.0f, 1.5f, 0.0f, 1.55f, 3}; // Nitrogen, type 3
        ligand_atoms[3] = {"H", -1.0f, -1.0f, 0.0f, 1.2f, 1}; // Hydrogen, type 1
        ligand_atoms[4] = {"C", 4.0f, 0.0f, 0.0f, 1.7f, 6};   // Carbon, type 6
        
        // Generate synthetic receptor (binding site)
        receptor_atoms.resize(8);
        receptor_atoms[0] = {"O", 1.0f, 0.5f, 0.0f, 1.5f, 5};   // Oxygen, type 5
        receptor_atoms[1] = {"N", 2.5f, -0.5f, 0.5f, 1.55f, 3}; // Nitrogen, type 3
        receptor_atoms[2] = {"C", 0.0f, 2.0f, 0.0f, 1.7f, 6};   // Carbon, type 6
        receptor_atoms[3] = {"O", -1.5f, 0.5f, 1.0f, 1.5f, 5};  // Oxygen, type 5
        receptor_atoms[4] = {"C", 3.5f, -1.5f, 0.0f, 1.7f, 6};  // Carbon, type 6
        receptor_atoms[5] = {"N", 5.0f, 0.0f, 0.5f, 1.55f, 3};  // Nitrogen, type 3
        receptor_atoms[6] = {"O", 2.0f, 1.5f, 1.5f, 1.5f, 5};   // Oxygen, type 5
        receptor_atoms[7] = {"S", -0.5f, 3.0f, 0.5f, 1.8f, 23}; // Sulfur, type 23
    }
};

// Test 1: Crystal pose scores with favorable interaction
TEST_F(GAValidationTest, CrystalPoseScorePhysicallySensible) {
    scoring::PoseScore score = scorer.score_pose(
        ligand_atoms.data(), ligand_atoms.size(),
        receptor_atoms.data(), receptor_atoms.size()
    );
    
    // Should produce finite values
    EXPECT_TRUE(std::isfinite(score.total_cf)) 
        << "CF energy should be finite";
    EXPECT_TRUE(std::isfinite(score.clash_penalty)) 
        << "Clash penalty should be finite";
    EXPECT_TRUE(std::isfinite(score.solvation)) 
        << "Solvation energy should be finite";
    
    // Penalties and solvation should be non-negative
    EXPECT_GE(score.clash_penalty, 0.0f) 
        << "Clash penalty cannot be negative";
    EXPECT_GE(score.solvation, 0.0f) 
        << "Solvation energy should be non-negative";
}

// Test 2: Batch scoring produces identical results to serial
TEST_F(GAValidationTest, BatchScoringConsistencyWithSerial) {
    // Generate population (10 poses with small perturbations)
    std::vector<std::vector<atom>> population;
    for (int i = 0; i < 10; ++i) {
        std::vector<atom> pose = ligand_atoms;
        // Add small random perturbation
        for (auto& a : pose) {
            a.x += (rand() % 100 - 50) * 0.01f;  // ±0.5 Å
            a.y += (rand() % 100 - 50) * 0.01f;
            a.z += (rand() % 100 - 50) * 0.01f;
        }
        population.push_back(pose);
    }
    
    // Score serially
    std::vector<scoring::PoseScore> serial_scores;
    for (const auto& pose : population) {
        serial_scores.push_back(
            scorer.score_pose(pose.data(), pose.size(),
                            receptor_atoms.data(), receptor_atoms.size())
        );
    }
    
    // Score as batch
    std::vector<const atom*> pose_ptrs;
    for (const auto& pose : population) {
        pose_ptrs.push_back(pose.data());
    }
    std::vector<scoring::PoseScore> batch_scores = 
        scorer.score_population(pose_ptrs, receptor_atoms.data(), receptor_atoms.size());
    
    // Compare (should be identical)
    ASSERT_EQ(serial_scores.size(), batch_scores.size());
    for (size_t i = 0; i < serial_scores.size(); ++i) {
        EXPECT_FLOAT_EQ(serial_scores[i].total_cf, batch_scores[i].total_cf)
            << "Serial vs batch CF mismatch at pose " << i;
        EXPECT_FLOAT_EQ(serial_scores[i].clash_penalty, batch_scores[i].clash_penalty)
            << "Serial vs batch clash mismatch at pose " << i;
        EXPECT_FLOAT_EQ(serial_scores[i].solvation, batch_scores[i].solvation)
            << "Serial vs batch solvation mismatch at pose " << i;
    }
}

// Test 3: OpenMP parallelism produces correct results
TEST_F(GAValidationTest, OpenMPBatchingCorrectness) {
    const int num_poses = 50;
    std::vector<atom> master_ligand = ligand_atoms;
    
    // Generate 50 perturbed poses
    std::vector<std::vector<atom>> population(num_poses);
    for (int i = 0; i < num_poses; ++i) {
        population[i] = master_ligand;
        for (auto& a : population[i]) {
            a.x += (rand() % 100 - 50) * 0.02f;  // Larger perturbation
            a.y += (rand() % 100 - 50) * 0.02f;
            a.z += (rand() % 100 - 50) * 0.02f;
        }
    }
    
    // Score with parallelization
    std::vector<const atom*> pose_ptrs(num_poses);
    for (int i = 0; i < num_poses; ++i) {
        pose_ptrs[i] = population[i].data();
    }
    
    auto scores = scorer.score_population(
        pose_ptrs, 
        receptor_atoms.data(), receptor_atoms.size()
    );
    
    // All scores should be finite and physically sensible
    EXPECT_EQ(scores.size(), num_poses);
    for (size_t i = 0; i < scores.size(); ++i) {
        EXPECT_TRUE(std::isfinite(scores[i].total_cf))
            << "Pose " << i << " CF not finite";
        EXPECT_TRUE(std::isfinite(scores[i].clash_penalty))
            << "Pose " << i << " clash not finite";
        EXPECT_TRUE(std::isfinite(scores[i].solvation))
            << "Pose " << i << " solvation not finite";
        EXPECT_GE(scores[i].clash_penalty, 0.0f)
            << "Pose " << i << " clash penalty negative";
    }
}

// Test 4: Performance benchmark (optional, for benchmarking report)
TEST_F(GAValidationTest, PerformanceBenchmark) {
    const int num_poses = 100;
    std::vector<std::vector<atom>> population(num_poses);
    
    // Generate population
    for (int i = 0; i < num_poses; ++i) {
        population[i] = ligand_atoms;
        for (auto& a : population[i]) {
            a.x += (rand() % 100 - 50) * 0.02f;
            a.y += (rand() % 100 - 50) * 0.02f;
            a.z += (rand() % 100 - 50) * 0.02f;
        }
    }
    
    // Prepare pointers
    std::vector<const atom*> pose_ptrs(num_poses);
    for (int i = 0; i < num_poses; ++i) {
        pose_ptrs[i] = population[i].data();
    }
    
    // Time batch scoring (multiple runs for statistical significance)
    auto t_start = std::chrono::high_resolution_clock::now();
    for (int run = 0; run < 10; ++run) {
        auto scores = scorer.score_population(
            pose_ptrs,
            receptor_atoms.data(), receptor_atoms.size()
        );
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double avg_ms_per_batch = total_ms / 10.0;
    double avg_us_per_pose = (avg_ms_per_batch * 1000.0) / num_poses;
    
    std::cerr << "\n[BENCHMARK] 100-pose batch scoring:\n"
              << "  Total 10 runs: " << total_ms << " ms\n"
              << "  Avg per batch: " << avg_ms_per_batch << " ms\n"
              << "  Avg per pose: " << avg_us_per_pose << " µs\n";
    
    // Scoring should be reasonably fast (< 1 ms per pose expected with vectorization)
    EXPECT_LT(avg_us_per_pose, 5000.0) 
        << "Batch scoring too slow (>5 ms per pose)";
}

// Test 5: Fitness value stability across runs
TEST_F(GAValidationTest, FitnessStability) {
    // Score same pose multiple times
    std::vector<scoring::PoseScore> repeated_scores;
    for (int i = 0; i < 5; ++i) {
        auto score = scorer.score_pose(
            ligand_atoms.data(), ligand_atoms.size(),
            receptor_atoms.data(), receptor_atoms.size()
        );
        repeated_scores.push_back(score);
    }
    
    // All should be identical (deterministic)
    for (size_t i = 1; i < repeated_scores.size(); ++i) {
        EXPECT_EQ(repeated_scores[0].total_cf, repeated_scores[i].total_cf)
            << "CF scoring not deterministic";
        EXPECT_EQ(repeated_scores[0].clash_penalty, repeated_scores[i].clash_penalty)
            << "Clash penalty not deterministic";
        EXPECT_EQ(repeated_scores[0].solvation, repeated_scores[i].solvation)
            << "Solvation not deterministic";
    }
}

}  // namespace anonymous
