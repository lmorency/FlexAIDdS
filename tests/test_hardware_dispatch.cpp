// tests/test_hardware_dispatch.cpp
// Unit tests for the ShannonThermoStack hardware dispatch layer
// Validates Shannon entropy computation, dispatch backend reporting,
// edge cases, and the ShannonEnergyMatrix singleton.
// Apache-2.0 © 2026 Le Bonhomme Pharma

#include <gtest/gtest.h>
#include "../LIB/ShannonThermoStack/ShannonThermoStack.h"
#include "../LIB/statmech.h"
#include "../LIB/tencm.h"
#include <cmath>
#include <vector>
#include <numeric>
#include <random>
#include <string>

using namespace shannon_thermo;

// ===========================================================================
// CONSTANTS
// ===========================================================================

static constexpr double EPSILON = 1e-6;

// ===========================================================================
// SHANNON ENTROPY — BASIC PROPERTIES
// ===========================================================================

TEST(ShannonEntropy, EmptyInputReturnsZero) {
    std::vector<double> empty;
    EXPECT_DOUBLE_EQ(compute_shannon_entropy(empty), 0.0);
}

TEST(ShannonEntropy, SingleValueReturnsZero) {
    // One sample → one bin with 100% → H = 0
    std::vector<double> single = {5.0};
    EXPECT_DOUBLE_EQ(compute_shannon_entropy(single), 0.0);
}

TEST(ShannonEntropy, IdenticalValuesReturnZero) {
    // All values in one bin → H = 0
    std::vector<double> same(100, 3.14);
    EXPECT_DOUBLE_EQ(compute_shannon_entropy(same), 0.0);
}

TEST(ShannonEntropy, NonNegative) {
    // Shannon entropy is always >= 0
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 5.0);
    std::vector<double> values(500);
    for (auto& v : values) v = dist(rng);

    double H = compute_shannon_entropy(values);
    EXPECT_GE(H, 0.0);
}

TEST(ShannonEntropy, UniformDistributionMaxEntropy) {
    // For a uniform distribution across num_bins bins:
    //   H = log2(num_bins)
    int num_bins = 10;
    int per_bin = 100;
    std::vector<double> values;
    values.reserve(num_bins * per_bin);

    // Create values that land evenly in each bin
    for (int b = 0; b < num_bins; ++b) {
        double center = static_cast<double>(b) + 0.5;
        for (int i = 0; i < per_bin; ++i)
            values.push_back(center);
    }

    double H = compute_shannon_entropy(values, num_bins);
    double H_max = std::log2(static_cast<double>(num_bins));
    EXPECT_NEAR(H, H_max, 0.1);  // tolerance for binning edge effects
}

TEST(ShannonEntropy, UpperBound) {
    // H <= log2(num_bins) for any distribution
    int num_bins = 20;
    std::mt19937 rng(123);
    std::normal_distribution<double> dist(0.0, 10.0);
    std::vector<double> values(1000);
    for (auto& v : values) v = dist(rng);

    double H = compute_shannon_entropy(values, num_bins);
    double H_max = std::log2(static_cast<double>(num_bins));
    EXPECT_LE(H, H_max + EPSILON);
}

TEST(ShannonEntropy, MoreBinsHigherEntropy) {
    // For the same data, more bins generally gives higher entropy
    std::mt19937 rng(99);
    std::uniform_real_distribution<double> dist(0.0, 100.0);
    std::vector<double> values(5000);
    for (auto& v : values) v = dist(rng);

    double H_5  = compute_shannon_entropy(values, 5);
    double H_20 = compute_shannon_entropy(values, 20);

    EXPECT_GT(H_20, H_5);
}

TEST(ShannonEntropy, DefaultBinsIsValid) {
    // Calling without explicit num_bins should work (DEFAULT_HIST_BINS = 20)
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0};
    double H = compute_shannon_entropy(values);
    EXPECT_GE(H, 0.0);
    EXPECT_TRUE(std::isfinite(H));
}

TEST(ShannonEntropy, NegativeBinsDefaultsGracefully) {
    // num_bins <= 0 should be corrected to DEFAULT_HIST_BINS
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0};
    double H = compute_shannon_entropy(values, -1);
    EXPECT_GE(H, 0.0);
    EXPECT_TRUE(std::isfinite(H));
}

// ===========================================================================
// SHANNON ENTROPY — DISCRETE VERSION
// ===========================================================================

TEST(ShannonEntropyDiscrete, EmptyInput) {
    std::vector<int> empty;
    EXPECT_DOUBLE_EQ(compute_shannon_entropy_discrete(empty), 0.0);
}

TEST(ShannonEntropyDiscrete, AllInOneBin) {
    std::vector<int> counts = {100, 0, 0, 0};
    EXPECT_DOUBLE_EQ(compute_shannon_entropy_discrete(counts), 0.0);
}

TEST(ShannonEntropyDiscrete, UniformCounts) {
    // 4 bins, equal counts → H = log2(4) = 2 bits
    std::vector<int> counts = {100, 100, 100, 100};
    double H = compute_shannon_entropy_discrete(counts);
    EXPECT_NEAR(H, 2.0, 0.01);
}

TEST(ShannonEntropyDiscrete, TwoBinsEqual) {
    // 2 equal bins → H = 1 bit
    std::vector<int> counts = {50, 50};
    double H = compute_shannon_entropy_discrete(counts);
    EXPECT_NEAR(H, 1.0, 0.01);
}

TEST(ShannonEntropyDiscrete, NonNegative) {
    std::vector<int> counts = {10, 20, 30, 5, 1};
    EXPECT_GE(compute_shannon_entropy_discrete(counts), 0.0);
}

// ===========================================================================
// SHANNON ENERGY MATRIX — SINGLETON & INITIALIZATION
// ===========================================================================

TEST(ShannonEnergyMatrix, SingletonReturnsSameInstance) {
    auto& m1 = ShannonEnergyMatrix::instance();
    auto& m2 = ShannonEnergyMatrix::instance();
    EXPECT_EQ(&m1, &m2);
}

TEST(ShannonEnergyMatrix, InitialiseIsIdempotent) {
    auto& mat = ShannonEnergyMatrix::instance();
    mat.initialise();
    EXPECT_TRUE(mat.is_initialised());

    // Second call should be no-op (early return)
    double v_before = mat.lookup(0, 0);
    mat.initialise();
    double v_after = mat.lookup(0, 0);
    EXPECT_DOUBLE_EQ(v_before, v_after);
}

TEST(ShannonEnergyMatrix, LookupValuesAreFinite) {
    auto& mat = ShannonEnergyMatrix::instance();
    mat.initialise();

    // Spot-check several entries
    for (int i = 0; i < SHANNON_BINS; i += 32) {
        for (int j = 0; j < SHANNON_BINS; j += 32) {
            double v = mat.lookup(i, j);
            EXPECT_TRUE(std::isfinite(v))
                << "Non-finite value at (" << i << "," << j << ")";
        }
    }
}

TEST(ShannonEnergyMatrix, LookupDeterministic) {
    // Matrix is seeded with 42 → same values every time
    auto& mat = ShannonEnergyMatrix::instance();
    mat.initialise();

    double v1 = mat.lookup(10, 20);
    double v2 = mat.lookup(10, 20);
    EXPECT_DOUBLE_EQ(v1, v2);
}

TEST(ShannonEnergyMatrix, DiagonalEntriesFinite) {
    auto& mat = ShannonEnergyMatrix::instance();
    mat.initialise();

    for (int i = 0; i < SHANNON_BINS; i += 16) {
        EXPECT_TRUE(std::isfinite(mat.lookup(i, i)));
    }
}

// ===========================================================================
// TORSIONAL VIBRATIONAL ENTROPY
// ===========================================================================

TEST(TorsionalVibEntropy, EmptyModesReturnsZero) {
    std::vector<tencm::NormalMode> empty;
    EXPECT_DOUBLE_EQ(compute_torsional_vibrational_entropy(empty), 0.0);
}

TEST(TorsionalVibEntropy, SkipsFirstSixModes) {
    // First 6 modes (translation+rotation) should be skipped
    // If we provide exactly 6 modes, result is 0
    std::vector<tencm::NormalMode> modes(6);
    for (int i = 0; i < 6; ++i)
        modes[i].eigenvalue = 1.0;  // non-trivial eigenvalues

    EXPECT_DOUBLE_EQ(compute_torsional_vibrational_entropy(modes), 0.0);
}

TEST(TorsionalVibEntropy, SkipsNearZeroEigenvalues) {
    // Modes with eigenvalue < 1e-6 are skipped
    std::vector<tencm::NormalMode> modes(10);
    for (int i = 0; i < 10; ++i)
        modes[i].eigenvalue = 1e-9;  // effectively zero

    EXPECT_DOUBLE_EQ(compute_torsional_vibrational_entropy(modes), 0.0);
}

TEST(TorsionalVibEntropy, ValidModesProducePositiveEntropy) {
    // Modes 6+ with reasonable eigenvalues should give S > 0
    std::vector<tencm::NormalMode> modes(12);
    for (int i = 0; i < 12; ++i)
        modes[i].eigenvalue = 0.5 + 0.1 * i;

    double S = compute_torsional_vibrational_entropy(modes, 300.0);
    EXPECT_GT(S, 0.0);
    EXPECT_TRUE(std::isfinite(S));
}

TEST(TorsionalVibEntropy, HigherTemperatureHigherEntropy) {
    std::vector<tencm::NormalMode> modes(10);
    for (int i = 0; i < 10; ++i)
        modes[i].eigenvalue = 1.0;

    double S_low  = compute_torsional_vibrational_entropy(modes, 200.0);
    double S_high = compute_torsional_vibrational_entropy(modes, 500.0);

    EXPECT_GT(S_high, S_low);
}

TEST(TorsionalVibEntropy, ResultIsFiniteForLargeModes) {
    std::vector<tencm::NormalMode> modes(100);
    for (int i = 0; i < 100; ++i)
        modes[i].eigenvalue = 0.01 * (i + 1);

    double S = compute_torsional_vibrational_entropy(modes, 298.15);
    EXPECT_TRUE(std::isfinite(S));
}

// ===========================================================================
// RUN_SHANNON_THERMO_STACK — FULL PIPELINE
// ===========================================================================

class ShannonThermoStackTest : public ::testing::Test {
protected:
    statmech::StatMechEngine engine{298.15};
    tencm::TorsionalENM tencm_model;  // default-constructed, not built

    void SetUp() override {
        // Populate with a realistic ensemble of energies
        engine.add_sample(-15.0);
        engine.add_sample(-12.0);
        engine.add_sample(-10.0);
        engine.add_sample(-8.0);
        engine.add_sample(-6.0);
    }
};

TEST_F(ShannonThermoStackTest, ProducesFiniteResults) {
    auto result = run_shannon_thermo_stack(engine, tencm_model, -10.0);

    EXPECT_TRUE(std::isfinite(result.deltaG));
    EXPECT_TRUE(std::isfinite(result.shannonEntropy));
    EXPECT_TRUE(std::isfinite(result.torsionalVibEntropy));
    EXPECT_TRUE(std::isfinite(result.entropyContribution));
}

TEST_F(ShannonThermoStackTest, ShannonEntropyNonNegative) {
    auto result = run_shannon_thermo_stack(engine, tencm_model, -10.0);
    EXPECT_GE(result.shannonEntropy, 0.0);
}

TEST_F(ShannonThermoStackTest, TorsionalEntropyZeroWhenNotBuilt) {
    // Default-constructed TorsionalENM is not built → S_vib = 0
    EXPECT_FALSE(tencm_model.is_built());
    auto result = run_shannon_thermo_stack(engine, tencm_model, -10.0);
    EXPECT_DOUBLE_EQ(result.torsionalVibEntropy, 0.0);
}

TEST_F(ShannonThermoStackTest, EntropyContributionIsNegative) {
    // -T*S <= 0 (entropy always contributes favorably to free energy)
    auto result = run_shannon_thermo_stack(engine, tencm_model, -10.0);
    EXPECT_LE(result.entropyContribution, 0.0 + EPSILON);
}

TEST_F(ShannonThermoStackTest, DeltaGIncorporatesEntropy) {
    double base_dG = -10.0;
    auto result = run_shannon_thermo_stack(engine, tencm_model, base_dG);

    // deltaG = base_dG + entropy_contribution
    EXPECT_NEAR(result.deltaG, base_dG + result.entropyContribution, EPSILON);
}

TEST_F(ShannonThermoStackTest, ReportContainsBackendName) {
    auto result = run_shannon_thermo_stack(engine, tencm_model, -10.0);

    // Report must contain "ShannonThermoStack["
    EXPECT_NE(result.report.find("ShannonThermoStack["), std::string::npos);

    // Must contain one of the known backend names
    bool has_backend =
        result.report.find("CUDA") != std::string::npos ||
        result.report.find("Metal") != std::string::npos ||
        result.report.find("AVX-512") != std::string::npos ||
        result.report.find("OpenMP") != std::string::npos ||
        result.report.find("scalar") != std::string::npos;
    EXPECT_TRUE(has_backend) << "Report missing backend: " << result.report;
}

TEST_F(ShannonThermoStackTest, ReportContainsMetrics) {
    auto result = run_shannon_thermo_stack(engine, tencm_model, -10.0);

    EXPECT_NE(result.report.find("S_conf="), std::string::npos);
    EXPECT_NE(result.report.find("S_vib="), std::string::npos);
    EXPECT_NE(result.report.find("kcal/mol"), std::string::npos);
}

// ===========================================================================
// RUN_SHANNON_THERMO_STACK — EDGE CASES
// ===========================================================================

TEST(ShannonThermoStackEdge, SingleSampleEnsemble) {
    statmech::StatMechEngine eng(298.15);
    eng.add_sample(-10.0);
    tencm::TorsionalENM tencm;

    auto result = run_shannon_thermo_stack(eng, tencm, -10.0);
    EXPECT_TRUE(std::isfinite(result.deltaG));
    EXPECT_TRUE(std::isfinite(result.shannonEntropy));
}

TEST(ShannonThermoStackEdge, LargeEnsemble) {
    statmech::StatMechEngine eng(298.15);
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(-15.0, 5.0);

    for (int i = 0; i < 10000; ++i)
        eng.add_sample(dist(rng));

    tencm::TorsionalENM tencm;
    auto result = run_shannon_thermo_stack(eng, tencm, -10.0);

    EXPECT_TRUE(std::isfinite(result.deltaG));
    EXPECT_GT(result.shannonEntropy, 0.0);
}

TEST(ShannonThermoStackEdge, ZeroBaseDeltaG) {
    statmech::StatMechEngine eng(298.15);
    eng.add_sample(-10.0);
    eng.add_sample(-5.0);
    tencm::TorsionalENM tencm;

    auto result = run_shannon_thermo_stack(eng, tencm, 0.0);
    // deltaG should equal just the entropy contribution
    EXPECT_NEAR(result.deltaG, result.entropyContribution, EPSILON);
}

TEST(ShannonThermoStackEdge, CustomTemperature) {
    statmech::StatMechEngine eng(310.0);
    eng.add_sample(-12.0);
    eng.add_sample(-8.0);
    tencm::TorsionalENM tencm;

    auto result = run_shannon_thermo_stack(eng, tencm, -10.0, 310.0);
    EXPECT_TRUE(std::isfinite(result.deltaG));
}

TEST(ShannonThermoStackEdge, DegenerateEnergyEnsemble) {
    // All samples at same energy → Boltzmann weights are equal
    // → -log(w) all equal → single unique value → H = 0
    statmech::StatMechEngine eng(298.15);
    for (int i = 0; i < 50; ++i)
        eng.add_sample(-10.0);

    tencm::TorsionalENM tencm;
    auto result = run_shannon_thermo_stack(eng, tencm, -10.0);

    EXPECT_TRUE(std::isfinite(result.deltaG));
    EXPECT_NEAR(result.shannonEntropy, 0.0, 0.01);
}

// ===========================================================================
// NUMERICAL STABILITY
// ===========================================================================

TEST(ShannonEntropyStability, VeryLargeValues) {
    std::vector<double> values(100);
    for (int i = 0; i < 100; ++i)
        values[i] = 1e6 + i * 0.01;

    double H = compute_shannon_entropy(values, 10);
    EXPECT_TRUE(std::isfinite(H));
    EXPECT_GE(H, 0.0);
}

TEST(ShannonEntropyStability, VerySmallRange) {
    std::vector<double> values(100);
    for (int i = 0; i < 100; ++i)
        values[i] = 1.0 + i * 1e-15;

    double H = compute_shannon_entropy(values, 10);
    EXPECT_TRUE(std::isfinite(H));
}

TEST(ShannonEntropyStability, NegativeValues) {
    std::vector<double> values = {-100.0, -50.0, -25.0, -10.0, -5.0};
    double H = compute_shannon_entropy(values, 5);
    EXPECT_TRUE(std::isfinite(H));
    EXPECT_GE(H, 0.0);
}

TEST(ShannonEntropyStability, MixedSignValues) {
    std::vector<double> values = {-50.0, -25.0, 0.0, 25.0, 50.0};
    double H = compute_shannon_entropy(values, 5);
    EXPECT_TRUE(std::isfinite(H));
    EXPECT_GE(H, 0.0);
}

TEST(ShannonEntropyStability, TwoValues) {
    // Minimum non-trivial case
    std::vector<double> values = {0.0, 1.0};
    double H = compute_shannon_entropy(values, 2);
    EXPECT_TRUE(std::isfinite(H));
}

// ===========================================================================
// DISPATCH BACKEND CONSISTENCY
// ===========================================================================

TEST(DispatchConsistency, ReproducibleResults) {
    // Same input → same output (deterministic regardless of backend)
    std::mt19937 rng(77);
    std::normal_distribution<double> dist(0.0, 10.0);
    std::vector<double> values(1000);
    for (auto& v : values) v = dist(rng);

    double H1 = compute_shannon_entropy(values, 20);
    double H2 = compute_shannon_entropy(values, 20);
    EXPECT_DOUBLE_EQ(H1, H2);
}

TEST(DispatchConsistency, EntropyMonotonicWithDataSpread) {
    // Wider spread → more bins occupied → higher entropy
    std::vector<double> narrow(500), wide(500);
    std::mt19937 rng(42);

    std::normal_distribution<double> n_dist(0.0, 1.0);
    std::normal_distribution<double> w_dist(0.0, 50.0);
    for (int i = 0; i < 500; ++i) {
        narrow[i] = n_dist(rng);
        wide[i]   = w_dist(rng);
    }

    double H_narrow = compute_shannon_entropy(narrow, 20);
    double H_wide   = compute_shannon_entropy(wide, 20);

    EXPECT_GT(H_wide, H_narrow);
}

// ===========================================================================
// COMPILE-TIME BACKEND DETECTION
// ===========================================================================

TEST(BackendDetection, ActiveBackendReported) {
    // Verify that the runtime report correctly reflects compilation flags
    statmech::StatMechEngine eng(298.15);
    eng.add_sample(-10.0);
    eng.add_sample(-5.0);
    tencm::TorsionalENM tencm;

    auto result = run_shannon_thermo_stack(eng, tencm, -10.0);

#if defined(FLEXAIDS_USE_CUDA)
    EXPECT_NE(result.report.find("CUDA"), std::string::npos)
        << "Expected CUDA in report: " << result.report;
#elif defined(ENABLE_METAL_CORE)
    EXPECT_NE(result.report.find("Metal"), std::string::npos)
        << "Expected Metal in report: " << result.report;
#elif defined(__AVX512F__)
    EXPECT_NE(result.report.find("AVX-512"), std::string::npos)
        << "Expected AVX-512 in report: " << result.report;
#elif defined(_OPENMP)
    EXPECT_NE(result.report.find("OpenMP"), std::string::npos)
        << "Expected OpenMP in report: " << result.report;
#else
    EXPECT_NE(result.report.find("scalar"), std::string::npos)
        << "Expected scalar in report: " << result.report;
#endif
}

#ifdef FLEXAIDS_HAS_EIGEN
TEST(BackendDetection, EigenTagInReport) {
    statmech::StatMechEngine eng(298.15);
    eng.add_sample(-10.0);
    tencm::TorsionalENM tencm;

    auto result = run_shannon_thermo_stack(eng, tencm, -10.0);
    EXPECT_NE(result.report.find("Eigen"), std::string::npos)
        << "Expected +Eigen in report: " << result.report;
}
#endif

// ===========================================================================
// MAIN
// ===========================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
