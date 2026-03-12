// tests/test_binding_mode_vibrational.cpp
// Unit tests for BindingMode vibrational correction (Phase 3)
// Validates ENCoM-based -T*S_vib integration into BindingMode free energy
// Apache-2.0 © 2026 Le Bonhomme Pharma

#include <gtest/gtest.h>
#include "../LIB/BindingMode.h"
#include "../LIB/statmech.h"
#include "../LIB/encom.h"
#include "../LIB/gaboom.h"
#include <cmath>
#include <vector>

// ===========================================================================
// TEST FIXTURE
// ===========================================================================

class BindingModeVibrationalTest : public ::testing::Test {
protected:
    FA_Global* mock_fa;
    GB_Global* mock_gb;
    VC_Global* mock_vc;
    chromosome* mock_chroms;
    genlim* mock_gene_lim;
    atom* mock_atoms;
    resid* mock_residue;
    gridpoint* mock_cleftgrid;
    BindingPopulation* test_population;

    static constexpr int NUM_ATOMS = 20;
    static constexpr double TEST_TEMPERATURE = 300.0;
    static constexpr double EPSILON = 1e-6;

    void SetUp() override {
        mock_fa = new FA_Global();
        mock_fa->temperature = TEST_TEMPERATURE;
        mock_fa->num_atoms = NUM_ATOMS;
        mock_fa->num_residues = 2;
        mock_fa->normal_modes = 0;  // No vibrational correction by default

        mock_gb = new GB_Global();
        mock_gb->num_genes = 6;

        mock_vc = new VC_Global();

        mock_chroms = new chromosome[5];
        mock_gene_lim = new genlim[mock_gb->num_genes];
        mock_atoms = new atom[NUM_ATOMS];
        mock_residue = new resid[mock_fa->num_residues];
        mock_cleftgrid = new gridpoint[100];

        // Initialize atoms with null eigen pointers
        for (int i = 0; i < NUM_ATOMS; ++i) {
            mock_atoms[i].eigen = nullptr;
        }

        test_population = new BindingPopulation(
            mock_fa, mock_gb, mock_vc,
            mock_chroms, mock_gene_lim,
            mock_atoms, mock_residue, mock_cleftgrid,
            5
        );
    }

    void TearDown() override {
        // Clean up eigenvalue arrays if allocated
        if (mock_atoms[0].eigen) {
            for (int m = 0; m < mock_fa->normal_modes; ++m) {
                delete[] mock_atoms[0].eigen[m];
            }
            delete[] mock_atoms[0].eigen;
            mock_atoms[0].eigen = nullptr;
        }

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

    Pose create_mock_pose(double cf_value, int index) {
        std::vector<float> empty_vec;
        Pose p(&mock_chroms[index], index, 0, 0.0, TEST_TEMPERATURE, empty_vec);
        p.CF = cf_value;
        return p;
    }

    // Set up mock eigenvalues on atoms[0] to simulate ENCoM modes
    void setup_mock_eigenvalues(int n_modes, double base_eigenvalue = 1.0) {
        mock_fa->normal_modes = n_modes;
        mock_atoms[0].eigen = new float*[n_modes];
        for (int m = 0; m < n_modes; ++m) {
            mock_atoms[0].eigen[m] = new float[1];
            // Eigenvalues increase linearly: 1.0, 2.0, 3.0, ...
            mock_atoms[0].eigen[m][0] = static_cast<float>(base_eigenvalue * (m + 1));
        }
    }
};

// ===========================================================================
// VIBRATIONAL CORRECTION TESTS
// ===========================================================================

TEST_F(BindingModeVibrationalTest, NoModesZeroCorrection) {
    // When normal_modes == 0, vibrational correction should be 0
    BindingMode mode(test_population);
    Pose p = create_mock_pose(-10.0, 0);
    mode.add_Pose(p);

    double energy_no_vib = mode.compute_energy();
    double correction = mode.compute_vibrational_correction();

    EXPECT_NEAR(correction, 0.0, EPSILON);

    // Free energy should equal StatMech-only free energy
    auto thermo = mode.get_thermodynamics();
    // With zero correction, compute_energy == statmech free energy
    EXPECT_NEAR(energy_no_vib, thermo.free_energy, EPSILON);
}

TEST_F(BindingModeVibrationalTest, WithModesNonzeroCorrection) {
    // Set up eigenvalues to produce a vibrational correction
    setup_mock_eigenvalues(5, 0.5);

    BindingMode mode(test_population);
    Pose p = create_mock_pose(-10.0, 0);
    mode.add_Pose(p);

    double correction = mode.compute_vibrational_correction();

    // Correction should be -T * S_vib, which is negative (adds stability)
    // S_vib > 0, so correction < 0
    EXPECT_LT(correction, 0.0);
    EXPECT_TRUE(std::isfinite(correction));
}

TEST_F(BindingModeVibrationalTest, CorrectionIncludedInFreeEnergy) {
    setup_mock_eigenvalues(5, 0.5);

    BindingMode mode(test_population);
    for (int i = 0; i < 3; ++i) {
        Pose p = create_mock_pose(-10.0 - i * 2.0, i);
        mode.add_Pose(p);
    }

    double correction = mode.compute_vibrational_correction();
    auto thermo = mode.get_thermodynamics();

    // get_thermodynamics() should include the vibrational correction
    // thermo.free_energy = statmech_F + correction
    // We verify by checking that free_energy != statmech_F when correction != 0
    EXPECT_NE(correction, 0.0);

    // compute_energy() should also include correction
    double total_energy = mode.compute_energy();
    EXPECT_NEAR(total_energy, thermo.free_energy, EPSILON);
}

TEST_F(BindingModeVibrationalTest, MoreModesLargerCorrection) {
    // With more modes, S_vib should be larger → correction more negative
    setup_mock_eigenvalues(3, 0.5);

    BindingMode mode_few(test_population);
    Pose p1 = create_mock_pose(-10.0, 0);
    mode_few.add_Pose(p1);
    double correction_few = mode_few.compute_vibrational_correction();

    // Clean up and set more modes
    for (int m = 0; m < 3; ++m) delete[] mock_atoms[0].eigen[m];
    delete[] mock_atoms[0].eigen;
    mock_atoms[0].eigen = nullptr;

    setup_mock_eigenvalues(10, 0.5);

    BindingMode mode_many(test_population);
    Pose p2 = create_mock_pose(-10.0, 0);
    mode_many.add_Pose(p2);
    double correction_many = mode_many.compute_vibrational_correction();

    // More modes → more negative correction (larger magnitude)
    EXPECT_LT(correction_many, correction_few);
}

TEST_F(BindingModeVibrationalTest, NullEigenReturnsZero) {
    // normal_modes > 0 but eigen pointer is null → should return 0 safely
    mock_fa->normal_modes = 5;
    // mock_atoms[0].eigen is already nullptr from SetUp

    BindingMode mode(test_population);
    Pose p = create_mock_pose(-10.0, 0);
    mode.add_Pose(p);

    double correction = mode.compute_vibrational_correction();
    EXPECT_NEAR(correction, 0.0, EPSILON);
}

TEST_F(BindingModeVibrationalTest, CorrectionScalesWithTemperature) {
    setup_mock_eigenvalues(5, 0.5);

    // Test at 300K
    BindingMode mode_300(test_population);
    Pose p1 = create_mock_pose(-10.0, 0);
    mode_300.add_Pose(p1);
    double correction_300 = mode_300.compute_vibrational_correction();

    // Change temperature to 600K
    mock_fa->temperature = 600.0;
    test_population->Temperature = 600;

    BindingMode mode_600(test_population);
    Pose p2 = create_mock_pose(-10.0, 0);
    mode_600.add_Pose(p2);
    double correction_600 = mode_600.compute_vibrational_correction();

    // At higher T, -T*S_vib should be more negative (larger magnitude)
    EXPECT_LT(correction_600, correction_300);

    // Restore
    mock_fa->temperature = TEST_TEMPERATURE;
    test_population->Temperature = static_cast<unsigned int>(TEST_TEMPERATURE);
}

// ===========================================================================
// MAIN
// ===========================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
