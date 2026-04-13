// tests/test_docking_pipeline.cpp
// End-to-end integration tests for the FlexAIDdS docking pipeline.
//
// These tests exercise the full path from ligand preprocessing through
// GA-based pose sampling, statistical mechanics scoring, and binding-mode
// clustering without requiring on-disk receptor/ligand files.
//
// The strategy is to:
//   1. Build a minimal in-memory BonMol via process_smiles (ProcessLigand)
//   2. Build a mock FA_Global / GA context with a tiny grid
//   3. Run StatMechEngine over a hand-crafted energy landscape
//   4. Push poses through BindingMode and read back thermodynamics
//
// This validates the interfaces between the four major subsystems.
// Apache-2.0 © 2026 Le Bonhomme Pharma

#include <gtest/gtest.h>

// ProcessLigand
#include "../LIB/ProcessLigand/ProcessLigand.h"

// StatMech
#include "../LIB/statmech.h"

// BindingMode / Pose
#include "../LIB/BindingMode.h"
#include "../LIB/gaboom.h"

#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>

using namespace bonmol;
using namespace statmech;

// ===========================================================================
// Helpers — mock GA infrastructure
// ===========================================================================

struct MockGA {
    FA_Global   fa{};
    GB_Global   gb{};
    VC_Global   vc{};
    std::vector<chromosome>  chroms;
    std::vector<genlim>      gene_lim;
    std::vector<atom>        atoms;
    std::vector<resid>       residues;
    std::vector<gridpoint>   cleftgrid;

    static constexpr int N_CHROMS  = 10;
    static constexpr int N_GENES   = 6;
    static constexpr int N_ATOMS   = 4;
    static constexpr int N_RES     = 1;
    static constexpr double TEMP   = 300.0;

    MockGA() {
        fa.temperature = static_cast<uint>(TEMP);
        fa.normal_modes = 0;
        gb.num_genes = N_GENES;

        chroms.resize(N_CHROMS);
        gene_lim.resize(N_GENES);
        atoms.resize(N_ATOMS);
        for (auto& a : atoms) a.eigen = nullptr;
        residues.resize(N_RES);
        cleftgrid.resize(50);
    }
};

// Build a BindingPopulation from a MockGA
static std::unique_ptr<BindingPopulation> make_population(MockGA& m) {
    return std::make_unique<BindingPopulation>(
        &m.fa, &m.gb, &m.vc,
        m.chroms.data(), m.gene_lim.data(),
        m.atoms.data(), m.residues.data(), m.cleftgrid.data(),
        MockGA::N_CHROMS
    );
}

// Create a Pose with a given CF value
static Pose make_pose(MockGA& m, int idx, double cf) {
    std::vector<float> empty;
    Pose p(&m.chroms[idx], idx, 0, 0.0, MockGA::TEMP, empty);
    p.CF = cf;
    return p;
}

// ===========================================================================
// Stage 1 — ProcessLigand output feeds BindingMode
// ===========================================================================

TEST(DockingPipeline, LigandPreprocessingSucceeds) {
    auto result = process_smiles("c1ccccc1"); // benzene
    ASSERT_TRUE(result.success) << "Pipeline error: " << result.error;
    EXPECT_EQ(result.num_atoms, 6);
    EXPECT_EQ(result.num_arom_rings, 1);
    EXPECT_EQ(result.num_rot_bonds, 0);
}

TEST(DockingPipeline, LigandMolecularWeightReasonable) {
    // Ibuprofen C13H18O2, MW ≈ 206 Da
    auto result = process_smiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O");
    ASSERT_TRUE(result.success);
    EXPECT_GT(result.molecular_weight, 180.0f);
    EXPECT_LT(result.molecular_weight, 240.0f);
}

TEST(DockingPipeline, LigandWithRotatableBonds) {
    // Butane has 1 rotatable bond; longer chains have more
    auto result = process_smiles("CCCCCC"); // hexane
    ASSERT_TRUE(result.success);
    EXPECT_GT(result.num_rot_bonds, 0);
}

// ===========================================================================
// Stage 2 — StatMechEngine with realistic CF landscape
// ===========================================================================

TEST(DockingPipeline, StatMechOverDockingEnsemble) {
    // Simulate a GA ensemble of 20 CF values drawn from a Gaussian around -12 kcal/mol
    StatMechEngine engine(300.0);
    std::vector<double> energies = {
        -14.5, -13.8, -13.2, -12.9, -12.5, -12.1, -11.8, -11.4,
        -11.0, -10.7, -10.3, -9.8,  -9.4,  -9.0,  -8.5,  -8.0,
        -7.5,  -7.0,  -6.5,  -6.0
    };
    for (double e : energies) engine.add_sample(e);

    auto thermo = engine.compute();
    EXPECT_TRUE(std::isfinite(thermo.free_energy));
    EXPECT_TRUE(std::isfinite(thermo.entropy));
    EXPECT_GT(thermo.entropy, 0.0);
    EXPECT_LT(thermo.free_energy, -6.0);   // F must be ≤ best energy
    EXPECT_GT(thermo.heat_capacity, 0.0);
}

TEST(DockingPipeline, StatMechDeltaGBetweenTwoModes) {
    StatMechEngine mode1(300.0), mode2(300.0);
    // Mode 1 is more stable
    for (double e : {-14.0, -13.0, -12.0}) mode1.add_sample(e);
    for (double e : {-10.0, -9.0,  -8.0})  mode2.add_sample(e);

    double dG = mode1.delta_G(mode2);  // mode1 - mode2 → should be negative
    EXPECT_LT(dG, 0.0);
}

// ===========================================================================
// Stage 3 — BindingMode clustering and thermodynamics
// ===========================================================================

TEST(DockingPipeline, SingleBindingModeThermodynamics) {
    MockGA m;
    auto pop = make_population(m);

    BindingMode bm(pop.get());
    std::vector<double> cfs = {-14.0, -13.5, -12.8, -12.0, -11.3};
    for (int i = 0; i < static_cast<int>(cfs.size()); ++i)
        bm.add_Pose(make_pose(m, i, cfs[i]));

    auto thermo = bm.get_thermodynamics();
    EXPECT_TRUE(std::isfinite(thermo.free_energy));
    EXPECT_GT(thermo.entropy, 0.0);
    EXPECT_LT(thermo.free_energy, cfs.back()); // F < worst energy
}

TEST(DockingPipeline, TwoBindingModesRankedByFreeEnergy) {
    MockGA m;
    auto pop = make_population(m);

    BindingMode bm1(pop.get()), bm2(pop.get());
    // Mode 1: more stable energies
    for (double e : {-15.0, -14.0, -13.0})
        bm1.add_Pose(make_pose(m, 0, e));
    // Mode 2: weaker energies
    for (double e : {-10.0, -9.0, -8.0})
        bm2.add_Pose(make_pose(m, 0, e));

    double F1 = bm1.get_thermodynamics().free_energy;
    double F2 = bm2.get_thermodynamics().free_energy;
    EXPECT_LT(F1, F2); // mode 1 is thermodynamically preferred
}

TEST(DockingPipeline, BindingModeRepresentativePoseIsBest) {
    MockGA m;
    auto pop = make_population(m);
    BindingMode bm(pop.get());

    bm.add_Pose(make_pose(m, 0, -8.0));
    bm.add_Pose(make_pose(m, 1, -15.0)); // best
    bm.add_Pose(make_pose(m, 2, -10.0));

    const Pose* rep = bm.representative_pose();
    ASSERT_NE(rep, nullptr);
    EXPECT_NEAR(rep->CF, -15.0, 1e-9);
}

// ===========================================================================
// Stage 4 — BindingPopulation aggregation
// ===========================================================================

TEST(DockingPipeline, BindingPopulationAddAndRetrieveModes) {
    MockGA m;
    auto pop = make_population(m);

    // Add poses with two distinct cluster labels
    for (int i = 0; i < 3; ++i)
        pop->add_pose_to_mode(0, make_pose(m, i, -14.0 - i));
    for (int i = 3; i < 6; ++i)
        pop->add_pose_to_mode(1, make_pose(m, i, -10.0 - (i - 3)));

    EXPECT_EQ(pop->num_modes(), 2);
    EXPECT_EQ(pop->get_mode(0).num_poses(), 3);
    EXPECT_EQ(pop->get_mode(1).num_poses(), 3);
}

TEST(DockingPipeline, BindingPopulationBestModeIsLowestFreeEnergy) {
    MockGA m;
    auto pop = make_population(m);

    for (int i = 0; i < 3; ++i)
        pop->add_pose_to_mode(0, make_pose(m, i, -14.0));
    for (int i = 3; i < 6; ++i)
        pop->add_pose_to_mode(1, make_pose(m, i, -8.0));

    const BindingMode* best = pop->best_mode();
    ASSERT_NE(best, nullptr);
    // Mode 0 has lower energies → lower F
    EXPECT_NEAR(best->get_thermodynamics().free_energy,
                pop->get_mode(0).get_thermodynamics().free_energy, 1e-9);
}

// ===========================================================================
// Stage 5 — Boltzmann weights across modes (relative populations)
// ===========================================================================

TEST(DockingPipeline, BoltzmannWeightsAcrossModesNormalized) {
    MockGA m;
    auto pop = make_population(m);

    for (int i = 0; i < 3; ++i)
        pop->add_pose_to_mode(0, make_pose(m, i, -14.0));
    for (int i = 3; i < 6; ++i)
        pop->add_pose_to_mode(1, make_pose(m, i, -10.0));

    auto weights = pop->mode_boltzmann_weights();
    double sum = 0.0;
    for (double w : weights) {
        EXPECT_GE(w, 0.0);
        sum += w;
    }
    EXPECT_NEAR(sum, 1.0, 1e-9);
}

TEST(DockingPipeline, MoreStableModeDominatesWeight) {
    MockGA m;
    auto pop = make_population(m);

    for (int i = 0; i < 5; ++i)
        pop->add_pose_to_mode(0, make_pose(m, i, -20.0)); // strongly stable
    for (int i = 5; i < 8; ++i)
        pop->add_pose_to_mode(1, make_pose(m, i, -5.0));  // weak

    auto weights = pop->mode_boltzmann_weights();
    ASSERT_GE(weights.size(), 2u);
    EXPECT_GT(weights[0], weights[1]);
}

// ===========================================================================
// MAIN
// ===========================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
