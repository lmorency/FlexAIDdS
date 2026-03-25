// tests/test_mif_grid.cpp
// Unit tests for MIFGrid — Molecular Interaction Field for cleft grid points
// Tests MIF computation, Boltzmann sampling, and grid prioritization.

#include <gtest/gtest.h>
#include "MIFGrid.h"
#include <cmath>
#include <numeric>
#include <map>

using namespace mif;

// ===========================================================================
// HELPER: build a minimal set of atoms and grid points for testing
// ===========================================================================

// Create protein atoms at known positions.
// Returns vector of FlexAID atom structs.
static std::vector<atom> make_protein_atoms(const float coords[][3], int n_atoms,
                                             float radius = 1.7f) {
    std::vector<atom> atoms(static_cast<std::size_t>(n_atoms));
    for (int i = 0; i < n_atoms; ++i) {
        memset(&atoms[static_cast<std::size_t>(i)], 0, sizeof(atom));
        atoms[static_cast<std::size_t>(i)].coor[0] = coords[i][0];
        atoms[static_cast<std::size_t>(i)].coor[1] = coords[i][1];
        atoms[static_cast<std::size_t>(i)].coor[2] = coords[i][2];
        atoms[static_cast<std::size_t>(i)].radius = radius;
        atoms[static_cast<std::size_t>(i)].type = 1;
    }
    return atoms;
}

// Create grid points at known positions. Index 0 is a dummy (ligand reference).
static std::vector<gridpoint> make_grid_points(const float coords[][3], int n_points) {
    // n_points + 1: index 0 is reserved
    std::vector<gridpoint> grid(static_cast<std::size_t>(n_points + 1));
    memset(grid.data(), 0, grid.size() * sizeof(gridpoint));
    for (int i = 0; i < n_points; ++i) {
        grid[static_cast<std::size_t>(i + 1)].coor[0] = coords[i][0];
        grid[static_cast<std::size_t>(i + 1)].coor[1] = coords[i][1];
        grid[static_cast<std::size_t>(i + 1)].coor[2] = coords[i][2];
        grid[static_cast<std::size_t>(i + 1)].index = i + 1;
    }
    return grid;
}

// ===========================================================================
// LJ ENERGY
// ===========================================================================

TEST(MIFGrid, LJEnergyAtContactDistance) {
    // At contact distance (dist = sigma), LJ should be at minimum (-1.0)
    float sigma = 3.4f;
    float dist_sq = sigma * sigma;
    float e = lj_energy(dist_sq, sigma);
    EXPECT_NEAR(e, -1.0f, 1e-5f);
}

TEST(MIFGrid, LJEnergyRepulsiveAtCloseRange) {
    // At half the contact distance, energy should be strongly positive
    float sigma = 3.4f;
    float dist = sigma * 0.8f;
    float e = lj_energy(dist * dist, sigma);
    EXPECT_GT(e, 0.0f);
}

TEST(MIFGrid, LJEnergyApproachesZeroFarAway) {
    float sigma = 3.4f;
    float dist = 20.0f;
    float e = lj_energy(dist * dist, sigma);
    EXPECT_NEAR(e, 0.0f, 0.01f);
}

// ===========================================================================
// MIF COMPUTATION
// ===========================================================================

TEST(MIFGrid, ComputeMIF_EmptyGrid) {
    std::vector<atom> atoms;
    cavity_detect::SpatialGrid sg;
    sg.build(atoms);

    gridpoint gp;
    memset(&gp, 0, sizeof(gp));

    MIFResult result = compute_mif(&gp, 1, nullptr, 0, sg);
    EXPECT_EQ(result.num_grd, 1);
    EXPECT_TRUE(result.sorted_indices.empty());
}

TEST(MIFGrid, ComputeMIF_FavorableNearProtein) {
    // Place 4 protein atoms around origin forming a pocket
    float protein_coords[][3] = {
        { 4.0f,  0.0f,  0.0f},
        {-4.0f,  0.0f,  0.0f},
        { 0.0f,  4.0f,  0.0f},
        { 0.0f, -4.0f,  0.0f},
    };
    auto atoms = make_protein_atoms(protein_coords, 4, 1.7f);

    cavity_detect::SpatialGrid sg;
    sg.build(atoms);

    // Grid point A: at origin (center of pocket — favorable distance from all atoms)
    // Grid point B: far away (50, 50, 50) — no interactions
    float grid_coords[][3] = {
        { 0.0f, 0.0f, 0.0f},  // center of pocket
        {50.0f, 50.0f, 50.0f}, // far away
    };
    auto grid = make_grid_points(grid_coords, 2);

    MIFResult result = compute_mif(grid.data(), 3, atoms.data(),
                                    static_cast<int>(atoms.size()), sg);

    EXPECT_EQ(result.num_grd, 3);
    // Point at center of pocket should have lower (more favorable) energy
    // than point far away (which has ~0 energy)
    float e_center = result.energies[1];
    float e_far = result.energies[2];
    EXPECT_LT(e_center, e_far);
    // Far point should have ~0 energy (no nearby atoms)
    EXPECT_NEAR(e_far, 0.0f, 0.01f);
    // Center point should be negative (favorable)
    EXPECT_LT(e_center, 0.0f);
}

TEST(MIFGrid, ComputeMIF_RepulsiveInsideAtom) {
    // Single protein atom at origin
    float protein_coords[][3] = {{ 0.0f, 0.0f, 0.0f }};
    auto atoms = make_protein_atoms(protein_coords, 1, 1.7f);

    cavity_detect::SpatialGrid sg;
    sg.build(atoms);

    // Grid point inside the atom (very close) — should be repulsive
    // Grid point at optimal distance — should be attractive
    float grid_coords[][3] = {
        { 0.5f, 0.0f, 0.0f},  // inside atom (dist < sigma)
        { 3.4f, 0.0f, 0.0f},  // at contact distance (sigma = 1.7 + 1.7 = 3.4)
    };
    auto grid = make_grid_points(grid_coords, 2);

    MIFResult result = compute_mif(grid.data(), 3, atoms.data(), 1, sg);

    float e_inside = result.energies[1];
    float e_contact = result.energies[2];

    // Inside should be strongly repulsive
    EXPECT_GT(e_inside, 0.0f);
    // Contact distance should be at energy minimum (-1.0)
    EXPECT_NEAR(e_contact, -1.0f, 0.05f);
}

TEST(MIFGrid, SortedIndicesAreCorrect) {
    // 3 protein atoms in a line: at x=0, x=5, x=10
    float protein_coords[][3] = {
        {0.0f, 0.0f, 0.0f},
        {5.0f, 0.0f, 0.0f},
        {10.0f, 0.0f, 0.0f},
    };
    auto atoms = make_protein_atoms(protein_coords, 3, 1.7f);
    cavity_detect::SpatialGrid sg;
    sg.build(atoms);

    // Grid point near protein (favorable), between atoms (good), far away (neutral)
    float grid_coords[][3] = {
        {50.0f, 50.0f, 50.0f},  // index 1: far away
        { 3.4f,  0.0f,  0.0f},  // index 2: at contact distance to atom 0
        { 2.5f,  2.5f,  0.0f},  // index 3: near pocket between atoms
    };
    auto grid = make_grid_points(grid_coords, 3);

    MIFResult result = compute_mif(grid.data(), 4, atoms.data(), 3, sg);

    ASSERT_EQ(result.sorted_indices.size(), 3u);
    // First in sorted order should be the most favorable (lowest energy)
    int best_idx = result.sorted_indices[0];
    for (int idx : result.sorted_indices) {
        EXPECT_LE(result.energies[static_cast<std::size_t>(best_idx)],
                  result.energies[static_cast<std::size_t>(idx)]);
    }
}

// ===========================================================================
// BOLTZMANN SAMPLING
// ===========================================================================

TEST(MIFGrid, BoltzmannCDF_MonotonicallyIncreasing) {
    float protein_coords[][3] = {
        {0.0f, 0.0f, 0.0f},
        {5.0f, 0.0f, 0.0f},
    };
    auto atoms = make_protein_atoms(protein_coords, 2);
    cavity_detect::SpatialGrid sg;
    sg.build(atoms);

    float grid_coords[][3] = {
        {3.4f, 0.0f, 0.0f},
        {7.0f, 0.0f, 0.0f},
        {50.0f, 50.0f, 50.0f},
    };
    auto grid = make_grid_points(grid_coords, 3);

    MIFResult result = compute_mif(grid.data(), 4, atoms.data(), 2, sg);
    build_sampling_cdf(result);

    ASSERT_EQ(result.cdf.size(), 3u);
    for (std::size_t i = 1; i < result.cdf.size(); ++i)
        EXPECT_GE(result.cdf[i], result.cdf[i-1]);
    EXPECT_NEAR(result.cdf.back(), 1.0, 1e-10);
}

TEST(MIFGrid, SampleGridIndex_FavorsFavorablePoints) {
    // Single atom at origin. Grid point at contact distance should be sampled
    // more often than one far away.
    float protein_coords[][3] = {{ 0.0f, 0.0f, 0.0f }};
    auto atoms = make_protein_atoms(protein_coords, 1);
    cavity_detect::SpatialGrid sg;
    sg.build(atoms);

    float grid_coords[][3] = {
        {3.4f, 0.0f, 0.0f},    // favorable (at contact distance)
        {50.0f, 50.0f, 50.0f}, // neutral (far away)
    };
    auto grid = make_grid_points(grid_coords, 2);

    MIFResult result = compute_mif(grid.data(), 3, atoms.data(), 1, sg);
    build_sampling_cdf(result, 300.0f);

    std::mt19937 rng(42);
    std::map<int, int> counts;
    for (int i = 0; i < 10000; ++i)
        counts[sample_grid_index(result, rng)]++;

    // Favorable point (index 1 or 2, whichever has lower energy) should
    // be sampled more often
    int favorable_idx = result.sorted_indices[0];
    int neutral_idx = result.sorted_indices[1];
    EXPECT_GT(counts[favorable_idx], counts[neutral_idx]);
}

// ===========================================================================
// GRID PRIORITIZATION
// ===========================================================================

TEST(MIFGrid, PrioritizeGrid_TopK) {
    float protein_coords[][3] = {
        {0.0f, 0.0f, 0.0f},
        {5.0f, 0.0f, 0.0f},
    };
    auto atoms = make_protein_atoms(protein_coords, 2);
    cavity_detect::SpatialGrid sg;
    sg.build(atoms);

    // 4 grid points: 2 near protein (favorable), 2 far away
    float grid_coords[][3] = {
        { 3.4f,  0.0f,  0.0f},  // favorable
        { 2.5f,  0.0f,  0.0f},  // favorable (closer)
        {50.0f, 50.0f, 50.0f},  // neutral
        {60.0f, 60.0f, 60.0f},  // neutral
    };
    auto grid = make_grid_points(grid_coords, 4);

    MIFResult result = compute_mif(grid.data(), 5, atoms.data(), 2, sg);

    // Keep top 50% → should keep 2 of 4 points
    auto kept = prioritize_grid(result, 50.0f);
    ASSERT_EQ(kept.size(), 2u);

    // The kept points should be the ones with lowest energy
    for (int idx : kept) {
        EXPECT_LT(result.energies[static_cast<std::size_t>(idx)],
                  result.energies[static_cast<std::size_t>(result.sorted_indices.back())]);
    }
}

TEST(MIFGrid, PrioritizeGrid_KeepAll) {
    float protein_coords[][3] = {{ 0.0f, 0.0f, 0.0f }};
    auto atoms = make_protein_atoms(protein_coords, 1);
    cavity_detect::SpatialGrid sg;
    sg.build(atoms);

    float grid_coords[][3] = {
        {3.0f, 0.0f, 0.0f},
        {4.0f, 0.0f, 0.0f},
    };
    auto grid = make_grid_points(grid_coords, 2);

    MIFResult result = compute_mif(grid.data(), 3, atoms.data(), 1, sg);

    auto kept = prioritize_grid(result, 100.0f);
    EXPECT_EQ(kept.size(), 2u);
}

TEST(MIFGrid, PrioritizeGrid_KeepAtLeastOne) {
    float protein_coords[][3] = {{ 0.0f, 0.0f, 0.0f }};
    auto atoms = make_protein_atoms(protein_coords, 1);
    cavity_detect::SpatialGrid sg;
    sg.build(atoms);

    float grid_coords[][3] = {{ 3.0f, 0.0f, 0.0f }};
    auto grid = make_grid_points(grid_coords, 1);

    MIFResult result = compute_mif(grid.data(), 2, atoms.data(), 1, sg);

    auto kept = prioritize_grid(result, 1.0f);
    EXPECT_GE(kept.size(), 1u);
}

// ===========================================================================
// REBUILD CLEFTGRID
// ===========================================================================

TEST(MIFGrid, RebuildCleftgrid_PreservesIndex0) {
    float protein_coords[][3] = {{ 0.0f, 0.0f, 0.0f }};
    auto atoms = make_protein_atoms(protein_coords, 1);
    cavity_detect::SpatialGrid sg;
    sg.build(atoms);

    float grid_coords[][3] = {
        {3.0f, 0.0f, 0.0f},
        {4.0f, 0.0f, 0.0f},
        {50.0f, 50.0f, 50.0f},
    };
    auto grid = make_grid_points(grid_coords, 3);
    // Set index 0 to a recognizable value
    grid[0].coor[0] = 99.0f;

    MIFResult result = compute_mif(grid.data(), 4, atoms.data(), 1, sg);
    auto kept = prioritize_grid(result, 50.0f);  // keep ~1-2 of 3

    gridpoint* new_grid = nullptr;
    int new_count = rebuild_cleftgrid(grid.data(), 4, kept, &new_grid);

    ASSERT_NE(new_grid, nullptr);
    EXPECT_EQ(new_count, 1 + static_cast<int>(kept.size()));
    // Index 0 preserved
    EXPECT_FLOAT_EQ(new_grid[0].coor[0], 99.0f);
    // New grid has sequential indices
    for (int i = 1; i < new_count; ++i)
        EXPECT_EQ(new_grid[i].index, i);

    free(new_grid);
}

// ===========================================================================
// THREAD SAFETY
// ===========================================================================

TEST(MIFGrid, ComputeMIF_DeterministicAcrossRuns) {
    // Verify same inputs produce same outputs (thread-safe)
    float protein_coords[][3] = {
        { 0.0f, 0.0f, 0.0f},
        { 5.0f, 0.0f, 0.0f},
        { 0.0f, 5.0f, 0.0f},
        { 5.0f, 5.0f, 0.0f},
    };
    auto atoms = make_protein_atoms(protein_coords, 4);
    cavity_detect::SpatialGrid sg;
    sg.build(atoms);

    float grid_coords[][3] = {
        {2.5f, 2.5f, 0.0f},
        {3.4f, 0.0f, 0.0f},
        {50.0f, 50.0f, 50.0f},
    };
    auto grid = make_grid_points(grid_coords, 3);

    MIFResult r1 = compute_mif(grid.data(), 4, atoms.data(), 4, sg);
    MIFResult r2 = compute_mif(grid.data(), 4, atoms.data(), 4, sg);

    ASSERT_EQ(r1.energies.size(), r2.energies.size());
    for (std::size_t i = 0; i < r1.energies.size(); ++i)
        EXPECT_FLOAT_EQ(r1.energies[i], r2.energies[i]);
    EXPECT_EQ(r1.sorted_indices, r2.sorted_indices);
}
