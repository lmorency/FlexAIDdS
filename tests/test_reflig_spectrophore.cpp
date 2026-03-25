// tests/test_reflig_spectrophore.cpp
// Unit tests for RefLigSeed and Spectrophore
// Tests reference ligand parsing, centroid computation, nearest grid finding,
// and spectrophore descriptor computation and comparison.

#include <gtest/gtest.h>
#include "RefLigSeed.h"
#include "Spectrophore.h"
#include "MIFGrid.h"
#include <cmath>
#include <fstream>
#include <cstring>

// ===========================================================================
// HELPERS
// ===========================================================================

static void write_test_pdb(const std::string& path,
                            const float coords[][3], int n_atoms) {
    std::ofstream out(path);
    for (int i = 0; i < n_atoms; ++i) {
        char line[120];
        std::snprintf(line, sizeof(line),
            "HETATM%5d  C1  LIG A   1    %8.3f%8.3f%8.3f  1.00  1.70\n",
            i + 1, coords[i][0], coords[i][1], coords[i][2]);
        out << line;
    }
    out << "END\n";
}

static void write_test_mol2(const std::string& path,
                             const float coords[][3], int n_atoms) {
    std::ofstream out(path);
    out << "@<TRIPOS>MOLECULE\n";
    out << "test_lig\n";
    out << n_atoms << " 0 0 0 0\n";
    out << "SMALL\nNO_CHARGES\n\n";
    out << "@<TRIPOS>ATOM\n";
    for (int i = 0; i < n_atoms; ++i) {
        char line[120];
        std::snprintf(line, sizeof(line),
            "%7d C%-3d   %8.3f %8.3f %8.3f C.3\n",
            i + 1, i + 1, coords[i][0], coords[i][1], coords[i][2]);
        out << line;
    }
}

static std::vector<gridpoint> make_grid(const float coords[][3], int n) {
    std::vector<gridpoint> grid(static_cast<std::size_t>(n + 1));
    memset(grid.data(), 0, grid.size() * sizeof(gridpoint));
    for (int i = 0; i < n; ++i) {
        grid[static_cast<std::size_t>(i + 1)].coor[0] = coords[i][0];
        grid[static_cast<std::size_t>(i + 1)].coor[1] = coords[i][1];
        grid[static_cast<std::size_t>(i + 1)].coor[2] = coords[i][2];
        grid[static_cast<std::size_t>(i + 1)].index = i + 1;
    }
    return grid;
}

// ===========================================================================
// REFLIGSEED — PDB PARSING
// ===========================================================================

TEST(RefLigSeed, ParsePDB) {
    float coords[][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    write_test_pdb("test_reflig.pdb", coords, 2);

    auto atoms = reflig::parse_pdb_coords("test_reflig.pdb");
    ASSERT_EQ(atoms.size(), 2u);
    EXPECT_NEAR(atoms[0].x, 1.0f, 0.01f);
    EXPECT_NEAR(atoms[1].z, 6.0f, 0.01f);
}

TEST(RefLigSeed, ParseMOL2) {
    float coords[][3] = {{10.0f, 20.0f, 30.0f}};
    write_test_mol2("test_reflig.mol2", coords, 1);

    auto atoms = reflig::parse_mol2_coords("test_reflig.mol2");
    ASSERT_EQ(atoms.size(), 1u);
    EXPECT_NEAR(atoms[0].x, 10.0f, 0.01f);
    EXPECT_NEAR(atoms[0].y, 20.0f, 0.01f);
}

// ===========================================================================
// REFLIGSEED — CENTROID
// ===========================================================================

TEST(RefLigSeed, ComputeCentroid) {
    std::vector<reflig::RefLigAtom> atoms = {
        {0.0f, 0.0f, 0.0f},
        {10.0f, 0.0f, 0.0f},
        {0.0f, 10.0f, 0.0f},
        {10.0f, 10.0f, 0.0f},
    };
    float c[3];
    reflig::compute_centroid(atoms, c);
    EXPECT_NEAR(c[0], 5.0f, 0.01f);
    EXPECT_NEAR(c[1], 5.0f, 0.01f);
    EXPECT_NEAR(c[2], 0.0f, 0.01f);
}

TEST(RefLigSeed, ComputeCentroidEmpty) {
    std::vector<reflig::RefLigAtom> atoms;
    float c[3] = {99.0f, 99.0f, 99.0f};
    reflig::compute_centroid(atoms, c);
    EXPECT_FLOAT_EQ(c[0], 0.0f);
}

// ===========================================================================
// REFLIGSEED — NEAREST GRID POINTS
// ===========================================================================

TEST(RefLigSeed, FindNearestGridPoints) {
    float grid_coords[][3] = {
        {1.0f, 0.0f, 0.0f},   // index 1: close to centroid (0,0,0)
        {2.0f, 0.0f, 0.0f},   // index 2
        {100.0f, 0.0f, 0.0f}, // index 3: far
        {0.5f, 0.5f, 0.0f},   // index 4: closest
    };
    auto grid = make_grid(grid_coords, 4);

    float centroid[3] = {0.0f, 0.0f, 0.0f};
    auto nearest = reflig::find_nearest_grid_points(centroid, grid.data(), 5, 2);

    ASSERT_EQ(nearest.size(), 2u);
    // Index 4 (0.5, 0.5, 0) should be closest, then index 1 (1, 0, 0)
    EXPECT_EQ(nearest[0], 4);
    EXPECT_EQ(nearest[1], 1);
}

TEST(RefLigSeed, PrepareRefLigSeed) {
    float lig_coords[][3] = {{5.0f, 5.0f, 0.0f}};
    write_test_pdb("test_reflig_seed.pdb", lig_coords, 1);

    float grid_coords[][3] = {
        {5.0f, 5.0f, 0.0f},    // index 1: at centroid
        {5.5f, 5.0f, 0.0f},    // index 2: near
        {100.0f, 100.0f, 0.0f}, // index 3: far
    };
    auto grid = make_grid(grid_coords, 3);

    auto data = reflig::prepare_reflig_seed("test_reflig_seed.pdb",
                                             grid.data(), 4, 2);

    EXPECT_NEAR(data.centroid[0], 5.0f, 0.01f);
    ASSERT_EQ(data.nearest_grid.size(), 2u);
    EXPECT_EQ(data.nearest_grid[0], 1);  // grid point at (5,5,0) is closest
    EXPECT_EQ(data.nearest_grid[1], 2);
}

// ===========================================================================
// SPECTROPHORE — BASIC COMPUTATION
// ===========================================================================

TEST(Spectrophore, DescriptorSize) {
    EXPECT_EQ(spectrophore::DESCRIPTOR_SIZE, 144);
}

TEST(Spectrophore, DefaultIsZero) {
    spectrophore::Spectrophore sp;
    for (int i = 0; i < spectrophore::DESCRIPTOR_SIZE; ++i)
        EXPECT_FLOAT_EQ(sp.values[i], 0.0f);
}

TEST(Spectrophore, TanimotoIdentical) {
    spectrophore::Spectrophore sp;
    sp.values[0] = 1.0f;
    sp.values[10] = 2.0f;
    sp.values[50] = 3.0f;
    EXPECT_NEAR(sp.tanimoto(sp), 1.0f, 1e-6f);
}

TEST(Spectrophore, TanimotoOrthogonal) {
    spectrophore::Spectrophore a, b;
    a.values[0] = 1.0f;
    b.values[1] = 1.0f;
    EXPECT_NEAR(a.tanimoto(b), 0.0f, 1e-6f);
}

TEST(Spectrophore, ComputeFromGrid) {
    // 4 grid points around center
    float grid_coords[][3] = {
        {1.0f, 0.0f, 0.0f},
        {-1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, -1.0f, 0.0f},
    };
    auto grid = make_grid(grid_coords, 4);

    float mif_energies[] = {0.0f, -1.0f, -0.5f, -2.0f, 0.5f};
    float center[3] = {0.0f, 0.0f, 0.0f};

    auto sp = spectrophore::compute_from_grid(grid.data(), 5, mif_energies, center);

    // Should have non-zero shape values
    bool has_shape = false;
    for (int i = 0; i < spectrophore::N_ANGULAR * spectrophore::N_RADIAL; ++i) {
        if (sp.values[i] > 0.0f) has_shape = true;
    }
    EXPECT_TRUE(has_shape);
}

TEST(Spectrophore, ComputeFromAtoms) {
    spectrophore::SimpleAtom atoms[] = {
        {1.0f, 0.0f, 0.0f, 1.7f, 0.0f},
        {-1.0f, 0.0f, 0.0f, 1.5f, -0.5f},
        {0.0f, 1.0f, 0.0f, 1.8f, 0.3f},
    };
    float center[3] = {0.0f, 0.0f, 0.0f};

    auto sp = spectrophore::compute_from_atoms(atoms, 3, center);

    // Shape bins should be populated
    bool has_shape = false;
    for (int i = 0; i < spectrophore::N_ANGULAR * spectrophore::N_RADIAL; ++i) {
        if (sp.values[i] > 0.0f) has_shape = true;
    }
    EXPECT_TRUE(has_shape);
}

TEST(Spectrophore, SimilarDescriptorsHighTanimoto) {
    // Two similar configurations should have high Tanimoto
    spectrophore::SimpleAtom atoms1[] = {
        {1.0f, 0.0f, 0.0f, 1.7f, 0.0f},
        {-1.0f, 0.0f, 0.0f, 1.7f, 0.0f},
    };
    spectrophore::SimpleAtom atoms2[] = {
        {1.1f, 0.0f, 0.0f, 1.7f, 0.0f},
        {-1.1f, 0.0f, 0.0f, 1.7f, 0.0f},
    };
    float center[3] = {0.0f, 0.0f, 0.0f};

    auto sp1 = spectrophore::compute_from_atoms(atoms1, 2, center);
    auto sp2 = spectrophore::compute_from_atoms(atoms2, 2, center);

    float sim = sp1.tanimoto(sp2);
    EXPECT_GT(sim, 0.8f);
}

TEST(Spectrophore, DissimilarDescriptorsLowTanimoto) {
    // Very different configurations should have low Tanimoto
    spectrophore::SimpleAtom atoms1[] = {
        {1.0f, 0.0f, 0.0f, 1.7f, 0.0f},
    };
    spectrophore::SimpleAtom atoms2[] = {
        {0.0f, 0.0f, 5.0f, 1.2f, -1.0f},
    };
    float center[3] = {0.0f, 0.0f, 0.0f};

    auto sp1 = spectrophore::compute_from_atoms(atoms1, 1, center);
    auto sp2 = spectrophore::compute_from_atoms(atoms2, 1, center);

    float sim = sp1.tanimoto(sp2);
    EXPECT_LT(sim, 0.5f);
}

TEST(Spectrophore, EuclideanDistance) {
    spectrophore::Spectrophore a, b;
    a.values[0] = 3.0f;
    b.values[0] = 0.0f;
    a.values[1] = 4.0f;
    b.values[1] = 0.0f;
    EXPECT_NEAR(a.euclidean(b), 5.0f, 1e-5f);
}
