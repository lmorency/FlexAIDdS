// test_gist_grid.cpp — Unit tests for GIST desolvation grid
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <fstream>
#include <cmath>
#include "GISTGrid.h"

class GISTGridTest : public ::testing::Test {
protected:
    std::string test_dx_file;

    void SetUp() override {
        // Create a small synthetic 4x4x4 DX grid file
        test_dx_file = "test_gist_grid.dx";
        std::ofstream ofs(test_dx_file);
        ofs << "# OpenDX test grid\n";
        ofs << "object 1 class gridpositions counts 4 4 4\n";
        ofs << "origin 0.0 0.0 0.0\n";
        ofs << "delta 1.0 0.0 0.0\n";
        ofs << "delta 0.0 1.0 0.0\n";
        ofs << "delta 0.0 0.0 1.0\n";
        ofs << "object 2 class gridconnections counts 4 4 4\n";
        ofs << "object 3 class array type float rank 0 items 64 data follows\n";
        // Fill with values: value = x + 10*y + 100*z (for easy verification)
        for (int x = 0; x < 4; ++x)
            for (int y = 0; y < 4; ++y)
                for (int z = 0; z < 4; ++z)
                    ofs << (x + 10.0 * y + 100.0 * z) << " ";
        ofs << "\n";
        ofs << "attribute \"dep\" string \"positions\"\n";
        ofs.close();
    }

    void TearDown() override {
        std::remove(test_dx_file.c_str());
    }
};

TEST_F(GISTGridTest, LoadValidDXFile) {
    gist::GISTGrid grid;
    EXPECT_TRUE(grid.load_dx(test_dx_file));
    EXPECT_TRUE(grid.is_loaded());
    EXPECT_EQ(grid.nx(), 4);
    EXPECT_EQ(grid.ny(), 4);
    EXPECT_EQ(grid.nz(), 4);
}

TEST_F(GISTGridTest, LoadNonExistentFile) {
    gist::GISTGrid grid;
    EXPECT_FALSE(grid.load_dx("nonexistent_file.dx"));
    EXPECT_FALSE(grid.is_loaded());
}

TEST_F(GISTGridTest, GridCornerValues) {
    gist::GISTGrid grid;
    ASSERT_TRUE(grid.load_dx(test_dx_file));

    // At grid origin (0,0,0): value = 0 + 0 + 0 = 0
    EXPECT_NEAR(grid.desolvation_energy(0.0f, 0.0f, 0.0f), 0.0, 0.01);

    // At (1,0,0): value = 1 + 0 + 0 = 1
    EXPECT_NEAR(grid.desolvation_energy(1.0f, 0.0f, 0.0f), 1.0, 0.01);

    // At (0,1,0): value = 0 + 10 + 0 = 10
    EXPECT_NEAR(grid.desolvation_energy(0.0f, 1.0f, 0.0f), 10.0, 0.01);

    // At (0,0,1): value = 0 + 0 + 100 = 100
    EXPECT_NEAR(grid.desolvation_energy(0.0f, 0.0f, 1.0f), 100.0, 0.01);
}

TEST_F(GISTGridTest, TrilinearInterpolation) {
    gist::GISTGrid grid;
    ASSERT_TRUE(grid.load_dx(test_dx_file));

    // At midpoint (0.5, 0, 0): should interpolate between 0 and 1 = 0.5
    double E = grid.desolvation_energy(0.5f, 0.0f, 0.0f);
    EXPECT_NEAR(E, 0.5, 0.01);

    // At (0.5, 0.5, 0): interpolate between 0, 1, 10, 11
    // Expected: (0+1+10+11)/4 = 5.5
    double E2 = grid.desolvation_energy(0.5f, 0.5f, 0.0f);
    EXPECT_NEAR(E2, 5.5, 0.01);
}

TEST_F(GISTGridTest, OutOfBoundsReturnsZero) {
    gist::GISTGrid grid;
    ASSERT_TRUE(grid.load_dx(test_dx_file));

    EXPECT_DOUBLE_EQ(grid.desolvation_energy(-1.0f, 0.0f, 0.0f), 0.0);
    EXPECT_DOUBLE_EQ(grid.desolvation_energy(0.0f, -1.0f, 0.0f), 0.0);
    EXPECT_DOUBLE_EQ(grid.desolvation_energy(10.0f, 0.0f, 0.0f), 0.0);
}

TEST_F(GISTGridTest, UnloadedGridReturnsZero) {
    gist::GISTGrid grid;
    EXPECT_DOUBLE_EQ(grid.desolvation_energy(1.0f, 1.0f, 1.0f), 0.0);
}
