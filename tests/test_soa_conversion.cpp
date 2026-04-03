// test_soa_conversion.cpp — Unit tests for AoS↔SoA atom data conversion
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cmath>
#include "flexaid.h"
#include "AtomSoA.h"

class SoAConversionTest : public ::testing::Test {
protected:
    static constexpr int N_ATOMS = 100;
    atom atoms[N_ATOMS];

    void SetUp() override {
        for (int i = 0; i < N_ATOMS; ++i) {
            atoms[i].coor[0] = static_cast<float>(i) * 1.5f;
            atoms[i].coor[1] = static_cast<float>(i) * 2.0f;
            atoms[i].coor[2] = static_cast<float>(i) * 0.5f;
            atoms[i].radius  = 1.5f + (i % 5) * 0.1f;
            atoms[i].charge  = -0.5f + (i % 10) * 0.1f;
            atoms[i].resp_charge = 0.0f;
            atoms[i].has_resp = 0;
            atoms[i].type256 = static_cast<uint8_t>(i % 256);
        }
    }
};

TEST_F(SoAConversionTest, RoundTripPreservesCoordinates) {
    atom_soa::AtomArrays soa;
    soa.from_aos(atoms, N_ATOMS);

    ASSERT_EQ(soa.count, N_ATOMS);

    for (int i = 0; i < N_ATOMS; ++i) {
        EXPECT_FLOAT_EQ(soa.x[i], atoms[i].coor[0]);
        EXPECT_FLOAT_EQ(soa.y[i], atoms[i].coor[1]);
        EXPECT_FLOAT_EQ(soa.z[i], atoms[i].coor[2]);
        EXPECT_FLOAT_EQ(soa.radius[i], atoms[i].radius);
        EXPECT_FLOAT_EQ(soa.charge[i], atoms[i].charge);
        EXPECT_EQ(soa.type256[i], atoms[i].type256);
    }

    // Modify SoA coordinates
    for (int i = 0; i < N_ATOMS; ++i) {
        soa.x[i] += 1.0f;
        soa.y[i] += 2.0f;
        soa.z[i] += 3.0f;
    }

    // Write back to AoS
    atom atoms_copy[N_ATOMS];
    std::memcpy(atoms_copy, atoms, sizeof(atoms));
    soa.to_aos(atoms_copy, N_ATOMS);

    for (int i = 0; i < N_ATOMS; ++i) {
        EXPECT_FLOAT_EQ(atoms_copy[i].coor[0], atoms[i].coor[0] + 1.0f);
        EXPECT_FLOAT_EQ(atoms_copy[i].coor[1], atoms[i].coor[1] + 2.0f);
        EXPECT_FLOAT_EQ(atoms_copy[i].coor[2], atoms[i].coor[2] + 3.0f);
    }
}

TEST_F(SoAConversionTest, RESPChargePreference) {
    // Set RESP charges on some atoms
    atoms[0].resp_charge = 0.42f;
    atoms[0].has_resp = 1;
    atoms[1].resp_charge = -0.33f;
    atoms[1].has_resp = 1;

    atom_soa::AtomArrays soa;
    soa.from_aos(atoms, N_ATOMS);

    // SoA should prefer RESP charges when available
    EXPECT_FLOAT_EQ(soa.charge[0], 0.42f);
    EXPECT_FLOAT_EQ(soa.charge[1], -0.33f);
    // Non-RESP atoms should use standard charge
    EXPECT_FLOAT_EQ(soa.charge[2], atoms[2].charge);
}

TEST_F(SoAConversionTest, Distance2Computation) {
    atom_soa::AtomArrays a, b;
    a.resize(1);
    b.resize(1);
    a.x[0] = 0.0f; a.y[0] = 0.0f; a.z[0] = 0.0f;
    b.x[0] = 3.0f; b.y[0] = 4.0f; b.z[0] = 0.0f;

    float d2 = atom_soa::distance2(a, 0, b, 0);
    EXPECT_FLOAT_EQ(d2, 25.0f); // 3² + 4² = 25
}

TEST_F(SoAConversionTest, EmptyConversion) {
    atom_soa::AtomArrays soa;
    soa.from_aos(atoms, 0);
    EXPECT_EQ(soa.count, 0);
}

TEST_F(SoAConversionTest, AlignmentCheck) {
    atom_soa::AtomArrays soa;
    soa.from_aos(atoms, N_ATOMS);

    // Check 64-byte alignment (AVX-512)
    EXPECT_EQ(reinterpret_cast<uintptr_t>(soa.x.data()) % 64, 0u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(soa.y.data()) % 64, 0u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(soa.z.data()) % 64, 0u);
}
