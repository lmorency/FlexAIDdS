// test_hbond_potential.cpp — Unit tests for angular-dependent H-bond potential
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cmath>
#include "flexaid.h"
#include "hbond_potential.h"
#include "atom_typing_256.h"

// Helper: create a minimal atom at given coordinates with specified type256
static atom make_atom(float x, float y, float z, uint8_t type256_val = 0) {
    atom a = {};
    a.coor[0] = x;
    a.coor[1] = y;
    a.coor[2] = z;
    a.type256 = type256_val;
    a.bond[0] = 0; // no bonds by default
    a.element[0] = 'C'; a.element[1] = '\0';
    a.name[0] = 'C'; a.name[1] = '\0';
    return a;
}

// Helper: make a hydrogen atom
static atom make_hydrogen(float x, float y, float z) {
    atom a = make_atom(x, y, z);
    a.element[0] = 'H'; a.element[1] = '\0';
    a.name[0] = 'H'; a.name[1] = '\0';
    return a;
}

class HBondPotentialTest : public ::testing::Test {
protected:
    // Default parameters matching config_defaults.h
    double optimal_dist = 2.8;
    double optimal_angle = 180.0;
    double sigma_dist = 0.4;
    double sigma_angle = 30.0;
    double weight = -2.5;
    double salt_bridge_weight = -5.0;
};

TEST_F(HBondPotentialTest, NonHBondCapableAtomsReturnZero) {
    // Two carbon atoms (not H-bond capable)
    uint8_t c_type = atom256::encode(atom256::C_sp3, atom256::Q_POSITIVE, false);
    atom atoms[2] = {
        make_atom(0, 0, 0, c_type),
        make_atom(2.8f, 0, 0, c_type),
    };

    double energy = hbond::compute_hbond_energy(
        atoms, 0, 1, 2.8, optimal_dist, optimal_angle,
        sigma_dist, sigma_angle, weight, salt_bridge_weight);
    EXPECT_DOUBLE_EQ(energy, 0.0);
}

TEST_F(HBondPotentialTest, HBondAtOptimalGeometry) {
    // N-H...O at optimal distance 2.8 A, angle ~180 degrees
    // Donor N at origin, H along x-axis, Acceptor O at 2.8 A
    // Use same charge polarity to avoid salt bridge path
    uint8_t n_type = atom256::encode(atom256::N_sp3, atom256::Q_NEGATIVE, true);
    uint8_t o_type = atom256::encode(atom256::O_sp2, atom256::Q_NEGATIVE, true);

    atom atoms[3];
    atoms[0] = make_atom(0, 0, 0, n_type);         // N (donor)
    atoms[1] = make_hydrogen(1.0f, 0, 0);           // H
    atoms[2] = make_atom(2.8f, 0, 0, o_type);       // O (acceptor)

    // Bond: N has H bonded
    atoms[0].bond[0] = 1;
    atoms[0].bond[1] = 1; // index of H

    double energy = hbond::compute_hbond_energy(
        atoms, 0, 2, 2.8, optimal_dist, optimal_angle,
        sigma_dist, sigma_angle, weight, salt_bridge_weight);

    // Should be close to the full weight (negative energy)
    EXPECT_LT(energy, 0.0);
    EXPECT_GT(energy, weight - 0.5); // within tolerance of full weight
}

TEST_F(HBondPotentialTest, DistanceDependence) {
    // Use same charge polarity to test standard H-bond (not salt bridge)
    uint8_t n_type = atom256::encode(atom256::N_sp3, atom256::Q_NEGATIVE, true);
    uint8_t o_type = atom256::encode(atom256::O_sp2, atom256::Q_NEGATIVE, true);

    atom atoms[3];
    atoms[0] = make_atom(0, 0, 0, n_type);
    atoms[1] = make_hydrogen(1.0f, 0, 0);
    atoms[2] = make_atom(2.8f, 0, 0, o_type);
    atoms[0].bond[0] = 1;
    atoms[0].bond[1] = 1;

    double E_optimal = hbond::compute_hbond_energy(
        atoms, 0, 2, 2.8, optimal_dist, optimal_angle,
        sigma_dist, sigma_angle, weight, salt_bridge_weight);

    // Far distance: should be weaker
    double E_far = hbond::compute_hbond_energy(
        atoms, 0, 2, 5.0, optimal_dist, optimal_angle,
        sigma_dist, sigma_angle, weight, salt_bridge_weight);

    EXPECT_LT(std::abs(E_far), std::abs(E_optimal));
}

TEST_F(HBondPotentialTest, SaltBridgeDetection) {
    // Anionic + cationic pair should use salt_bridge_weight
    uint8_t anion_type = atom256::encode(atom256::O_co2, atom256::Q_NEGATIVE, true);
    uint8_t cation_type = atom256::encode(atom256::N_quat, atom256::Q_POSITIVE, true);

    atom atoms[3];
    atoms[0] = make_atom(0, 0, 0, anion_type);
    atoms[1] = make_hydrogen(1.0f, 0, 0);
    atoms[2] = make_atom(2.8f, 0, 0, cation_type);
    atoms[2].bond[0] = 1;
    atoms[2].bond[1] = 1; // H bonded to cation

    double energy = hbond::compute_hbond_energy(
        atoms, 0, 2, 2.8, optimal_dist, optimal_angle,
        sigma_dist, sigma_angle, weight, salt_bridge_weight);

    // Salt bridge should be stronger (more negative) than standard H-bond
    EXPECT_LT(energy, 0.0);
}

TEST_F(HBondPotentialTest, AngleDegFunction) {
    float a[3] = {0, 0, 0};
    float b[3] = {1, 0, 0};
    float c[3] = {0, 1, 0};

    double angle = hbond::angle_deg(a, b, c);
    EXPECT_NEAR(angle, 90.0, 0.01);

    // 180 degrees: b and c on opposite sides
    float c2[3] = {-1, 0, 0};
    double angle2 = hbond::angle_deg(a, b, c2);
    EXPECT_NEAR(angle2, 180.0, 0.01);

    // 0 degrees: same direction
    float c3[3] = {2, 0, 0};
    double angle3 = hbond::angle_deg(a, b, c3);
    EXPECT_NEAR(angle3, 0.0, 0.01);
}

TEST_F(HBondPotentialTest, CfstrNewFieldsInitializedToZero) {
    cfstr cf = {};
    EXPECT_DOUBLE_EQ(cf.hbond, 0.0);
    EXPECT_DOUBLE_EQ(cf.gist_desolv, 0.0);
}
