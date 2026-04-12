// tests/test_metal_coordination.cpp
// Unit tests for the metal ion coordination scoring potential
// Apache-2.0 (c) 2026 Le Bonhomme Pharma

#include <gtest/gtest.h>
#include "../LIB/metal_coordination.h"
#include <cmath>
#include <cstring>
#include <vector>

// Undefine the single-letter macro from flexaid.h to avoid name collisions
#ifdef E
#undef E
#endif

using namespace metal_coord;

static constexpr double EPS = 1e-6;

// ─── Helper: create a minimal atom with the given SYBYL type ────────────────
static atom_struct make_atom(int sybyl_type, float x = 0.f, float y = 0.f,
                              float z = 0.f) {
    atom_struct a{};
    a.type = sybyl_type;
    a.coor[0] = x;
    a.coor[1] = y;
    a.coor[2] = z;
    a.charge = 0.0f;
    a.type256 = 0;
    a.bond[0] = 0;
    std::memset(a.name, 0, sizeof(a.name));
    std::memset(a.element, 0, sizeof(a.element));
    return a;
}

// ===========================================================================
// MORSE POTENTIAL — UNIT TESTS
// ===========================================================================

TEST(MorsePotential, MinimumAtIdealDistance) {
    // At r = r0, Morse potential = well_depth
    double E = morse_potential(2.36, 2.36, -15.0, 2.0);
    EXPECT_NEAR(E, -15.0, EPS);
}

TEST(MorsePotential, AsymptoteAtLargeDistance) {
    // At large r, Morse potential -> 0
    double E = morse_potential(10.0, 2.36, -15.0, 2.0);
    EXPECT_NEAR(E, 0.0, 0.01);
}

TEST(MorsePotential, RepulsiveAtShortDistance) {
    // At r << r0, Morse potential > 0 (repulsive)
    double E = morse_potential(1.5, 2.36, -15.0, 2.0);
    EXPECT_GT(E, 0.0);
}

TEST(MorsePotential, SymmetricWellShape) {
    // Verify that the well is deeper at r0 than at r0 +/- delta
    double E_center = morse_potential(2.36, 2.36, -15.0, 2.0);
    double E_plus   = morse_potential(2.36 + 0.3, 2.36, -15.0, 2.0);
    double E_minus  = morse_potential(2.36 - 0.3, 2.36, -15.0, 2.0);
    EXPECT_LT(E_center, E_plus);
    EXPECT_LT(E_center, E_minus);
}

// ===========================================================================
// METAL PARAMETER LOOKUP
// ===========================================================================

TEST(MetalParams, Ca2PlusRecognized) {
    const MetalParams* p = get_metal_params(36);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->ideal_cn, 7);
    EXPECT_EQ(p->geometry, Geometry::PENTAGONAL_BIPYRAMIDAL);
    EXPECT_NEAR(p->angle_primary, 72.0, EPS);
}

TEST(MetalParams, Zn2PlusRecognized) {
    const MetalParams* p = get_metal_params(35);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->ideal_cn, 4);
    EXPECT_EQ(p->geometry, Geometry::TETRAHEDRAL);
}

TEST(MetalParams, Mg2PlusRecognized) {
    const MetalParams* p = get_metal_params(28);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->ideal_cn, 6);
    EXPECT_EQ(p->geometry, Geometry::OCTAHEDRAL);
}

TEST(MetalParams, NonMetalReturnsNull) {
    EXPECT_EQ(get_metal_params(1), nullptr);   // C.1
    EXPECT_EQ(get_metal_params(14), nullptr);  // O.3
    EXPECT_EQ(get_metal_params(0), nullptr);   // out of range
    EXPECT_EQ(get_metal_params(99), nullptr);  // out of range
}

TEST(MetalParams, PlaceholderMetalReturnsNull) {
    EXPECT_EQ(get_metal_params(29), nullptr);  // Sr — no params
}

// ===========================================================================
// DONOR AFFINITY LOOKUP
// ===========================================================================

TEST(DonorAffinity, Ca2PlusCarboxylateO) {
    const DonorAffinity* aff = get_donor_affinity(36, 15); // Ca2+ - O.CO2
    ASSERT_NE(aff, nullptr);
    EXPECT_NEAR(aff->ideal_dist, 2.36, EPS);
    EXPECT_NEAR(aff->well_depth, -15.0, EPS);
}

TEST(DonorAffinity, Ca2PlusCarbonylO) {
    const DonorAffinity* aff = get_donor_affinity(36, 13); // Ca2+ - O.2
    ASSERT_NE(aff, nullptr);
    EXPECT_NEAR(aff->ideal_dist, 2.36, EPS);
    EXPECT_NEAR(aff->well_depth, -12.0, EPS);
}

TEST(DonorAffinity, Ca2PlusHydroxylO) {
    const DonorAffinity* aff = get_donor_affinity(36, 14); // Ca2+ - O.3
    ASSERT_NE(aff, nullptr);
    EXPECT_NEAR(aff->ideal_dist, 2.38, EPS);
    EXPECT_NEAR(aff->well_depth, -10.0, EPS);
}

TEST(DonorAffinity, Ca2PlusNitrogenFallback) {
    // N types (6-12) should fall back to generic N parameters
    const DonorAffinity* aff = get_donor_affinity(36, 11); // Ca2+ - N.AM
    ASSERT_NE(aff, nullptr);
    EXPECT_NEAR(aff->ideal_dist, 2.50, EPS);
    EXPECT_NEAR(aff->well_depth, -4.0, EPS);
}

TEST(DonorAffinity, Zn2PlusSulfur) {
    const DonorAffinity* aff = get_donor_affinity(35, 18); // Zn2+ - S.3
    ASSERT_NE(aff, nullptr);
    EXPECT_NEAR(aff->ideal_dist, 2.30, EPS);
    EXPECT_NEAR(aff->well_depth, -16.0, EPS);
}

TEST(DonorAffinity, Zn2PlusNitrogenFallback) {
    const DonorAffinity* aff = get_donor_affinity(35, 10); // Zn2+ - N.AR
    ASSERT_NE(aff, nullptr);
    EXPECT_NEAR(aff->ideal_dist, 2.05, EPS);
    EXPECT_NEAR(aff->well_depth, -14.0, EPS);
}

TEST(DonorAffinity, UnknownMetalDonorPair) {
    // Ca2+ with a halogen (no affinity defined)
    EXPECT_EQ(get_donor_affinity(36, 23), nullptr); // Ca2+ - F
    EXPECT_EQ(get_donor_affinity(36, 24), nullptr); // Ca2+ - Cl
}

TEST(DonorAffinity, NonMetalReturnsNull) {
    EXPECT_EQ(get_donor_affinity(1, 14), nullptr);  // C.1 is not a metal
    EXPECT_EQ(get_donor_affinity(14, 36), nullptr); // O.3 is not a metal
}

// ===========================================================================
// TYPE CLASSIFICATION
// ===========================================================================

TEST(TypeClassification, MetalTypes) {
    EXPECT_TRUE(is_metal_type(36));  // Ca
    EXPECT_TRUE(is_metal_type(35));  // Zn
    EXPECT_TRUE(is_metal_type(28));  // Mg
    EXPECT_TRUE(is_metal_type(37));  // Fe
    EXPECT_FALSE(is_metal_type(1));  // C.1
    EXPECT_FALSE(is_metal_type(14)); // O.3
    EXPECT_FALSE(is_metal_type(29)); // Sr (placeholder)
}

TEST(TypeClassification, DonorTypes) {
    EXPECT_TRUE(is_coord_donor_type(13));  // O.2
    EXPECT_TRUE(is_coord_donor_type(14));  // O.3
    EXPECT_TRUE(is_coord_donor_type(15));  // O.CO2
    EXPECT_TRUE(is_coord_donor_type(18));  // S.3
    EXPECT_TRUE(is_coord_donor_type(6));   // N.1
    EXPECT_TRUE(is_coord_donor_type(22));  // P.3
    EXPECT_FALSE(is_coord_donor_type(1));  // C.1
    EXPECT_FALSE(is_coord_donor_type(3));  // C.3
    EXPECT_FALSE(is_coord_donor_type(36)); // Ca
}

// ===========================================================================
// FULL SCORING FUNCTION (compute_metal_coord_energy)
// ===========================================================================

TEST(MetalCoordEnergy, Ca2PlusO_CO2_IdealDistance) {
    atom_struct atoms[2];
    atoms[0] = make_atom(36, 0.f, 0.f, 0.f);       // Ca2+
    atoms[1] = make_atom(15, 2.36f, 0.f, 0.f);      // O.CO2 at ideal distance
    double E = compute_metal_coord_energy(atoms, 0, 1, 2.36, 1.0, 2.0);
    // At ideal distance: E = weight * well_depth = 1.0 * -15.0
    EXPECT_NEAR(E, -15.0, EPS);
}

TEST(MetalCoordEnergy, Ca2PlusO_CO2_FarAway) {
    atom_struct atoms[2];
    atoms[0] = make_atom(36, 0.f, 0.f, 0.f);
    atoms[1] = make_atom(15, 8.0f, 0.f, 0.f);
    double E = compute_metal_coord_energy(atoms, 0, 1, 8.0, 1.0, 2.0);
    EXPECT_NEAR(E, 0.0, 0.01);
}

TEST(MetalCoordEnergy, Ca2PlusO_CO2_TooClose) {
    atom_struct atoms[2];
    atoms[0] = make_atom(36, 0.f, 0.f, 0.f);
    atoms[1] = make_atom(15, 1.5f, 0.f, 0.f);
    double E = compute_metal_coord_energy(atoms, 0, 1, 1.5, 1.0, 2.0);
    EXPECT_GT(E, 0.0);  // repulsive at short distance
}

TEST(MetalCoordEnergy, ReverseAtomOrder) {
    // Should work regardless of which atom is metal vs donor
    atom_struct atoms[2];
    atoms[0] = make_atom(15, 2.36f, 0.f, 0.f);      // O.CO2
    atoms[1] = make_atom(36, 0.f, 0.f, 0.f);         // Ca2+
    double E = compute_metal_coord_energy(atoms, 0, 1, 2.36, 1.0, 2.0);
    EXPECT_NEAR(E, -15.0, EPS);
}

TEST(MetalCoordEnergy, NonMetalPairReturnsZero) {
    atom_struct atoms[2];
    atoms[0] = make_atom(3, 0.f, 0.f, 0.f);   // C.3
    atoms[1] = make_atom(14, 2.0f, 0.f, 0.f);  // O.3
    double E = compute_metal_coord_energy(atoms, 0, 1, 2.0, 1.0, 2.0);
    EXPECT_EQ(E, 0.0);
}

TEST(MetalCoordEnergy, MetalMetalReturnsZero) {
    atom_struct atoms[2];
    atoms[0] = make_atom(36, 0.f, 0.f, 0.f);  // Ca2+
    atoms[1] = make_atom(35, 3.0f, 0.f, 0.f);  // Zn2+
    double E = compute_metal_coord_energy(atoms, 0, 1, 3.0, 1.0, 2.0);
    EXPECT_EQ(E, 0.0);
}

TEST(MetalCoordEnergy, WeightScaling) {
    atom_struct atoms[2];
    atoms[0] = make_atom(36, 0.f, 0.f, 0.f);
    atoms[1] = make_atom(15, 2.36f, 0.f, 0.f);
    double E1 = compute_metal_coord_energy(atoms, 0, 1, 2.36, 1.0, 2.0);
    double E2 = compute_metal_coord_energy(atoms, 0, 1, 2.36, 2.0, 2.0);
    EXPECT_NEAR(E2, 2.0 * E1, EPS);
}

TEST(MetalCoordEnergy, Zn2PlusS3) {
    atom_struct atoms[2];
    atoms[0] = make_atom(35, 0.f, 0.f, 0.f);       // Zn2+
    atoms[1] = make_atom(18, 2.30f, 0.f, 0.f);      // S.3
    double E = compute_metal_coord_energy(atoms, 0, 1, 2.30, 1.0, 2.0);
    EXPECT_NEAR(E, -16.0, EPS);
}

TEST(MetalCoordEnergy, Mg2PlusO_CO2) {
    atom_struct atoms[2];
    atoms[0] = make_atom(28, 0.f, 0.f, 0.f);       // Mg2+
    atoms[1] = make_atom(15, 2.06f, 0.f, 0.f);      // O.CO2
    double E = compute_metal_coord_energy(atoms, 0, 1, 2.06, 1.0, 2.0);
    EXPECT_NEAR(E, -14.0, EPS);
}

// ===========================================================================
// COORDINATION NUMBER PENALTY
// ===========================================================================

TEST(CNPenalty, IdealCNZeroPenalty) {
    EXPECT_NEAR(cn_penalty(7, 7), 0.0, EPS);
    EXPECT_NEAR(cn_penalty(4, 4), 0.0, EPS);
    EXPECT_NEAR(cn_penalty(6, 6), 0.0, EPS);
}

TEST(CNPenalty, OneOffPenalty) {
    // delta=1 → -0.5 * 1 = -0.5
    EXPECT_NEAR(cn_penalty(6, 7), -0.5, EPS);
    EXPECT_NEAR(cn_penalty(8, 7), -0.5, EPS);
}

TEST(CNPenalty, TwoOffPenalty) {
    // delta=2 → -0.5 * 4 = -2.0
    EXPECT_NEAR(cn_penalty(5, 7), -2.0, EPS);
}

TEST(CNPenalty, SymmetricPenalty) {
    // Under-coordination and over-coordination penalized equally
    EXPECT_EQ(cn_penalty(5, 7), cn_penalty(9, 7));
}

// ===========================================================================
// COORDINATION CUTOFF
// ===========================================================================

TEST(CoordCutoff, WithinCutoff) {
    EXPECT_TRUE(is_coordinating(2.36, 2.36));
    EXPECT_TRUE(is_coordinating(3.0, 2.36));   // 2.36 + 0.8 = 3.16
}

TEST(CoordCutoff, OutsideCutoff) {
    EXPECT_FALSE(is_coordinating(3.5, 2.36));  // > 3.16
}

// ===========================================================================
// MULTIPLE METALS — SIMULTANEOUS SCORING
// ===========================================================================

TEST(MetalCoordEnergy, FeNitrogen) {
    atom_struct atoms[2];
    atoms[0] = make_atom(37, 0.f, 0.f, 0.f);       // Fe
    atoms[1] = make_atom(10, 2.15f, 0.f, 0.f);      // N.AR (fallback to generic N)
    double E = compute_metal_coord_energy(atoms, 0, 1, 2.15, 1.0, 2.0);
    EXPECT_NEAR(E, -14.0, EPS);
}

TEST(MetalCoordEnergy, Cu2PlusSulfur) {
    atom_struct atoms[2];
    atoms[0] = make_atom(30, 0.f, 0.f, 0.f);       // Cu2+
    atoms[1] = make_atom(18, 2.15f, 0.f, 0.f);      // S.3
    double E = compute_metal_coord_energy(atoms, 0, 1, 2.15, 1.0, 2.0);
    EXPECT_NEAR(E, -14.0, EPS);
}

TEST(MetalCoordEnergy, UnknownDonorForMetal) {
    atom_struct atoms[2];
    atoms[0] = make_atom(36, 0.f, 0.f, 0.f);       // Ca2+
    atoms[1] = make_atom(25, 3.0f, 0.f, 0.f);       // BR — no affinity
    double E = compute_metal_coord_energy(atoms, 0, 1, 3.0, 1.0, 2.0);
    EXPECT_EQ(E, 0.0);
}
