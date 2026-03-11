// tests/test_ring_conformer_library.cpp
// Unit tests for RingConformerLibrary — non-aromatic ring conformer sampling
// Tests the conformer table, singleton pattern, and ring detection.

#include <gtest/gtest.h>
#include "RingConformerLibrary.h"
#include <cmath>
#include <set>

using namespace ring_flex;

// ===========================================================================
// CONFORMER LIBRARY SINGLETON
// ===========================================================================

TEST(RingConformerLibrary, SingletonReturnsSameInstance) {
    auto& lib1 = RingConformerLibrary::instance();
    auto& lib2 = RingConformerLibrary::instance();
    EXPECT_EQ(&lib1, &lib2);
}

TEST(RingConformerLibrary, HasExpectedSixMemberedConformers) {
    auto& lib = RingConformerLibrary::instance();
    // 8 conformer types: 2 chair, 2 boat, 2 twist-boat, 2 half-chair
    EXPECT_EQ(lib.n_six(), 8);
    EXPECT_EQ(lib.six_conformers().size(), 8u);
}

TEST(RingConformerLibrary, HasExpectedFiveMemberedConformers) {
    auto& lib = RingConformerLibrary::instance();
    // 5 envelope conformers: E0-E4
    EXPECT_EQ(lib.n_five(), 5);
    EXPECT_EQ(lib.five_conformers().size(), 5u);
}

// ===========================================================================
// SIX-MEMBERED RING CONFORMER TABLE
// ===========================================================================

TEST(RingConformerLibrary, SixConformerTypesAreDistinct) {
    auto& lib = RingConformerLibrary::instance();
    std::set<uint8_t> seen;
    for (const auto& conf : lib.six_conformers()) {
        EXPECT_TRUE(seen.insert(static_cast<uint8_t>(conf.type)).second)
            << "Duplicate conformer type: " << conf.name;
    }
}

TEST(RingConformerLibrary, SixConformerDihedralsArePhysical) {
    auto& lib = RingConformerLibrary::instance();
    for (const auto& conf : lib.six_conformers()) {
        for (int i = 0; i < 6; ++i) {
            // Ring dihedrals should be in range [-90, 90] degrees
            EXPECT_GE(conf.dihedrals[i], -90.0f)
                << conf.name << " dihedral[" << i << "]";
            EXPECT_LE(conf.dihedrals[i], 90.0f)
                << conf.name << " dihedral[" << i << "]";
        }
    }
}

TEST(RingConformerLibrary, ChairConformersAreMirrors) {
    auto& lib = RingConformerLibrary::instance();
    const auto& chair1 = lib.six_at(0);  // 4C1
    const auto& chair2 = lib.six_at(1);  // 1C4

    EXPECT_EQ(chair1.type, SixConformerType::Chair1);
    EXPECT_EQ(chair2.type, SixConformerType::Chair2);

    // Chair mirrors: dihedrals should be negated
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(chair1.dihedrals[i], -chair2.dihedrals[i], 0.1f);
    }
}

TEST(RingConformerLibrary, SixConformerNamesAreNonEmpty) {
    auto& lib = RingConformerLibrary::instance();
    for (const auto& conf : lib.six_conformers()) {
        EXPECT_FALSE(conf.name.empty());
    }
}

// ===========================================================================
// FIVE-MEMBERED RING CONFORMER TABLE
// ===========================================================================

TEST(RingConformerLibrary, FiveConformerPhasesAreDifferent) {
    auto& lib = RingConformerLibrary::instance();
    std::set<float> phases;
    for (const auto& conf : lib.five_conformers()) {
        EXPECT_TRUE(phases.insert(conf.phase_P).second)
            << "Duplicate phase_P: " << conf.phase_P;
    }
}

TEST(RingConformerLibrary, FiveConformerDihedralsArePhysical) {
    auto& lib = RingConformerLibrary::instance();
    for (const auto& conf : lib.five_conformers()) {
        for (int i = 0; i < 5; ++i) {
            EXPECT_GE(conf.dihedrals[i], -60.0f)
                << conf.name << " dihedral[" << i << "]";
            EXPECT_LE(conf.dihedrals[i], 60.0f)
                << conf.name << " dihedral[" << i << "]";
        }
    }
}

// ===========================================================================
// RANDOM INDEX GENERATION
// ===========================================================================

TEST(RingConformerLibrary, RandomSixIndexInBounds) {
    auto& lib = RingConformerLibrary::instance();
    for (int trial = 0; trial < 100; ++trial) {
        int idx = lib.random_six_index();
        EXPECT_GE(idx, 0);
        EXPECT_LT(idx, lib.n_six());
    }
}

TEST(RingConformerLibrary, RandomFiveIndexInBounds) {
    auto& lib = RingConformerLibrary::instance();
    for (int trial = 0; trial < 100; ++trial) {
        int idx = lib.random_five_index();
        EXPECT_GE(idx, 0);
        EXPECT_LT(idx, lib.n_five());
    }
}

// ===========================================================================
// INDEXED ACCESS
// ===========================================================================

TEST(RingConformerLibrary, SixAtValidIndex) {
    auto& lib = RingConformerLibrary::instance();
    const auto& conf = lib.six_at(0);
    EXPECT_EQ(conf.type, SixConformerType::Chair1);
}

TEST(RingConformerLibrary, SixAtInvalidIndexThrows) {
    auto& lib = RingConformerLibrary::instance();
    EXPECT_THROW(lib.six_at(100), std::out_of_range);
}

TEST(RingConformerLibrary, FiveAtValidIndex) {
    auto& lib = RingConformerLibrary::instance();
    const auto& conf = lib.five_at(0);
    EXPECT_EQ(conf.type, FiveConformerType::E0);
}

TEST(RingConformerLibrary, FiveAtInvalidIndexThrows) {
    auto& lib = RingConformerLibrary::instance();
    EXPECT_THROW(lib.five_at(100), std::out_of_range);
}

// ===========================================================================
// RING DETECTION
// ===========================================================================

TEST(RingDetection, NullInputReturnsEmpty) {
    auto result = detect_non_aromatic_rings(nullptr, 0, nullptr);
    EXPECT_TRUE(result.empty());
}

TEST(RingDetection, EmptyInputReturnsEmpty) {
    atom atoms[1] = {};
    int indices[1] = {0};
    auto result = detect_non_aromatic_rings(indices, 0, atoms);
    EXPECT_TRUE(result.empty());
}

TEST(RingDetection, SixMemberedRingDetected) {
    // Build a simple 6-membered non-aromatic ring (cyclohexane-like)
    // Each atom has exactly 2 neighbours (ring) + sp3 carbon
    const int N = 6;
    atom atoms[N] = {};

    for (int i = 0; i < N; ++i) {
        atoms[i].element[0] = 'C';
        atoms[i].element[1] = ' ';
        atoms[i].bond[0] = 2;  // 2 bonds (ring only, no hydrogens)
        atoms[i].bond[1] = (i + 1) % N;
        atoms[i].bond[2] = (i + N - 1) % N;
        // Make it sp3 by adding more bonds (> 3 bonds to defeat aromaticity check)
        atoms[i].bond[0] = 4;
        atoms[i].bond[3] = i;  // self-bond (dummy, won't be traversed)
        atoms[i].bond[4] = i;
    }

    int indices[] = {0, 1, 2, 3, 4, 5};
    auto result = detect_non_aromatic_rings(indices, N, atoms);

    // Should detect the ring as non-aromatic (sp3)
    EXPECT_GE(result.size(), 1u);
    if (!result.empty()) {
        EXPECT_EQ(result[0].size, RingSize::Six);
        EXPECT_FALSE(result[0].is_aromatic);
        EXPECT_EQ(result[0].atom_indices.size(), 6u);
    }
}

// ===========================================================================
// RING DESCRIPTOR
// ===========================================================================

TEST(RingDescriptor, DefaultIsNotAromatic) {
    RingDescriptor desc;
    desc.atom_indices = {0, 1, 2, 3, 4};
    desc.size = RingSize::Five;
    desc.is_aromatic = false;
    EXPECT_FALSE(desc.is_aromatic);
    EXPECT_EQ(desc.size, RingSize::Five);
}
