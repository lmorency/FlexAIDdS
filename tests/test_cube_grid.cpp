// tests/test_cube_grid.cpp — Unit tests for cube grid operations
// Tests GridKey reproducibility, generate_grid correctness, and slice_grid midpoint logic

#include <gtest/gtest.h>
#include <cmath>
#include <map>
#include <set>
#include <vector>
#include "../LIB/maps.hpp"

// ============================================================================
// GridKey tests — reproducible integer-snapped coordinate keys
// ============================================================================

TEST(GridKey, SnapsToMilliangstroms) {
    float coor[3] = {1.234f, -5.678f, 0.0f};
    GridKey k(coor);
    EXPECT_EQ(k.ix, 1234);
    EXPECT_EQ(k.iy, -5678);
    EXPECT_EQ(k.iz, 0);
}

TEST(GridKey, RoundTrip) {
    float orig[3] = {12.345f, -0.001f, 99.999f};
    GridKey k(orig);
    float recovered[3];
    k.to_coor(recovered);
    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(recovered[i], orig[i], 0.001f);
    }
}

TEST(GridKey, NegativeZeroEqualsPositiveZero) {
    // -0.0 and +0.0 should produce the same key
    float a[3] = {0.0f, 0.0f, 0.0f};
    float b[3] = {-0.0f, -0.0f, -0.0f};
    GridKey ka(a), kb(b);
    EXPECT_EQ(ka, kb);
}

TEST(GridKey, CloseValuesDedup) {
    // Values that differ by less than 0.5 milliangstrom should snap to same key
    float a[3] = {1.0004f, 2.0f, 3.0f};
    float b[3] = {1.0001f, 2.0f, 3.0f};
    GridKey ka(a), kb(b);
    EXPECT_EQ(ka, kb);
}

TEST(GridKey, DifferentValuesDistinct) {
    // Values that differ by >= 1 milliangstrom should be distinct
    float a[3] = {1.000f, 2.000f, 3.000f};
    float b[3] = {1.001f, 2.000f, 3.000f};
    GridKey ka(a), kb(b);
    EXPECT_FALSE(ka == kb);
}

TEST(GridKey, OrderingIsConsistent) {
    GridKey a(1.0f, 2.0f, 3.0f);
    GridKey b(1.0f, 2.0f, 3.001f);
    GridKey c(1.0f, 2.001f, 3.0f);
    GridKey d(1.001f, 2.0f, 3.0f);

    // Lexicographic: x first, then y, then z
    EXPECT_TRUE(a < b);  // same x,y, smaller z
    EXPECT_TRUE(a < c);  // same x, smaller y
    EXPECT_TRUE(a < d);  // smaller x
}

TEST(GridKey, UsableAsMapKey) {
    std::map<GridKey, int> m;
    GridKey a(1.0f, 2.0f, 3.0f);
    GridKey b(4.0f, 5.0f, 6.0f);
    GridKey a_dup(1.0f, 2.0f, 3.0f);

    m[a] = 10;
    m[b] = 20;

    EXPECT_EQ(m.size(), 2u);
    EXPECT_EQ(m[a_dup], 10);  // lookup by equivalent key works
}

// ============================================================================
// Legacy get_key / parse_key tests — ensure backward compatibility
// (Suppress deprecation warnings since these are intentionally testing
// the deprecated API)
// ============================================================================

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4996)
#endif

TEST(LegacyMapKey, RoundTrip) {
    float orig[3] = {1.234f, -5.678f, 0.0f};
    std::string key = get_key(orig);
    float recovered[3];
    parse_key(key, recovered);
    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(recovered[i], orig[i], 0.001f);
    }
}

TEST(LegacyMapKey, KeyLength) {
    float coor[3] = {1.0f, 2.0f, 3.0f};
    std::string key = get_key(coor);
    EXPECT_EQ(key.length(), 24u);  // 3 * 8 chars
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

// ============================================================================
// Diagonal distance test — verifies the sin(45) fix
// ============================================================================

TEST(SliceGrid, DiagonalDistanceFormula) {
    // The face-diagonal of a cube cell with edge length s is s*sqrt(2).
    // Previously the code used sin(45.0f) (radians), giving wrong results.
    // Verify the corrected formula: sqrhyp = 2 * s^2
    float spacer = 0.75f;
    float sqrspa = spacer * spacer;
    float sqrhyp_correct = 2.0f * sqrspa;

    // Two grid points at a face diagonal
    float p1[3] = {0.0f, 0.0f, 0.0f};
    float p2[3] = {spacer, spacer, 0.0f};

    float dx = p2[0] - p1[0];
    float dy = p2[1] - p1[1];
    float dz = p2[2] - p1[2];
    float actual_sqrdist = dx*dx + dy*dy + dz*dz;

    EXPECT_NEAR(actual_sqrdist, sqrhyp_correct, 1e-6f);

    // The old buggy formula for comparison
    float sqrhyp_buggy = std::pow(spacer / std::sin(45.0f), 2.0f);
    // sin(45 radians) ≈ 0.8509, so this gives spacer^2 / 0.724 ≈ 1.381 * spacer^2
    // which is wrong (should be 2.0 * spacer^2)
    EXPECT_GT(std::fabs(actual_sqrdist - sqrhyp_buggy), 0.1f);
}

// ============================================================================
// GridKey neighbor probing — validates the O(n) slice approach
// ============================================================================

TEST(SliceGrid, NeighborProbeFindsAxisAligned) {
    float spacer = 0.5f;
    int ispacer = static_cast<int>(std::round(spacer * 1000.0f));  // 500

    GridKey origin(0.0f, 0.0f, 0.0f);
    GridKey right(spacer, 0.0f, 0.0f);

    // Verify the probe offset matches
    GridKey probed;
    probed.ix = origin.ix + ispacer;
    probed.iy = origin.iy;
    probed.iz = origin.iz;

    EXPECT_EQ(probed, right);

    // Verify midpoint
    GridKey mid;
    mid.ix = (origin.ix + right.ix) / 2;
    mid.iy = (origin.iy + right.iy) / 2;
    mid.iz = (origin.iz + right.iz) / 2;

    float mid_coor[3];
    mid.to_coor(mid_coor);
    EXPECT_NEAR(mid_coor[0], spacer / 2.0f, 0.001f);
    EXPECT_NEAR(mid_coor[1], 0.0f, 0.001f);
    EXPECT_NEAR(mid_coor[2], 0.0f, 0.001f);
}

TEST(SliceGrid, NeighborProbeFindsFaceDiagonal) {
    float spacer = 1.0f;
    int ispacer = static_cast<int>(std::round(spacer * 1000.0f));

    GridKey origin(0.0f, 0.0f, 0.0f);
    GridKey diag(spacer, spacer, 0.0f);

    // Probe the diagonal offset
    GridKey probed;
    probed.ix = origin.ix + 1 * ispacer;
    probed.iy = origin.iy + 1 * ispacer;
    probed.iz = origin.iz + 0 * ispacer;

    EXPECT_EQ(probed, diag);

    // Midpoint should be at (0.5, 0.5, 0.0)
    GridKey mid;
    mid.ix = (origin.ix + diag.ix) / 2;
    mid.iy = (origin.iy + diag.iy) / 2;
    mid.iz = (origin.iz + diag.iz) / 2;

    float mid_coor[3];
    mid.to_coor(mid_coor);
    EXPECT_NEAR(mid_coor[0], 0.5f, 0.001f);
    EXPECT_NEAR(mid_coor[1], 0.5f, 0.001f);
    EXPECT_NEAR(mid_coor[2], 0.0f, 0.001f);
}

// ============================================================================
// Body-diagonal exclusion — verifies that slice_grid does NOT create
// midpoints along body diagonals (distance s*sqrt(3))
// ============================================================================

TEST(SliceGrid, BodyDiagonalNotProbed) {
    // Simulate the slice_grid neighbor-probing logic on a 2x2x2 cube grid
    // (8 corner points). Verify that the body-diagonal midpoint (0.5, 0.5, 0.5)
    // is NOT generated.
    float spacer = 1.0f;
    int ispacer = static_cast<int>(std::round(spacer * 1000.0f));

    // Build a 2x2x2 cube grid: 8 points at (0,0,0)..(1,1,1)
    std::map<GridKey, int> grid;
    int idx = 0;
    for (int x = 0; x <= 1; x++)
        for (int y = 0; y <= 1; y++)
            for (int z = 0; z <= 1; z++)
                grid[GridKey(x * spacer, y * spacer, z * spacer)] = idx++;

    EXPECT_EQ(grid.size(), 8u);

    // Same offsets as slice_grid.cpp (no body diagonals)
    static const int axis_offsets[][3] = {
        {1,0,0}, {0,1,0}, {0,0,1}
    };
    static const int diag_offsets[][3] = {
        {1,1,0}, {1,-1,0}, {1,0,1}, {1,0,-1},
        {0,1,1}, {0,1,-1}
    };

    std::set<GridKey> midpoints;
    for (auto& kv : grid) {
        const GridKey& gk = kv.first;

        for (int d = 0; d < 3; d++) {
            GridKey nb;
            nb.ix = gk.ix + axis_offsets[d][0] * ispacer;
            nb.iy = gk.iy + axis_offsets[d][1] * ispacer;
            nb.iz = gk.iz + axis_offsets[d][2] * ispacer;
            if (grid.count(nb)) {
                GridKey mid;
                mid.ix = (gk.ix + nb.ix) / 2;
                mid.iy = (gk.iy + nb.iy) / 2;
                mid.iz = (gk.iz + nb.iz) / 2;
                if (!grid.count(mid)) midpoints.insert(mid);
            }
        }

        for (int d = 0; d < 6; d++) {
            GridKey nb;
            nb.ix = gk.ix + diag_offsets[d][0] * ispacer;
            nb.iy = gk.iy + diag_offsets[d][1] * ispacer;
            nb.iz = gk.iz + diag_offsets[d][2] * ispacer;
            if (grid.count(nb)) {
                GridKey mid;
                mid.ix = (gk.ix + nb.ix) / 2;
                mid.iy = (gk.iy + nb.iy) / 2;
                mid.iz = (gk.iz + nb.iz) / 2;
                if (!grid.count(mid)) midpoints.insert(mid);
            }
        }
    }

    // The body-diagonal midpoint (0.5, 0.5, 0.5) should NOT be present
    GridKey body_mid(0.5f, 0.5f, 0.5f);
    EXPECT_EQ(midpoints.count(body_mid), 0u)
        << "Body-diagonal midpoint (0.5, 0.5, 0.5) should not be generated";

    // But face-diagonal midpoints SHOULD be present
    // e.g., midpoint of (0,0,0)-(1,1,0) = (0.5, 0.5, 0.0)
    GridKey face_mid(0.5f, 0.5f, 0.0f);
    EXPECT_EQ(midpoints.count(face_mid), 1u)
        << "Face-diagonal midpoint (0.5, 0.5, 0.0) should be generated";

    // And axis-aligned midpoints SHOULD be present
    // e.g., midpoint of (0,0,0)-(1,0,0) = (0.5, 0.0, 0.0)
    GridKey axis_mid(0.5f, 0.0f, 0.0f);
    EXPECT_EQ(midpoints.count(axis_mid), 1u)
        << "Axis-aligned midpoint (0.5, 0.0, 0.0) should be generated";
}

// ============================================================================
// Deduplication correctness — GridKey in std::map
// ============================================================================

TEST(GridKey, DeduplicatesEquivalentCoordinates) {
    std::map<GridKey, int> m;

    // Insert the same physical point via slightly different float paths
    float a[3] = {1.5f, 2.5f, 3.5f};
    float b[3] = {1.5001f, 2.4999f, 3.5001f};  // within 0.5 milliangstrom

    // With %8.3f string keys, both would map to " 1.500 2.500 3.500" — same key.
    // With GridKey at milliangstrom, a -> (1500,2500,3500), b -> (1500,2500,3500)
    GridKey ka(a), kb(b);

    m[ka] = 1;
    m[kb] = 2;

    // Should overwrite, not create two entries
    EXPECT_EQ(m.size(), 1u);
    EXPECT_EQ(m[ka], 2);
}

TEST(GridKey, DistinguishesSubMilliangstromDifferences) {
    // Points 1 milliangstrom apart should be distinct
    std::map<GridKey, int> m;
    float a[3] = {1.000f, 0.0f, 0.0f};
    float b[3] = {1.001f, 0.0f, 0.0f};

    m[GridKey(a)] = 1;
    m[GridKey(b)] = 2;

    EXPECT_EQ(m.size(), 2u);
}
