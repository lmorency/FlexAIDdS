// tests/test_ga_validation.cpp — Voronoi geometry & GA validation tests
// Tests: solve_3x3, solve_2xS, cosPQR, spherical_arc, add_vertex, VoronoiCFBatch workspace
// Apache-2.0 © 2026 Le Bonhomme Pharma

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <cstring>

#include "flexaid.h"
#include "Vcontacts.h"

namespace {

static constexpr double EPSILON = 1e-6;

// ===========================================================================
// solve_3x3: solves Ax = b via Cramer's rule (4-element rows: A0,A1,A2,D)
// ===========================================================================

class Solve3x3Test : public ::testing::Test {};

// Identity-like system: x=1, y=2, z=3
TEST_F(Solve3x3Test, IdentitySystem) {
    // Row format: [A, B, C, D] for Ax + By + Cz + D = 0
    // x = 1  =>  1*x + 0*y + 0*z - 1 = 0
    double eq0[4] = {1.0, 0.0, 0.0, -1.0};
    double eq1[4] = {0.0, 1.0, 0.0, -2.0};
    double eq2[4] = {0.0, 0.0, 1.0, -3.0};
    double pt[3];

    int result = solve_3x3(eq0, eq1, eq2, pt);
    EXPECT_EQ(result, 0);
    EXPECT_NEAR(pt[0], 1.0, EPSILON);
    EXPECT_NEAR(pt[1], 2.0, EPSILON);
    EXPECT_NEAR(pt[2], 3.0, EPSILON);
}

// Coupled system: 2x + y = 5, x + 3y = 10, z = 4
TEST_F(Solve3x3Test, CoupledSystem) {
    double eq0[4] = {2.0, 1.0, 0.0, -5.0};
    double eq1[4] = {1.0, 3.0, 0.0, -10.0};
    double eq2[4] = {0.0, 0.0, 1.0, -4.0};
    double pt[3];

    int result = solve_3x3(eq0, eq1, eq2, pt);
    EXPECT_EQ(result, 0);
    // 2x + y = 5, x + 3y = 10 => x = 1, y = 3
    EXPECT_NEAR(pt[0], 1.0, EPSILON);
    EXPECT_NEAR(pt[1], 3.0, EPSILON);
    EXPECT_NEAR(pt[2], 4.0, EPSILON);
}

// Singular system (parallel planes) returns -1
TEST_F(Solve3x3Test, SingularSystemReturnsNegOne) {
    double eq0[4] = {1.0, 2.0, 3.0, -1.0};
    double eq1[4] = {2.0, 4.0, 6.0, -2.0};  // 2x eq0
    double eq2[4] = {0.0, 0.0, 1.0, -1.0};
    double pt[3];

    int result = solve_3x3(eq0, eq1, eq2, pt);
    EXPECT_EQ(result, -1);
}

// ===========================================================================
// solve_2xS: intersection of two planes with a sphere
// ===========================================================================

class Solve2xSTest : public ::testing::Test {};

// Two orthogonal planes through origin, sphere of radius 1
TEST_F(Solve2xSTest, OrthogonalPlanesUnitSphere) {
    plane p0{}, p1{};
    // Plane x = 0
    p0.Ai[0] = 1.0; p0.Ai[1] = 0.0; p0.Ai[2] = 0.0; p0.Ai[3] = 0.0;
    // Plane y = 0
    p1.Ai[0] = 0.0; p1.Ai[1] = 1.0; p1.Ai[2] = 0.0; p1.Ai[3] = 0.0;

    double pt0[3], pt1[3];
    int result = solve_2xS(&p0, &p1, 1.0f, pt0, pt1);
    EXPECT_EQ(result, 0);

    // Intersection line is the z-axis; sphere intersections at (0,0,±1)
    EXPECT_NEAR(pt0[0], 0.0, EPSILON);
    EXPECT_NEAR(pt0[1], 0.0, EPSILON);
    EXPECT_NEAR(std::abs(pt0[2]), 1.0, EPSILON);

    EXPECT_NEAR(pt1[0], 0.0, EPSILON);
    EXPECT_NEAR(pt1[1], 0.0, EPSILON);
    EXPECT_NEAR(std::abs(pt1[2]), 1.0, EPSILON);

    // The two solutions should be distinct (opposite z)
    EXPECT_NEAR(pt0[2] + pt1[2], 0.0, EPSILON);
}

// Parallel planes have no intersection => returns -1
TEST_F(Solve2xSTest, ParallelPlanesReturnsNegOne) {
    plane p0{}, p1{};
    p0.Ai[0] = 1.0; p0.Ai[1] = 0.0; p0.Ai[2] = 0.0; p0.Ai[3] = -1.0;
    p1.Ai[0] = 1.0; p1.Ai[1] = 0.0; p1.Ai[2] = 0.0; p1.Ai[3] = -2.0;

    double pt0[3], pt1[3];
    int result = solve_2xS(&p0, &p1, 1.0f, pt0, pt1);
    EXPECT_EQ(result, -1);
}

// Planes far from origin, small sphere => no intersection
TEST_F(Solve2xSTest, NoIntersectionWithSmallSphere) {
    plane p0{}, p1{};
    // Plane x = 10
    p0.Ai[0] = 1.0; p0.Ai[1] = 0.0; p0.Ai[2] = 0.0; p0.Ai[3] = -10.0;
    // Plane y = 10
    p1.Ai[0] = 0.0; p1.Ai[1] = 1.0; p1.Ai[2] = 0.0; p1.Ai[3] = -10.0;

    double pt0[3], pt1[3];
    int result = solve_2xS(&p0, &p1, 1.0f, pt0, pt1);
    EXPECT_EQ(result, -1);
}

// ===========================================================================
// cosPQR: cosine of angle PQR (Q is center point)
// ===========================================================================

class CosPQRTest : public ::testing::Test {};

// Right angle: P=(1,0,0), Q=(0,0,0), R=(0,1,0) => cos(90°) = 0
TEST_F(CosPQRTest, RightAngle) {
    double P[3] = {1.0, 0.0, 0.0};
    double Q[3] = {0.0, 0.0, 0.0};
    double R[3] = {0.0, 1.0, 0.0};

    double cos_val = cosPQR(P, Q, R);
    EXPECT_NEAR(cos_val, 0.0, EPSILON);
}

// Collinear same direction: QP=(1,0,0) QR=(2,0,0) => cos(0°) = 1
TEST_F(CosPQRTest, CollinearSameDirection) {
    double P[3] = {2.0, 0.0, 0.0};
    double Q[3] = {1.0, 0.0, 0.0};
    double R[3] = {3.0, 0.0, 0.0};

    double cos_val = cosPQR(P, Q, R);
    EXPECT_NEAR(cos_val, 1.0, EPSILON);  // QP and QR point in the same direction
}

// 60-degree angle
TEST_F(CosPQRTest, SixtyDegreeAngle) {
    double P[3] = {1.0, 0.0, 0.0};
    double Q[3] = {0.0, 0.0, 0.0};
    double R[3] = {0.5, std::sqrt(3.0) / 2.0, 0.0};

    double cos_val = cosPQR(P, Q, R);
    EXPECT_NEAR(cos_val, 0.5, EPSILON);  // cos(60°) = 0.5
}

// 3D angle
TEST_F(CosPQRTest, ThreeDimensionalAngle) {
    double P[3] = {1.0, 0.0, 0.0};
    double Q[3] = {0.0, 0.0, 0.0};
    double R[3] = {0.0, 0.0, 1.0};

    double cos_val = cosPQR(P, Q, R);
    EXPECT_NEAR(cos_val, 0.0, EPSILON);  // 90° in 3D
}

// ===========================================================================
// add_vertex: populates a vertex struct from coordinates
// ===========================================================================

class AddVertexTest : public ::testing::Test {};

TEST_F(AddVertexTest, StoresCoordinatesAndPlanes) {
    vertex poly[2];
    double coor[3] = {3.0, 4.0, 0.0};

    int result = add_vertex(poly, 0, coor, 1, 2, 3);
    EXPECT_EQ(result, 0);
    EXPECT_DOUBLE_EQ(poly[0].xi[0], 3.0);
    EXPECT_DOUBLE_EQ(poly[0].xi[1], 4.0);
    EXPECT_DOUBLE_EQ(poly[0].xi[2], 0.0);
    EXPECT_EQ(poly[0].plane[0], 1);
    EXPECT_EQ(poly[0].plane[1], 2);
    EXPECT_EQ(poly[0].plane[2], 3);
    EXPECT_NEAR(poly[0].dist, 5.0, EPSILON);  // sqrt(9+16) = 5
}

TEST_F(AddVertexTest, ComputesDistFromOrigin) {
    vertex poly[2];
    double coor[3] = {1.0, 1.0, 1.0};

    add_vertex(poly, 0, coor, 0, 0, 0);
    EXPECT_NEAR(poly[0].dist, std::sqrt(3.0), EPSILON);
}

TEST_F(AddVertexTest, OriginVertexHasZeroDist) {
    vertex poly[2];
    double coor[3] = {0.0, 0.0, 0.0};

    add_vertex(poly, 0, coor, 0, 0, 0);
    EXPECT_DOUBLE_EQ(poly[0].dist, 0.0);
}

// ===========================================================================
// solve_3x3 determinism: same input always gives same output
// ===========================================================================

TEST(VoronoiDeterminismTest, Solve3x3IsDeterministic) {
    double eq0[4] = {1.0, 2.0, 1.0, -7.0};
    double eq1[4] = {3.0, 1.0, 2.0, -11.0};
    double eq2[4] = {2.0, 3.0, 1.0, -10.0};

    double pt_first[3], pt_second[3];

    solve_3x3(eq0, eq1, eq2, pt_first);

    for (int i = 0; i < 5; ++i) {
        solve_3x3(eq0, eq1, eq2, pt_second);
        EXPECT_EQ(pt_first[0], pt_second[0]);
        EXPECT_EQ(pt_first[1], pt_second[1]);
        EXPECT_EQ(pt_first[2], pt_second[2]);
    }

    gene genes[3];
    genes[0].to_ic = 45.0;
    genes[1].to_ic = -90.0;
    genes[2].to_ic = 0.0;

    atom a;
    std::memset(&a, 0, sizeof(atom));
    resid r;
    std::memset(&r, 0, sizeof(resid));

    cfstr result = voronoi_cf::eval_span(
        &fa, &gb, &vc,
        std::span<const genlim>(gl, 3),
        std::span<atom>(&a, 1),
        std::span<resid>(&r, 1),
        nullptr,
        std::span<const gene>(genes, 3),
        test_sum_function
    );

    // Values within bounds: 45 + (-90) + 0 = -45.0
    EXPECT_DOUBLE_EQ(result.com, -45.0);
}

// ═══════════════════════════════════════════════════════════════════════
// Scoring ordering invariant: more negative com → better score
// ═══════════════════════════════════════════════════════════════════════

TEST(ScoringInvariant, MoreNegativeComIsBetter) {
    cfstr good, bad;
    std::memset(&good, 0, sizeof(cfstr));
    std::memset(&bad, 0, sizeof(cfstr));

    good.com = -20.0;  // strong complementarity
    bad.com  = -5.0;   // weak complementarity

    EXPECT_LT(get_apparent_cf_evalue(&good), get_apparent_cf_evalue(&bad))
        << "More negative com should give lower (better) apparent evalue";
    EXPECT_LT(get_cf_evalue(&good), get_cf_evalue(&bad))
        << "More negative com should give lower (better) full evalue";
}

TEST(ScoringInvariant, WallPenaltyWorsensScore) {
    cfstr no_clash, clash;
    std::memset(&no_clash, 0, sizeof(cfstr));
    std::memset(&clash, 0, sizeof(cfstr));

    no_clash.com = -10.0;
    no_clash.wal = 0.0;

    clash.com = -10.0;
    clash.wal = 5.0;  // steric clash penalty

    EXPECT_LT(get_apparent_cf_evalue(&no_clash), get_apparent_cf_evalue(&clash))
        << "Wall penalty should worsen the score";
}

// ===========================================================================
// solve_2xS results lie on the sphere
// ===========================================================================

TEST(VoronoiGeometryTest, Solve2xSResultsLieOnSphere) {
    plane p0{}, p1{};
    // Plane x + y = 0
    p0.Ai[0] = 1.0; p0.Ai[1] = 1.0; p0.Ai[2] = 0.0; p0.Ai[3] = 0.0;
    // Plane z = 0
    p1.Ai[0] = 0.0; p1.Ai[1] = 0.0; p1.Ai[2] = 1.0; p1.Ai[3] = 0.0;

    const float radius = 2.5f;
    double pt0[3], pt1[3];
    int result = solve_2xS(&p0, &p1, radius, pt0, pt1);
    ASSERT_EQ(result, 0);

    // Both points should lie on the sphere: |pt| = radius
    double dist0 = std::sqrt(pt0[0]*pt0[0] + pt0[1]*pt0[1] + pt0[2]*pt0[2]);
    double dist1 = std::sqrt(pt1[0]*pt1[0] + pt1[1]*pt1[1] + pt1[2]*pt1[2]);
    EXPECT_NEAR(dist0, static_cast<double>(radius), EPSILON);
    EXPECT_NEAR(dist1, static_cast<double>(radius), EPSILON);

    // Both points should satisfy plane equations
    EXPECT_NEAR(p0.Ai[0]*pt0[0] + p0.Ai[1]*pt0[1] + p0.Ai[2]*pt0[2] + p0.Ai[3], 0.0, EPSILON);
    EXPECT_NEAR(p1.Ai[0]*pt0[0] + p1.Ai[1]*pt0[1] + p1.Ai[2]*pt0[2] + p1.Ai[3], 0.0, EPSILON);
}

}  // namespace anonymous
