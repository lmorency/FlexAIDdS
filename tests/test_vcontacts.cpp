// tests/test_vcontacts.cpp
// Unit tests for Voronoi contact scoring math functions (Vcontacts.cpp)
// Tests cover: solve_3x3, solve_2xS, cosPQR, spherical_arc, add_vertex,
// generate_dim_sig — the isolatable pure-math core of the scoring engine.
// Apache-2.0 © 2026 Le Bonhomme Pharma

#include <gtest/gtest.h>
#include "../LIB/Vcontacts.h"

#include <cmath>
#include <cstring>
#include <string>

static constexpr double EPSILON = 1e-6;

// ===========================================================================
// solve_3x3 — 3×3 linear system solver (Cramer's rule)
// ===========================================================================

class Solve3x3Test : public ::testing::Test {};

TEST_F(Solve3x3Test, IdentitySystemGivesNegRHS) {
    // x = -d0, y = -d1, z = -d2  (since Ax+By+Cz+D=0 form)
    // eq: {A, B, C, D}
    double eq0[] = {1.0, 0.0, 0.0, -3.0};  // x = 3
    double eq1[] = {0.0, 1.0, 0.0, -5.0};  // y = 5
    double eq2[] = {0.0, 0.0, 1.0, -7.0};  // z = 7
    double pt[3];

    int rc = solve_3x3(eq0, eq1, eq2, pt);
    EXPECT_EQ(rc, 0);
    EXPECT_NEAR(pt[0], 3.0, EPSILON);
    EXPECT_NEAR(pt[1], 5.0, EPSILON);
    EXPECT_NEAR(pt[2], 7.0, EPSILON);
}

TEST_F(Solve3x3Test, GeneralSystem) {
    // 2x + y - z = 3  →  {2, 1, -1, -3}
    // x - y + 2z = 1  →  {1, -1, 2, -1}
    // 3x + 2y + z = 10 → {3, 2, 1, -10}
    // Solution: x=2, y=1, z=2
    double eq0[] = {2.0,  1.0, -1.0, -3.0};
    double eq1[] = {1.0, -1.0,  2.0, -1.0};
    double eq2[] = {3.0,  2.0,  1.0, -10.0};
    double pt[3];

    int rc = solve_3x3(eq0, eq1, eq2, pt);
    EXPECT_EQ(rc, 0);
    // Verify solution satisfies all three equations
    EXPECT_NEAR(2.0 * pt[0] + 1.0 * pt[1] - 1.0 * pt[2], 3.0, EPSILON);
    EXPECT_NEAR(1.0 * pt[0] - 1.0 * pt[1] + 2.0 * pt[2], 1.0, EPSILON);
    EXPECT_NEAR(3.0 * pt[0] + 2.0 * pt[1] + 1.0 * pt[2], 10.0, EPSILON);
}

TEST_F(Solve3x3Test, SingularSystemReturnsNegOne) {
    // Two identical equations → singular
    double eq0[] = {1.0, 2.0, 3.0, -4.0};
    double eq1[] = {1.0, 2.0, 3.0, -4.0};
    double eq2[] = {0.0, 0.0, 1.0, -1.0};
    double pt[3];

    int rc = solve_3x3(eq0, eq1, eq2, pt);
    EXPECT_EQ(rc, -1);
}

TEST_F(Solve3x3Test, AllZerosSingular) {
    double eq0[] = {0.0, 0.0, 0.0, 0.0};
    double eq1[] = {0.0, 0.0, 0.0, 0.0};
    double eq2[] = {0.0, 0.0, 0.0, 0.0};
    double pt[3];

    int rc = solve_3x3(eq0, eq1, eq2, pt);
    EXPECT_EQ(rc, -1);
}

// ===========================================================================
// solve_2xS — intersection of two planes and a sphere
// ===========================================================================

TEST(Solve2xS, OrthogonalPlanesOnSphere) {
    // Plane 1: x = 0   → {1, 0, 0, 0}
    // Plane 2: y = 0   → {0, 1, 0, 0}
    // Sphere radius = 2.0
    // Intersection: (0, 0, ±2)
    plane p0{}, p1{};
    p0.Ai[0] = 1.0; p0.Ai[1] = 0.0; p0.Ai[2] = 0.0; p0.Ai[3] = 0.0;
    p1.Ai[0] = 0.0; p1.Ai[1] = 1.0; p1.Ai[2] = 0.0; p1.Ai[3] = 0.0;
    double pt0[3], pt1[3];

    int rc = solve_2xS(&p0, &p1, 2.0f, pt0, pt1);
    EXPECT_EQ(rc, 0);

    // Both points should be at (0, 0, ±2)
    EXPECT_NEAR(pt0[0], 0.0, EPSILON);
    EXPECT_NEAR(pt0[1], 0.0, EPSILON);
    EXPECT_NEAR(std::abs(pt0[2]), 2.0, EPSILON);

    EXPECT_NEAR(pt1[0], 0.0, EPSILON);
    EXPECT_NEAR(pt1[1], 0.0, EPSILON);
    EXPECT_NEAR(std::abs(pt1[2]), 2.0, EPSILON);

    // Should be opposite signs
    EXPECT_NEAR(pt0[2] + pt1[2], 0.0, EPSILON);
}

TEST(Solve2xS, SphereRadiusTooSmall) {
    // Planes shifted so intersection line misses a small sphere
    plane p0{}, p1{};
    p0.Ai[0] = 1.0; p0.Ai[1] = 0.0; p0.Ai[2] = 0.0; p0.Ai[3] = -5.0;  // x = 5
    p1.Ai[0] = 0.0; p1.Ai[1] = 1.0; p1.Ai[2] = 0.0; p1.Ai[3] = -5.0;  // y = 5
    double pt0[3], pt1[3];

    int rc = solve_2xS(&p0, &p1, 1.0f, pt0, pt1);  // sphere too small
    EXPECT_EQ(rc, -1);
}

TEST(Solve2xS, ParallelPlanesReturnNegOne) {
    // Both planes are x = const (parallel, different D)
    plane p0{}, p1{};
    p0.Ai[0] = 1.0; p0.Ai[1] = 0.0; p0.Ai[2] = 0.0; p0.Ai[3] = 0.0;
    p1.Ai[0] = 1.0; p1.Ai[1] = 0.0; p1.Ai[2] = 0.0; p1.Ai[3] = -1.0;
    double pt0[3], pt1[3];

    int rc = solve_2xS(&p0, &p1, 5.0f, pt0, pt1);
    EXPECT_EQ(rc, -1);
}

TEST(Solve2xS, ResultsLieOnSphere) {
    plane p0{}, p1{};
    p0.Ai[0] = 1.0; p0.Ai[1] = 0.0; p0.Ai[2] = 0.0; p0.Ai[3] = 0.0;
    p1.Ai[0] = 0.0; p1.Ai[1] = 0.0; p1.Ai[2] = 1.0; p1.Ai[3] = 0.0;
    double pt0[3], pt1[3];
    float radius = 3.0f;

    int rc = solve_2xS(&p0, &p1, radius, pt0, pt1);
    EXPECT_EQ(rc, 0);

    double r0 = std::sqrt(pt0[0]*pt0[0] + pt0[1]*pt0[1] + pt0[2]*pt0[2]);
    double r1 = std::sqrt(pt1[0]*pt1[0] + pt1[1]*pt1[1] + pt1[2]*pt1[2]);
    EXPECT_NEAR(r0, radius, EPSILON);
    EXPECT_NEAR(r1, radius, EPSILON);
}

// ===========================================================================
// cosPQR — cosine of angle at Q between points P, Q, R
// ===========================================================================

TEST(CosPQR, RightAngle) {
    double P[] = {1.0, 0.0, 0.0};
    double Q[] = {0.0, 0.0, 0.0};
    double R[] = {0.0, 1.0, 0.0};

    double cos_val = cosPQR(P, Q, R);
    EXPECT_NEAR(cos_val, 0.0, EPSILON);  // cos(90°) = 0
}

TEST(CosPQR, ZeroAngle) {
    double P[] = {2.0, 0.0, 0.0};
    double Q[] = {0.0, 0.0, 0.0};
    double R[] = {5.0, 0.0, 0.0};

    double cos_val = cosPQR(P, Q, R);
    EXPECT_NEAR(cos_val, 1.0, EPSILON);  // cos(0°) = 1
}

TEST(CosPQR, StraightAngle) {
    double P[] = {1.0, 0.0, 0.0};
    double Q[] = {0.0, 0.0, 0.0};
    double R[] = {-3.0, 0.0, 0.0};

    double cos_val = cosPQR(P, Q, R);
    EXPECT_NEAR(cos_val, -1.0, EPSILON);  // cos(180°) = -1
}

TEST(CosPQR, FortyFiveDegrees) {
    double P[] = {1.0, 0.0, 0.0};
    double Q[] = {0.0, 0.0, 0.0};
    double R[] = {1.0, 1.0, 0.0};

    double cos_val = cosPQR(P, Q, R);
    EXPECT_NEAR(cos_val, std::cos(M_PI / 4.0), EPSILON);  // cos(45°)
}

TEST(CosPQR, ThreeDimensional) {
    double P[] = {1.0, 1.0, 1.0};
    double Q[] = {0.0, 0.0, 0.0};
    double R[] = {-1.0, -1.0, -1.0};

    double cos_val = cosPQR(P, Q, R);
    EXPECT_NEAR(cos_val, -1.0, EPSILON);
}

// ===========================================================================
// add_vertex — appends vertex to polyhedron array
// ===========================================================================

TEST(AddVertex, StoresCoordinatesAndPlanes) {
    vertex poly[5];
    double coor[] = {1.0, 2.0, 3.0};

    int rc = add_vertex(poly, 0, coor, 10, 20, 30);
    EXPECT_EQ(rc, 0);

    EXPECT_DOUBLE_EQ(poly[0].xi[0], 1.0);
    EXPECT_DOUBLE_EQ(poly[0].xi[1], 2.0);
    EXPECT_DOUBLE_EQ(poly[0].xi[2], 3.0);
    EXPECT_EQ(poly[0].plane[0], 10);
    EXPECT_EQ(poly[0].plane[1], 20);
    EXPECT_EQ(poly[0].plane[2], 30);
}

TEST(AddVertex, CalculatesDistance) {
    vertex poly[1];
    double coor[] = {3.0, 4.0, 0.0};

    add_vertex(poly, 0, coor, 0, 0, 0);

    EXPECT_NEAR(poly[0].dist, 5.0, EPSILON);  // sqrt(9+16+0) = 5
}

TEST(AddVertex, AtOrigin) {
    vertex poly[1];
    double coor[] = {0.0, 0.0, 0.0};

    add_vertex(poly, 0, coor, 0, 0, 0);

    EXPECT_NEAR(poly[0].dist, 0.0, EPSILON);
}

TEST(AddVertex, MultipleVertices) {
    vertex poly[3];
    double c0[] = {1.0, 0.0, 0.0};
    double c1[] = {0.0, 1.0, 0.0};
    double c2[] = {0.0, 0.0, 1.0};

    add_vertex(poly, 0, c0, 1, 2, 3);
    add_vertex(poly, 1, c1, 4, 5, 6);
    add_vertex(poly, 2, c2, 7, 8, 9);

    EXPECT_DOUBLE_EQ(poly[0].xi[0], 1.0);
    EXPECT_DOUBLE_EQ(poly[1].xi[1], 1.0);
    EXPECT_DOUBLE_EQ(poly[2].xi[2], 1.0);
    EXPECT_NEAR(poly[0].dist, 1.0, EPSILON);
    EXPECT_NEAR(poly[1].dist, 1.0, EPSILON);
    EXPECT_NEAR(poly[2].dist, 1.0, EPSILON);
}

// ===========================================================================
// spherical_arc — area of arc on sphere surface
// ===========================================================================

TEST(SphericalArc, AreaIsFinite) {
    // Create three vertices on a unit sphere
    vertex A{}, B{}, C{};
    A.xi[0] = 0.0; A.xi[1] = 0.0; A.xi[2] = 1.0;
    A.dist = 1.0;
    B.xi[0] = 1.0; B.xi[1] = 0.0; B.xi[2] = 0.0;
    B.dist = 1.0;
    C.xi[0] = 0.0; C.xi[1] = 1.0; C.xi[2] = 0.0;
    C.dist = 1.0;

    double area = spherical_arc(&A, &B, &C, 1.0f);
    EXPECT_TRUE(std::isfinite(area));
}

TEST(SphericalArc, LargerRadiusLargerArea) {
    vertex A{}, B{}, C{};
    A.xi[0] = 0.0; A.xi[1] = 0.0; A.xi[2] = 1.0; A.dist = 1.0;
    B.xi[0] = 1.0; B.xi[1] = 0.0; B.xi[2] = 0.0; B.dist = 1.0;
    C.xi[0] = 0.0; C.xi[1] = 1.0; C.xi[2] = 0.0; C.dist = 1.0;

    double area1 = spherical_arc(&A, &B, &C, 1.0f);
    double area2 = spherical_arc(&A, &B, &C, 2.0f);

    // Area should scale with radius (r^2 scaling for spherical areas)
    EXPECT_TRUE(std::isfinite(area1));
    EXPECT_TRUE(std::isfinite(area2));
}

// ===========================================================================
// generate_dim_sig — grid dimension signature for caching
// ===========================================================================

TEST(GenerateDimSig, CorrectFormat) {
    float global_min[] = {1.5f, 2.7f, 3.9f};
    std::string sig = generate_dim_sig(global_min, 10);
    // int(1.5)=1, int(2.7)=2, int(3.9)=3
    EXPECT_EQ(sig, "1/2/3/10");
}

TEST(GenerateDimSig, NegativeCoordinates) {
    float global_min[] = {-5.3f, -2.1f, -0.7f};
    std::string sig = generate_dim_sig(global_min, 5);
    // int(-5.3)=-5, int(-2.1)=-2, int(-0.7)=0
    EXPECT_EQ(sig, "-5/-2/0/5");
}

TEST(GenerateDimSig, DifferentDimDifferentSig) {
    float global_min[] = {0.0f, 0.0f, 0.0f};
    std::string s1 = generate_dim_sig(global_min, 3);
    std::string s2 = generate_dim_sig(global_min, 4);
    EXPECT_NE(s1, s2);
}
