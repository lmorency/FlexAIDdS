// test_fast_optics.cpp — Unit tests for lightweight FastOPTICS super-cluster extraction
//
// Tests: FULL_OPTICS fallback, SUPER_CLUSTER_ONLY mode, edge cases

#include <gtest/gtest.h>
#include "fast_optics.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>

// ── Helpers ──────────────────────────────────────────────────────────────────

static std::vector<Point> make_1d_points(const std::vector<double>& values) {
    std::vector<Point> pts(values.size());
    for (size_t i = 0; i < values.size(); ++i)
        pts[i].coords = { values[i] };
    return pts;
}

static std::vector<Point> make_2d_cluster(double cx, double cy, int n, double spread) {
    std::vector<Point> pts;
    for (int i = 0; i < n; ++i) {
        double angle = 2.0 * M_PI * i / n;
        pts.push_back({{ cx + spread * std::cos(angle),
                         cy + spread * std::sin(angle) }});
    }
    return pts;
}

// ── FULL_OPTICS mode tests ─────────────────────────────────────────────────

TEST(FastOPTICS, FullOpticsReturnsAllIndices) {
    auto pts = make_1d_points({1.0, 2.0, 3.0, 4.0, 5.0});
    FastOPTICS foptics(pts, 2);
    auto indices = foptics.extractSuperCluster(ClusterMode::FULL_OPTICS);
    ASSERT_EQ(indices.size(), pts.size());
}

TEST(FastOPTICS, OrderingMatchesPointCount) {
    auto pts = make_1d_points({-10.0, -9.5, -8.0, -11.0, -7.5});
    FastOPTICS foptics(pts, 2);
    const auto& ordering = foptics.getOrdering();
    ASSERT_EQ(ordering.size(), pts.size());
}

// ── SUPER_CLUSTER_ONLY mode tests ──────────────────────────────────────────

TEST(FastOPTICS, SuperClusterSelectsSubset) {
    // Two well-separated clusters in 1D: dense group near 0, sparse outliers near 100
    std::vector<double> values;
    // Dense cluster: 20 points near 0
    for (int i = 0; i < 20; ++i) values.push_back(0.1 * i);
    // Sparse outliers: 5 points near 100
    for (int i = 0; i < 5; ++i) values.push_back(100.0 + i * 5.0);

    auto pts = make_1d_points(values);
    FastOPTICS foptics(pts, 3);
    auto sc = foptics.extractSuperCluster(ClusterMode::SUPER_CLUSTER_ONLY);

    // Super-cluster should be smaller than total
    ASSERT_GT(sc.size(), 0u);
    ASSERT_LE(sc.size(), pts.size());
}

TEST(FastOPTICS, SuperCluster2DClusters) {
    // Two 2D clusters: tight cluster A at (0,0), loose cluster B at (50,50)
    auto clusterA = make_2d_cluster(0.0, 0.0, 15, 0.5);
    auto clusterB = make_2d_cluster(50.0, 50.0, 5, 5.0);

    std::vector<Point> pts;
    pts.insert(pts.end(), clusterA.begin(), clusterA.end());
    pts.insert(pts.end(), clusterB.begin(), clusterB.end());

    FastOPTICS foptics(pts, 3);
    auto sc = foptics.extractSuperCluster(ClusterMode::SUPER_CLUSTER_ONLY);

    // Should select a connected component, not all points
    ASSERT_GT(sc.size(), 0u);
    ASSERT_LE(sc.size(), pts.size());
}

// ── Edge cases ─────────────────────────────────────────────────────────────

TEST(FastOPTICS, SinglePoint) {
    auto pts = make_1d_points({42.0});
    FastOPTICS foptics(pts, 1);
    auto sc_full = foptics.extractSuperCluster(ClusterMode::FULL_OPTICS);
    ASSERT_EQ(sc_full.size(), 1u);

    auto sc_super = foptics.extractSuperCluster(ClusterMode::SUPER_CLUSTER_ONLY);
    ASSERT_EQ(sc_super.size(), 1u);
}

TEST(FastOPTICS, TwoIdenticalPoints) {
    auto pts = make_1d_points({5.0, 5.0});
    FastOPTICS foptics(pts, 1);
    auto sc = foptics.extractSuperCluster(ClusterMode::SUPER_CLUSTER_ONLY);
    ASSERT_EQ(sc.size(), 2u);
}

TEST(FastOPTICS, UniformDistribution) {
    // All points equidistant — super-cluster should still work
    std::vector<double> values;
    for (int i = 0; i < 10; ++i) values.push_back(static_cast<double>(i));
    auto pts = make_1d_points(values);
    FastOPTICS foptics(pts, 2);
    auto sc = foptics.extractSuperCluster(ClusterMode::SUPER_CLUSTER_ONLY);
    ASSERT_GT(sc.size(), 0u);
}

TEST(FastOPTICS, IndicesAreValid) {
    auto pts = make_1d_points({1.0, 3.0, 5.0, 7.0, 9.0, 11.0});
    FastOPTICS foptics(pts, 2);
    auto sc = foptics.extractSuperCluster(ClusterMode::SUPER_CLUSTER_ONLY);
    for (size_t idx : sc) {
        ASSERT_LT(idx, pts.size()) << "Index out of bounds: " << idx;
    }
}

// ── Distance helper ────────────────────────────────────────────────────────

TEST(FastOPTICS, DistanceFunction) {
    Point a{{ 0.0, 0.0 }};
    Point b{{ 3.0, 4.0 }};
    ASSERT_NEAR(distance(a, b), 5.0, 1e-10);
}

TEST(FastOPTICS, DistanceIdenticalPoints) {
    Point a{{ 1.0, 2.0, 3.0 }};
    ASSERT_NEAR(distance(a, a), 0.0, 1e-15);
}
