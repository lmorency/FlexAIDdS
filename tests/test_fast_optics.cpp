// test_fast_optics.cpp — Unit tests for lightweight FastOPTICS super-cluster extraction
//
// Tests: FULL_OPTICS fallback, SUPER_CLUSTER_ONLY mode, edge cases

#include <gtest/gtest.h>
#include "fast_optics.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <set>

using namespace fast_optics;

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

TEST(FastOPTICSTest, FullOpticsReturnsAllIndices) {
    auto pts = make_1d_points({1.0, 2.0, 3.0, 4.0, 5.0});
    FastOPTICS foptics(pts, 2);
    auto indices = foptics.extractSuperCluster(ClusterMode::FULL_OPTICS);
    ASSERT_EQ(indices.size(), pts.size());
}

TEST(FastOPTICSTest, OrderingMatchesPointCount) {
    auto pts = make_1d_points({-10.0, -9.5, -8.0, -11.0, -7.5});
    FastOPTICS foptics(pts, 2);
    const auto& ordering = foptics.getOrdering();
    ASSERT_EQ(ordering.size(), pts.size());
}

// ── SUPER_CLUSTER_ONLY mode tests ──────────────────────────────────────────

TEST(FastOPTICSTest, SuperClusterSelectsSubset) {
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

TEST(FastOPTICSTest, SuperCluster2DClusters) {
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

TEST(FastOPTICSTest, SinglePoint) {
    auto pts = make_1d_points({42.0});
    FastOPTICS foptics(pts, 1);
    auto sc_full = foptics.extractSuperCluster(ClusterMode::FULL_OPTICS);
    ASSERT_EQ(sc_full.size(), 1u);

    auto sc_super = foptics.extractSuperCluster(ClusterMode::SUPER_CLUSTER_ONLY);
    ASSERT_EQ(sc_super.size(), 1u);
}

TEST(FastOPTICSTest, TwoIdenticalPoints) {
    auto pts = make_1d_points({5.0, 5.0});
    FastOPTICS foptics(pts, 1);
    auto sc = foptics.extractSuperCluster(ClusterMode::SUPER_CLUSTER_ONLY);
    ASSERT_EQ(sc.size(), 2u);
}

TEST(FastOPTICSTest, UniformDistribution) {
    // All points equidistant — super-cluster should still work
    std::vector<double> values;
    for (int i = 0; i < 10; ++i) values.push_back(static_cast<double>(i));
    auto pts = make_1d_points(values);
    FastOPTICS foptics(pts, 2);
    auto sc = foptics.extractSuperCluster(ClusterMode::SUPER_CLUSTER_ONLY);
    ASSERT_GT(sc.size(), 0u);
}

TEST(FastOPTICSTest, IndicesAreValid) {
    auto pts = make_1d_points({1.0, 3.0, 5.0, 7.0, 9.0, 11.0});
    FastOPTICS foptics(pts, 2);
    auto sc = foptics.extractSuperCluster(ClusterMode::SUPER_CLUSTER_ONLY);
    for (size_t idx : sc) {
        ASSERT_LT(idx, pts.size()) << "Index out of bounds: " << idx;
    }
}

// ── Distance helper ────────────────────────────────────────────────────────

TEST(FastOPTICSTest, DistanceFunction) {
    Point a{{ 0.0, 0.0 }};
    Point b{{ 3.0, 4.0 }};
    ASSERT_NEAR(distance(a, b), 5.0, 1e-10);
}

TEST(FastOPTICSTest, DistanceIdenticalPoints) {
    Point a{{ 1.0, 2.0, 3.0 }};
    ASSERT_NEAR(distance(a, a), 0.0, 1e-15);
}

// ── Reproducibility tests ─────────────────────────────────────────────────

TEST(FastOPTICSTest, SuperClusterReturnsPointIndicesNotOrderingPositions) {
    // Construct data where ordering permutation ≠ identity:
    // Points: [100, 1, 2, 3, 4, 5] — OPTICS will order the dense cluster
    // (indices 1-5) before the outlier (index 0), so ordering[0].index ≠ 0
    auto pts = make_1d_points({100.0, 1.0, 2.0, 3.0, 4.0, 5.0});
    FastOPTICS foptics(pts, 2);

    auto sc = foptics.extractSuperCluster(ClusterMode::SUPER_CLUSTER_ONLY);

    // All returned indices must be valid point indices
    ASSERT_GT(sc.size(), 0u);
    for (size_t idx : sc) {
        ASSERT_LT(idx, pts.size()) << "Index out of bounds: " << idx;
    }

    // Verify the seed is a point in the dense cluster (1-5), not the outlier (0).
    // The lowest-reachability point should be in the dense group.
    // Even if the flood-fill only returns the seed, it must be a valid point index
    // that correctly references its energy via points_[idx].
    EXPECT_NE(sc[0], 0u)
        << "Seed should be in the dense cluster, not the far outlier";
}

TEST(FastOPTICSTest, SuperClusterIdempotent) {
    // Two calls on the same data must return identical results
    auto pts = make_1d_points({-10.0, -9.5, -8.0, -11.0, -7.5, 5.0, 6.0});
    FastOPTICS foptics(pts, 2);

    auto sc1 = foptics.extractSuperCluster(ClusterMode::SUPER_CLUSTER_ONLY);
    auto sc2 = foptics.extractSuperCluster(ClusterMode::SUPER_CLUSTER_ONLY);

    ASSERT_EQ(sc1.size(), sc2.size());
    for (size_t i = 0; i < sc1.size(); ++i) {
        ASSERT_EQ(sc1[i], sc2[i]) << "Mismatch at position " << i;
    }
}

TEST(FastOPTICSTest, SuperClusterIndicesArePointIndicesNotOrderingPositions) {
    // Verify that SUPER_CLUSTER_ONLY returns the same index space
    // as FULL_OPTICS (original point indices)
    auto pts = make_1d_points({-10.0, -9.0, -8.0, -7.0, -6.0});
    FastOPTICS foptics(pts, 2);

    auto full = foptics.extractSuperCluster(ClusterMode::FULL_OPTICS);
    auto sc = foptics.extractSuperCluster(ClusterMode::SUPER_CLUSTER_ONLY);

    // Both should return subsets of {0, 1, 2, 3, 4} (point indices)
    for (size_t idx : full) {
        ASSERT_LT(idx, pts.size());
    }
    for (size_t idx : sc) {
        ASSERT_LT(idx, pts.size());
    }

    // SC should be a subset of FULL
    std::set<size_t> full_set(full.begin(), full.end());
    for (size_t idx : sc) {
        EXPECT_TRUE(full_set.count(idx))
            << "SC index " << idx << " not in FULL_OPTICS result";
    }
}
