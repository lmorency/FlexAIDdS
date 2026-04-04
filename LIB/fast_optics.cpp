#include "fast_optics.hpp"

#include <numeric>
#include <functional>

namespace fast_optics {

FastOPTICS::FastOPTICS(const std::vector<Point>& points, int minPts, double eps)
    : minPts_(minPts), eps_(eps), points_(points)
{
    computeOrdering();
}

void FastOPTICS::computeOrdering() {
    size_t n = points_.size();
    if (n == 0) return;

    std::vector<bool> processed(n, false);
    ordering_.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        if (processed[i]) continue;

        // Priority queue: smallest reachability first
        using Entry = std::pair<double, size_t>;
        std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> seeds;

        processed[i] = true;
        ordering_.push_back({i, std::numeric_limits<double>::infinity()});

        double cd = coreDist(i);
        if (cd < std::numeric_limits<double>::infinity()) {
            // Update seeds with neighbors
            for (size_t j = 0; j < points_.size(); ++j) {
                if (processed[j]) continue;
                double d = distance(points_[i], points_[j]);
                if (eps_ <= 0.0 || d <= eps_) {
                    double newReach = std::max(cd, d);
                    seeds.push({newReach, j});
                }
            }

            while (!seeds.empty()) {
                auto [reach, idx] = seeds.top();
                seeds.pop();
                if (processed[idx]) continue;

                processed[idx] = true;
                ordering_.push_back({idx, reach});

                double cd2 = coreDist(idx);
                if (cd2 < std::numeric_limits<double>::infinity()) {
                    for (size_t j = 0; j < points_.size(); ++j) {
                        if (processed[j]) continue;
                        double d = distance(points_[idx], points_[j]);
                        if (eps_ <= 0.0 || d <= eps_) {
                            double newReach = std::max(cd2, d);
                            seeds.push({newReach, j});
                        }
                    }
                }
            }
        }
    }
}

double FastOPTICS::coreDist(size_t idx) const {
    // minPts=1: every point is its own core with distance 0
    if (minPts_ <= 1) return 0.0;

    if (static_cast<int>(points_.size()) < minPts_) {
        return std::numeric_limits<double>::infinity();
    }

    std::vector<double> dists;
    dists.reserve(points_.size());
    for (size_t j = 0; j < points_.size(); ++j) {
        if (j == idx) continue;
        double d = distance(points_[idx], points_[j]);
        if (eps_ <= 0.0 || d <= eps_) {
            dists.push_back(d);
        }
    }

    if (static_cast<int>(dists.size()) < minPts_ - 1) {
        return std::numeric_limits<double>::infinity();
    }

    std::nth_element(dists.begin(), dists.begin() + minPts_ - 2, dists.end());
    return dists[minPts_ - 2];
}

const std::vector<Reachability>& FastOPTICS::getOrdering() const {
    return ordering_;
}

std::vector<size_t> FastOPTICS::extractSuperCluster(ClusterMode mode) const {
    if (mode == ClusterMode::SUPER_CLUSTER_ONLY) {
        size_t n = points_.size();
        if (n == 0) return {};

        // Connected-components flood-fill at 0.8 * core-dist cutoff.
        // Start from the point with lowest reachability (most central).
        std::vector<size_t> superCluster;
        std::vector<bool> visited(n, false);

        // Find seed: ordering entry with lowest reachability distance,
        // breaking ties by original point index for determinism.
        size_t seed_pos = 0;
        for (size_t i = 1; i < ordering_.size(); ++i) {
            if (ordering_[i].reach < ordering_[seed_pos].reach ||
                (ordering_[i].reach == ordering_[seed_pos].reach &&
                 ordering_[i].index < ordering_[seed_pos].index))
                seed_pos = i;
        }
        // Convert ordering position → original point index
        size_t seed = ordering_[seed_pos].index;

        // Flood-fill over point indices
        std::queue<size_t> q;
        q.push(seed);
        visited[seed] = true;
        while (!q.empty()) {
            size_t u = q.front(); q.pop();
            superCluster.push_back(u);
            for (size_t v = 0; v < n; ++v) {
                if (!visited[v] && distance(points_[u], points_[v]) <= 0.8 * coreDist(v)) {
                    visited[v] = true;
                    q.push(v);
                }
            }
        }
        return superCluster;
    }
    // fallback to full hierarchy walk (original behaviour)
    return extractClustersFromOrdering();
}

std::vector<size_t> FastOPTICS::extractClustersFromOrdering() const {
    std::vector<size_t> indices;
    indices.reserve(ordering_.size());
    for (const auto& r : ordering_) {
        indices.push_back(r.index);
    }
    return indices;
}

} // namespace fast_optics
