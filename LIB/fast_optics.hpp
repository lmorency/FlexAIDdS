#ifndef FAST_OPTICS_HPP
#define FAST_OPTICS_HPP

#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <limits>

struct Point {
    std::vector<double> coords;
};

inline double distance(const Point& a, const Point& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.coords.size(); ++i) {
        double d = a.coords[i] - b.coords[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

struct Reachability {
    size_t index;
    double reach;
};

// New lightweight mode for Shannon collapse
enum class ClusterMode {
    FULL_OPTICS,      // original behaviour
    SUPER_CLUSTER_ONLY // new 40 % faster mode
};

class FastOPTICS {
public:
    FastOPTICS(const std::vector<Point>& points, int minPts = 4, double eps = 0.0);
    const std::vector<Reachability>& getOrdering() const;
    std::vector<size_t> extractSuperCluster(ClusterMode mode = ClusterMode::FULL_OPTICS) const; // new
private:
    int minPts_;
    double eps_;
    std::vector<Point> points_;
    std::vector<Reachability> ordering_;

    void computeOrdering();
    double coreDist(size_t idx) const;
    std::vector<size_t> extractClustersFromOrdering() const;
};

#endif // FAST_OPTICS_HPP
