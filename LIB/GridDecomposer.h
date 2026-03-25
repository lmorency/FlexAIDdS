// GridDecomposer.h — Octree spatial decomposition of cube grid into regions
// for massively parallel independent GA simulations.
#pragma once

#include "flexaid.h"
#include "maps.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <cstring>

struct GridRegion {
    int region_id;
    std::vector<int> grid_indices;   // indices into original cleftgrid[]
    float center[3];                 // centroid of region
    float radius;                    // bounding sphere radius
    int num_points;

    GridRegion() : region_id(-1), center{0,0,0}, radius(0), num_points(0) {}
};

class GridDecomposer {
public:
    // Decompose cleftgrid into spatial regions using octree subdivision.
    // Grid indices 1..num_grd-1 are partitioned (index 0 is the reference point).
    // target_regions: approximate number of output regions (power of 8 not required)
    // min_points_per_region: regions smaller than this are merged into neighbors
    static std::vector<GridRegion> decompose_octree(
        const gridpoint* cleftgrid,
        int num_grd,
        int target_regions,
        int min_points_per_region = 20
    );

    // Merge undersized regions into their nearest neighbor by centroid distance.
    static void balance_regions(
        std::vector<GridRegion>& regions,
        const gridpoint* cleftgrid,
        int min_points
    );

    // Extract a subgrid for a region: allocates new gridpoint array with
    // points renumbered starting from index 1 (index 0 = reference from original).
    // Caller must free() the returned pointer.
    static gridpoint* extract_subgrid(
        const gridpoint* cleftgrid,
        const GridRegion& region,
        int& out_num_grd
    );

    // Compute centroid and bounding sphere radius for a region.
    static void compute_region_bounds(
        GridRegion& region,
        const gridpoint* cleftgrid
    );

private:
    struct OctreeNode {
        float center[3];
        float half_extent;
        std::vector<int> point_indices;
        int depth;
        std::array<OctreeNode*, 8> children;

        OctreeNode() : center{0,0,0}, half_extent(0), depth(0) {
            children.fill(nullptr);
        }
        ~OctreeNode() {
            for (auto* c : children) delete c;
        }
    };

    // Determine which octant a point belongs to relative to center.
    static int octant_index(const float* point, const float* center);

    // Recursively build octree.
    static void build_octree(
        OctreeNode* node,
        const gridpoint* cleftgrid,
        int max_points_per_leaf,
        int max_depth
    );

    // Collect leaf nodes into regions.
    static void collect_leaves(
        const OctreeNode* node,
        std::vector<GridRegion>& regions,
        int& next_id
    );
};
