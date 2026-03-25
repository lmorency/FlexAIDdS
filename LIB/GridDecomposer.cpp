// GridDecomposer.cpp — Octree spatial decomposition of cube grid
#include "GridDecomposer.h"
#include <cstdio>
#include <cfloat>
#include <numeric>

// ============================================================================
// Octree construction helpers
// ============================================================================

int GridDecomposer::octant_index(const float* point, const float* center) {
    int idx = 0;
    if (point[0] >= center[0]) idx |= 1;
    if (point[1] >= center[1]) idx |= 2;
    if (point[2] >= center[2]) idx |= 4;
    return idx;
}

void GridDecomposer::build_octree(
    OctreeNode* node,
    const gridpoint* cleftgrid,
    int max_points_per_leaf,
    int max_depth)
{
    // Leaf condition: few enough points or max depth reached
    if ((int)node->point_indices.size() <= max_points_per_leaf ||
        node->depth >= max_depth) {
        return;
    }

    // Distribute points to 8 children
    float he = node->half_extent * 0.5f;  // child half-extent

    for (int i = 0; i < 8; i++) {
        node->children[i] = new OctreeNode();
        node->children[i]->half_extent = he;
        node->children[i]->depth = node->depth + 1;

        // Child center offset
        node->children[i]->center[0] = node->center[0] + ((i & 1) ? he : -he);
        node->children[i]->center[1] = node->center[1] + ((i & 2) ? he : -he);
        node->children[i]->center[2] = node->center[2] + ((i & 4) ? he : -he);
    }

    // Assign each point to a child
    for (int idx : node->point_indices) {
        int oct = octant_index(cleftgrid[idx].coor, node->center);
        node->children[oct]->point_indices.push_back(idx);
    }

    // Clear parent's points (they now live in children)
    node->point_indices.clear();
    node->point_indices.shrink_to_fit();

    // Recurse into non-empty children; delete empty ones
    for (int i = 0; i < 8; i++) {
        if (node->children[i]->point_indices.empty()) {
            delete node->children[i];
            node->children[i] = nullptr;
        } else {
            build_octree(node->children[i], cleftgrid, max_points_per_leaf, max_depth);
        }
    }
}

void GridDecomposer::collect_leaves(
    const OctreeNode* node,
    std::vector<GridRegion>& regions,
    int& next_id)
{
    if (!node) return;

    // If this is a leaf (has points), collect it
    if (!node->point_indices.empty()) {
        GridRegion r;
        r.region_id = next_id++;
        r.grid_indices = node->point_indices;
        r.num_points = (int)node->point_indices.size();
        regions.push_back(std::move(r));
        return;
    }

    // Internal node: recurse into children
    for (auto* child : node->children) {
        collect_leaves(child, regions, next_id);
    }
}

// ============================================================================
// Public API
// ============================================================================

std::vector<GridRegion> GridDecomposer::decompose_octree(
    const gridpoint* cleftgrid,
    int num_grd,
    int target_regions,
    int min_points_per_region)
{
    if (num_grd <= 1) return {};

    int n_points = num_grd - 1;  // indices 1..num_grd-1

    // Compute bounding box of all grid points
    float bb_min[3] = { FLT_MAX,  FLT_MAX,  FLT_MAX};
    float bb_max[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (int i = 1; i < num_grd; i++) {
        for (int d = 0; d < 3; d++) {
            if (cleftgrid[i].coor[d] < bb_min[d]) bb_min[d] = cleftgrid[i].coor[d];
            if (cleftgrid[i].coor[d] > bb_max[d]) bb_max[d] = cleftgrid[i].coor[d];
        }
    }

    // Root node: centered on bounding box, half-extent = max dimension / 2
    OctreeNode root;
    for (int d = 0; d < 3; d++) {
        root.center[d] = (bb_min[d] + bb_max[d]) * 0.5f;
    }
    float max_span = 0;
    for (int d = 0; d < 3; d++) {
        float span = bb_max[d] - bb_min[d];
        if (span > max_span) max_span = span;
    }
    root.half_extent = max_span * 0.5f + 0.001f;  // small pad to include boundary
    root.depth = 0;

    // Populate root with all grid point indices
    root.point_indices.resize(n_points);
    std::iota(root.point_indices.begin(), root.point_indices.end(), 1);

    // Determine max_points_per_leaf to approximate target_regions
    int max_points_per_leaf = std::max(1, n_points / target_regions);
    // Max depth: log8(target_regions) + 2, capped at 10
    int max_depth = std::min(10, (int)std::ceil(std::log2(target_regions) / 3.0) + 2);

    // Build octree
    build_octree(&root, cleftgrid, max_points_per_leaf, max_depth);

    // Collect leaf regions
    std::vector<GridRegion> regions;
    int next_id = 0;
    collect_leaves(&root, regions, next_id);

    // Compute bounds for each region
    for (auto& r : regions) {
        compute_region_bounds(r, cleftgrid);
    }

    // Balance: merge tiny regions
    if (min_points_per_region > 0) {
        balance_regions(regions, cleftgrid, min_points_per_region);
    }

    printf("GridDecomposer: %d grid points → %d regions\n",
           n_points, (int)regions.size());

    return regions;
}

void GridDecomposer::balance_regions(
    std::vector<GridRegion>& regions,
    const gridpoint* cleftgrid,
    int min_points)
{
    bool merged = true;
    while (merged) {
        merged = false;
        for (size_t i = 0; i < regions.size(); i++) {
            if (regions[i].num_points >= min_points) continue;
            if (regions.size() <= 1) break;

            // Find nearest neighbor by centroid distance
            float best_dist = FLT_MAX;
            size_t best_j = 0;
            for (size_t j = 0; j < regions.size(); j++) {
                if (j == i) continue;
                float dx = regions[i].center[0] - regions[j].center[0];
                float dy = regions[i].center[1] - regions[j].center[1];
                float dz = regions[i].center[2] - regions[j].center[2];
                float d2 = dx*dx + dy*dy + dz*dz;
                if (d2 < best_dist) {
                    best_dist = d2;
                    best_j = j;
                }
            }

            // Merge i into best_j
            regions[best_j].grid_indices.insert(
                regions[best_j].grid_indices.end(),
                regions[i].grid_indices.begin(),
                regions[i].grid_indices.end()
            );
            regions[best_j].num_points = (int)regions[best_j].grid_indices.size();
            compute_region_bounds(regions[best_j], cleftgrid);

            // Remove i
            regions.erase(regions.begin() + (long)i);
            merged = true;
            break;  // restart scan
        }
    }

    // Reassign sequential IDs
    for (int i = 0; i < (int)regions.size(); i++) {
        regions[i].region_id = i;
    }
}

gridpoint* GridDecomposer::extract_subgrid(
    const gridpoint* cleftgrid,
    const GridRegion& region,
    int& out_num_grd)
{
    int n = region.num_points;
    out_num_grd = n + 1;  // index 0 = reference, 1..n = region points

    gridpoint* subgrid = (gridpoint*)malloc(out_num_grd * sizeof(gridpoint));
    if (!subgrid) return nullptr;

    // Copy reference point (index 0)
    memset(subgrid, 0, out_num_grd * sizeof(gridpoint));
    subgrid[0] = cleftgrid[0];

    // Copy region points starting at index 1
    for (int i = 0; i < n; i++) {
        subgrid[i + 1] = cleftgrid[region.grid_indices[i]];
    }

    return subgrid;
}

void GridDecomposer::compute_region_bounds(
    GridRegion& region,
    const gridpoint* cleftgrid)
{
    if (region.grid_indices.empty()) {
        region.center[0] = region.center[1] = region.center[2] = 0;
        region.radius = 0;
        return;
    }

    // Centroid
    double cx = 0, cy = 0, cz = 0;
    for (int idx : region.grid_indices) {
        cx += cleftgrid[idx].coor[0];
        cy += cleftgrid[idx].coor[1];
        cz += cleftgrid[idx].coor[2];
    }
    int n = (int)region.grid_indices.size();
    region.center[0] = (float)(cx / n);
    region.center[1] = (float)(cy / n);
    region.center[2] = (float)(cz / n);

    // Bounding sphere radius
    float max_r2 = 0;
    for (int idx : region.grid_indices) {
        float dx = cleftgrid[idx].coor[0] - region.center[0];
        float dy = cleftgrid[idx].coor[1] - region.center[1];
        float dz = cleftgrid[idx].coor[2] - region.center[2];
        float r2 = dx*dx + dy*dy + dz*dz;
        if (r2 > max_r2) max_r2 = r2;
    }
    region.radius = std::sqrt(max_r2);
    region.num_points = n;
}
