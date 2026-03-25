// MIFGrid.h — Molecular Interaction Field for cleft grid points
// Computes per-grid-point vdW energy using SpatialGrid neighbor queries.
// Provides Boltzmann-weighted sampling for GA initialization and
// grid prioritization (filtering to top-K% favorable points).
// Header-only, thread-safe, Apache-2.0 © 2026 Le Bonhomme Pharma

#pragma once

#include <vector>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <numeric>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "flexaid.h"
#include "CavityDetect/SpatialGrid.h"

namespace mif {

// ── Result of MIF computation ───────────────────────────────────────────────
struct MIFResult {
    std::vector<float> energies;       // per-grid-point energy (lower = more favorable)
    std::vector<int>   sorted_indices; // grid indices sorted by energy (ascending)
    std::vector<double> cdf;           // CDF for Boltzmann-weighted sampling
    int num_grd = 0;                   // total grid points (including index 0)
};

// ── Lennard-Jones 6-12 potential ────────────────────────────────────────────
// sigma = r_probe + r_atom (vdW contact distance)
// well_depth chosen so the scoring has a clear minimum at contact distance
inline float lj_energy(float dist_sq, float sigma) {
    if (dist_sq < 1.0f) return 1e6f;  // hard core repulsion
    float sigma_sq = sigma * sigma;
    float ratio2 = sigma_sq / dist_sq;
    float ratio6 = ratio2 * ratio2 * ratio2;
    // Standard LJ: E = eps * (ratio^12 - 2*ratio^6)
    // eps = 1.0 (arbitrary units — only relative ordering matters)
    return ratio6 * ratio6 - 2.0f * ratio6;
}

// ── Compute MIF ─────────────────────────────────────────────────────────────
// For each grid point, sum vdW interaction energy with nearby protein atoms.
// probe_radius: vdW radius of the probe (default 1.7 Å ≈ carbon)
// cutoff: distance cutoff for interactions (default 6.0 Å)
inline MIFResult compute_mif(
    const gridpoint* cleftgrid, int num_grd,
    const atom* atoms, int atm_cnt,
    const cavity_detect::SpatialGrid& spatial_grid,
    float probe_radius = 1.7f,
    float cutoff = 6.0f)
{
    MIFResult result;
    result.num_grd = num_grd;
    result.energies.resize(static_cast<std::size_t>(num_grd), 0.0f);

    if (num_grd <= 1 || spatial_grid.empty()) return result;

    const float cutoff_sq = cutoff * cutoff;

    // Thread-safe: SpatialGrid is immutable after build()
    #ifdef _OPENMP
    #pragma omp parallel
    {
        std::vector<std::size_t> neighbor_buf(512);

        #pragma omp for schedule(dynamic, 64)
        for (int gp = 1; gp < num_grd; ++gp) {
            std::size_t n_neighbors = spatial_grid.query_neighbors(
                cleftgrid[gp].coor, neighbor_buf.data(), neighbor_buf.size());

            float total_energy = 0.0f;
            for (std::size_t n = 0; n < n_neighbors; ++n) {
                std::size_t ai = neighbor_buf[n];
                float dx = cleftgrid[gp].coor[0] - atoms[ai].coor[0];
                float dy = cleftgrid[gp].coor[1] - atoms[ai].coor[1];
                float dz = cleftgrid[gp].coor[2] - atoms[ai].coor[2];
                float dist_sq = dx*dx + dy*dy + dz*dz;

                if (dist_sq > cutoff_sq) continue;

                float sigma = probe_radius + atoms[ai].radius;
                total_energy += lj_energy(dist_sq, sigma);
            }
            result.energies[static_cast<std::size_t>(gp)] = total_energy;
        }
    }
    #else
    std::vector<std::size_t> neighbor_buf(512);
    for (int gp = 1; gp < num_grd; ++gp) {
        std::size_t n_neighbors = spatial_grid.query_neighbors(
            cleftgrid[gp].coor, neighbor_buf.data(), neighbor_buf.size());

        float total_energy = 0.0f;
        for (std::size_t n = 0; n < n_neighbors; ++n) {
            std::size_t ai = neighbor_buf[n];
            float dx = cleftgrid[gp].coor[0] - atoms[ai].coor[0];
            float dy = cleftgrid[gp].coor[1] - atoms[ai].coor[1];
            float dz = cleftgrid[gp].coor[2] - atoms[ai].coor[2];
            float dist_sq = dx*dx + dy*dy + dz*dz;

            if (dist_sq > cutoff_sq) continue;

            float sigma = probe_radius + atoms[ai].radius;
            total_energy += lj_energy(dist_sq, sigma);
        }
        result.energies[static_cast<std::size_t>(gp)] = total_energy;
    }
    #endif

    // Build sorted indices (ascending energy = most favorable first)
    result.sorted_indices.resize(static_cast<std::size_t>(num_grd - 1));
    std::iota(result.sorted_indices.begin(), result.sorted_indices.end(), 1);
    std::sort(result.sorted_indices.begin(), result.sorted_indices.end(),
        [&](int a, int b) {
            return result.energies[static_cast<std::size_t>(a)]
                 < result.energies[static_cast<std::size_t>(b)];
        });

    return result;
}

// ── Build CDF for Boltzmann-weighted sampling ───────────────────────────────
// temperature in Kelvin. kB_kcal ≈ 0.001987 kcal/(mol·K).
inline void build_sampling_cdf(MIFResult& mif, float temperature = 300.0f) {
    if (mif.sorted_indices.empty()) return;

    const double kBT = 0.001987 * static_cast<double>(temperature);
    const std::size_t N = mif.sorted_indices.size();

    // Find minimum energy for numerical stability (log-sum-exp trick)
    float e_min = mif.energies[static_cast<std::size_t>(mif.sorted_indices[0])];

    mif.cdf.resize(N);
    double cumulative = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        float e = mif.energies[static_cast<std::size_t>(mif.sorted_indices[i])];
        double boltzmann = std::exp(-static_cast<double>(e - e_min) / kBT);
        cumulative += boltzmann;
        mif.cdf[i] = cumulative;
    }
    // Normalize
    for (std::size_t i = 0; i < N; ++i)
        mif.cdf[i] /= cumulative;
}

// ── Sample a grid index from the MIF-weighted distribution ──────────────────
inline int sample_grid_index(const MIFResult& mif, std::mt19937& rng) {
    if (mif.cdf.empty()) return 1;  // fallback to grid point 1

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double u = dist(rng);

    // Binary search in CDF
    auto it = std::lower_bound(mif.cdf.begin(), mif.cdf.end(), u);
    std::size_t idx = static_cast<std::size_t>(
        std::distance(mif.cdf.begin(), it));
    if (idx >= mif.sorted_indices.size())
        idx = mif.sorted_indices.size() - 1;

    return mif.sorted_indices[idx];
}

// ── Grid prioritization: filter to top K% most favorable points ─────────────
// Returns a vector of grid indices (1-based) sorted by energy, keeping only
// the top_k_percent most favorable. The caller can use this to rebuild or
// reindex the cleftgrid if desired.
inline std::vector<int> prioritize_grid(const MIFResult& mif,
                                         float top_k_percent = 50.0f) {
    if (mif.sorted_indices.empty()) return {};

    std::size_t n_keep = std::max(
        static_cast<std::size_t>(1),
        static_cast<std::size_t>(
            static_cast<float>(mif.sorted_indices.size()) * top_k_percent / 100.0f));

    if (n_keep > mif.sorted_indices.size())
        n_keep = mif.sorted_indices.size();

    return std::vector<int>(
        mif.sorted_indices.begin(),
        mif.sorted_indices.begin() + static_cast<std::ptrdiff_t>(n_keep));
}

// ── Rebuild a compacted cleftgrid from prioritized indices ───────────────────
// Copies selected grid points into a new array. Index 0 (ligand reference
// conformation) is always preserved. Returns the new grid point count.
// The caller must free the returned array with free().
inline int rebuild_cleftgrid(const gridpoint* old_grid, int old_num_grd,
                              const std::vector<int>& keep_indices,
                              gridpoint** new_grid_out) {
    int new_count = 1 + static_cast<int>(keep_indices.size());  // index 0 + kept points

    auto* new_grid = static_cast<gridpoint*>(
        malloc(static_cast<std::size_t>(new_count) * sizeof(gridpoint)));
    if (!new_grid) return -1;

    // Copy index 0 (ligand reference conformation)
    new_grid[0] = old_grid[0];

    // Copy kept grid points with new sequential indices
    for (std::size_t i = 0; i < keep_indices.size(); ++i) {
        int old_idx = keep_indices[i];
        new_grid[i + 1] = old_grid[old_idx];
        new_grid[i + 1].index = static_cast<int>(i + 1);
    }

    *new_grid_out = new_grid;
    return new_count;
}

} // namespace mif
