// BindingResidues.h — Identify key binding-site residues from MIF scores
//
// Given MIF energies on a cleftgrid, finds the protein residues that contribute
// most to favorable binding interactions. Uses SpatialGrid for fast atom lookup.
//
// Header-only, C++20, Apache-2.0 © 2026 Le Bonhomme Pharma

#pragma once

#include "flexaid.h"
#include "MIFGrid.h"
#include "CavityDetect/SpatialGrid.h"

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>

namespace binding_residues {

// ── Result: one residue's contribution to binding ──────────────────────────

struct ResidueContribution {
    int    res_index;        // internal residue index (into resid[])
    char   name[4];          // 3-letter residue name (e.g. "ASP")
    int    number;           // PDB residue number
    char   chain;            // chain ID
    float  mif_score;        // summed MIF energy from nearby grid points (more negative = more favorable)
    int    contact_count;    // number of favorable grid points near this residue
    float  min_distance;     // closest distance from any residue atom to a favorable grid point
};

// ── Identify key binding-site residues ─────────────────────────────────────
//
// Algorithm:
//   1. Select top-K% most favorable grid points by MIF energy
//   2. For each favorable grid point, find nearby protein atoms (within cutoff)
//   3. Map atoms → residues, accumulate MIF score per residue
//   4. Sort residues by total MIF score (most favorable first)
//
// Returns a sorted vector of ResidueContribution (most favorable first).

inline std::vector<ResidueContribution> identify_key_residues(
    const gridpoint* cleftgrid, int num_grd,
    const float* mif_energies,
    const atom* atoms, int atm_cnt,
    const resid* residues,
    const cavity_detect::SpatialGrid& spatial_grid,
    float top_k_percent = 30.0f,
    float contact_cutoff = 4.5f)
{
    if (num_grd <= 1 || !mif_energies) return {};

    // Step 1: Find favorable grid points (top-K% by MIF energy)
    const int n_points = num_grd - 1;  // index 0 is unused
    std::vector<int> sorted_indices(static_cast<size_t>(n_points));
    std::iota(sorted_indices.begin(), sorted_indices.end(), 1);

    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&](int a, int b) { return mif_energies[a] < mif_energies[b]; });

    int k = std::max(1, static_cast<int>(
        static_cast<float>(n_points) * top_k_percent / 100.0f));
    if (k > n_points) k = n_points;

    // Step 2 & 3: For each favorable grid point, find nearby atoms → residues
    // Use a map indexed by residue index for accumulation
    struct ResAccum {
        float  score = 0.0f;
        int    count = 0;
        float  min_dist = 1e10f;
    };
    std::vector<ResAccum> res_accum(static_cast<size_t>(atm_cnt > 0 ? atm_cnt : 1));

    // Track which residue indices we've seen
    std::vector<bool> res_seen(static_cast<size_t>(atm_cnt > 0 ? atm_cnt : 1), false);
    int max_res_idx = 0;

    const float cutoff_sq = contact_cutoff * contact_cutoff;

    for (int ki = 0; ki < k; ++ki) {
        int gp = sorted_indices[static_cast<size_t>(ki)];
        float e = mif_energies[gp];
        if (e >= 0.0f) continue;  // only consider favorable (negative) energies

        float gx = cleftgrid[gp].coor[0];
        float gy = cleftgrid[gp].coor[1];
        float gz = cleftgrid[gp].coor[2];

        // Query nearby atoms using SpatialGrid (returns indices within cell_size range)
        float qcoord[3] = {gx, gy, gz};
        auto neighbors = spatial_grid.query_neighbors(qcoord);

        for (std::size_t nb_idx : neighbors) {
            int atom_idx = static_cast<int>(nb_idx);
            if (atom_idx < 0 || atom_idx >= atm_cnt) continue;

            float dx = atoms[atom_idx].coor[0] - gx;
            float dy = atoms[atom_idx].coor[1] - gy;
            float dz = atoms[atom_idx].coor[2] - gz;
            float dist_sq = dx * dx + dy * dy + dz * dz;

            if (dist_sq > cutoff_sq) continue;

            int ri = atoms[atom_idx].ofres;
            if (ri < 0 || ri >= atm_cnt) continue;

            auto uri = static_cast<size_t>(ri);
            if (uri >= res_accum.size()) {
                res_accum.resize(uri + 1);
                res_seen.resize(uri + 1, false);
            }

            float dist = std::sqrt(dist_sq);
            // Weight contribution by inverse distance (closer atoms contribute more)
            float weight = 1.0f / (1.0f + dist);
            res_accum[uri].score += e * weight;
            res_accum[uri].count += 1;
            if (dist < res_accum[uri].min_dist) {
                res_accum[uri].min_dist = dist;
            }
            res_seen[uri] = true;
            if (ri > max_res_idx) max_res_idx = ri;
        }
    }

    // Step 4: Collect and sort results
    std::vector<ResidueContribution> results;
    for (int ri = 0; ri <= max_res_idx; ++ri) {
        auto uri = static_cast<size_t>(ri);
        if (uri >= res_seen.size() || !res_seen[uri]) continue;

        ResidueContribution rc{};
        rc.res_index = ri;
        rc.mif_score = res_accum[uri].score;
        rc.contact_count = res_accum[uri].count;
        rc.min_distance = res_accum[uri].min_dist;

        // Copy residue info
        std::strncpy(rc.name, residues[ri].name, 3);
        rc.name[3] = '\0';
        rc.number = residues[ri].number;
        rc.chain = residues[ri].chn;

        results.push_back(rc);
    }

    // Sort by MIF score (most negative = most favorable = first)
    std::sort(results.begin(), results.end(),
              [](const ResidueContribution& a, const ResidueContribution& b) {
                  return a.mif_score < b.mif_score;
              });

    return results;
}

// ── Convenience: identify from FA_Global (uses pre-computed MIF) ───────────

inline std::vector<ResidueContribution> identify_key_residues_from_fa(
    const FA_Global* FA,
    const gridpoint* cleftgrid,
    const atom* atoms,
    const resid* residues,
    float top_k_percent = 30.0f,
    float contact_cutoff = 4.5f)
{
    if (!FA->mif_energies || FA->mif_count == 0) return {};

    std::vector<atom> protein_atoms(atoms, atoms + FA->atm_cnt_real);
    cavity_detect::SpatialGrid sg;
    sg.build(protein_atoms);

    return identify_key_residues(
        cleftgrid, FA->num_grd, FA->mif_energies,
        atoms, FA->atm_cnt_real, residues, sg,
        top_k_percent, contact_cutoff);
}

// ── Print summary to stdout ────────────────────────────────────────────────

inline void print_key_residues(const std::vector<ResidueContribution>& residues,
                                int max_display = 20) {
    printf("─── Key Binding-Site Residues (by MIF score) ───\n");
    printf("%-5s %-4s %5s %6s  %8s  %5s  %7s\n",
           "Rank", "Name", "Num", "Chain", "MIF_Score", "Contacts", "MinDist");

    int n = std::min(static_cast<int>(residues.size()), max_display);
    for (int i = 0; i < n; ++i) {
        const auto& r = residues[static_cast<size_t>(i)];
        printf("%-5d %-4s %5d %4c    %8.2f  %5d   %6.2f\n",
               i + 1, r.name, r.number, r.chain,
               r.mif_score, r.contact_count, r.min_distance);
    }
    if (static_cast<int>(residues.size()) > max_display) {
        printf("... and %d more residues\n",
               static_cast<int>(residues.size()) - max_display);
    }
}

// ── Auto-flex: add key binding residues to FA->flex_res[] ──────────────────
//
// Adds the top-N most favorable binding residues as flexible side-chains.
// Skips GLY, ALA, PRO (no rotameric freedom), ligand residues, and
// residues already in flex_res[]. Returns count of residues added.
//
// Call AFTER compute_mif_and_reflig() and BEFORE build_rotamers()/add2_optimiz_vec("SC").

inline int add_key_residues_as_flexible(
    FA_Global* FA,
    const gridpoint* cleftgrid,
    const atom* atoms,
    const resid* residues,
    int max_auto_flex = 5,
    float top_k_percent = 30.0f,
    float contact_cutoff = 4.5f)
{
    if (!FA->mif_energies || FA->mif_count == 0 || max_auto_flex <= 0) return 0;

    // Identify key residues
    auto key_res = identify_key_residues_from_fa(
        FA, cleftgrid, atoms, residues, top_k_percent, contact_cutoff);

    if (key_res.empty()) return 0;

    // Residues to skip (no side-chain rotamers)
    auto is_skip_residue = [](const char* name) {
        return std::strcmp(name, "GLY") == 0 ||
               std::strcmp(name, "ALA") == 0 ||
               std::strcmp(name, "PRO") == 0;
    };

    // Check if residue is already in flex_res[]
    auto already_flexible = [&](int res_index) {
        for (int i = 0; i < FA->nflxsc; ++i) {
            if (FA->flex_res[i].inum == res_index) return true;
        }
        return false;
    };

    // Ensure flex_res array is allocated
    if (!FA->flex_res) {
        FA->MIN_FLEX_RESIDUE = std::max(FA->MIN_FLEX_RESIDUE, max_auto_flex + 5);
        FA->flex_res = static_cast<flxsc*>(
            calloc(static_cast<size_t>(FA->MIN_FLEX_RESIDUE), sizeof(flxsc)));
        if (!FA->flex_res) return 0;
    }

    int added = 0;
    for (const auto& rc : key_res) {
        if (added >= max_auto_flex) break;
        if (is_skip_residue(rc.name)) continue;
        if (residues[rc.res_index].type != 0) continue;  // skip ligand residues
        if (already_flexible(rc.res_index)) continue;

        // Grow flex_res if needed
        if (FA->nflxsc >= FA->MIN_FLEX_RESIDUE) {
            FA->MIN_FLEX_RESIDUE += 5;
            FA->flex_res = static_cast<flxsc*>(
                realloc(FA->flex_res,
                        static_cast<size_t>(FA->MIN_FLEX_RESIDUE) * sizeof(flxsc)));
            if (!FA->flex_res) return added;
            std::memset(&FA->flex_res[FA->MIN_FLEX_RESIDUE - 5], 0, 5 * sizeof(flxsc));
        }

        // Add to flex_res
        flxsc& fr = FA->flex_res[FA->nflxsc];
        std::strncpy(fr.name, rc.name, 3);
        fr.name[3] = '\0';
        fr.chn = rc.chain;
        fr.num = rc.number;
        fr.inum = rc.res_index;
        set_intprob(&fr);

        FA->nflxsc++;
        added++;

        printf("AUTOFLEX: %s %d:%c added as flexible (MIF=%.2f, %d contacts)\n",
               rc.name, rc.number, rc.chain, rc.mif_score, rc.contact_count);
    }

    return added;
}

} // namespace binding_residues
