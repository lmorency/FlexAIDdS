// TargetKnowledgeBase.h — Cross-ligand knowledge accumulation
//
// Accumulates target-intrinsic knowledge from completed docking sessions
// that does NOT violate per-ligand thermodynamic independence:
//
// 1. Conformer population statistics: which receptor conformers are sampled
//    across all ligands. Bayesian posterior serves as prior for future docking.
//
// 2. Grid energy statistics: running mean of energies observed at each grid
//    point across ligands. Identifies structural "hot spots".
//
// 3. Binding subsite hits: spatial coordinates where diverse ligands bind.
//    Cross-ligand spatial clustering reveals pharmacophoric anchor points.
//
// Thread-safe: all writes are mutex-protected.
//
// Copyright 2026 Le Bonhomme Pharma
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>
#include <string>
#include <array>
#include <mutex>
#include <cmath>

namespace target {

class TargetKnowledgeBase {
public:
    TargetKnowledgeBase() = default;
    TargetKnowledgeBase(int n_models, int n_grid_points);

    // ── Conformer weights (CCBM) ──────────────────────────────────────

    /// Accumulate a ligand's conformer population vector
    /// (from BindingMode::conformer_populations()).
    void accumulate_conformer_weights(const std::vector<double>& weights);

    /// Bayesian posterior: uniform prior + accumulated observations.
    /// Returns normalized weights (sum = 1).
    std::vector<double> conformer_posterior() const;

    /// Number of ligands that contributed conformer data.
    int conformer_observation_count() const;

    // ── Grid energy statistics ─────────────────────────────────────────

    /// Accumulate per-grid-point best energies from one ligand's docking.
    void accumulate_grid_energies(const std::vector<float>& energies);

    /// Running mean energy per grid point.
    std::vector<float> grid_mean_energies() const;

    /// Fraction of ligands with energy < cutoff at each grid point ("hot spots").
    std::vector<float> grid_hotspot_fraction(float energy_cutoff = -1.0f) const;

    int grid_observation_count() const;

    // ── Binding subsite hits ───────────────────────────────────────────

    struct SubsiteHit {
        std::array<float, 3> center;
        double energy;
        std::string ligand_name;
    };

    /// Record where a ligand's best-scoring pose was placed.
    void accumulate_binding_center(float x, float y, float z,
                                    double energy,
                                    const std::string& ligand_name);

    /// All recorded binding centers (for external clustering).
    const std::vector<SubsiteHit>& all_hits() const;

    // ── State ──────────────────────────────────────────────────────────

    int n_models() const { return n_models_; }
    int n_grid_points() const { return n_grid_points_; }

private:
    int n_models_ = 0;
    int n_grid_points_ = 0;

    // Conformer accumulation (Welford online mean)
    std::vector<double> conformer_sum_;
    int conformer_count_ = 0;

    // Grid energy accumulation
    std::vector<double> grid_energy_sum_;
    std::vector<int>    grid_energy_count_;
    std::vector<int>    grid_favorable_count_;
    int total_ligand_count_ = 0;

    // Binding center hits
    std::vector<SubsiteHit> subsite_hits_;

    mutable std::mutex mtx_;
};

} // namespace target
