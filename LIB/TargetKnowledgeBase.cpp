// TargetKnowledgeBase.cpp — Cross-ligand knowledge accumulation
//
// Copyright 2026 Le Bonhomme Pharma
// SPDX-License-Identifier: Apache-2.0

#include "TargetKnowledgeBase.h"

namespace target {

TargetKnowledgeBase::TargetKnowledgeBase(int n_models, int n_grid_points)
    : n_models_(n_models)
    , n_grid_points_(n_grid_points)
    , conformer_sum_(n_models > 0 ? n_models : 0, 0.0)
    , conformer_count_(0)
    , grid_energy_sum_(n_grid_points > 0 ? n_grid_points : 0, 0.0)
    , grid_energy_count_(n_grid_points > 0 ? n_grid_points : 0, 0)
    , grid_favorable_count_(n_grid_points > 0 ? n_grid_points : 0, 0)
    , total_ligand_count_(0)
{
}

// ── Conformer weights ──────────────────────────────────────────────────

void TargetKnowledgeBase::accumulate_conformer_weights(const std::vector<double>& weights)
{
    std::lock_guard<std::mutex> lock(mtx_);

    if (n_models_ == 0) {
        // First call: initialize dimensions
        n_models_ = static_cast<int>(weights.size());
        conformer_sum_.assign(n_models_, 0.0);
    }

    if (static_cast<int>(weights.size()) != n_models_) return;  // dimension mismatch

    for (int i = 0; i < n_models_; ++i) {
        conformer_sum_[i] += weights[i];
    }
    ++conformer_count_;
}

std::vector<double> TargetKnowledgeBase::conformer_posterior() const
{
    std::lock_guard<std::mutex> lock(mtx_);

    if (n_models_ <= 0) return {};

    std::vector<double> posterior(n_models_);

    // Bayesian posterior with uniform (1/n_models) Dirichlet prior
    // and accumulated observations (pseudo-count = 1 per model)
    double total = 0.0;
    for (int i = 0; i < n_models_; ++i) {
        posterior[i] = conformer_sum_[i] + 1.0;  // prior pseudo-count
        total += posterior[i];
    }

    if (total > 0.0) {
        for (auto& p : posterior) p /= total;
    }

    return posterior;
}

int TargetKnowledgeBase::conformer_observation_count() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return conformer_count_;
}

// ── Grid energy statistics ─────────────────────────────────────────────

void TargetKnowledgeBase::accumulate_grid_energies(const std::vector<float>& energies)
{
    std::lock_guard<std::mutex> lock(mtx_);

    if (n_grid_points_ == 0) {
        // First call: initialize
        n_grid_points_ = static_cast<int>(energies.size());
        grid_energy_sum_.assign(n_grid_points_, 0.0);
        grid_energy_count_.assign(n_grid_points_, 0);
        grid_favorable_count_.assign(n_grid_points_, 0);
    }

    if (static_cast<int>(energies.size()) != n_grid_points_) return;

    for (int i = 0; i < n_grid_points_; ++i) {
        if (std::isfinite(energies[i])) {
            grid_energy_sum_[i] += energies[i];
            grid_energy_count_[i]++;
            // Default cutoff used during accumulation is -1.0 kcal/mol
            if (energies[i] < -1.0f) {
                grid_favorable_count_[i]++;
            }
        }
    }
    ++total_ligand_count_;
}

std::vector<float> TargetKnowledgeBase::grid_mean_energies() const
{
    std::lock_guard<std::mutex> lock(mtx_);

    std::vector<float> means(n_grid_points_, 0.0f);
    for (int i = 0; i < n_grid_points_; ++i) {
        if (grid_energy_count_[i] > 0) {
            means[i] = static_cast<float>(grid_energy_sum_[i] / grid_energy_count_[i]);
        }
    }
    return means;
}

std::vector<float> TargetKnowledgeBase::grid_hotspot_fraction(float energy_cutoff) const
{
    std::lock_guard<std::mutex> lock(mtx_);

    std::vector<float> fractions(n_grid_points_, 0.0f);
    if (total_ligand_count_ == 0) return fractions;

    for (int i = 0; i < n_grid_points_; ++i) {
        fractions[i] = static_cast<float>(grid_favorable_count_[i])
                       / static_cast<float>(total_ligand_count_);
    }
    return fractions;
}

int TargetKnowledgeBase::grid_observation_count() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return total_ligand_count_;
}

// ── Binding subsite hits ───────────────────────────────────────────────

void TargetKnowledgeBase::accumulate_binding_center(float x, float y, float z,
                                                      double energy,
                                                      const std::string& ligand_name)
{
    std::lock_guard<std::mutex> lock(mtx_);
    subsite_hits_.push_back({{x, y, z}, energy, ligand_name});
}

const std::vector<TargetKnowledgeBase::SubsiteHit>&
TargetKnowledgeBase::all_hits() const
{
    // Note: not locked because vector is append-only from accumulate_binding_center
    // and callers typically call this after all docking is complete.
    // For thread-safety during active accumulation, the caller should
    // ensure no concurrent writes.
    return subsite_hits_;
}

} // namespace target
