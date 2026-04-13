// GrandPartitionFunction.cpp — Grand canonical partition function
//
// Copyright 2026 Le Bonhomme Pharma
// SPDX-License-Identifier: Apache-2.0

#include "GrandPartitionFunction.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace target {

GrandPartitionFunction::GrandPartitionFunction(double temperature_K)
    : T_(temperature_K)
    , beta_(1.0 / (statmech::kB_kcal * temperature_K))
{
    if (temperature_K <= 0.0)
        throw std::invalid_argument("Temperature must be positive");
}

// ── Ligand registration ────────────────────────────────────────────────

void GrandPartitionFunction::add_ligand(const std::string& name, double log_Z,
                                         double concentration_M)
{
    if (concentration_M <= 0.0)
        throw std::invalid_argument("Concentration must be positive");
    double log_zZ = std::log(concentration_M) + log_Z;

    std::lock_guard<std::mutex> lock(mtx_);
    if (ligands_.count(name))
        throw std::invalid_argument("Ligand '" + name + "' already registered");
    ligands_[name] = {log_Z, log_zZ};
}

void GrandPartitionFunction::add_ligand(const std::string& name,
                                         const statmech::StatMechEngine& engine,
                                         double concentration_M)
{
    if (engine.size() == 0)
        throw std::invalid_argument("Cannot add ligand with empty ensemble");
    auto thermo = engine.compute();
    add_ligand(name, thermo.log_Z, concentration_M);
}

void GrandPartitionFunction::overwrite_ligand(const std::string& name, double new_log_Z)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = ligands_.find(name);
    if (it == ligands_.end())
        throw std::invalid_argument("Ligand '" + name + "' not found");
    // Preserve concentration; update both log_Z and log_zZ
    double log_c = it->second.log_zZ - it->second.log_Z;  // ln(c/c°)
    it->second.log_Z = new_log_Z;
    it->second.log_zZ = log_c + new_log_Z;
}

void GrandPartitionFunction::merge_ligand(const std::string& name, double new_log_Z)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = ligands_.find(name);
    if (it == ligands_.end())
        throw std::invalid_argument("Ligand '" + name + "' not found");

    // Merge log_zZ (which is what participates in Ξ)
    double a = it->second.log_zZ;
    double log_c = a - it->second.log_Z;
    double b = log_c + new_log_Z;  // same concentration for the new ensemble
    double max_val = std::max(a, b);
    it->second.log_zZ = max_val + std::log(std::exp(a - max_val) + std::exp(b - max_val));
    // Update intrinsic log_Z to the merged value too
    it->second.log_Z = it->second.log_zZ - log_c;
}

void GrandPartitionFunction::remove_ligand(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (!ligands_.erase(name))
        throw std::invalid_argument("Ligand '" + name + "' not found");
}

// ── Thermodynamic queries ──────────────────────────────────────────────

double GrandPartitionFunction::compute_log_Xi_unlocked() const
{
    // Ξ = 1 + Σ_i z_i·Z_i = 1 + Σ_i exp(ln(z_i·Z_i))
    // ln Ξ = log_sum_exp(0, ln(z_1·Z_1), ln(z_2·Z_2), ...)

    if (ligands_.empty()) return 0.0;

    double max_val = 0.0;
    for (const auto& [name, entry] : ligands_) {
        if (entry.log_zZ > max_val) max_val = entry.log_zZ;
    }

    double sum = std::exp(0.0 - max_val);  // empty site: ln(1) = 0
    for (const auto& [name, entry] : ligands_) {
        sum += std::exp(entry.log_zZ - max_val);
    }
    return max_val + std::log(sum);
}

double GrandPartitionFunction::log_Xi() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return compute_log_Xi_unlocked();
}

double GrandPartitionFunction::binding_probability(const std::string& name) const
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = ligands_.find(name);
    if (it == ligands_.end())
        throw std::invalid_argument("Ligand '" + name + "' not found");
    double log_xi = compute_log_Xi_unlocked();
    return std::exp(it->second.log_zZ - log_xi);
}

double GrandPartitionFunction::empty_probability() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    double log_xi = compute_log_Xi_unlocked();
    return std::exp(-log_xi);
}

double GrandPartitionFunction::free_energy(const std::string& name) const
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = ligands_.find(name);
    if (it == ligands_.end())
        throw std::invalid_argument("Ligand '" + name + "' not found");
    // F = −kT · ln Z_i  (concentration-independent intrinsic free energy)
    return -(1.0 / beta_) * it->second.log_Z;
}

double GrandPartitionFunction::delta_G_bind(const std::string& name, double F_ref) const
{
    return free_energy(name) - F_ref;
}

double GrandPartitionFunction::selectivity(const std::string& a,
                                            const std::string& b) const
{
    double diff = log_selectivity(a, b);
    if (diff > 700.0)  return std::numeric_limits<double>::infinity();
    if (diff < -700.0) return 0.0;
    return std::exp(diff);
}

double GrandPartitionFunction::log_selectivity(const std::string& a,
                                                const std::string& b) const
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it_a = ligands_.find(a);
    auto it_b = ligands_.find(b);
    if (it_a == ligands_.end())
        throw std::invalid_argument("Ligand '" + a + "' not found");
    if (it_b == ligands_.end())
        throw std::invalid_argument("Ligand '" + b + "' not found");
    // (z_A·Z_A) / (z_B·Z_B)
    return it_a->second.log_zZ - it_b->second.log_zZ;
}

// ── Ranking ────────────────────────────────────────────────────────────

std::vector<GrandPartitionFunction::LigandRank> GrandPartitionFunction::rank() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    double log_xi = compute_log_Xi_unlocked();
    double kT = 1.0 / beta_;

    std::vector<LigandRank> ranks;
    ranks.reserve(ligands_.size());
    for (const auto& [name, entry] : ligands_) {
        ranks.push_back({
            name,
            entry.log_Z,
            -kT * entry.log_Z,                          // ΔG (intrinsic)
            std::exp(entry.log_zZ - log_xi)             // p_bound
        });
    }

    std::sort(ranks.begin(), ranks.end(),
              [](const LigandRank& a, const LigandRank& b) {
                  return a.dG < b.dG;
              });

    return ranks;
}

// ── State queries ──────────────────────────────────────────────────────

int GrandPartitionFunction::num_ligands() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return static_cast<int>(ligands_.size());
}

bool GrandPartitionFunction::has_ligand(const std::string& name) const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return ligands_.count(name) > 0;
}

std::vector<std::pair<std::string, double>> GrandPartitionFunction::all_log_Z() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<std::pair<std::string, double>> result;
    result.reserve(ligands_.size());
    for (const auto& [name, entry] : ligands_) {
        result.emplace_back(name, entry.log_Z);
    }
    return result;
}

} // namespace target
