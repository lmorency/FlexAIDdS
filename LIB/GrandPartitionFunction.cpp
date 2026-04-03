// GrandPartitionFunction.cpp — Grand canonical partition function
//
// Copyright 2026 Le Bonhomme Pharma
// SPDX-License-Identifier: Apache-2.0

#include "GrandPartitionFunction.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace target {

GrandPartitionFunction::GrandPartitionFunction(double temperature_K)
    : T_(temperature_K)
    , beta_(1.0 / (statmech::kB_kcal * temperature_K))
{
    if (temperature_K <= 0.0)
        throw std::invalid_argument("Temperature must be positive");
}

// ── Ligand registration ────────────────────────────────────────────────

void GrandPartitionFunction::add_ligand(const std::string& name, double log_Z)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (log_Z_.count(name))
        throw std::invalid_argument("Ligand '" + name + "' already registered");
    log_Z_[name] = log_Z;
}

void GrandPartitionFunction::add_ligand(const std::string& name,
                                         const statmech::StatMechEngine& engine)
{
    if (engine.size() == 0)
        throw std::invalid_argument("Cannot add ligand with empty ensemble");
    auto thermo = engine.compute();
    add_ligand(name, thermo.log_Z);
}

void GrandPartitionFunction::update_ligand(const std::string& name, double new_log_Z)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = log_Z_.find(name);
    if (it == log_Z_.end())
        throw std::invalid_argument("Ligand '" + name + "' not found");

    // Merge: Z_merged = Z_old + Z_new  →  ln Z_merged = log_sum_exp(ln Z_old, ln Z_new)
    double a = it->second;
    double b = new_log_Z;
    double max_val = std::max(a, b);
    it->second = max_val + std::log(std::exp(a - max_val) + std::exp(b - max_val));
}

void GrandPartitionFunction::remove_ligand(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (!log_Z_.erase(name))
        throw std::invalid_argument("Ligand '" + name + "' not found");
}

// ── Thermodynamic queries ──────────────────────────────────────────────

double GrandPartitionFunction::compute_log_Xi_unlocked() const
{
    // Ξ = 1 + Σ_i Z_i = 1 + Σ_i exp(ln Z_i)
    // ln Ξ = log_sum_exp(0, ln Z_1, ln Z_2, ...)
    //   where 0 = ln(1) represents the empty site

    if (log_Z_.empty()) return 0.0;  // ln(1) = 0, only empty site

    // Collect all log(Z) values plus 0.0 (empty site)
    std::vector<double> values;
    values.reserve(log_Z_.size() + 1);
    values.push_back(0.0);  // empty site: ln(1) = 0
    for (const auto& [name, lz] : log_Z_) {
        values.push_back(lz);
    }

    // Numerically stable log-sum-exp
    double max_val = *std::max_element(values.begin(), values.end());
    double sum = 0.0;
    for (double v : values) {
        sum += std::exp(v - max_val);
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
    auto it = log_Z_.find(name);
    if (it == log_Z_.end())
        throw std::invalid_argument("Ligand '" + name + "' not found");
    double log_xi = compute_log_Xi_unlocked();
    return std::exp(it->second - log_xi);
}

double GrandPartitionFunction::empty_probability() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    double log_xi = compute_log_Xi_unlocked();
    return std::exp(-log_xi);  // exp(ln(1) - ln(Ξ)) = 1/Ξ
}

double GrandPartitionFunction::delta_G(const std::string& name) const
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = log_Z_.find(name);
    if (it == log_Z_.end())
        throw std::invalid_argument("Ligand '" + name + "' not found");
    // ΔG = −kT · ln Z_i
    return -(1.0 / beta_) * it->second;
}

double GrandPartitionFunction::selectivity(const std::string& a,
                                            const std::string& b) const
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it_a = log_Z_.find(a);
    auto it_b = log_Z_.find(b);
    if (it_a == log_Z_.end())
        throw std::invalid_argument("Ligand '" + a + "' not found");
    if (it_b == log_Z_.end())
        throw std::invalid_argument("Ligand '" + b + "' not found");
    // Z_a / Z_b = exp(ln Z_a - ln Z_b)
    return std::exp(it_a->second - it_b->second);
}

// ── Ranking ────────────────────────────────────────────────────────────

std::vector<GrandPartitionFunction::LigandRank> GrandPartitionFunction::rank() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    double log_xi = compute_log_Xi_unlocked();
    double kT = 1.0 / beta_;

    std::vector<LigandRank> ranks;
    ranks.reserve(log_Z_.size());
    for (const auto& [name, lz] : log_Z_) {
        ranks.push_back({
            name,
            lz,
            -kT * lz,                         // ΔG
            std::exp(lz - log_xi)              // p_bound
        });
    }

    // Sort by ΔG ascending (most favorable first)
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
    return static_cast<int>(log_Z_.size());
}

bool GrandPartitionFunction::has_ligand(const std::string& name) const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return log_Z_.count(name) > 0;
}

std::vector<std::pair<std::string, double>> GrandPartitionFunction::all_log_Z() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<std::pair<std::string, double>> result;
    result.reserve(log_Z_.size());
    for (const auto& [name, lz] : log_Z_) {
        result.emplace_back(name, lz);
    }
    return result;
}

} // namespace target
