// GrandPartitionFunction.h — Grand canonical partition function for competitive binding
//
// For a binding site that can be empty or occupied by one of N ligands:
//
//   Ξ = 1 + Z_A + Z_B + Z_C + ...
//
// where Z_i is ligand i's canonical partition function (from StatMechEngine).
// The "1" represents the empty (apo) site.
//
// From Ξ we compute:
//   p(empty)     = 1 / Ξ
//   p(ligand_i)  = Z_i / Ξ
//   ΔG(ligand_i) = −kT ln Z_i
//   selectivity(A/B) = Z_A / Z_B = exp(β(ΔG_B − ΔG_A))
//
// All arithmetic uses log-space (stores ln Z_i) for numerical stability
// since partition functions can span hundreds of orders of magnitude.
//
// Thread-safe: all mutations are mutex-protected.
//
// Copyright 2026 Le Bonhomme Pharma
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "statmech.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <cmath>
#include <stdexcept>

namespace target {

class GrandPartitionFunction {
public:
    explicit GrandPartitionFunction(double temperature_K = 300.0);

    // ── Ligand registration ────────────────────────────────────────────

    /// Register a ligand's partition function from its completed StatMechEngine.
    /// Extracts log(Z) from engine.compute().log_Z.
    void add_ligand(const std::string& name, double log_Z);

    /// Convenience: extract log_Z from a StatMechEngine
    void add_ligand(const std::string& name, const statmech::StatMechEngine& engine);

    /// Update an existing ligand's Z (e.g., after re-docking with more samples).
    /// Uses log-sum-exp to merge old + new Z values.
    void update_ligand(const std::string& name, double new_log_Z);

    /// Remove a ligand from the ensemble.
    void remove_ligand(const std::string& name);

    // ── Thermodynamic queries ──────────────────────────────────────────

    /// ln(Ξ) = ln(1 + Σ_i Z_i)  using log-sum-exp for stability
    double log_Xi() const;

    /// p(ligand_i bound) = Z_i / Ξ = exp(ln Z_i − ln Ξ)
    double binding_probability(const std::string& name) const;

    /// p(empty) = 1/Ξ = exp(−ln Ξ)
    double empty_probability() const;

    /// ΔG(ligand_i) = −kT · ln Z_i
    double delta_G(const std::string& name) const;

    /// Selectivity ratio: Z_a / Z_b = exp(ln Z_a − ln Z_b)
    double selectivity(const std::string& a, const std::string& b) const;

    // ── Ranking ────────────────────────────────────────────────────────

    struct LigandRank {
        std::string name;
        double log_Z;       // ln Z_i
        double dG;           // −kT ln Z_i (kcal/mol)
        double p_bound;      // Z_i / Ξ
    };

    /// Rank all ligands by ΔG (ascending = most favorable first).
    std::vector<LigandRank> rank() const;

    // ── State queries ──────────────────────────────────────────────────

    int  num_ligands() const;
    bool has_ligand(const std::string& name) const;
    double temperature() const noexcept { return T_; }

    /// Get all ligand names and their log(Z) values.
    std::vector<std::pair<std::string, double>> all_log_Z() const;

private:
    double T_;
    double beta_;    // 1/(kT)
    std::unordered_map<std::string, double> log_Z_;
    mutable std::mutex mtx_;

    // Compute ln(Ξ) without holding lock (caller must hold lock)
    double compute_log_Xi_unlocked() const;
};

} // namespace target
