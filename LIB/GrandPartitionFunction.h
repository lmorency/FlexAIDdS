// GrandPartitionFunction.h — Grand canonical partition function for competitive binding
//
// For a binding site that can be empty or occupied by one of N ligands:
//
//   Ξ = 1 + z_A·Z_A + z_B·Z_B + z_C·Z_C + ...
//
// where Z_i is ligand i's canonical partition function (from StatMechEngine),
// z_i = c_i / c° is the fugacity (activity at concentration c_i, standard
// state c° = 1 M), and the "1" represents the empty (apo) site.
//
// Internally stores ln(z_i · Z_i) = ln(c_i/c°) + ln Z_i for each ligand,
// so all existing log-space arithmetic is preserved.
//
// From Ξ we compute:
//   p(empty)     = 1 / Ξ
//   p(ligand_i)  = z_i·Z_i / Ξ
//   ΔG(ligand_i) = −kT ln Z_i          (concentration-independent)
//   selectivity(A/B) = (z_A·Z_A) / (z_B·Z_B)
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
#include <algorithm>
#include <stdexcept>

namespace target {

class GrandPartitionFunction {
public:
    explicit GrandPartitionFunction(double temperature_K = 300.0);

    // ── Ligand registration ────────────────────────────────────────────

    /// Register a ligand with its partition function and optional concentration.
    /// @param name       Ligand identifier
    /// @param log_Z      ln(Z_i) from StatMechEngine
    /// @param concentration_M  Ligand concentration in molar (default 1.0 M = standard state)
    void add_ligand(const std::string& name, double log_Z, double concentration_M = 1.0);

    /// Convenience: extract log_Z from a StatMechEngine
    void add_ligand(const std::string& name, const statmech::StatMechEngine& engine,
                    double concentration_M = 1.0);

    /// Overwrite an existing ligand's Z (e.g., after re-docking with a better estimate).
    void overwrite_ligand(const std::string& name, double new_log_Z);

    /// Merge an independent ensemble into an existing ligand's Z (log-sum-exp).
    void merge_ligand(const std::string& name, double new_log_Z);

    /// Remove a ligand from the ensemble.
    void remove_ligand(const std::string& name);

    // ── Thermodynamic queries ──────────────────────────────────────────

    /// ln(Ξ) = ln(1 + Σ_i Z_i)  using log-sum-exp for stability
    double log_Xi() const;

    /// p(ligand_i bound) = Z_i / Ξ = exp(ln Z_i − ln Ξ)
    double binding_probability(const std::string& name) const;

    /// p(empty) = 1/Ξ = exp(−ln Ξ)
    double empty_probability() const;

    /// Intrinsic free energy: F = −kT · ln Z_i  (kcal/mol).
    /// This is the Helmholtz free energy of the bound ensemble,
    /// NOT the binding free energy (which requires an unbound reference).
    double free_energy(const std::string& name) const;

    /// Binding free energy: ΔG_bind = F_bound − F_ref.
    /// @param name    Ligand name
    /// @param F_ref   Reference-state free energy (unbound ligand in solution).
    ///                Typically obtained from a separate StatMechEngine on the
    ///                unbound ensemble. If 0.0, returns −kT ln Z_i directly.
    double delta_G_bind(const std::string& name, double F_ref = 0.0) const;

    /// Selectivity ratio: Z_a / Z_b = exp(ln Z_a − ln Z_b).
    /// Returns +Inf or 0.0 for extreme ratios (|ΔΔG| > ~700 kT).
    double selectivity(const std::string& a, const std::string& b) const;

    /// ln(Z_a / Z_b) = ln Z_a − ln Z_b.  Overflow-safe.
    double log_selectivity(const std::string& a, const std::string& b) const;

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

    struct LigandEntry {
        double log_Z;          // intrinsic ln(Z_i)
        double log_zZ;         // ln(z_i · Z_i) = ln(c_i/c°) + ln(Z_i)
    };
    std::unordered_map<std::string, LigandEntry> ligands_;
    mutable std::mutex mtx_;

    // Compute ln(Ξ) without holding lock (caller must hold lock)
    double compute_log_Xi_unlocked() const;
};

} // namespace target
