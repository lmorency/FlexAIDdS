// MultiSiteGPF.h — Multi-site grand canonical partition function
//
// For a receptor with S independent (or coupled) binding sites:
//
//   Ξ_total = Π_s Ξ_s + coupling terms
//
// Each site has its own GrandPartitionFunction. Sites can be:
//   - Independent: Ξ = Π_s Ξ_s
//   - Cooperatively coupled: extra ω terms for simultaneous occupancy
//
// Cooperativity model:
//   Ξ = Π_s (1 + Σ_i z_is·Z_is) + Σ_pairs ω_ab · (z_a·Z_a)(z_b·Z_b) + ...
//
// where ω_ab > 1 for positive cooperativity, < 1 for negative.
//
// Thread-safe: each site's GPF manages its own lock; the MultiSiteGPF
// adds a separate lock for the coupling matrix.
//
// Copyright 2026 Le Bonhomme Pharma
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "GrandPartitionFunction.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <stdexcept>

namespace target {

class MultiSiteGPF {
public:
    explicit MultiSiteGPF(double temperature_K = 300.0);

    // ── Site management ─────────────────────────────────────────────────

    /// Add a named binding site. Returns the site index.
    int add_site(const std::string& site_name);

    /// Number of registered sites.
    int num_sites() const;

    /// Get site index by name (-1 if not found).
    int site_index(const std::string& name) const;

    /// Access a site's GPF by index.
    GrandPartitionFunction& site(int idx);
    const GrandPartitionFunction& site(int idx) const;

    // ── Ligand registration (convenience: register to all sites) ────────

    /// Register a ligand to a specific site.
    void add_ligand(int site_idx, const std::string& ligand_name,
                    double log_Z, double concentration_M = 1.0);

    /// Register a ligand to a specific site by name.
    void add_ligand(const std::string& site_name, const std::string& ligand_name,
                    double log_Z, double concentration_M = 1.0);

    // ── Cooperativity ───────────────────────────────────────────────────

    /// Set cooperativity factor between two sites.
    /// ω > 1: positive cooperativity (binding at one site enhances the other)
    /// ω < 1: negative cooperativity (binding at one site inhibits the other)
    /// ω = 1: independent (default)
    void set_cooperativity(int site_a, int site_b, double omega);

    /// Get cooperativity factor between two sites (default 1.0).
    double cooperativity(int site_a, int site_b) const;

    // ── Thermodynamic queries ───────────────────────────────────────────

    /// ln(Ξ_total) for the full multi-site system.
    /// For independent sites: Σ_s ln Ξ_s
    /// For coupled sites: includes coupling contributions
    double log_Xi() const;

    /// p(all sites empty) = 1 / Ξ_total
    double empty_probability() const;

    /// p(ligand bound at specific site) — marginal probability.
    double binding_probability(int site_idx, const std::string& ligand_name) const;

    /// p(any ligand bound at specific site) — site occupancy.
    double site_occupancy(int site_idx) const;

    /// Selectivity of ligand A vs B at a specific site.
    double selectivity(int site_idx,
                       const std::string& a, const std::string& b) const;

    // ── Cross-site analysis ─────────────────────────────────────────────

    struct CrossSiteSelectivity {
        std::string ligand_name;
        int best_site_idx;
        double best_site_selectivity;  // vs next-best site
        std::vector<double> site_probabilities;
    };

    /// For each ligand present at multiple sites, compute cross-site selectivity.
    std::vector<CrossSiteSelectivity> cross_site_analysis() const;

private:
    double T_;
    std::vector<std::unique_ptr<GrandPartitionFunction>> sites_;
    std::unordered_map<std::string, int> site_name_to_idx_;

    // Coupling matrix: coupling_[a][b] = ω_ab (symmetric)
    std::vector<std::vector<double>> coupling_;
    mutable std::mutex coupling_mtx_;  // protects coupling_ reads/writes

    void validate_site_idx(int idx) const;
};

} // namespace target
