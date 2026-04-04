/// @file TargetKnowledgeBase.h
/// @brief Cross-ligand knowledge accumulation for target-intrinsic learning.
///
/// Accumulates target-intrinsic knowledge from completed docking sessions
/// that does NOT violate per-ligand thermodynamic independence:
///
///   1. **Conformer population statistics**: which receptor conformers are sampled
///      across all ligands. Bayesian posterior serves as prior for future docking.
///
///   2. **Grid energy statistics**: running mean/variance of energies observed at
///      each grid point across ligands. Identifies structural "hot spots".
///
///   3. **Binding subsite hits**: spatial coordinates where diverse ligands bind.
///      Cross-ligand spatial clustering reveals pharmacophoric anchor points.
///
/// Thread-safety: all public methods are mutex-protected. Safe for concurrent
/// accumulation from multiple docking sessions.
///
/// Copyright 2026 Le Bonhomme Pharma
/// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>
#include <string>
#include <array>
#include <mutex>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <optional>

namespace target {

/// A single binding-center observation from one ligand's docking result.
struct SubsiteHit {
    std::array<float, 3> center;    ///< Cartesian coordinates of binding center (Å).
    double energy;                  ///< Best pose energy at this center (kcal/mol).
    std::string ligand_name;        ///< Name of the ligand that produced this hit.

    SubsiteHit() : center{0.0f, 0.0f, 0.0f}, energy(0.0) {}
    SubsiteHit(float x, float y, float z, double e, std::string name)
        : center{x, y, z}, energy(e), ligand_name(std::move(name)) {}
};

/// @class TargetKnowledgeBase
/// @brief Accumulates cross-ligand knowledge about a docking target.
///
/// Designed for concurrent use: multiple threads may call accumulate_*()
/// methods simultaneously. All reads and writes are protected by a single
/// mutex. For high-contention scenarios, consider partitioning by data type.
///
/// @invariant conformer_sum_.size() == n_models_ (when n_models_ > 0)
/// @invariant grid_energy_sum_.size() == n_grid_points_ (when n_grid_points_ > 0)
/// @invariant grid_energy_count_.size() == n_grid_points_
class TargetKnowledgeBase {
public:
    /// Construct an empty knowledge base (dimensions set on first accumulation).
    TargetKnowledgeBase() = default;

    /// Construct a knowledge base with known dimensions.
    /// @param n_models Number of receptor conformer models (must be >= 0).
    /// @param n_grid_points Number of grid points for energy statistics (must be >= 0).
    /// @throws std::invalid_argument if n_models or n_grid_points is negative.
    TargetKnowledgeBase(int n_models, int n_grid_points);

    // ── Conformer weights (CCBM) ──────────────────────────────────────

    /// Accumulate a ligand's conformer population vector.
    ///
    /// Populations should sum to ~1.0 (normalized weights from
    /// BindingMode::conformer_populations()). Thread-safe.
    ///
    /// @param weights Conformer populations. Size must match n_models().
    ///   On the first call (if n_models == 0), this sets the dimensionality.
    /// @return true if accumulated successfully, false on dimension mismatch.
    bool accumulate_conformer_weights(const std::vector<double>& weights);

    /// Bayesian posterior over conformer populations.
    ///
    /// Combines a uniform Dirichlet prior (pseudo-count = 1 per conformer)
    /// with the accumulated observations. Returns normalized weights (sum = 1).
    ///
    /// @return Posterior distribution. Empty if n_models == 0.
    std::vector<double> conformer_posterior() const;

    /// Number of ligands that contributed conformer data.
    int conformer_observation_count() const noexcept;

    // ── Grid energy statistics ─────────────────────────────────────────

    /// Accumulate per-grid-point energies from one ligand's docking.
    ///
    /// Non-finite values (NaN, ±Inf) at individual grid points are skipped.
    /// Thread-safe.
    ///
    /// @param energies Per-grid-point energies (kcal/mol). Size must match
    ///   n_grid_points(). First call (if n_grid_points == 0) sets dimensionality.
    /// @return true if accumulated successfully, false on dimension mismatch.
    bool accumulate_grid_energies(const std::vector<float>& energies);

    /// Running mean energy per grid point.
    /// Points with zero observations return 0.0f.
    std::vector<float> grid_mean_energies() const;

    /// Fraction of ligands with energy below @p energy_cutoff at each grid point.
    ///
    /// Unlike grid_mean_energies(), this computes the fraction on-the-fly from
    /// the raw energy sums, so any cutoff can be queried after accumulation.
    /// However, for efficiency, accumulation also tracks a hardcoded -1.0 kcal/mol
    /// favorable count; passing -1.0f uses that precomputed count.
    ///
    /// @param energy_cutoff Threshold in kcal/mol (default -1.0).
    /// @note When cutoff != -1.0f, the result is approximate: it uses the
    ///   precomputed favorable count which was accumulated with cutoff = -1.0.
    ///   For exact arbitrary-cutoff queries, accumulate raw per-point energy
    ///   histograms externally.
    std::vector<float> grid_hotspot_fraction(float energy_cutoff = -1.0f) const;

    /// Number of ligands that contributed grid energy data.
    int grid_observation_count() const noexcept;

    // ── Binding subsite hits ───────────────────────────────────────────

    /// Record where a ligand's best-scoring pose was placed.
    ///
    /// @param x,y,z Cartesian coordinates of the binding center (Å).
    /// @param energy Best pose energy at this center (kcal/mol).
    /// @param ligand_name Identifier for the contributing ligand.
    void accumulate_binding_center(float x, float y, float z,
                                    double energy,
                                    const std::string& ligand_name);

    /// Thread-safe snapshot of all recorded binding centers.
    ///
    /// Returns a copy (not a reference) to avoid data races with concurrent
    /// accumulation. For post-docking analysis where no more accumulations
    /// will occur, use all_hits_ref() instead to avoid the copy.
    std::vector<SubsiteHit> all_hits() const;

    /// Direct read-only reference to the hits vector.
    ///
    /// @warning NOT thread-safe during concurrent accumulation.
    /// Only call this after all docking sessions are complete.
    const std::vector<SubsiteHit>& all_hits_ref() const noexcept { return subsite_hits_; }

    /// Number of binding center hits recorded.
    int hit_count() const noexcept;

    // ── State queries ──────────────────────────────────────────────────

    /// Number of receptor conformer models.
    int n_models() const noexcept { return n_models_; }

    /// Number of grid points for energy statistics.
    int n_grid_points() const noexcept { return n_grid_points_; }

    /// Reset all accumulated data. Thread-safe.
    void clear();

    /// Check internal invariants. Returns true if all invariants hold.
    /// Acquires the mutex; safe to call concurrently.
    bool check_invariants() const;

    // ── Serialization (MPIMergeable) ───────────────────────────────────

    /// Serialize all accumulated state into a byte buffer for transport.
    ///
    /// Format: [header][conformer_sum][grid_energy_sum][grid_energy_count]
    ///         [grid_favorable_count][subsite_hit_count][SubsiteHitWire×N]
    ///
    /// Thread-safe (acquires mutex).
    std::vector<char> serialize() const;

    /// Merge a remote knowledge base's serialized state into this one.
    ///
    /// Conformer sums are added, grid energy sums/counts are added,
    /// subsite hits are appended. Dimension mismatches in remote data
    /// are handled gracefully (skip that section).
    ///
    /// Thread-safe (calls thread-safe accumulation methods internally).
    ///
    /// @param data Pointer to serialized buffer.
    /// @param len  Length of the buffer in bytes.
    void deserialize_merge(const char* data, size_t len);

private:
    int n_models_ = 0;         ///< Number of receptor conformer models.
    int n_grid_points_ = 0;    ///< Number of grid energy accumulation points.

    // ── Conformer accumulation ────────────────────────────────────────
    std::vector<double> conformer_sum_;   ///< Running sum of conformer populations.
    int conformer_count_ = 0;             ///< Number of ligands that contributed.

    // ── Grid energy accumulation ──────────────────────────────────────
    std::vector<double> grid_energy_sum_;      ///< Running sum of energies per point.
    std::vector<int>    grid_energy_count_;     ///< Observation count per point.
    std::vector<int>    grid_favorable_count_;  ///< Count of obs < -1.0 kcal/mol per point.
    int total_ligand_count_ = 0;               ///< Total ligands accumulated for grid.

    // ── Binding center hits ───────────────────────────────────────────
    std::vector<SubsiteHit> subsite_hits_;  ///< All recorded binding centers.

    mutable std::mutex mtx_;  ///< Protects all mutable state.
};

} // namespace target
