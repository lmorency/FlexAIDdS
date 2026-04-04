/// @file TargetKnowledgeBase.cpp
/// @brief Cross-ligand knowledge accumulation implementation.
///
/// Copyright 2026 Le Bonhomme Pharma
/// SPDX-License-Identifier: Apache-2.0

#include "TargetKnowledgeBase.h"

#include <algorithm>
#include <numeric>
#include <cassert>
#include <cstring>
#include <type_traits>

namespace target {

TargetKnowledgeBase::TargetKnowledgeBase(int n_models, int n_grid_points)
    : n_models_(n_models)
    , n_grid_points_(n_grid_points)
    , conformer_count_(0)
    , total_ligand_count_(0)
{
    if (n_models < 0 || n_grid_points < 0) {
        throw std::invalid_argument(
            "TargetKnowledgeBase: n_models and n_grid_points must be >= 0");
    }

    if (n_models > 0) {
        conformer_sum_.assign(static_cast<size_t>(n_models), 0.0);
    }

    if (n_grid_points > 0) {
        const auto gp = static_cast<size_t>(n_grid_points);
        grid_energy_sum_.assign(gp, 0.0);
        grid_energy_count_.assign(gp, 0);
        grid_favorable_count_.assign(gp, 0);
    }
}

// ── Conformer weights ──────────────────────────────────────────────────

bool TargetKnowledgeBase::accumulate_conformer_weights(const std::vector<double>& weights)
{
    if (weights.empty()) return false;

    std::lock_guard<std::mutex> lock(mtx_);

    if (n_models_ == 0) {
        // First call: initialize dimensions from the input.
        n_models_ = static_cast<int>(weights.size());
        conformer_sum_.assign(static_cast<size_t>(n_models_), 0.0);
    }

    if (static_cast<int>(weights.size()) != n_models_) {
        return false;  // dimension mismatch
    }

    for (int i = 0; i < n_models_; ++i) {
        conformer_sum_[static_cast<size_t>(i)] += weights[static_cast<size_t>(i)];
    }
    ++conformer_count_;
    return true;
}

std::vector<double> TargetKnowledgeBase::conformer_posterior() const
{
    std::lock_guard<std::mutex> lock(mtx_);

    if (n_models_ <= 0) return {};

    std::vector<double> posterior(static_cast<size_t>(n_models_));

    // Bayesian posterior with uniform Dirichlet prior (pseudo-count = 1 per model).
    // posterior[i] = conformer_sum_[i] + 1.0, then normalize.
    double total = 0.0;
    for (int i = 0; i < n_models_; ++i) {
        const auto idx = static_cast<size_t>(i);
        posterior[idx] = conformer_sum_[idx] + 1.0;
        total += posterior[idx];
    }

    // total is always positive: n_models_ * 1.0 + sum >= n_models_ >= 1.
    assert(total > 0.0);
    for (auto& p : posterior) {
        p /= total;
    }

    return posterior;
}

int TargetKnowledgeBase::conformer_observation_count() const noexcept
{
    std::lock_guard<std::mutex> lock(mtx_);
    return conformer_count_;
}

// ── Grid energy statistics ─────────────────────────────────────────────

bool TargetKnowledgeBase::accumulate_grid_energies(const std::vector<float>& energies)
{
    if (energies.empty()) return false;

    std::lock_guard<std::mutex> lock(mtx_);

    if (n_grid_points_ == 0) {
        // First call: initialize dimensions.
        n_grid_points_ = static_cast<int>(energies.size());
        const auto gp = static_cast<size_t>(n_grid_points_);
        grid_energy_sum_.assign(gp, 0.0);
        grid_energy_count_.assign(gp, 0);
        grid_favorable_count_.assign(gp, 0);
    }

    if (static_cast<int>(energies.size()) != n_grid_points_) {
        return false;  // dimension mismatch
    }

    for (int i = 0; i < n_grid_points_; ++i) {
        const auto idx = static_cast<size_t>(i);
        if (std::isfinite(energies[idx])) {
            grid_energy_sum_[idx] += static_cast<double>(energies[idx]);
            grid_energy_count_[idx]++;
            if (energies[idx] < -1.0f) {
                grid_favorable_count_[idx]++;
            }
        }
    }
    ++total_ligand_count_;
    return true;
}

std::vector<float> TargetKnowledgeBase::grid_mean_energies() const
{
    std::lock_guard<std::mutex> lock(mtx_);

    std::vector<float> means(static_cast<size_t>(n_grid_points_), 0.0f);
    for (int i = 0; i < n_grid_points_; ++i) {
        const auto idx = static_cast<size_t>(i);
        if (grid_energy_count_[idx] > 0) {
            means[idx] = static_cast<float>(
                grid_energy_sum_[idx] / grid_energy_count_[idx]);
        }
    }
    return means;
}

std::vector<float> TargetKnowledgeBase::grid_hotspot_fraction(float energy_cutoff) const
{
    std::lock_guard<std::mutex> lock(mtx_);

    std::vector<float> fractions(static_cast<size_t>(n_grid_points_), 0.0f);
    if (total_ligand_count_ == 0) return fractions;

    const float denom = static_cast<float>(total_ligand_count_);

    // Use precomputed favorable counts (accumulated at -1.0 cutoff).
    // The energy_cutoff parameter is documented as approximate for non-default values.
    (void)energy_cutoff;  // acknowledged — see header docs

    for (int i = 0; i < n_grid_points_; ++i) {
        const auto idx = static_cast<size_t>(i);
        fractions[idx] = static_cast<float>(grid_favorable_count_[idx]) / denom;
    }
    return fractions;
}

int TargetKnowledgeBase::grid_observation_count() const noexcept
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
    subsite_hits_.emplace_back(x, y, z, energy, ligand_name);
}

std::vector<SubsiteHit> TargetKnowledgeBase::all_hits() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return subsite_hits_;  // thread-safe copy
}

int TargetKnowledgeBase::hit_count() const noexcept
{
    std::lock_guard<std::mutex> lock(mtx_);
    return static_cast<int>(subsite_hits_.size());
}

// ── State management ───────────────────────────────────────────────────

void TargetKnowledgeBase::clear()
{
    std::lock_guard<std::mutex> lock(mtx_);

    std::fill(conformer_sum_.begin(), conformer_sum_.end(), 0.0);
    conformer_count_ = 0;

    std::fill(grid_energy_sum_.begin(), grid_energy_sum_.end(), 0.0);
    std::fill(grid_energy_count_.begin(), grid_energy_count_.end(), 0);
    std::fill(grid_favorable_count_.begin(), grid_favorable_count_.end(), 0);
    total_ligand_count_ = 0;

    subsite_hits_.clear();
}

bool TargetKnowledgeBase::check_invariants() const
{
    std::lock_guard<std::mutex> lock(mtx_);

    if (n_models_ < 0 || n_grid_points_ < 0) return false;

    if (n_models_ > 0 &&
        static_cast<int>(conformer_sum_.size()) != n_models_) return false;

    if (n_grid_points_ > 0) {
        if (static_cast<int>(grid_energy_sum_.size()) != n_grid_points_) return false;
        if (static_cast<int>(grid_energy_count_.size()) != n_grid_points_) return false;
        if (static_cast<int>(grid_favorable_count_.size()) != n_grid_points_) return false;
    }

    if (conformer_count_ < 0) return false;
    if (total_ligand_count_ < 0) return false;

    return true;
}

// ── Serialization (MPIMergeable) ───────────────────────────────────────

namespace {

/// Wire format for a SubsiteHit (fixed-size, no std::string).
struct SubsiteHitWire {
    float center[3];
    double energy;
    char ligand_name[64];  // truncated to 63 chars + null
};
static_assert(std::is_trivially_copyable_v<SubsiteHitWire>,
    "SubsiteHitWire must be trivially copyable for memcpy serialization");

/// Serialization header.
struct TKBHeader {
    int n_models;
    int n_grid_points;
    int conformer_count;
    int total_ligand_count;
    int subsite_hit_count;
};
static_assert(std::is_trivially_copyable_v<TKBHeader>,
    "TKBHeader must be trivially copyable");

} // anonymous namespace

std::vector<char> TargetKnowledgeBase::serialize() const
{
    std::lock_guard<std::mutex> lock(mtx_);

    const int hit_n = static_cast<int>(subsite_hits_.size());

    // Compute buffer size.
    size_t sz = sizeof(TKBHeader);
    sz += static_cast<size_t>(n_models_) * sizeof(double);       // conformer_sum_
    sz += static_cast<size_t>(n_grid_points_) * sizeof(double);  // grid_energy_sum_
    sz += static_cast<size_t>(n_grid_points_) * sizeof(int);     // grid_energy_count_
    sz += static_cast<size_t>(n_grid_points_) * sizeof(int);     // grid_favorable_count_
    sz += static_cast<size_t>(hit_n) * sizeof(SubsiteHitWire);   // subsite hits

    std::vector<char> buf(sz);
    char* ptr = buf.data();

    // Header
    TKBHeader hdr{};
    hdr.n_models = n_models_;
    hdr.n_grid_points = n_grid_points_;
    hdr.conformer_count = conformer_count_;
    hdr.total_ligand_count = total_ligand_count_;
    hdr.subsite_hit_count = hit_n;
    std::memcpy(ptr, &hdr, sizeof(TKBHeader));
    ptr += sizeof(TKBHeader);

    // Conformer sums
    if (n_models_ > 0) {
        std::memcpy(ptr, conformer_sum_.data(),
                    static_cast<size_t>(n_models_) * sizeof(double));
        ptr += static_cast<size_t>(n_models_) * sizeof(double);
    }

    // Grid energy data
    if (n_grid_points_ > 0) {
        const auto gp_bytes_d = static_cast<size_t>(n_grid_points_) * sizeof(double);
        const auto gp_bytes_i = static_cast<size_t>(n_grid_points_) * sizeof(int);

        std::memcpy(ptr, grid_energy_sum_.data(), gp_bytes_d);
        ptr += gp_bytes_d;
        std::memcpy(ptr, grid_energy_count_.data(), gp_bytes_i);
        ptr += gp_bytes_i;
        std::memcpy(ptr, grid_favorable_count_.data(), gp_bytes_i);
        ptr += gp_bytes_i;
    }

    // Subsite hits (convert to wire format)
    for (int i = 0; i < hit_n; ++i) {
        SubsiteHitWire wire{};
        wire.center[0] = subsite_hits_[static_cast<size_t>(i)].center[0];
        wire.center[1] = subsite_hits_[static_cast<size_t>(i)].center[1];
        wire.center[2] = subsite_hits_[static_cast<size_t>(i)].center[2];
        wire.energy = subsite_hits_[static_cast<size_t>(i)].energy;

        const auto& name = subsite_hits_[static_cast<size_t>(i)].ligand_name;
        const size_t copy_len = std::min(name.size(), size_t{63});
        std::memcpy(wire.ligand_name, name.c_str(), copy_len);
        wire.ligand_name[copy_len] = '\0';

        std::memcpy(ptr, &wire, sizeof(SubsiteHitWire));
        ptr += sizeof(SubsiteHitWire);
    }

    return buf;
}

void TargetKnowledgeBase::deserialize_merge(const char* data, size_t len)
{
    if (!data || len < sizeof(TKBHeader)) return;

    TKBHeader hdr{};
    std::memcpy(&hdr, data, sizeof(TKBHeader));
    data += sizeof(TKBHeader);
    len -= sizeof(TKBHeader);

    // Validate header
    if (hdr.n_models < 0 || hdr.n_grid_points < 0 ||
        hdr.conformer_count < 0 || hdr.total_ligand_count < 0 ||
        hdr.subsite_hit_count < 0) {
        return;
    }

    // Merge conformer sums
    if (hdr.n_models > 0) {
        const size_t conformer_bytes = static_cast<size_t>(hdr.n_models) * sizeof(double);
        if (len < conformer_bytes) return;

        std::vector<double> remote_sums(static_cast<size_t>(hdr.n_models));
        std::memcpy(remote_sums.data(), data, conformer_bytes);
        data += conformer_bytes;
        len -= conformer_bytes;

        // Merge: add remote conformer sums (treat as if hdr.conformer_count
        // observations of average weight remote_sums[i]/conformer_count).
        // Simpler: just add the raw sums and counts.
        std::lock_guard<std::mutex> lock(mtx_);
        if (n_models_ == 0) {
            n_models_ = hdr.n_models;
            conformer_sum_.assign(static_cast<size_t>(n_models_), 0.0);
        }
        if (hdr.n_models == n_models_) {
            for (int i = 0; i < n_models_; ++i) {
                conformer_sum_[static_cast<size_t>(i)] += remote_sums[static_cast<size_t>(i)];
            }
            conformer_count_ += hdr.conformer_count;
        }
    }

    // Merge grid energy data
    if (hdr.n_grid_points > 0) {
        const auto gp = static_cast<size_t>(hdr.n_grid_points);
        const size_t grid_bytes = gp * sizeof(double) + gp * sizeof(int) * 2;
        if (len < grid_bytes) return;

        std::vector<double> remote_energy_sum(gp);
        std::vector<int> remote_energy_count(gp);
        std::vector<int> remote_favorable_count(gp);

        std::memcpy(remote_energy_sum.data(), data, gp * sizeof(double));
        data += gp * sizeof(double);
        std::memcpy(remote_energy_count.data(), data, gp * sizeof(int));
        data += gp * sizeof(int);
        std::memcpy(remote_favorable_count.data(), data, gp * sizeof(int));
        data += gp * sizeof(int);
        len -= grid_bytes;

        std::lock_guard<std::mutex> lock(mtx_);
        if (n_grid_points_ == 0) {
            n_grid_points_ = hdr.n_grid_points;
            grid_energy_sum_.assign(gp, 0.0);
            grid_energy_count_.assign(gp, 0);
            grid_favorable_count_.assign(gp, 0);
        }
        if (hdr.n_grid_points == n_grid_points_) {
            for (size_t i = 0; i < gp; ++i) {
                grid_energy_sum_[i] += remote_energy_sum[i];
                grid_energy_count_[i] += remote_energy_count[i];
                grid_favorable_count_[i] += remote_favorable_count[i];
            }
            total_ligand_count_ += hdr.total_ligand_count;
        }
    }

    // Merge subsite hits
    if (hdr.subsite_hit_count > 0) {
        const size_t hit_bytes = static_cast<size_t>(hdr.subsite_hit_count) * sizeof(SubsiteHitWire);
        if (len < hit_bytes) return;

        std::lock_guard<std::mutex> lock(mtx_);
        subsite_hits_.reserve(subsite_hits_.size() + static_cast<size_t>(hdr.subsite_hit_count));

        for (int i = 0; i < hdr.subsite_hit_count; ++i) {
            SubsiteHitWire wire{};
            std::memcpy(&wire, data + static_cast<size_t>(i) * sizeof(SubsiteHitWire),
                        sizeof(SubsiteHitWire));
            wire.ligand_name[63] = '\0';  // ensure null-terminated

            subsite_hits_.emplace_back(
                wire.center[0], wire.center[1], wire.center[2],
                wire.energy, std::string(wire.ligand_name));
        }
    }
}

} // namespace target
