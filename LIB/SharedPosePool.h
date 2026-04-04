/// @file SharedPosePool.h
/// @brief Thread-safe best-pose sharing pool for parallel GA regions.
///
/// Maintains a fixed-capacity sorted pool of the best docking poses across
/// all parallel GA regions. Designed for high-throughput concurrent publish
/// from multiple threads with minimal lock contention.
///
/// For MPI: provides serialize/deserialize_merge for Allgather exchange.
///
/// Copyright 2026 Le Bonhomme Pharma
/// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <cassert>
#include <stdexcept>
#include <limits>
#include <type_traits>

/// @brief A single docking pose shared between parallel GA regions.
///
/// POD/trivially-copyable type — safe for memcpy-based MPI serialization.
/// Sorted by ascending energy in the pool (best = lowest energy first).
struct SharedPose {
    double energy;          ///< CF score in kcal/mol (lower = better).
    float  grid_coor[3];   ///< Grid-space coordinates (Å, region-independent).
    int    source_region;   ///< Which GA region produced this pose.
    int    generation;      ///< GA generation when this pose was published.

    SharedPose()
        : energy(std::numeric_limits<double>::max())
        , grid_coor{0.0f, 0.0f, 0.0f}
        , source_region(-1)
        , generation(-1) {}

    SharedPose(double e, float x, float y, float z, int region, int gen)
        : energy(e), grid_coor{x, y, z}, source_region(region), generation(gen) {}
};

static_assert(std::is_trivially_copyable_v<SharedPose>,
    "SharedPose must be trivially copyable for MPI serialization");

/// @class SharedPosePool
/// @brief Thread-safe fixed-capacity pool of best docking poses.
///
/// Maintains an ascending-sorted array of the best (lowest energy) poses.
/// When the pool is full, new poses are only accepted if they are better
/// than the current worst (highest energy) pose.
///
/// Thread safety: all public methods are mutex-protected.
///
/// @invariant pool_.size() == capacity_ (always pre-allocated)
/// @invariant 0 <= used_ <= capacity_
/// @invariant pool_[0..used_-1] is sorted by ascending energy
class SharedPosePool {
public:
    /// Construct a pool with the given capacity.
    /// @param pool_size Maximum number of poses to retain (must be > 0).
    /// @throws std::invalid_argument if pool_size <= 0.
    explicit SharedPosePool(int pool_size = 256);

    // Non-copyable (owns mutex), movable.
    SharedPosePool(const SharedPosePool&) = delete;
    SharedPosePool& operator=(const SharedPosePool&) = delete;
    SharedPosePool(SharedPosePool&&) noexcept;
    SharedPosePool& operator=(SharedPosePool&&) noexcept;

    /// Publish a pose into the pool.
    ///
    /// If the pool is not full, the pose is inserted in sorted position.
    /// If the pool is full, the pose replaces the worst entry only if
    /// it has strictly lower energy.
    ///
    /// NaN energies are rejected silently.
    ///
    /// @param pose The pose to publish.
    /// @return true if the pose was inserted, false if rejected.
    bool publish(const SharedPose& pose);

    /// Get the top @p k poses from the pool (best = lowest energy first).
    ///
    /// Returns a thread-safe snapshot. If k > count(), returns all available.
    /// @param k Maximum number of poses to return (clamped to [0, count()]).
    std::vector<SharedPose> get_top(int k) const;

    /// Pool capacity (maximum number of poses).
    int capacity() const noexcept { return capacity_; }

    /// Current number of poses in the pool.
    int count() const noexcept;

    /// Whether the pool has reached its capacity.
    bool is_full() const noexcept;

    /// Energy of the best (lowest) pose, or +inf if empty.
    double best_energy() const noexcept;

    /// Energy of the worst (highest) pose in the pool, or +inf if empty.
    double worst_energy() const noexcept;

    // ── Serialization for MPI transport ───────────────────────────────

    /// Serialize the pool contents into a byte buffer.
    /// Format: [int32 count][SharedPose × count]
    std::vector<char> serialize() const;

    /// Deserialize a remote pool's buffer and merge into this pool.
    ///
    /// Invalid or truncated buffers are silently ignored (no crash).
    /// Each remote pose is published individually (respecting capacity).
    ///
    /// @param data Pointer to serialized buffer.
    /// @param len  Length of the buffer in bytes.
    void deserialize_merge(const char* data, size_t len);

    /// Clear all poses from the pool. Thread-safe.
    void clear();

private:
    int capacity_;                  ///< Maximum pool size.
    std::vector<SharedPose> pool_;  ///< Pre-allocated pose storage.
    int used_ = 0;                  ///< Number of valid entries in pool_[0..used_-1].
    mutable std::mutex mtx_;        ///< Protects pool_ and used_.

    /// Insert a pose into the sorted pool (caller must hold mtx_).
    /// Uses binary search for O(log n) insertion point lookup.
    /// @pre used_ < capacity_
    void insert_sorted_unlocked(const SharedPose& pose);
};
