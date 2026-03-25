// SharedPosePool.h — Lock-free async best-pose sharing between parallel GA regions
#pragma once

#include <vector>
#include <atomic>
#include <algorithm>
#include <cstring>
#include <mutex>

struct SharedPose {
    double energy;          // CF score (lower = better)
    float  grid_coor[3];   // grid point coordinates (region-independent)
    int    source_region;   // which region produced this pose
    int    generation;      // GA generation when published

    SharedPose() : energy(1e30), grid_coor{0,0,0}, source_region(-1), generation(-1) {}
};

// Thread-safe pool of best poses shared across parallel GA regions.
// Uses a sorted fixed-size pool with mutex protection.
// For MPI: serialize/deserialize for Allgather exchange.
class SharedPosePool {
public:
    explicit SharedPosePool(int pool_size = 256);

    // Publish a pose: inserts if better than worst in pool.
    // Thread-safe (mutex-protected).
    void publish(const SharedPose& pose);

    // Get top K poses from pool (by ascending energy = best first).
    // Thread-safe snapshot.
    std::vector<SharedPose> get_top(int k) const;

    // Get pool size
    int capacity() const { return pool_size_; }
    int count() const;

    // Serialization for MPI transport
    std::vector<char> serialize() const;
    void deserialize_merge(const char* data, size_t len);

    // Clear pool
    void clear();

private:
    int pool_size_;
    std::vector<SharedPose> pool_;
    int used_;  // how many slots are filled
    mutable std::mutex mtx_;

    // Insert maintaining sorted order (ascending energy)
    void insert_sorted(const SharedPose& pose);
};
