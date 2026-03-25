// SharedPosePool.cpp — Lock-free async best-pose sharing
#include "SharedPosePool.h"
#include <cstring>

SharedPosePool::SharedPosePool(int pool_size)
    : pool_size_(pool_size), used_(0) {
    pool_.resize(pool_size);
}

void SharedPosePool::publish(const SharedPose& pose) {
    std::lock_guard<std::mutex> lock(mtx_);

    if (used_ < pool_size_) {
        // Pool not full: insert in sorted position
        insert_sorted(pose);
        return;
    }

    // Pool full: only insert if better than worst (last element)
    if (pose.energy < pool_[used_ - 1].energy) {
        pool_[used_ - 1] = pose;  // overwrite worst
        // Re-sort the last element into position
        for (int i = used_ - 1; i > 0; i--) {
            if (pool_[i].energy < pool_[i-1].energy) {
                std::swap(pool_[i], pool_[i-1]);
            } else {
                break;
            }
        }
    }
}

std::vector<SharedPose> SharedPosePool::get_top(int k) const {
    std::lock_guard<std::mutex> lock(mtx_);
    int n = std::min(k, used_);
    return std::vector<SharedPose>(pool_.begin(), pool_.begin() + n);
}

int SharedPosePool::count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return used_;
}

void SharedPosePool::clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    used_ = 0;
    for (auto& p : pool_) p = SharedPose();
}

void SharedPosePool::insert_sorted(const SharedPose& pose) {
    // Find insertion point (ascending energy order)
    int pos = used_;
    for (int i = 0; i < used_; i++) {
        if (pose.energy < pool_[i].energy) {
            pos = i;
            break;
        }
    }

    // Shift elements right
    if (used_ < pool_size_) {
        for (int i = used_; i > pos; i--) {
            pool_[i] = pool_[i-1];
        }
        pool_[pos] = pose;
        used_++;
    }
}

// ── Serialization for MPI ────────────────────────────────────────────────────

std::vector<char> SharedPosePool::serialize() const {
    std::lock_guard<std::mutex> lock(mtx_);

    // Format: [int used_][SharedPose × used_]
    size_t sz = sizeof(int) + used_ * sizeof(SharedPose);
    std::vector<char> buf(sz);
    char* ptr = buf.data();

    std::memcpy(ptr, &used_, sizeof(int));
    ptr += sizeof(int);

    if (used_ > 0) {
        std::memcpy(ptr, pool_.data(), used_ * sizeof(SharedPose));
    }

    return buf;
}

void SharedPosePool::deserialize_merge(const char* data, size_t len) {
    if (len < sizeof(int)) return;

    int remote_count;
    std::memcpy(&remote_count, data, sizeof(int));
    data += sizeof(int);

    if (len < sizeof(int) + remote_count * sizeof(SharedPose)) return;

    // Merge each remote pose into this pool
    for (int i = 0; i < remote_count; i++) {
        SharedPose pose;
        std::memcpy(&pose, data + i * sizeof(SharedPose), sizeof(SharedPose));
        publish(pose);  // publish handles thread safety
    }
}
