// GPUContextPool.h — Thread-safe singleton pool for GPU evaluation contexts
//
// GPU contexts (CudaEvalCtx, MetalEvalCtx) are expensive to create (~100ms).
// This pool manages a single context per GPU backend, allowing multiple
// concurrent GA instances to share the same device context safely.
//
// Usage:
//   auto& pool = GPUContextPool::instance();
//   auto handle = pool.acquire_cuda(n_atoms, n_types, ...);
//   // ... use handle.ctx ...
//   pool.release_cuda(handle);
//
// For ParallelDock (same system, different regions): all threads share one
// context since n_atoms and n_types are identical.
//
// For ParallelCampaign (different ligands): acquire() waits for exclusive
// access when dimensions change, serializing GPU rebuilds naturally.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>
#include <condition_variable>
#include <functional>

#ifdef FLEXAIDS_USE_CUDA
#include "cuda_eval.cuh"
#endif

#ifdef FLEXAIDS_USE_METAL
#include "metal_eval.h"
#endif

class GPUContextPool {
public:
    static GPUContextPool& instance() {
        static GPUContextPool pool;
        return pool;
    }

#ifdef FLEXAIDS_USE_CUDA
    struct CudaHandle {
        CudaEvalCtx* ctx;
    };

    // Acquire the shared CUDA context, rebuilding if dimensions changed.
    // Blocks if another thread is rebuilding.
    // init_fn is called to create/recreate the context when dimensions change.
    CudaHandle acquire_cuda(
        int n_atoms, int n_types,
        std::function<CudaEvalCtx*(void)> init_fn
    ) {
        std::unique_lock<std::mutex> lock(cuda_mtx_);
        // Wait if someone is rebuilding with different dimensions
        cuda_cv_.wait(lock, [&] {
            return !cuda_rebuilding_ || (cuda_natom_ == n_atoms && cuda_ntype_ == n_types);
        });

        if (!cuda_ctx_ || cuda_natom_ != n_atoms || cuda_ntype_ != n_types) {
            cuda_rebuilding_ = true;
            // Wait for all users to release before rebuilding
            cuda_cv_.wait(lock, [&] { return cuda_ref_count_ == 0; });

            if (cuda_ctx_) {
                cuda_eval_shutdown(cuda_ctx_);
                cuda_ctx_ = nullptr;
            }
            lock.unlock();
            auto* new_ctx = init_fn();
            lock.lock();

            cuda_ctx_   = new_ctx;
            cuda_natom_ = n_atoms;
            cuda_ntype_ = n_types;
            cuda_rebuilding_ = false;
            cuda_cv_.notify_all();
        }

        cuda_ref_count_++;
        return CudaHandle{cuda_ctx_};
    }

    void release_cuda(const CudaHandle& /*handle*/) {
        std::lock_guard<std::mutex> lock(cuda_mtx_);
        cuda_ref_count_--;
        if (cuda_ref_count_ == 0) {
            cuda_cv_.notify_all();
        }
    }
#endif

#ifdef FLEXAIDS_USE_METAL
    struct MetalHandle {
        MetalEvalCtx* ctx;
    };

    MetalHandle acquire_metal(
        int n_atoms, int n_types,
        std::function<MetalEvalCtx*(void)> init_fn
    ) {
        std::unique_lock<std::mutex> lock(metal_mtx_);
        metal_cv_.wait(lock, [&] {
            return !metal_rebuilding_ || (metal_natom_ == n_atoms && metal_ntype_ == n_types);
        });

        if (!metal_ctx_ || metal_natom_ != n_atoms || metal_ntype_ != n_types) {
            metal_rebuilding_ = true;
            metal_cv_.wait(lock, [&] { return metal_ref_count_ == 0; });

            if (metal_ctx_) {
                metal_eval_shutdown(metal_ctx_);
                metal_ctx_ = nullptr;
            }
            lock.unlock();
            auto* new_ctx = init_fn();
            lock.lock();

            metal_ctx_   = new_ctx;
            metal_natom_ = n_atoms;
            metal_ntype_ = n_types;
            metal_rebuilding_ = false;
            metal_cv_.notify_all();
        }

        metal_ref_count_++;
        return MetalHandle{metal_ctx_};
    }

    void release_metal(const MetalHandle& /*handle*/) {
        std::lock_guard<std::mutex> lock(metal_mtx_);
        metal_ref_count_--;
        if (metal_ref_count_ == 0) {
            metal_cv_.notify_all();
        }
    }
#endif

private:
    GPUContextPool() = default;
    ~GPUContextPool() {
#ifdef FLEXAIDS_USE_CUDA
        if (cuda_ctx_) cuda_eval_shutdown(cuda_ctx_);
#endif
#ifdef FLEXAIDS_USE_METAL
        if (metal_ctx_) metal_eval_shutdown(metal_ctx_);
#endif
    }

    GPUContextPool(const GPUContextPool&) = delete;
    GPUContextPool& operator=(const GPUContextPool&) = delete;

#ifdef FLEXAIDS_USE_CUDA
    std::mutex cuda_mtx_;
    std::condition_variable cuda_cv_;
    CudaEvalCtx* cuda_ctx_ = nullptr;
    int cuda_natom_ = 0;
    int cuda_ntype_ = 0;
    int cuda_ref_count_ = 0;
    bool cuda_rebuilding_ = false;
#endif

#ifdef FLEXAIDS_USE_METAL
    std::mutex metal_mtx_;
    std::condition_variable metal_cv_;
    MetalEvalCtx* metal_ctx_ = nullptr;
    int metal_natom_ = 0;
    int metal_ntype_ = 0;
    int metal_ref_count_ = 0;
    bool metal_rebuilding_ = false;
#endif
};
