// async_dispatch.h — Async GPU/CPU dispatch with thread pool and GPUEvent
//
// Provides AsyncDispatcher: a lightweight task submission layer that returns
// GPUEvent completion tokens.  GPU work is submitted to the appropriate
// stream/command-buffer; CPU work is offloaded to a bounded thread pool.
//
// Usage:
//   auto& disp = AsyncDispatcher::instance();
//
//   // Submit GPU work (returns immediately, GPU runs asynchronously)
//   GPUEvent ev = disp.submit_gpu(EventBackend::CUDA, [&](cudaStream_t s) {
//       my_kernel<<<grid, block, 0, s>>>(args);
//   });
//
//   // Submit CPU work (runs on thread pool)
//   GPUEvent ev2 = disp.submit_cpu([&] {
//       heavy_cpu_computation(data, n);
//   });
//
//   // ... do other work ...
//   ev.wait();
//   ev2.wait();
//
// The thread pool is sized to hardware_concurrency / 2 (leaving cores for
// OpenMP parallel regions in the GA).
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gpu_event.h"

#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <memory>
#include <algorithm>

#ifdef FLEXAIDS_USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef FLEXAIDS_USE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace flexaids {

// ─── Thread pool (internal) ────────────────────────────────────────────────

namespace detail {

class ThreadPool {
public:
    explicit ThreadPool(int n_threads)
        : stop_(false)
    {
        n_threads = std::max(1, n_threads);
        workers_.reserve(n_threads);
        for (int i = 0; i < n_threads; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_)
            if (t.joinable()) t.join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Enqueue a task.  Returns immediately.
    void enqueue(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            tasks_.push(std::move(task));
        }
        cv_.notify_one();
    }

private:
    void worker_loop() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait(lk, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool stop_;
};

}  // namespace detail

// ─── AsyncDispatcher ───────────────────────────────────────────────────────

class AsyncDispatcher {
public:
    static AsyncDispatcher& instance() {
        static AsyncDispatcher disp;
        return disp;
    }

    // ── CPU async submission ────────────────────────────────────────────

    // Submit a CPU task to the thread pool.
    // Returns a shared completion token that the caller can wait on.
    // Internally uses a shared atomic flag so both the pool thread
    // (producer) and the caller (consumer) reference the same state.
    std::shared_ptr<GPUEvent> submit_cpu(std::function<void()> task) {
        // Metal backend uses manual signal() — not auto-complete like CPU.
        auto ev = std::make_shared<GPUEvent>(EventBackend::METAL);

        pool_.enqueue([ev, fn = std::move(task)]() mutable {
            fn();
            ev->signal();
        });

        return ev;
    }

    // ── GPU async submission (CUDA) ─────────────────────────────────────

#ifdef FLEXAIDS_USE_CUDA
    // Submit CUDA work on the default stream (or caller-provided stream).
    // The callable receives the stream to launch kernels on.
    // Returns a GPUEvent that completes when the stream reaches the event.
    GPUEvent submit_cuda(std::function<void(cudaStream_t)> kernel_launcher,
                         cudaStream_t stream = nullptr) {
        GPUEvent ev(EventBackend::CUDA);
        kernel_launcher(stream);
        ev.record(stream);
        return ev;
    }
#endif

    // ── GPU async submission (ROCm/HIP) ─────────────────────────────────

#ifdef FLEXAIDS_USE_ROCM
    // Submit HIP work on the default stream (or caller-provided stream).
    GPUEvent submit_rocm(std::function<void(hipStream_t)> kernel_launcher,
                         hipStream_t stream = nullptr) {
        GPUEvent ev(EventBackend::ROCM);
        kernel_launcher(stream);
        ev.record(stream);
        return ev;
    }
#endif

    // ── GPU async submission (Metal) ────────────────────────────────────

    // Metal completion is handled via command-buffer completion handlers.
    // The caller is responsible for calling ev.signal() in the completion
    // handler of the MTLCommandBuffer.
    GPUEvent create_metal_event() {
        return GPUEvent(EventBackend::METAL);
    }

    // ── Convenience: auto-dispatch ──────────────────────────────────────

    // Submit GPU work and fall back to CPU thread pool if no GPU callable
    // is provided.  GPU paths return a stack GPUEvent; CPU path returns a
    // shared_ptr<GPUEvent> (caller should call ->wait()).
    //
    // Prefer the typed submit_cuda / submit_rocm / submit_cpu methods
    // directly for clearer ownership semantics.

    // ── Pool info ───────────────────────────────────────────────────────

    int pool_size() const noexcept { return pool_threads_; }

private:
    AsyncDispatcher()
        : pool_threads_(std::max(1, static_cast<int>(
              std::thread::hardware_concurrency()) / 2))
        , pool_(pool_threads_)
    {}

    AsyncDispatcher(const AsyncDispatcher&) = delete;
    AsyncDispatcher& operator=(const AsyncDispatcher&) = delete;

    int pool_threads_;
    detail::ThreadPool pool_;
};

}  // namespace flexaids
