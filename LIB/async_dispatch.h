// async_dispatch.h — Asynchronous GPU/CPU dispatch with thread pool
//
// Provides AsyncDispatcher for submitting kernel work asynchronously:
//   - GPU kernels: launches on GPU stream, returns GPUEvent for tracking
//   - CPU fallback: submits to internal thread pool, returns GPUEvent wrapping future
//   - Batching: multiple independent dispatches can be submitted and waited together
//
// Usage:
//   auto& dispatcher = AsyncDispatcher::instance();
//   GPUEvent event = dispatcher.submit_cpu([&]() {
//       cpu_eval_batch(ctx, pop_size, n_genes, genes, com, wal, sas);
//   });
//   // ... do other work ...
//   event.wait();
//
// Apache-2.0 © 2026 Le Bonhomme Pharma
#pragma once

#include "gpu_event.h"

#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// ─── Thread Pool (internal) ─────────────────────────────────────────────────

namespace detail {

class ThreadPool {
public:
    explicit ThreadPool(int n_threads = 0) {
        if (n_threads <= 0) {
            n_threads = static_cast<int>(std::thread::hardware_concurrency());
            if (n_threads <= 0) n_threads = 2;
        }
        for (int i = 0; i < n_threads; ++i) {
            workers_.emplace_back([this]() { worker_loop(); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            shutdown_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
    }

    // Non-copyable, non-movable
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /// Submit a task and get a future for completion tracking.
    std::future<void> submit(std::function<void()> task) {
        auto promise = std::make_shared<std::promise<void>>();
        auto future  = promise->get_future();

        {
            std::lock_guard<std::mutex> lock(mtx_);
            queue_.emplace_back([task = std::move(task),
                                 promise = std::move(promise)]() {
                try {
                    task();
                    promise->set_value();
                } catch (...) {
                    promise->set_exception(std::current_exception());
                }
            });
        }
        cv_.notify_one();
        return future;
    }

    /// Number of worker threads.
    [[nodiscard]] int size() const {
        return static_cast<int>(workers_.size());
    }

private:
    std::vector<std::thread>         workers_;
    std::deque<std::function<void()>> queue_;
    std::mutex                        mtx_;
    std::condition_variable           cv_;
    bool                              shutdown_ = false;

    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait(lock, [this]() {
                    return shutdown_ || !queue_.empty();
                });
                if (shutdown_ && queue_.empty()) return;
                task = std::move(queue_.front());
                queue_.pop_front();
            }
            task();
        }
    }
};

}  // namespace detail

// ─── AsyncDispatcher ────────────────────────────────────────────────────────

class AsyncDispatcher {
public:
    /// Meyers singleton.
    static AsyncDispatcher& instance() {
        static AsyncDispatcher inst;
        return inst;
    }

    // Non-copyable
    AsyncDispatcher(const AsyncDispatcher&) = delete;
    AsyncDispatcher& operator=(const AsyncDispatcher&) = delete;

    /// Submit a CPU task asynchronously.
    /// Returns a GPUEvent that can be waited on or queried.
    GPUEvent submit_cpu(std::function<void()> task) {
        GPUEvent event(Backend::SCALAR);
        auto future = pool_.submit([event_ptr = &event,
                                     task = std::move(task)]() {
            task();
        });
        event.set_future(std::move(future));
        return event;
    }

    /// Submit a task that returns a DispatchResult.
    /// The result is stored in the GPUEvent for retrieval.
    GPUEvent submit(Backend backend, std::function<DispatchResult()> task) {
        GPUEvent event(backend);

        if (backend == Backend::CUDA || backend == Backend::ROCM ||
            backend == Backend::METAL) {
            // For GPU backends, the task itself handles GPU submission.
            // The future wraps the host-side orchestration.
            auto future = pool_.submit([task = std::move(task), &event]() {
                DispatchResult r = task();
                event.mark_complete(std::move(r));
            });
            event.set_future(std::move(future));
        } else {
            // CPU backends: run task on the thread pool
            auto future = pool_.submit([task = std::move(task), &event]() {
                DispatchResult r = task();
                event.mark_complete(std::move(r));
            });
            event.set_future(std::move(future));
        }

        return event;
    }

    /// Submit multiple independent tasks and return events for each.
    std::vector<GPUEvent> submit_batch(
        Backend backend,
        const std::vector<std::function<DispatchResult()>>& tasks)
    {
        std::vector<GPUEvent> events;
        events.reserve(tasks.size());
        for (const auto& task : tasks) {
            events.push_back(submit(backend, task));
        }
        return events;
    }

    /// Wait for all events to complete.
    static void wait_all(std::vector<GPUEvent>& events) {
        for (auto& e : events) {
            e.wait();
        }
    }

    /// Query if all events are complete (non-blocking).
    [[nodiscard]] static bool all_complete(
        const std::vector<GPUEvent>& events)
    {
        for (const auto& e : events) {
            if (!e.query()) return false;
        }
        return true;
    }

    /// Number of worker threads in the pool.
    [[nodiscard]] int pool_size() const { return pool_.size(); }

private:
    AsyncDispatcher() : pool_(2) {}  // 2 threads for async dispatch

    detail::ThreadPool pool_;
};
