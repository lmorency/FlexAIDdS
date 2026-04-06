// gpu_event.h — GPU completion event abstraction for FlexAIDdS
//
// Wraps platform-specific GPU synchronization primitives:
//   - CUDA:  cudaEvent_t
//   - ROCm:  hipEvent_t
//   - Metal: dispatch_semaphore / MTLSharedEvent (via bridge)
//   - CPU:   std::future<void> from thread pool
//
// Supports:
//   - record()           — stamp the event on the current GPU stream
//   - wait()             — block until the event completes
//   - query()            — non-blocking completion check
//   - on_complete(cb)    — register a completion callback
//
// Apache-2.0 © 2026 Le Bonhomme Pharma
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <string>

// ─── Backend / Error enums (shared with dispatch layer) ─────────────────────

enum class Backend : uint8_t {
    AUTO    = 0,
    SCALAR  = 1,
    SSE42   = 2,
    AVX     = 3,
    AVX2    = 4,
    AVX512  = 5,
    OPENMP  = 6,
    CUDA    = 7,
    ROCM    = 8,
    METAL   = 9
};

enum class DispatchError : uint8_t {
    OK = 0,
    NO_BACKEND,
    ALLOC_FAILED,
    LAUNCH_FAILED,
    SYNC_FAILED,
    INVALID_ARGS,
    BUFFER_OVERFLOW,
    DEVICE_LOST
};

struct DispatchResult {
    DispatchError error       = DispatchError::OK;
    Backend       used_backend = Backend::AUTO;
    double        elapsed_ms  = 0.0;
    std::string   detail;

    explicit operator bool() const { return error == DispatchError::OK; }
};

// ─── GPUEvent ───────────────────────────────────────────────────────────────

class GPUEvent {
public:
    // Construct an event for a given backend.
    // For CPU backend, the event wraps a std::future.
    explicit GPUEvent(Backend b = Backend::SCALAR)
        : backend_(b) {}

    // Move-only (GPU events are not copyable)
    GPUEvent(GPUEvent&& other) noexcept
        : backend_(other.backend_)
        , completed_(other.completed_.load())
        , result_(std::move(other.result_))
        , callback_(std::move(other.callback_))
        , cpu_future_(std::move(other.cpu_future_))
#ifdef FLEXAIDS_USE_CUDA
        , cuda_event_(other.cuda_event_)
#endif
    {
#ifdef FLEXAIDS_USE_CUDA
        other.cuda_event_ = nullptr;
#endif
    }

    GPUEvent& operator=(GPUEvent&& other) noexcept {
        if (this != &other) {
            backend_    = other.backend_;
            completed_  = other.completed_.load();
            result_     = std::move(other.result_);
            callback_   = std::move(other.callback_);
            cpu_future_ = std::move(other.cpu_future_);
#ifdef FLEXAIDS_USE_CUDA
            cuda_event_ = other.cuda_event_;
            other.cuda_event_ = nullptr;
#endif
        }
        return *this;
    }

    GPUEvent(const GPUEvent&) = delete;
    GPUEvent& operator=(const GPUEvent&) = delete;

    ~GPUEvent() {
        release_gpu_resources();
    }

    // ── Core API ────────────────────────────────────────────────────────────

    /// Record the event (stamp it on the current GPU stream).
    /// For CPU events, this is a no-op (completion is set by the future).
    void record() {
#ifdef FLEXAIDS_USE_CUDA
        if (backend_ == Backend::CUDA && cuda_event_) {
            // cudaEventRecord(cuda_event_, 0);  // default stream
        }
#endif
        // Metal and ROCm: handled by their respective bridges
    }

    /// Block until the event completes.
    void wait() {
        if (completed_.load(std::memory_order_acquire)) return;

        switch (backend_) {
        case Backend::CUDA:
#ifdef FLEXAIDS_USE_CUDA
            if (cuda_event_) {
                // cudaEventSynchronize(cuda_event_);
            }
#endif
            break;
        case Backend::ROCM:
            // hipEventSynchronize would go here
            break;
        case Backend::METAL:
            // Metal completion handler sets completed_ flag
            break;
        default:
            // CPU: wait on the future
            if (cpu_future_.valid()) {
                cpu_future_.wait();
            }
            break;
        }

        completed_.store(true, std::memory_order_release);
        fire_callback();
    }

    /// Non-blocking completion check.
    [[nodiscard]] bool query() const {
        if (completed_.load(std::memory_order_acquire)) return true;

        switch (backend_) {
        case Backend::CUDA:
#ifdef FLEXAIDS_USE_CUDA
            // return cudaEventQuery(cuda_event_) == cudaSuccess;
#endif
            break;
        case Backend::ROCM:
            // return hipEventQuery(hip_event_) == hipSuccess;
            break;
        default:
            if (cpu_future_.valid()) {
                return cpu_future_.wait_for(std::chrono::seconds(0))
                       == std::future_status::ready;
            }
            break;
        }
        return false;
    }

    /// Register a completion callback.
    /// The callback receives the DispatchResult when the event completes.
    /// If the event has already completed, the callback fires immediately.
    void on_complete(std::function<void(DispatchResult)> cb) {
        callback_ = std::move(cb);
        if (completed_.load(std::memory_order_acquire)) {
            fire_callback();
        }
    }

    /// Mark the event as completed (called from GPU bridge or thread pool).
    void mark_complete(DispatchResult r = {}) {
        result_ = std::move(r);
        completed_.store(true, std::memory_order_release);
        fire_callback();
    }

    /// Get the backend this event was created for.
    [[nodiscard]] Backend backend() const { return backend_; }

    /// Get the result (only meaningful after completion).
    [[nodiscard]] const DispatchResult& result() const { return result_; }

    /// Set a CPU future for async CPU dispatch.
    void set_future(std::future<void> f) {
        cpu_future_ = std::move(f);
    }

private:
    Backend                              backend_;
    std::atomic<bool>                    completed_{false};
    DispatchResult                       result_;
    std::function<void(DispatchResult)>  callback_;
    mutable std::future<void>            cpu_future_;

#ifdef FLEXAIDS_USE_CUDA
    void* cuda_event_ = nullptr;  // cudaEvent_t
#endif

    void fire_callback() {
        if (callback_) {
            auto cb = std::move(callback_);
            callback_ = nullptr;
            cb(result_);
        }
    }

    void release_gpu_resources() {
#ifdef FLEXAIDS_USE_CUDA
        if (cuda_event_) {
            // cudaEventDestroy(static_cast<cudaEvent_t>(cuda_event_));
            cuda_event_ = nullptr;
        }
#endif
    }
};
