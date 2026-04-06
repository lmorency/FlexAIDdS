// gpu_buffer.h — RAII wrapper for GPU memory allocations
//
// GPUBuffer<T> owns a device-side allocation and automatically frees it on
// destruction.  Non-copyable, move-only.  Dispatches to cudaMalloc/hipMalloc
// based on the Backend tag supplied at construction time.
//
// Metal buffers are NOT managed here because MTLBuffer has its own ARC/MRC
// lifecycle via Objective-C++.  Callers that need Metal buffers should
// continue using the existing bridge pattern (ShannonMetalBridge, etc.).
//
// Usage:
//   GPUBuffer<float> buf(1024, GPUBackend::CUDA);
//   buf.upload(host_ptr, 1024);
//   kernel<<<...>>>(buf.data(), ...);
//   buf.download(host_ptr, 1024);
//   // freed automatically when buf goes out of scope
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "flexaid_exception.h"
#include <cstddef>
#include <string>
#include <utility>

#ifdef FLEXAIDS_USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef FLEXAIDS_USE_ROCM
#include <hip/hip_runtime.h>
#endif

// Backend tag for GPUBuffer allocation dispatch.
// Kept separate from the future UnifiedHardwareDispatch Backend enum so this
// header remains self-contained and lightweight.
enum class GPUBackend : unsigned char {
    CPU   = 0,
    CUDA  = 1,
    ROCm  = 2
};

template<typename T>
class GPUBuffer {
public:
    // Default-construct an empty (null) buffer.
    GPUBuffer() = default;

    // Allocate `count` elements on the given backend.
    // Throws FlexAIDException on allocation failure.
    GPUBuffer(std::size_t count, GPUBackend backend)
        : count_(count), backend_(backend)
    {
        if (count_ == 0) return;
        const std::size_t bytes = count_ * sizeof(T);

        switch (backend_) {
#ifdef FLEXAIDS_USE_CUDA
        case GPUBackend::CUDA: {
            cudaError_t err = cudaMalloc(&ptr_, bytes);
            if (err != cudaSuccess) {
                throw FlexAIDException(
                    std::string("GPUBuffer CUDA alloc failed (") +
                    std::to_string(bytes) + " bytes): " +
                    cudaGetErrorString(err));
            }
            break;
        }
#endif
#ifdef FLEXAIDS_USE_ROCM
        case GPUBackend::ROCm: {
            hipError_t err = hipMalloc(&ptr_, bytes);
            if (err != hipSuccess) {
                throw FlexAIDException(
                    std::string("GPUBuffer ROCm alloc failed (") +
                    std::to_string(bytes) + " bytes): " +
                    hipGetErrorString(err));
            }
            break;
        }
#endif
        case GPUBackend::CPU:
            ptr_ = std::malloc(bytes);
            if (!ptr_) {
                throw FlexAIDException(
                    "GPUBuffer CPU alloc failed (" +
                    std::to_string(bytes) + " bytes)");
            }
            break;

        default:
            throw FlexAIDException(
                "GPUBuffer: unsupported backend " +
                std::to_string(static_cast<int>(backend_)));
        }
    }

    // Destructor — frees device memory.
    ~GPUBuffer() { free_internal(); }

    // Move constructor.
    GPUBuffer(GPUBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_), backend_(other.backend_)
    {
        other.ptr_   = nullptr;
        other.count_ = 0;
    }

    // Move assignment.
    GPUBuffer& operator=(GPUBuffer&& other) noexcept {
        if (this != &other) {
            free_internal();
            ptr_     = other.ptr_;
            count_   = other.count_;
            backend_ = other.backend_;
            other.ptr_   = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    // Non-copyable.
    GPUBuffer(const GPUBuffer&)            = delete;
    GPUBuffer& operator=(const GPUBuffer&) = delete;

    // ── Data transfer ────────────────────────────────────────────────────────

    // Upload `count` elements from host memory to device.
    // Throws FlexAIDException on failure.
    void upload(const T* host_data, std::size_t count) {
        if (!ptr_ || count == 0) return;
        const std::size_t bytes = count * sizeof(T);

        switch (backend_) {
#ifdef FLEXAIDS_USE_CUDA
        case GPUBackend::CUDA: {
            cudaError_t err = cudaMemcpy(ptr_, host_data, bytes,
                                         cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                throw FlexAIDException(
                    std::string("GPUBuffer CUDA upload failed: ") +
                    cudaGetErrorString(err));
            }
            break;
        }
#endif
#ifdef FLEXAIDS_USE_ROCM
        case GPUBackend::ROCm: {
            hipError_t err = hipMemcpy(ptr_, host_data, bytes,
                                       hipMemcpyHostToDevice);
            if (err != hipSuccess) {
                throw FlexAIDException(
                    std::string("GPUBuffer ROCm upload failed: ") +
                    hipGetErrorString(err));
            }
            break;
        }
#endif
        case GPUBackend::CPU:
            std::memcpy(ptr_, host_data, bytes);
            break;

        default:
            break;
        }
    }

    // Download `count` elements from device to host memory.
    // Throws FlexAIDException on failure.
    void download(T* host_data, std::size_t count) const {
        if (!ptr_ || count == 0) return;
        const std::size_t bytes = count * sizeof(T);

        switch (backend_) {
#ifdef FLEXAIDS_USE_CUDA
        case GPUBackend::CUDA: {
            cudaError_t err = cudaMemcpy(host_data, ptr_, bytes,
                                         cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                throw FlexAIDException(
                    std::string("GPUBuffer CUDA download failed: ") +
                    cudaGetErrorString(err));
            }
            break;
        }
#endif
#ifdef FLEXAIDS_USE_ROCM
        case GPUBackend::ROCm: {
            hipError_t err = hipMemcpy(host_data, ptr_, bytes,
                                       hipMemcpyDeviceToHost);
            if (err != hipSuccess) {
                throw FlexAIDException(
                    std::string("GPUBuffer ROCm download failed: ") +
                    hipGetErrorString(err));
            }
            break;
        }
#endif
        case GPUBackend::CPU:
            std::memcpy(host_data, ptr_, bytes);
            break;

        default:
            break;
        }
    }

    // Zero the device buffer.
    void zero() {
        if (!ptr_ || count_ == 0) return;
        const std::size_t bytes = count_ * sizeof(T);

        switch (backend_) {
#ifdef FLEXAIDS_USE_CUDA
        case GPUBackend::CUDA:
            cudaMemset(ptr_, 0, bytes);
            break;
#endif
#ifdef FLEXAIDS_USE_ROCM
        case GPUBackend::ROCm:
            hipMemset(ptr_, 0, bytes);
            break;
#endif
        case GPUBackend::CPU:
            std::memset(ptr_, 0, bytes);
            break;
        default:
            break;
        }
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    T*          data()    noexcept { return static_cast<T*>(ptr_); }
    const T*    data()    const noexcept { return static_cast<const T*>(ptr_); }
    std::size_t size()    const noexcept { return count_; }
    bool        empty()   const noexcept { return ptr_ == nullptr || count_ == 0; }
    GPUBackend  backend() const noexcept { return backend_; }

    explicit operator bool() const noexcept { return ptr_ != nullptr; }

private:
    void free_internal() noexcept {
        if (!ptr_) return;

        switch (backend_) {
#ifdef FLEXAIDS_USE_CUDA
        case GPUBackend::CUDA:
            cudaFree(ptr_);
            break;
#endif
#ifdef FLEXAIDS_USE_ROCM
        case GPUBackend::ROCm:
            hipFree(ptr_);
            break;
#endif
        case GPUBackend::CPU:
            std::free(ptr_);
            break;
        default:
            break;
        }
        ptr_ = nullptr;
        count_ = 0;
    }

    void*       ptr_     = nullptr;
    std::size_t count_   = 0;
    GPUBackend  backend_ = GPUBackend::CPU;
};
