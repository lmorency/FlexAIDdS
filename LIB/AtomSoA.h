// AtomSoA.h — Structure-of-Arrays layout for atom coordinate data
//
// Provides SoA (Structure of Arrays) views of atom data for optimal
// SIMD vectorization. AVX-512 requires contiguous, aligned memory
// for efficient 512-bit vector loads. AoS→SoA conversion at batch
// boundary enables inner-loop vectorization.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstring>

// Forward-declare to avoid circular include with flexaid.h
struct atom_struct;

namespace atom_soa {

// 64-byte aligned allocator for AVX-512 compatibility
template<typename T>
struct AlignedAllocator {
    using value_type = T;
    static constexpr std::size_t alignment = 64;

    AlignedAllocator() noexcept = default;
    template<typename U>
    AlignedAllocator(const AlignedAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        void* ptr = nullptr;
#ifdef _WIN32
        ptr = _aligned_malloc(n * sizeof(T), alignment);
#else
        if (posix_memalign(&ptr, alignment, n * sizeof(T)) != 0)
            ptr = nullptr;
#endif
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, std::size_t) noexcept {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }
};

template<typename T, typename U>
bool operator==(const AlignedAllocator<T>&, const AlignedAllocator<U>&) { return true; }
template<typename T, typename U>
bool operator!=(const AlignedAllocator<T>&, const AlignedAllocator<U>&) { return false; }

using AlignedFloatVec  = std::vector<float,   AlignedAllocator<float>>;
using AlignedUint8Vec  = std::vector<uint8_t, AlignedAllocator<uint8_t>>;

// SoA representation of atom arrays for SIMD-friendly access
struct AtomArrays {
    AlignedFloatVec x;        // x coordinates
    AlignedFloatVec y;        // y coordinates
    AlignedFloatVec z;        // z coordinates
    AlignedFloatVec radius;   // atomic radii
    AlignedFloatVec charge;   // partial charges (RESP when available)
    AlignedUint8Vec type256;  // 256-class atom types
    int count = 0;

    void resize(int n) {
        count = n;
        x.resize(n, 0.0f);
        y.resize(n, 0.0f);
        z.resize(n, 0.0f);
        radius.resize(n, 0.0f);
        charge.resize(n, 0.0f);
        type256.resize(n, 0);
    }

    // Convert from AoS (atom_struct array) to SoA
    void from_aos(const atom_struct* atoms, int n) {
        resize(n);
        for (int i = 0; i < n; ++i) {
            x[i]       = atoms[i].coor[0];
            y[i]       = atoms[i].coor[1];
            z[i]       = atoms[i].coor[2];
            radius[i]  = atoms[i].radius;
            charge[i]  = atoms[i].has_resp ? atoms[i].resp_charge : atoms[i].charge;
            type256[i] = atoms[i].type256;
        }
    }

    // Copy coordinates back to AoS (for modified positions)
    void to_aos(atom_struct* atoms, int n) const {
        int lim = (n < count) ? n : count;
        for (int i = 0; i < lim; ++i) {
            atoms[i].coor[0] = x[i];
            atoms[i].coor[1] = y[i];
            atoms[i].coor[2] = z[i];
        }
    }
};

// Compute squared distance between atom i in SoA array A and atom j in SoA array B.
// Scalar fallback — AVX-512 batch version in VoronoiCFBatch_SoA.h.
inline float distance2(const AtomArrays& a, int i, const AtomArrays& b, int j) {
    float dx = a.x[i] - b.x[j];
    float dy = a.y[i] - b.y[j];
    float dz = a.z[i] - b.z[j];
    return dx * dx + dy * dy + dz * dz;
}

} // namespace atom_soa
