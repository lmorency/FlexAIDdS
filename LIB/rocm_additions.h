// Copyright 2026 Le Bonhomme Pharma
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// ============================================================================
// rocm_additions.h
//
// Patch instructions for LIB/HardwareDispatch.h to add AMD ROCm/HIP support.
//
// Apply the diff-style additions below to their indicated locations in
// HardwareDispatch.h.  The new Backend enum value ROCM = 6 is inserted
// between CUDA (5) and AUTO (255); the new HardwareInfo fields are appended
// after the CUDA block.
//
// After patching, the priority order in AUTO selection becomes:
//   CUDA > ROCM > METAL > AVX512 > AVX2 > OPENMP > SCALAR
// ============================================================================

#pragma once

// ============================================================================
// 1.  Backend enum  (add inside `enum class Backend : uint8_t { … }`)
//
//     EXISTING (keep):
//         CUDA    = 5,
//         AUTO    = 255
//
//     CHANGE TO:
//         CUDA    = 5,
//         ROCM    = 6,   // AMD ROCm/HIP GPU
//         AUTO    = 255
// ============================================================================

// ============================================================================
// 2.  HardwareInfo struct  (add after the `has_metal` / `metal_max_buffer`
//     block, before the closing brace of HardwareInfo)
//
//     ADD:
// ============================================================================
//
//     // ── ROCm/HIP (AMD GPU) ──────────────────────────────────────────────
//     bool        has_rocm           = false;
//     std::string rocm_device_name;
//     int         rocm_compute_units = 0;   // CUs (MultiProcessor count)
//     std::size_t rocm_global_mem    = 0;   // bytes
//     int         rocm_gcn_arch      = 0;   // numeric GCN arch, e.g. 908 → gfx908
//     int         rocm_wavefront     = 64;  // wavefront size (usually 64 on GCN/CDNA)
//
// ============================================================================
// 3.  AUTO backend selection  (in whatever function resolves Backend::AUTO)
//
//     Insert the ROCM check between CUDA and METAL:
//
//     if (hw.has_cuda)   return Backend::CUDA;
//  ++ if (hw.has_rocm)   return Backend::ROCM;
//     if (hw.has_metal)  return Backend::METAL;
//     if (hw.has_avx512) return Backend::AVX512;
//     if (hw.has_avx2)   return Backend::AVX2;
//     if (hw.has_openmp) return Backend::OPENMP;
//     return Backend::SCALAR;
//
// ============================================================================
// 4.  Convenience helper  (optional, add in the inline utilities section)
//
//     Converts a numeric GCN arch to the canonical gfxNNN string.
// ============================================================================

#include <string>

/// Returns the canonical HIP architecture string for a numeric GCN arch code.
/// Examples: 908 → "gfx908"  (MI100)
///           0x90a = 2314 → "gfx90a" (MI210/MI250)
///           942 → "gfx942"  (MI300X)
///           1100 → "gfx1100" (RX 7900 XT, RDNA3)
inline std::string rocm_gcn_arch_string(int gcn_arch) {
    // HIP encodes RDNA hexadecimal archs (e.g. 0x90a) as integers.
    // We reconstruct the gfx label used by the compiler / runtime.
    if (gcn_arch == 0x90a) return "gfx90a";  // MI210 / MI250 series

    // Decimal archs map directly: 908 → "gfx908", 942 → "gfx942", etc.
    return "gfx" + std::to_string(gcn_arch);
}

/// Human-readable product name for well-known GCN/CDNA/RDNA architectures.
inline std::string rocm_gcn_arch_product(int gcn_arch) {
    switch (gcn_arch) {
        case 803:   return "Fiji / Vega (gfx803)";
        case 900:   return "Vega 10 (gfx900, Vega 64)";
        case 906:   return "Vega 20 (gfx906, Radeon VII / MI50/MI60)";
        case 908:   return "CDNA1 (gfx908, Instinct MI100)";
        case 0x90a: return "CDNA2 (gfx90a, Instinct MI200 series)";
        case 940:   return "CDNA3 (gfx940, Instinct MI300)";
        case 941:   return "CDNA3 (gfx941, Instinct MI300)";
        case 942:   return "CDNA3 (gfx942, Instinct MI300X)";
        case 1010:  return "RDNA1 (gfx1010, Navi 10)";
        case 1030:  return "RDNA2 (gfx1030, Navi 21)";
        case 1100:  return "RDNA3 (gfx1100, Navi 31)";
        case 1101:  return "RDNA3 (gfx1101, Navi 32)";
        case 1102:  return "RDNA3 (gfx1102, Navi 33)";
        default:    return "AMD GPU (gfx" + std::to_string(gcn_arch) + ")";
    }
}
