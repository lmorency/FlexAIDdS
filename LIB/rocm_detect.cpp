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
// rocm_detect.cpp
//
// ROCm/HIP device detection for FlexAIDS HardwareDispatch.
//
// Integration instructions
// ───────────────────────────────────────────────────────────────────────────
// 1.  Add this file to the build system (see CMakeLists_rocm_patch.cmake).
// 2.  At the bottom of LIB/hardware_detect.cpp, add:
//
//       #ifdef FLEXAIDS_USE_ROCM
//       #include "rocm_detect.cpp"   // or link as a separate TU
//       #endif
//
//     Alternatively, compile rocm_detect.cpp as its own translation unit and
//     forward-declare `detect_rocm_devices` in hardware_detect.cpp:
//
//       #ifdef FLEXAIDS_USE_ROCM
//       void detect_rocm_devices(HardwareInfo& hw);
//       #endif
//
// 3.  Call detect_rocm_devices(hw) after detect_cuda_devices(hw) (or
//     wherever CUDA detection currently happens):
//
//       #ifdef FLEXAIDS_USE_ROCM
//       detect_rocm_devices(hw);
//       #endif
// ============================================================================

#ifdef FLEXAIDS_USE_ROCM

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <string>

// Pull in HardwareInfo.  Adjust the path if the layout differs.
#include "../LIB/HardwareDispatch.h"

// ── Internal helper ─────────────────────────────────────────────────────────

/// Log a HIP runtime error without aborting; detection is best-effort.
static void log_hip_error(const char* context, hipError_t err) {
    std::fprintf(stderr,
        "[FlexAIDS] ROCm detect: %s failed: %s (%d)\n",
        context,
        hipGetErrorString(err),
        static_cast<int>(err));
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Populates hw with information about the first visible HIP/ROCm device.
///
/// If multiple AMD GPUs are present, device 0 is used (the driver default).
/// Multi-GPU support can be added later via a device index parameter or by
/// iterating over all devices and selecting the one with the most VRAM.
///
/// Errors are soft-failures: if ROCm is unavailable the function returns
/// without modifying hw (has_rocm stays false).
void detect_rocm_devices(HardwareInfo& hw) {
    // ── 1. Count available HIP devices ─────────────────────────────────────
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess) {
        // hipErrorNoDevice is expected on non-AMD hardware; suppress noise.
        if (err != hipErrorNoDevice) {
            log_hip_error("hipGetDeviceCount", err);
        }
        return;
    }
    if (device_count == 0) {
        return;  // No AMD GPU visible
    }

    // ── 2. Select device 0 ─────────────────────────────────────────────────
    err = hipSetDevice(0);
    if (err != hipSuccess) {
        log_hip_error("hipSetDevice(0)", err);
        return;
    }

    // ── 3. Query device properties ─────────────────────────────────────────
    hipDeviceProp_t prop;
    std::memset(&prop, 0, sizeof(prop));
    err = hipGetDeviceProperties(&prop, 0);
    if (err != hipSuccess) {
        log_hip_error("hipGetDeviceProperties", err);
        return;
    }

    // ── 4. Populate HardwareInfo ────────────────────────────────────────────
    hw.has_rocm           = true;
    hw.rocm_device_name   = prop.name;
    hw.rocm_compute_units = prop.multiProcessorCount;
    hw.rocm_global_mem    = prop.totalGlobalMem;

    // gcnArch: integer encoding of the GPU architecture.
    // AMD HIP ≥ 5.x exposes gcnArch; older versions may set it to 0.
    // For ROCm ≥ 5.3 one should prefer gcnArchName (string), but the
    // integer is more convenient for switch/table lookups in the dispatcher.
    hw.rocm_gcn_arch = prop.gcnArch;

    // Wavefront size (AMD GCN/CDNA = 64, some RDNA modes allow 32).
    // hipDeviceProp_t exposes warpSize which the runtime sets correctly.
    hw.rocm_wavefront = prop.warpSize;

    // ── 5. Optional: log summary ────────────────────────────────────────────
#ifdef FLEXAIDS_VERBOSE_DETECT
    std::fprintf(stderr,
        "[FlexAIDS] ROCm device 0: %s  CUs=%d  VRAM=%.1f GiB  "
        "arch=gfx%d  wavefront=%d\n",
        hw.rocm_device_name.c_str(),
        hw.rocm_compute_units,
        static_cast<double>(hw.rocm_global_mem) / (1024.0 * 1024.0 * 1024.0),
        hw.rocm_gcn_arch,
        hw.rocm_wavefront);

    // Also enumerate additional devices for informational purposes.
    if (device_count > 1) {
        std::fprintf(stderr,
            "[FlexAIDS] %d additional ROCm device(s) present (using device 0).\n",
            device_count - 1);
        for (int i = 1; i < device_count; ++i) {
            hipDeviceProp_t p2;
            std::memset(&p2, 0, sizeof(p2));
            if (hipGetDeviceProperties(&p2, i) == hipSuccess) {
                std::fprintf(stderr,
                    "[FlexAIDS]   device %d: %s  CUs=%d  VRAM=%.1f GiB  arch=gfx%d\n",
                    i, p2.name, p2.multiProcessorCount,
                    static_cast<double>(p2.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0),
                    p2.gcnArch);
            }
        }
    }
#endif  // FLEXAIDS_VERBOSE_DETECT
}

// ── Multi-device variant (optional, exported for future use) ─────────────────

/// Returns the index of the AMD device with the largest global memory,
/// which is usually the best default for compute workloads.
///
/// Returns -1 if no device is available.
int rocm_best_device_index() {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) return -1;

    int    best_idx = 0;
    size_t best_mem = 0;

    for (int i = 0; i < count; ++i) {
        hipDeviceProp_t p;
        std::memset(&p, 0, sizeof(p));
        if (hipGetDeviceProperties(&p, i) != hipSuccess) continue;
        if (p.totalGlobalMem > best_mem) {
            best_mem = p.totalGlobalMem;
            best_idx = i;
        }
    }
    return best_idx;
}

#endif  // FLEXAIDS_USE_ROCM
