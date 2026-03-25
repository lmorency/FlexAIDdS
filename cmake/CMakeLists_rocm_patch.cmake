# Copyright 2026 Le Bonhomme Pharma
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# =============================================================================
# CMakeLists_rocm_patch.cmake
#
# CMake snippet to add ROCm/HIP support to the FlexAIDS target.
#
# Integration instructions
# ─────────────────────────────────────────────────────────────────────────────
# Paste the contents of this file into the project's root CMakeLists.txt
# AFTER the main FlexAID target has been defined and BEFORE install() rules.
#
# Alternatively, include it directly:
#
#   include(ROCm/CMakeLists_rocm_patch.cmake)
#
# Usage:
#   cmake .. -DFLEXAIDS_USE_ROCM=ON \
#            -DFLEXAIDS_HIP_ARCHITECTURES="gfx908;gfx90a;gfx942"
# =============================================================================

cmake_minimum_required(VERSION 3.21)  # HIP language support added in 3.21

# =============================================================================
# Option
# =============================================================================

option(FLEXAIDS_USE_ROCM "Enable AMD ROCm/HIP GPU acceleration" OFF)

if(NOT FLEXAIDS_USE_ROCM)
    message(STATUS "[FlexAIDS] ROCm/HIP support disabled (set -DFLEXAIDS_USE_ROCM=ON to enable)")
    return()
endif()

# =============================================================================
# ROCm root discovery
# =============================================================================

# Standard ROCm install locations (Linux).  Users can override by setting
# ROCM_PATH or CMAKE_PREFIX_PATH before running cmake.
if(NOT DEFINED ENV{ROCM_PATH})
    set(_rocm_hints /opt/rocm /opt/rocm-6.0.0 /opt/rocm-5.7.1 /usr/local/rocm)
else()
    set(_rocm_hints $ENV{ROCM_PATH})
endif()

list(APPEND CMAKE_PREFIX_PATH ${_rocm_hints})

# =============================================================================
# Enable HIP language in CMake
# =============================================================================

# This must come before find_package(hip) so that CMake knows how to compile
# .hip files.  If CUDA is also enabled, both languages can coexist; CMake
# selects the correct toolchain per file extension.
enable_language(HIP)

# =============================================================================
# Find HIP package
# =============================================================================

find_package(hip QUIET
    HINTS ${_rocm_hints}
    PATH_SUFFIXES lib/cmake/hip hip/lib/cmake/hip
)

if(NOT hip_FOUND)
    message(WARNING
        "[FlexAIDS] FLEXAIDS_USE_ROCM=ON but ROCm/HIP was not found. "
        "Install ROCm (https://rocm.docs.amd.com) or set ROCM_PATH. "
        "Building without ROCm support.")
    set(FLEXAIDS_USE_ROCM OFF CACHE BOOL "" FORCE)
    return()
endif()

message(STATUS "[FlexAIDS] ROCm/HIP found: ${hip_VERSION}")
message(STATUS "[FlexAIDS] HIP include dirs: ${hip_INCLUDE_DIRS}")

# =============================================================================
# GPU architecture targets
# =============================================================================

# Default: MI100 (gfx908), MI200 series (gfx90a), MI300X (gfx942).
# Add RDNA targets (gfx1100 etc.) if consumer GPU support is needed.
# Specifying more architectures increases compile time and binary size.
set(FLEXAIDS_HIP_ARCHITECTURES "gfx908;gfx90a;gfx942"
    CACHE STRING
    "Semicolon-separated list of AMD GPU architectures (e.g. gfx908;gfx90a;gfx942)")

message(STATUS "[FlexAIDS] HIP target architectures: ${FLEXAIDS_HIP_ARCHITECTURES}")

# =============================================================================
# Source files
# =============================================================================

# Paths are relative to the root CMakeLists.txt.  Adjust if the ROCm/
# directory is nested differently.
set(_rocm_sources
    ROCm/hip_eval.hip
    ROCm/rocm_detect.cpp
)

# =============================================================================
# Update the FlexAID target
# =============================================================================

# Add the ROCm source files.
target_sources(FlexAID PRIVATE ${_rocm_sources})

# Mark hip_eval.hip as a HIP source so CMake routes it through hipcc.
set_source_files_properties(ROCm/hip_eval.hip
    PROPERTIES
        LANGUAGE HIP
)

# Propagate the compile definition that gates all ROCm code.
target_compile_definitions(FlexAID PRIVATE FLEXAIDS_USE_ROCM)

# Add HIP include directories.
target_include_directories(FlexAID PRIVATE ${hip_INCLUDE_DIRS})

# Link the HIP device runtime.
target_link_libraries(FlexAID PRIVATE hip::device)

# Set the GPU architectures for the FlexAID target.
set_property(TARGET FlexAID
    PROPERTY HIP_ARCHITECTURES ${FLEXAIDS_HIP_ARCHITECTURES})

# =============================================================================
# Compiler flags for HIP sources
# =============================================================================

# -fgpu-rdc (relocatable device code) is required if __device__ functions
# defined in one TU are called from another (e.g., headers with device
# helpers).  Disable if all device code is in a single TU to reduce link time.
option(FLEXAIDS_HIP_RDC "Enable relocatable device code for HIP" OFF)

if(FLEXAIDS_HIP_RDC)
    set_property(TARGET FlexAID PROPERTY HIP_SEPARABLE_COMPILATION ON)
    message(STATUS "[FlexAIDS] HIP relocatable device code (RDC) enabled")
endif()

# Optimisation flags passed through to hipcc.
target_compile_options(FlexAID PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:
        -O3
        -std=c++20
        --gpu-max-threads-per-block=1024
    >
)

# Optional verbose detection logging.
option(FLEXAIDS_VERBOSE_DETECT "Enable verbose hardware detection output" OFF)
if(FLEXAIDS_VERBOSE_DETECT)
    target_compile_definitions(FlexAID PRIVATE FLEXAIDS_VERBOSE_DETECT)
endif()

# =============================================================================
# Sanity check: warn if CUDA is also enabled
# =============================================================================

if(FLEXAIDS_USE_CUDA AND FLEXAIDS_USE_ROCM)
    message(WARNING
        "[FlexAIDS] Both CUDA and ROCm are enabled.  At runtime only one "
        "GPU backend will be active (CUDA takes priority).  This is a "
        "supported configuration for cross-vendor testing but may increase "
        "build times significantly.")
endif()

# =============================================================================
# Summary
# =============================================================================

message(STATUS "──────────────────────────────────────────────────")
message(STATUS "[FlexAIDS] ROCm/HIP backend configuration:")
message(STATUS "  hip_VERSION            : ${hip_VERSION}")
message(STATUS "  HIP_ARCHITECTURES      : ${FLEXAIDS_HIP_ARCHITECTURES}")
message(STATUS "  HIP_RDC                : ${FLEXAIDS_HIP_RDC}")
message(STATUS "  VERBOSE_DETECT         : ${FLEXAIDS_VERBOSE_DETECT}")
message(STATUS "──────────────────────────────────────────────────")
