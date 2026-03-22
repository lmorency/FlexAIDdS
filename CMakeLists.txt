# Build: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build

cmake_minimum_required(VERSION 3.18)
project(FlexAID LANGUAGES CXX)

include(CheckCXXCompilerFlag)

# ─── C++20 ────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)   # -std=c++20 not -std=gnu++20

# Fail fast if compiler is too old to support C++20
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0")
        message(FATAL_ERROR
            "FlexAIDdS requires C++20 — found ${CMAKE_CXX_COMPILER_ID} "
            "${CMAKE_CXX_COMPILER_VERSION}. Upgrade to GCC ≥ 10 or Clang ≥ 10.")
    endif()
elseif(MSVC)
    if(MSVC_VERSION LESS 1930)
        message(FATAL_ERROR
            "FlexAIDdS requires MSVC ≥ 19.30 (Visual Studio 2022 17.0+) for C++20 — "
            "found MSVC ${MSVC_VERSION}.")
    endif()
endif()

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

# ─── options ────────────────────────────────────────────────────────────────
option(FLEXAIDS_USE_CUDA    "Enable CUDA GPU evaluation"          OFF)
option(FLEXAIDS_USE_METAL   "Enable Metal GPU acceleration"       OFF)
option(FLEXAIDS_USE_AVX2    "Enable AVX2 SIMD acceleration"       ON)
option(FLEXAIDS_USE_AVX512  "Enable AVX-512 SIMD acceleration"    OFF)
option(FLEXAIDS_USE_OPENMP  "Enable OpenMP thread parallelism"    ON)
option(FLEXAIDS_USE_EIGEN   "Enable Eigen3 vectorised linear algebra" ON)
option(FLEXAIDS_USE_256_MATRIX "Enable 256×256 soft contact matrix" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings via pybind11" OFF)

# ─── platform / SIMD helpers ───────────────────────────────────────────────
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" FLEXAIDS_SYSTEM_PROCESSOR_LOWER)
set(FLEXAIDS_IS_ARM FALSE)
set(FLEXAIDS_IS_X86 FALSE)

if(FLEXAIDS_SYSTEM_PROCESSOR_LOWER MATCHES "^(arm64|aarch64)$")
    set(FLEXAIDS_IS_ARM TRUE)
elseif(FLEXAIDS_SYSTEM_PROCESSOR_LOWER MATCHES "^(x86_64|amd64|x64|i[3-6]86)$")
    set(FLEXAIDS_IS_X86 TRUE)
endif()

if(APPLE AND FLEXAIDS_IS_ARM)
    if(FLEXAIDS_USE_AVX2 OR FLEXAIDS_USE_AVX512)
        message(STATUS "Apple Silicon detected — disabling x86 SIMD flags (-mavx2, -mfma, -mavx512*)")
    endif()
    set(FLEXAIDS_USE_AVX2 OFF CACHE BOOL "Enable AVX2 SIMD acceleration" FORCE)
    set(FLEXAIDS_USE_AVX512 OFF CACHE BOOL "Enable AVX-512 SIMD acceleration" FORCE)
endif()

function(flexaids_configure_simd target_name)
    if(MSVC)
        if(FLEXAIDS_IS_X86)
            if(FLEXAIDS_USE_AVX512)
                target_compile_options(${target_name} PRIVATE /arch:AVX512)
                target_compile_definitions(${target_name} PRIVATE FLEXAIDS_USE_AVX512)
                message(STATUS "${target_name}: AVX-512 enabled via /arch:AVX512")
            elseif(FLEXAIDS_USE_AVX2)
                target_compile_options(${target_name} PRIVATE /arch:AVX2)
                target_compile_definitions(${target_name} PRIVATE FLEXAIDS_USE_AVX2)
                message(STATUS "${target_name}: AVX2 enabled via /arch:AVX2")
            endif()
        else()
            message(STATUS "${target_name}: non-x86 MSVC target detected — building without x86 SIMD flags")
        endif()
        return()
    endif()

    if(NOT FLEXAIDS_IS_X86)
        message(STATUS "${target_name}: non-x86 target detected (${CMAKE_SYSTEM_PROCESSOR}) — building without x86 SIMD flags")
        return()
    endif()

    if(FLEXAIDS_USE_AVX512)
        check_cxx_compiler_flag("-mavx512f" FLEXAIDS_COMPILER_SUPPORTS_MAVX512F)
        check_cxx_compiler_flag("-mavx512dq" FLEXAIDS_COMPILER_SUPPORTS_MAVX512DQ)
        check_cxx_compiler_flag("-mavx512bw" FLEXAIDS_COMPILER_SUPPORTS_MAVX512BW)
        check_cxx_compiler_flag("-mavx2" FLEXAIDS_COMPILER_SUPPORTS_MAVX2)
        check_cxx_compiler_flag("-mfma" FLEXAIDS_COMPILER_SUPPORTS_MFMA)

        if(FLEXAIDS_COMPILER_SUPPORTS_MAVX512F AND FLEXAIDS_COMPILER_SUPPORTS_MAVX512DQ AND
           FLEXAIDS_COMPILER_SUPPORTS_MAVX512BW AND FLEXAIDS_COMPILER_SUPPORTS_MAVX2 AND
           FLEXAIDS_COMPILER_SUPPORTS_MFMA)
            target_compile_options(${target_name} PRIVATE -mavx512f -mavx512dq -mavx512bw -mavx2 -mfma)
            target_compile_definitions(${target_name} PRIVATE FLEXAIDS_USE_AVX512)
            message(STATUS "${target_name}: AVX-512 enabled")
        else()
            message(WARNING "${target_name}: AVX-512 requested but compiler/target does not support the required flags; building without AVX-512")
        endif()
    elseif(FLEXAIDS_USE_AVX2)
        check_cxx_compiler_flag("-mavx2" FLEXAIDS_COMPILER_SUPPORTS_MAVX2)
        check_cxx_compiler_flag("-mfma" FLEXAIDS_COMPILER_SUPPORTS_MFMA)

        if(FLEXAIDS_COMPILER_SUPPORTS_MAVX2 AND FLEXAIDS_COMPILER_SUPPORTS_MFMA)
            target_compile_options(${target_name} PRIVATE -mavx2 -mfma)
            target_compile_definitions(${target_name} PRIVATE FLEXAIDS_USE_AVX2)
            message(STATUS "${target_name}: AVX2/FMA enabled")
        else()
            message(WARNING "${target_name}: AVX2 requested but compiler/target does not support -mavx2/-mfma; building without x86 SIMD")
        endif()
    endif()
endfunction()

# ─── OpenMP ───────────────────────────────────────────────────────────────
if(FLEXAIDS_USE_OPENMP)
    find_package(OpenMP)
endif()

# ─── Eigen3 (header-only; optional but strongly recommended) ──────────────────────
if(FLEXAIDS_USE_EIGEN)
    find_package(Eigen3 3.4 QUIET NO_MODULE)
    if(Eigen3_FOUND)
        message(STATUS "Eigen3 ${Eigen3_VERSION} found — enabling vectorised entropy/geometry")
    else()
        # Try pkg-config fallback
        find_package(PkgConfig QUIET)
        if(PKG_CONFIG_FOUND)
            pkg_check_modules(EIGEN3 QUIET eigen3)
        endif()
        if(EIGEN3_FOUND)
            message(STATUS "Eigen3 found via pkg-config")
        else()
            message(STATUS "Eigen3 not found — scalar fallback will be used (install libeigen3-dev for acceleration)")
            set(FLEXAIDS_USE_EIGEN OFF)
        endif()
    endif()
endif()

# ─── nlohmann/json (header-only JSON parser, MIT license) ──────────────────
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nlohmann_json)
message(STATUS "nlohmann/json fetched — JSON config support enabled")

# ─── pybind11 (for Python bindings) ────────────────────────────────────────
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        message(WARNING "pybind11 not found. Install with: pip install pybind11[global]")
        set(BUILD_PYTHON_BINDINGS OFF)
    else()
        message(STATUS "pybind11 ${pybind11_VERSION} found — Python bindings enabled")
    endif()
endif()

# ─── CUDA (optional) ─────────────────────────────────────────────────────
if(FLEXAIDS_USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# ─── Metal (optional, macOS only) ────────────────────────────────────────
if(FLEXAIDS_USE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "Metal acceleration is only supported on macOS")
    endif()
    find_library(METAL_LIBRARY Metal REQUIRED)
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    find_library(METALKIT_LIBRARY MetalKit REQUIRED)
    message(STATUS "Metal acceleration enabled")
else()
    if(APPLE)
        message(STATUS "Metal acceleration available but disabled (use -DFLEXAIDS_USE_METAL=ON)")
    endif()
endif()

# ─── executable ─────────────────────────────────────────────────────────
add_executable(FlexAID "")
target_sources(FlexAID PRIVATE
        LIB/top.cpp
        LIB/fileio.cpp
        LIB/assign_shift.cpp
        LIB/buildic.cpp
        LIB/Vcontacts.cpp
        LIB/ic2cf.cpp
        LIB/vcfunction.cpp
        LIB/create_rebuild_list.cpp
        LIB/read_coor.cpp
        LIB/read_constraints.cpp
        LIB/assign_constraint.cpp
        LIB/modify_pdb.cpp
        LIB/update_constraint.cpp
        LIB/update_optres.cpp
        LIB/update_bonded.cpp
        LIB/read_grid.cpp
        LIB/read_input.cpp
        LIB/read_lig.cpp
        LIB/read_pdb.cpp
        LIB/add2_optimiz_vec.cpp
        LIB/assign_radius.cpp
        LIB/assign_radii.cpp
        LIB/assign_radii_types.cpp
        LIB/assign_types.cpp
        LIB/bondedlist.cpp
        LIB/buildcc.cpp
        LIB/buildlist.cpp
        LIB/geometry.cpp
        LIB/read_conect.cpp
        LIB/residue_conect.cpp
        LIB/wif083.cpp
        LIB/calc_cleftic.cpp
        LIB/write_pdb.cpp
        LIB/gaboom.cpp
        LIB/calc_center.cpp
        LIB/ic_bounds.cpp
        LIB/buildic_point.cpp
        LIB/buildcc_point.cpp
        LIB/read_rmsdst.cpp
        LIB/calc_rmsd_chrom.cpp
        LIB/calc_rmsd.cpp
        LIB/write_rrd.cpp
        LIB/write_sphere.cpp
        LIB/write_grid.cpp
        LIB/write_rrg.cpp
        LIB/read_rotlib.cpp
        LIB/number_of_dihedrals.cpp
        LIB/read_flexscfile.cpp
        LIB/read_rotobs.cpp
        LIB/set_intprob.cpp
        LIB/build_rotamers.cpp
        LIB/check_clash.cpp
        LIB/read_normalgrid.cpp
        LIB/read_eigen.cpp
        LIB/assign_eigen.cpp
        LIB/print_surfmat.cpp
        LIB/alter_mode.cpp
        LIB/dee_pivot.cpp
        LIB/read_emat.cpp
        LIB/read_spheres.cpp
        LIB/cluster.cpp
        LIB/DensityPeak_Cluster.cpp
        LIB/FastOPTICS_cluster.cpp
        LIB/FOPTICS.cpp
        LIB/fast_optics.cpp
        LIB/BindingMode.cpp
        LIB/rna_structure.cpp
        LIB/shortest_path.cpp
        LIB/assign_shortflex.cpp
        LIB/slice_grid.cpp
        LIB/partition_grid.cpp
        LIB/generate_grid.cpp
        LIB/maps.cpp
        # ── new C++20 modules ──
        LIB/tENCoM/tencm.cpp
        LIB/statmech.cpp
        LIB/encom.cpp             # ← NEW: ENCoM vibrational entropy
        # ── v1.5 thermodynamic + hardware acceleration modules ──
        LIB/hardware_detect.cpp
        LIB/hardware_dispatch.cpp
        LIB/ShannonThermoStack/ShannonThermoStack.cpp
        LIB/LigandRingFlex/RingConformerLibrary.cpp
        LIB/LigandRingFlex/SugarPucker.cpp
        LIB/LigandRingFlex/LigandRingFlex.cpp
        LIB/NATURaL/NATURaLDualAssembly.cpp
        LIB/NATURaL/RibosomeElongation.cpp
        LIB/NATURaL/TransloconInsertion.cpp
        LIB/ChiralCenter/ChiralCenterGene.cpp
        # ── CleftDetector + Mol2Reader + SdfReader ──
        LIB/CleftDetector.cpp
        LIB/Mol2Reader.cpp
        LIB/SdfReader.cpp
        # ── CavityDetect: native SURFNET cavity detection ──
        LIB/CavityDetect/CavityDetect.cpp
        # ── PTMAttachment: PTM/glycan modification on target chain ──
        LIB/PTMAttachment/PTMAttachment.cpp
        # ── JSON config parser (zero external dependencies) ──
        LIB/config_parser.cpp
        # ── Unified runtime hardware dispatch ──
        LIB/HardwareDispatch.cpp
)

target_include_directories(FlexAID PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/LIB
    ${CMAKE_CURRENT_SOURCE_DIR}/LIB/tENCoM
    ${CMAKE_CURRENT_SOURCE_DIR}/LIB/ShannonThermoStack
    ${CMAKE_CURRENT_SOURCE_DIR}/LIB/LigandRingFlex
    ${CMAKE_CURRENT_SOURCE_DIR}/LIB/NATURaL
    ${CMAKE_CURRENT_SOURCE_DIR}/LIB/ChiralCenter
    ${CMAKE_CURRENT_SOURCE_DIR}/LIB/CavityDetect
    ${CMAKE_CURRENT_SOURCE_DIR}/LIB/PTMAttachment
)

# ─── compiler flags ────────────────────────────────────────────────────
if(MSVC)
    target_compile_options(FlexAID PRIVATE /W4 /O2 /MT)
    target_compile_definitions(FlexAID PRIVATE
        _CRT_SECURE_NO_WARNINGS      # suppress MSVC sprintf/strcpy deprecation
        _USE_MATH_DEFINES             # expose M_PI in <cmath>
        NOMINMAX                      # prevent windows.h min/max macros
    )
else()
    target_compile_options(FlexAID PRIVATE -Wall -O3 -ffast-math)
endif()
flexaids_configure_simd(FlexAID)

# ─── Eigen3 link + definition ─────────────────────────────────────────────
if(FLEXAIDS_USE_EIGEN AND Eigen3_FOUND)
    target_link_libraries(FlexAID PRIVATE Eigen3::Eigen)
    target_compile_definitions(FlexAID PRIVATE FLEXAIDS_HAS_EIGEN)
elseif(FLEXAIDS_USE_EIGEN AND EIGEN3_FOUND)
    target_include_directories(FlexAID PRIVATE ${EIGEN3_INCLUDE_DIRS})
    target_compile_definitions(FlexAID PRIVATE FLEXAIDS_HAS_EIGEN)
endif()

# ─── nlohmann/json link ────────────────────────────────────────────────
target_link_libraries(FlexAID PRIVATE nlohmann_json::nlohmann_json)

# ─── OpenMP link ────────────────────────────────────────────────────────
if(FLEXAIDS_USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(FlexAID PRIVATE OpenMP::OpenMP_CXX)
endif()

# ─── CUDA sources ───────────────────────────────────────────────────────
if(FLEXAIDS_USE_CUDA)
    target_sources(FlexAID PRIVATE
        LIB/cuda_eval.cu                          # existing CF batch kernel
        LIB/ShannonThermoStack/shannon_cuda.cu    # Shannon entropy histogram kernel
        LIB/tENCoM/tencm_cuda.cu                  # TENCoM contact discovery + Hessian assembly
        LIB/gpu_fast_optics.cu                    # GPU-accelerated FastOPTICS neighbour search
    )
    target_compile_definitions(FlexAID PRIVATE FLEXAIDS_USE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
    endif()
    set_target_properties(FlexAID PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17)
    target_link_libraries(FlexAID PRIVATE CUDA::cudart CUDA::cudadevrt)
    message(STATUS "CUDA enabled — CF batch + Shannon histogram + TENCoM Hessian kernels")
endif()

# ─── Metal sources ──────────────────────────────────────────────────────
if(FLEXAIDS_USE_METAL)
    target_sources(FlexAID PRIVATE LIB/metal_eval.mm)
    target_compile_definitions(FlexAID PRIVATE FLEXAIDS_USE_METAL)
    target_compile_options(FlexAID PRIVATE -fno-objc-arc)
    target_link_libraries(FlexAID PRIVATE
        ${METAL_LIBRARY}
        ${FOUNDATION_LIBRARY}
        ${METALKIT_LIBRARY}
    )
    set_source_files_properties(LIB/metal_eval.mm PROPERTIES
        LANGUAGE OBJCXX
        COMPILE_FLAGS "-std=c++20 -fno-objc-arc"
    )
    # Shannon Metal GPU bridge (Apple Silicon Shannon entropy histogram)
    target_sources(FlexAID PRIVATE LIB/ShannonThermoStack/ShannonMetalBridge.mm)
    target_compile_definitions(FlexAID PRIVATE FLEXAIDS_HAS_METAL_SHANNON)
    set_source_files_properties(LIB/ShannonThermoStack/ShannonMetalBridge.mm PROPERTIES
        LANGUAGE OBJCXX
        COMPILE_FLAGS "-std=c++20 -fno-objc-arc"
    )
    # TENCoM Metal bridge (GPU contact discovery + Hessian assembly)
    target_sources(FlexAID PRIVATE LIB/tENCoM/tencm_metal.mm)
    target_compile_definitions(FlexAID PRIVATE FLEXAIDS_HAS_METAL_TENCM)
    set_source_files_properties(LIB/tENCoM/tencm_metal.mm PROPERTIES
        LANGUAGE OBJCXX
        COMPILE_FLAGS "-std=c++20 -fno-objc-arc"
    )
    # CavityDetect Metal bridge (Task 1.4: GPU probe-sphere generation)
    target_sources(FlexAID PRIVATE LIB/CavityDetect/CavityDetectMetalBridge.mm)
    set_source_files_properties(LIB/CavityDetect/CavityDetectMetalBridge.mm PROPERTIES
        LANGUAGE OBJCXX
        COMPILE_FLAGS "-std=c++20 -fno-objc-arc"
    )
    # Compile CavityDetect.metal shader → .metallib
    set(CAVITY_METAL_SRC  "${CMAKE_CURRENT_SOURCE_DIR}/LIB/CavityDetect/CavityDetect.metal")
    set(CAVITY_AIR        "${CMAKE_CURRENT_BINARY_DIR}/CavityDetect.air")
    set(CAVITY_METALLIB   "${CMAKE_CURRENT_BINARY_DIR}/CavityDetect.metallib")
    add_custom_command(
        OUTPUT  "${CAVITY_AIR}"
        COMMAND xcrun -sdk macosx metal -c "${CAVITY_METAL_SRC}" -o "${CAVITY_AIR}"
        DEPENDS "${CAVITY_METAL_SRC}"
        COMMENT "Compiling CavityDetect Metal shader"
        VERBATIM
    )
    add_custom_command(
        OUTPUT  "${CAVITY_METALLIB}"
        COMMAND xcrun -sdk macosx metallib "${CAVITY_AIR}" -o "${CAVITY_METALLIB}"
        DEPENDS "${CAVITY_AIR}"
        COMMENT "Linking CavityDetect.metallib"
        VERBATIM
    )
    add_custom_target(CavityDetectMetal DEPENDS "${CAVITY_METALLIB}")
    add_dependencies(FlexAID CavityDetectMetal)
    target_compile_definitions(FlexAID PRIVATE
        CAVITY_METALLIB_PATH="${CAVITY_METALLIB}"
    )
endif()

# ─── Python bindings (pybind11 module) ─────────────────────────────────────────
if(BUILD_PYTHON_BINDINGS AND pybind11_FOUND)
    pybind11_add_module(_core
        python/bindings/core_bindings.cpp
        LIB/statmech.cpp
        LIB/hardware_dispatch.cpp
        LIB/hardware_detect.cpp
        LIB/encom.cpp
        LIB/ShannonThermoStack/ShannonThermoStack.cpp
        LIB/tencm.cpp
        LIB/fast_optics.cpp
        # BindingMode requires full GA infrastructure
        # Partial exposure for now (thermodynamics only)
    )
    # Conditionally add 256×256 matrix bindings
    if(FLEXAIDS_USE_256_MATRIX)
        target_sources(_core PRIVATE python/bindings/bindings_matrix.cpp)
        target_compile_definitions(_core PRIVATE FLEXAIDS_USE_256_MATRIX)
    endif()
    target_include_directories(_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/ShannonThermoStack
    )
    target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})
    if(MSVC)
        target_compile_definitions(_core PRIVATE
            _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES NOMINMAX)
    endif()

    # Link Eigen3 if available
    if(FLEXAIDS_USE_EIGEN AND Eigen3_FOUND)
        target_link_libraries(_core PRIVATE Eigen3::Eigen)
        target_compile_definitions(_core PRIVATE FLEXAIDS_HAS_EIGEN)
    endif()

    message(STATUS "Python bindings enabled — build target: _core.so")
    message(STATUS "Install with: pip install -e python/")
endif()

# ─── TeNCoM benchmark (optional standalone binary) ────────────────────────────────
option(ENABLE_TENCOM_BENCHMARK "Build the TeNCoM benchmark binary" OFF)
if(ENABLE_TENCOM_BENCHMARK)
    add_executable(benchmark_tencom
        LIB/tENCoM/benchmark_tencom.cpp
        LIB/tENCoM/tencm.cpp
        LIB/statmech.cpp
        LIB/hardware_dispatch.cpp
        LIB/hardware_detect.cpp
        LIB/encom.cpp
    )
    target_include_directories(benchmark_tencom PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/tENCoM)
    if(MSVC)
        target_compile_options(benchmark_tencom PRIVATE /O2 /fp:fast)
    else()
        target_compile_options(benchmark_tencom PRIVATE -O3 -ffast-math)
    endif()
    flexaids_configure_simd(benchmark_tencom)
    if(FLEXAIDS_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(benchmark_tencom PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(FLEXAIDS_USE_EIGEN AND Eigen3_FOUND)
        target_link_libraries(benchmark_tencom PRIVATE Eigen3::Eigen)
        target_compile_definitions(benchmark_tencom PRIVATE FLEXAIDS_HAS_EIGEN)
    endif()
    if(MSVC)
        target_compile_definitions(benchmark_tencom PRIVATE
            _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES NOMINMAX)
    endif()
    message(STATUS "TeNCoM benchmark enabled — build target: benchmark_tencom")
endif()

# ─── tENCoM: ultra-fast vibrational entropy tool (HPC-ready) ──────────────────
# Enabled by default. Same LTO + native optimisation as FlexAIDdS.
option(ENABLE_TENCOM_TOOL "Build the tENCoM vibrational entropy differential tool" ON)
if(ENABLE_TENCOM_TOOL)
    add_executable(tENCoM
        LIB/tENCoM/tencom_main.cpp
        LIB/tENCoM/tencom_diff.cpp
        LIB/tENCoM/tencom_output.cpp
        LIB/tENCoM/pdb_calpha.cpp
        LIB/tENCoM/tencm.cpp
        LIB/statmech.cpp
        LIB/hardware_dispatch.cpp
        LIB/hardware_detect.cpp
        LIB/encom.cpp
    )
    target_include_directories(tENCoM PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/tENCoM)
    if(MSVC)
        target_compile_options(tENCoM PRIVATE /O2 /GL /fp:fast /DNDEBUG /MT)
        target_compile_definitions(tENCoM PRIVATE
            _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES NOMINMAX)
        target_link_options(tENCoM PRIVATE /LTCG)
    else()
        target_compile_options(tENCoM PRIVATE
            -O3 -ffast-math -march=native -flto -DNDEBUG)
        target_link_options(tENCoM PRIVATE -flto -s)
    endif()
    flexaids_configure_simd(tENCoM)
    if(FLEXAIDS_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(tENCoM PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(FLEXAIDS_USE_EIGEN AND Eigen3_FOUND)
        target_link_libraries(tENCoM PRIVATE Eigen3::Eigen)
        target_compile_definitions(tENCoM PRIVATE FLEXAIDS_HAS_EIGEN)
    endif()
    # CUDA GPU acceleration for large proteins (N > 256 residues)
    if(FLEXAIDS_USE_CUDA)
        target_sources(tENCoM PRIVATE LIB/tENCoM/tencm_cuda.cu)
        target_compile_definitions(tENCoM PRIVATE FLEXAIDS_USE_CUDA)
        set_target_properties(tENCoM PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_STANDARD 17)
        target_link_libraries(tENCoM PRIVATE CUDA::cudart CUDA::cudadevrt)
    endif()
    # Metal GPU acceleration (macOS)
    if(FLEXAIDS_USE_METAL)
        target_sources(tENCoM PRIVATE LIB/tENCoM/tencm_metal.mm)
        target_compile_definitions(tENCoM PRIVATE FLEXAIDS_HAS_METAL_TENCM)
        set_source_files_properties(LIB/tENCoM/tencm_metal.mm PROPERTIES
            LANGUAGE OBJCXX
            COMPILE_FLAGS "-std=c++20 -fno-objc-arc")
        target_link_libraries(tENCoM PRIVATE
            ${METAL_LIBRARY} ${FOUNDATION_LIBRARY} ${METALKIT_LIBRARY})
    endif()
    message(STATUS "tENCoM tool enabled — build target: tENCoM")
endif()

# ─── VoronoiCFBatch benchmark (Task 2.3) ───────────────────────────────────────
# VoronoiCFBatch.h is header-only; the benchmark binary exercises the std::span
# batch interface and prints a serial-vs-OpenMP speedup table.
option(ENABLE_VCFBATCH_BENCHMARK "Build the VoronoiCFBatch benchmark binary" OFF)
if(ENABLE_VCFBATCH_BENCHMARK)
    add_executable(benchmark_vcfbatch LIB/benchmark_vcfbatch.cpp)
    target_include_directories(benchmark_vcfbatch PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB)
    if(MSVC)
        target_compile_options(benchmark_vcfbatch PRIVATE /O2 /fp:fast)
    else()
        target_compile_options(benchmark_vcfbatch PRIVATE -O3 -ffast-math)
    endif()
    flexaids_configure_simd(benchmark_vcfbatch)
    if(FLEXAIDS_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(benchmark_vcfbatch PRIVATE OpenMP::OpenMP_CXX)
    endif()
    message(STATUS "VoronoiCFBatch benchmark enabled — build target: benchmark_vcfbatch")
endif()

# ─── FlexAIDdS: ultra-fast docking executable ─────────────────────────────
option(BUILD_FLEXAIDDS_FAST "Build ultra-fast FlexAIDdS docking executable (LTO + native)" ON)
if(BUILD_FLEXAIDDS_FAST)
    add_executable(FlexAIDdS "")
    get_target_property(_flexaid_sources FlexAID SOURCES)
    target_sources(FlexAIDdS PRIVATE ${_flexaid_sources})

    target_include_directories(FlexAIDdS PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/tENCoM
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/ShannonThermoStack
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/LigandRingFlex
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/NATURaL
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/ChiralCenter
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/CavityDetect
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/PTMAttachment
    )

    if(MSVC)
        target_compile_options(FlexAIDdS PRIVATE /O2 /GL /DNDEBUG /MT)
        target_link_options(FlexAIDdS PRIVATE /LTCG)
        target_compile_definitions(FlexAIDdS PRIVATE
            _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES NOMINMAX)
    else()
        target_compile_options(FlexAIDdS PRIVATE
            -O3 -ffast-math -march=native -flto -DNDEBUG)
        target_link_options(FlexAIDdS PRIVATE -flto -s)
    endif()

    flexaids_configure_simd(FlexAIDdS)
    target_link_libraries(FlexAIDdS PRIVATE nlohmann_json::nlohmann_json)

    if(FLEXAIDS_USE_EIGEN AND Eigen3_FOUND)
        target_link_libraries(FlexAIDdS PRIVATE Eigen3::Eigen)
        target_compile_definitions(FlexAIDdS PRIVATE FLEXAIDS_HAS_EIGEN)
    elseif(FLEXAIDS_USE_EIGEN AND EIGEN3_FOUND)
        target_include_directories(FlexAIDdS PRIVATE ${EIGEN3_INCLUDE_DIRS})
        target_compile_definitions(FlexAIDdS PRIVATE FLEXAIDS_HAS_EIGEN)
    endif()

    if(FLEXAIDS_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(FlexAIDdS PRIVATE OpenMP::OpenMP_CXX)
    endif()

    message(STATUS "FlexAIDdS fast executable enabled")
endif()

# ─── NATURaL Hammerhead Ribozyme simulation (standalone) ─────────────────────
# Runs the Zhao 2011 RNAP master equation on the 40-nt Schistosoma mansoni
# minimal hammerhead ribozyme in co-transcriptional (RNAP) mode.
# Only depends on RibosomeElongation (pure C++20, no FlexAID engine needed).
option(ENABLE_NATURAL_HAMMERHEAD "Build NATURaL co-transcriptional HHRz simulation" OFF)
if(ENABLE_NATURAL_HAMMERHEAD)
    add_executable(natural_hammerhead
        tools/natural_hammerhead.cpp
        LIB/NATURaL/RibosomeElongation.cpp
        LIB/NATURaL/TransloconInsertion.cpp
    )
    target_include_directories(natural_hammerhead PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/NATURaL
    )
    if(MSVC)
        target_compile_options(natural_hammerhead PRIVATE /O2 /fp:fast)
        target_compile_definitions(natural_hammerhead PRIVATE
            _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES NOMINMAX)
    else()
        target_compile_options(natural_hammerhead PRIVATE -O3 -ffast-math)
    endif()
    flexaids_configure_simd(natural_hammerhead)
    if(FLEXAIDS_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(natural_hammerhead PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(FLEXAIDS_USE_EIGEN AND Eigen3_FOUND)
        target_link_libraries(natural_hammerhead PRIVATE Eigen3::Eigen)
        target_compile_definitions(natural_hammerhead PRIVATE FLEXAIDS_HAS_EIGEN)
    elseif(FLEXAIDS_USE_EIGEN AND EIGEN3_FOUND)
        target_include_directories(natural_hammerhead PRIVATE ${EIGEN3_INCLUDE_DIRS})
        target_compile_definitions(natural_hammerhead PRIVATE FLEXAIDS_HAS_EIGEN)
    endif()
    message(STATUS "NATURaL HHRz simulation enabled — build target: natural_hammerhead")
endif()

# ─── Dispatch Benchmark (optional standalone binary) ─────────────────────────
option(ENABLE_DISPATCH_BENCHMARK "Build the hardware dispatch benchmark binary" OFF)
if(ENABLE_DISPATCH_BENCHMARK)
    add_executable(benchmark_dispatch
        LIB/benchmark_dispatch.cpp
        LIB/HardwareDispatch.cpp
    )
    target_include_directories(benchmark_dispatch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/LIB)
    if(MSVC)
        target_compile_options(benchmark_dispatch PRIVATE /O2 /fp:fast)
    else()
        target_compile_options(benchmark_dispatch PRIVATE -O3 -ffast-math)
    endif()
    flexaids_configure_simd(benchmark_dispatch)
    if(FLEXAIDS_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(benchmark_dispatch PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(FLEXAIDS_USE_EIGEN AND Eigen3_FOUND)
        target_link_libraries(benchmark_dispatch PRIVATE Eigen3::Eigen)
        target_compile_definitions(benchmark_dispatch PRIVATE FLEXAIDS_HAS_EIGEN)
    elseif(FLEXAIDS_USE_EIGEN AND EIGEN3_FOUND)
        target_include_directories(benchmark_dispatch PRIVATE ${EIGEN3_INCLUDE_DIRS})
        target_compile_definitions(benchmark_dispatch PRIVATE FLEXAIDS_HAS_EIGEN)
    endif()
    message(STATUS "Dispatch benchmark enabled — build target: benchmark_dispatch")
endif()

# ─── Unit Tests (GoogleTest) ───────────────────────────────────────────────
option(BUILD_TESTING "Build unit tests (requires GoogleTest)" OFF)
if(BUILD_TESTING)
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.14.0
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    enable_testing()

    # Helper: apply MSVC-specific definitions for test targets
    function(flexaids_configure_msvc_test target_name)
        if(MSVC)
            target_compile_definitions(${target_name} PRIVATE
                _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES NOMINMAX)
        endif()
    endfunction()

    # test_statmech — StatMechEngine unit tests
    add_executable(test_statmech tests/test_statmech.cpp LIB/statmech.cpp
                   LIB/hardware_dispatch.cpp LIB/hardware_detect.cpp)
    target_include_directories(test_statmech PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/LIB)
    target_link_libraries(test_statmech PRIVATE GTest::gtest GTest::gtest_main)
    if(MSVC)
        target_compile_options(test_statmech PRIVATE /O2)
    else()
        target_compile_options(test_statmech PRIVATE -O2)
    endif()
    flexaids_configure_simd(test_statmech)
    if(FLEXAIDS_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(test_statmech PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(FLEXAIDS_USE_EIGEN AND Eigen3_FOUND)
        target_link_libraries(test_statmech PRIVATE Eigen3::Eigen)
        target_compile_definitions(test_statmech PRIVATE FLEXAIDS_HAS_EIGEN)
    elseif(FLEXAIDS_USE_EIGEN AND EIGEN3_FOUND)
        target_include_directories(test_statmech PRIVATE ${EIGEN3_INCLUDE_DIRS})
        target_compile_definitions(test_statmech PRIVATE FLEXAIDS_HAS_EIGEN)
    endif()
    flexaids_configure_msvc_test(test_statmech)
    add_test(NAME StatMechTests COMMAND test_statmech)

    # test_json_config — JSON parser & config defaults tests (zero dependencies)
    add_executable(test_json_config tests/test_json_config.cpp)
    target_include_directories(test_json_config PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/LIB)
    target_link_libraries(test_json_config PRIVATE GTest::gtest GTest::gtest_main)
    if(MSVC)
        target_compile_options(test_json_config PRIVATE /O2)
    else()
        target_compile_options(test_json_config PRIVATE -O2)
    endif()
    flexaids_configure_msvc_test(test_json_config)
    add_test(NAME JsonConfigTests COMMAND test_json_config)

    # test_hardware_dispatch — ShannonThermoStack dispatch layer tests
    add_executable(test_hardware_dispatch
        tests/test_hardware_dispatch.cpp
        LIB/ShannonThermoStack/ShannonThermoStack.cpp
        LIB/statmech.cpp
        LIB/hardware_dispatch.cpp
        LIB/hardware_detect.cpp
        LIB/tENCoM/tencm.cpp)
    target_include_directories(test_hardware_dispatch PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/tENCoM
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/ShannonThermoStack)
    target_link_libraries(test_hardware_dispatch PRIVATE GTest::gtest GTest::gtest_main)
    if(MSVC)
        target_compile_options(test_hardware_dispatch PRIVATE /O2)
    else()
        target_compile_options(test_hardware_dispatch PRIVATE -O2)
    endif()
    if(FLEXAIDS_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(test_hardware_dispatch PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(FLEXAIDS_HAS_EIGEN)
        target_link_libraries(test_hardware_dispatch PRIVATE Eigen3::Eigen)
    endif()
    flexaids_configure_msvc_test(test_hardware_dispatch)
    add_test(NAME HardwareDispatchTests COMMAND test_hardware_dispatch)

    # test_hardware_detect_dispatch — unified dispatch layer + hardware detection tests
    add_executable(test_hardware_detect_dispatch
        tests/test_hardware_detect_dispatch.cpp
        LIB/hardware_detect.cpp
        LIB/hardware_dispatch.cpp
        LIB/statmech.cpp)
    target_include_directories(test_hardware_detect_dispatch PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB)
    target_link_libraries(test_hardware_detect_dispatch PRIVATE GTest::gtest GTest::gtest_main)
    if(MSVC)
        target_compile_options(test_hardware_detect_dispatch PRIVATE /O2)
    else()
        target_compile_options(test_hardware_detect_dispatch PRIVATE -O2)
    endif()
    flexaids_configure_simd(test_hardware_detect_dispatch)
    if(FLEXAIDS_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(test_hardware_detect_dispatch PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(FLEXAIDS_USE_EIGEN AND Eigen3_FOUND)
        target_link_libraries(test_hardware_detect_dispatch PRIVATE Eigen3::Eigen)
        target_compile_definitions(test_hardware_detect_dispatch PRIVATE FLEXAIDS_HAS_EIGEN)
    elseif(FLEXAIDS_USE_EIGEN AND EIGEN3_FOUND)
        target_include_directories(test_hardware_detect_dispatch PRIVATE ${EIGEN3_INCLUDE_DIRS})
        target_compile_definitions(test_hardware_detect_dispatch PRIVATE FLEXAIDS_HAS_EIGEN)
    endif()
    if(FLEXAIDS_USE_CUDA)
        target_compile_definitions(test_hardware_detect_dispatch PRIVATE FLEXAIDS_USE_CUDA)
        target_link_libraries(test_hardware_detect_dispatch PRIVATE CUDA::cudart)
    endif()
    if(FLEXAIDS_USE_METAL)
        target_compile_definitions(test_hardware_detect_dispatch PRIVATE FLEXAIDS_USE_METAL)
    endif()
    flexaids_configure_msvc_test(test_hardware_detect_dispatch)
    add_test(NAME HardwareDetectDispatchTests COMMAND test_hardware_detect_dispatch)

    # test_tencom_diff — tENCoM differential engine tests
    add_executable(test_tencom_diff
        tests/test_tencom_diff.cpp
        LIB/tENCoM/tencom_diff.cpp
        LIB/tENCoM/pdb_calpha.cpp
        LIB/tENCoM/tencm.cpp
        LIB/statmech.cpp
        LIB/hardware_dispatch.cpp
        LIB/hardware_detect.cpp
        LIB/encom.cpp
    )
    target_include_directories(test_tencom_diff PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/tENCoM)
    target_link_libraries(test_tencom_diff PRIVATE GTest::gtest GTest::gtest_main)
    if(MSVC)
        target_compile_options(test_tencom_diff PRIVATE /O2)
    else()
        target_compile_options(test_tencom_diff PRIVATE -O2)
    endif()
    flexaids_configure_simd(test_tencom_diff)
    if(FLEXAIDS_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(test_tencom_diff PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(FLEXAIDS_USE_EIGEN AND Eigen3_FOUND)
        target_link_libraries(test_tencom_diff PRIVATE Eigen3::Eigen)
        target_compile_definitions(test_tencom_diff PRIVATE FLEXAIDS_HAS_EIGEN)
    endif()
    flexaids_configure_msvc_test(test_tencom_diff)
    add_test(NAME TENCoMDiffTests COMMAND test_tencom_diff)

    # test_ptm_attachment — PTMAttachment module tests
    add_executable(test_ptm_attachment
        tests/test_ptm_attachment.cpp
        LIB/PTMAttachment/PTMAttachment.cpp
    )
    target_include_directories(test_ptm_attachment PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/PTMAttachment
    )
    target_link_libraries(test_ptm_attachment PRIVATE GTest::gtest GTest::gtest_main)
    if(MSVC)
        target_compile_options(test_ptm_attachment PRIVATE /O2)
    else()
        target_compile_options(test_ptm_attachment PRIVATE -O2)
    endif()
    flexaids_configure_msvc_test(test_ptm_attachment)
    add_test(NAME PTMAttachmentTests COMMAND test_ptm_attachment)

    # test_gaboom — Genetic Algorithm engine unit tests
    add_executable(test_gaboom
        tests/test_gaboom.cpp
        tests/stubs.cpp
        LIB/gaboom.cpp
        LIB/Vcontacts.cpp
        LIB/statmech.cpp
        LIB/geometry.cpp
        LIB/tENCoM/tencm.cpp
        LIB/encom.cpp
        LIB/BindingMode.cpp
        LIB/fast_optics.cpp
        LIB/ShannonThermoStack/ShannonThermoStack.cpp
        LIB/CleftDetector.cpp
        LIB/NATURaL/NATURaLDualAssembly.cpp
        LIB/NATURaL/RibosomeElongation.cpp
        LIB/NATURaL/TransloconInsertion.cpp
        LIB/hardware_dispatch.cpp
        LIB/hardware_detect.cpp
    )
    target_include_directories(test_gaboom PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/tENCoM
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/ShannonThermoStack
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/NATURaL
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/CavityDetect
    )
    target_link_libraries(test_gaboom PRIVATE GTest::gtest GTest::gtest_main)
    if(MSVC)
        target_compile_options(test_gaboom PRIVATE /O2)
    else()
        target_compile_options(test_gaboom PRIVATE -O2)
    endif()
    if(FLEXAIDS_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(test_gaboom PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(FLEXAIDS_USE_EIGEN AND Eigen3_FOUND)
        target_link_libraries(test_gaboom PRIVATE Eigen3::Eigen)
        target_compile_definitions(test_gaboom PRIVATE FLEXAIDS_HAS_EIGEN)
    endif()
    flexaids_configure_msvc_test(test_gaboom)
    add_test(NAME GABoomTests COMMAND test_gaboom)

    # test_vcontacts — Voronoi contact scoring math unit tests
    add_executable(test_vcontacts
        tests/test_vcontacts.cpp
        tests/stubs.cpp
        LIB/Vcontacts.cpp
        LIB/geometry.cpp
    )
    target_include_directories(test_vcontacts PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/LIB)
    target_link_libraries(test_vcontacts PRIVATE GTest::gtest GTest::gtest_main)
    if(MSVC)
        target_compile_options(test_vcontacts PRIVATE /O2)
    else()
        target_compile_options(test_vcontacts PRIVATE -O2)
    endif()
    if(FLEXAIDS_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(test_vcontacts PRIVATE OpenMP::OpenMP_CXX)
    endif()
    flexaids_configure_msvc_test(test_vcontacts)
    add_test(NAME VcontactsTests COMMAND test_vcontacts)

    # test_cleft_cavity — CleftDetector + CavityDetect unit tests
    add_executable(test_cleft_cavity
        tests/test_cleft_cavity.cpp
        LIB/CleftDetector.cpp
        LIB/CavityDetect/CavityDetect.cpp
        LIB/geometry.cpp
    )
    target_include_directories(test_cleft_cavity PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/CavityDetect
    )
    target_link_libraries(test_cleft_cavity PRIVATE GTest::gtest GTest::gtest_main)
    if(MSVC)
        target_compile_options(test_cleft_cavity PRIVATE /O2)
    else()
        target_compile_options(test_cleft_cavity PRIVATE -O2)
    endif()
    if(FLEXAIDS_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(test_cleft_cavity PRIVATE OpenMP::OpenMP_CXX)
    endif()
    flexaids_configure_msvc_test(test_cleft_cavity)
    add_test(NAME CleftCavityTests COMMAND test_cleft_cavity)

    # test_encom — ENCoM vibrational entropy engine tests
    add_executable(test_encom
        tests/test_encom.cpp
        LIB/encom.cpp
    )
    target_include_directories(test_encom PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/LIB)
    target_link_libraries(test_encom PRIVATE GTest::gtest GTest::gtest_main)
    if(MSVC)
        target_compile_options(test_encom PRIVATE /O2)
    else()
        target_compile_options(test_encom PRIVATE -O2)
    endif()
    flexaids_configure_msvc_test(test_encom)
    add_test(NAME ENCoMTests COMMAND test_encom)

    # test_ring_conformer_library — RingConformerLibrary + ring detection tests
    add_executable(test_ring_conformer_library
        tests/test_ring_conformer_library.cpp
        LIB/LigandRingFlex/RingConformerLibrary.cpp
    )
    target_include_directories(test_ring_conformer_library PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/LigandRingFlex
    )
    target_link_libraries(test_ring_conformer_library PRIVATE GTest::gtest GTest::gtest_main)
    if(MSVC)
        target_compile_options(test_ring_conformer_library PRIVATE /O2)
    else()
        target_compile_options(test_ring_conformer_library PRIVATE -O2)
    endif()
    flexaids_configure_msvc_test(test_ring_conformer_library)
    add_test(NAME RingConformerLibraryTests COMMAND test_ring_conformer_library)

    # test_sugar_pucker — SugarPucker Cremer-Pople pseudorotation tests
    add_executable(test_sugar_pucker
        tests/test_sugar_pucker.cpp
        LIB/LigandRingFlex/SugarPucker.cpp
    )
    target_include_directories(test_sugar_pucker PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/LigandRingFlex
    )
    target_link_libraries(test_sugar_pucker PRIVATE GTest::gtest GTest::gtest_main)
    if(MSVC)
        target_compile_options(test_sugar_pucker PRIVATE /O2)
    else()
        target_compile_options(test_sugar_pucker PRIVATE -O2)
    endif()
    if(FLEXAIDS_USE_EIGEN AND Eigen3_FOUND)
        target_link_libraries(test_sugar_pucker PRIVATE Eigen3::Eigen)
        target_compile_definitions(test_sugar_pucker PRIVATE FLEXAIDS_HAS_EIGEN)
    endif()
    flexaids_configure_msvc_test(test_sugar_pucker)
    add_test(NAME SugarPuckerTests COMMAND test_sugar_pucker)

    # test_chiral_center — ChiralCenterGene R/S stereocenter tests
    add_executable(test_chiral_center
        tests/test_chiral_center.cpp
        LIB/ChiralCenter/ChiralCenterGene.cpp
    )
    target_include_directories(test_chiral_center PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/ChiralCenter
    )
    target_link_libraries(test_chiral_center PRIVATE GTest::gtest GTest::gtest_main)
    if(MSVC)
        target_compile_options(test_chiral_center PRIVATE /O2)
    else()
        target_compile_options(test_chiral_center PRIVATE -O2)
    endif()
    if(FLEXAIDS_USE_EIGEN AND Eigen3_FOUND)
        target_link_libraries(test_chiral_center PRIVATE Eigen3::Eigen)
        target_compile_definitions(test_chiral_center PRIVATE FLEXAIDS_HAS_EIGEN)
    endif()
    flexaids_configure_msvc_test(test_chiral_center)
    add_test(NAME ChiralCenterTests COMMAND test_chiral_center)

    # test_cavity_detect — CavityDetect SURFNET cavity detection tests
    add_executable(test_cavity_detect
        tests/test_cavity_detect.cpp
        LIB/CavityDetect/CavityDetect.cpp
    )
    target_include_directories(test_cavity_detect PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/CavityDetect
    )
    target_link_libraries(test_cavity_detect PRIVATE GTest::gtest GTest::gtest_main)
    if(MSVC)
        target_compile_options(test_cavity_detect PRIVATE /O2)
    else()
        target_compile_options(test_cavity_detect PRIVATE -O2)
    endif()
    if(FLEXAIDS_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(test_cavity_detect PRIVATE OpenMP::OpenMP_CXX)
    endif()
    flexaids_configure_simd(test_cavity_detect)
    add_test(NAME CavityDetectTests COMMAND test_cavity_detect)

    # test_binding_mode_statmech — BindingMode ↔ StatMechEngine integration tests
    add_executable(test_binding_mode_statmech
        tests/test_binding_mode_statmech.cpp
        tests/stubs.cpp
        LIB/gaboom.cpp
        LIB/Vcontacts.cpp
        LIB/statmech.cpp
        LIB/geometry.cpp
        LIB/tENCoM/tencm.cpp
        LIB/encom.cpp
        LIB/BindingMode.cpp
        LIB/fast_optics.cpp
        LIB/ShannonThermoStack/ShannonThermoStack.cpp
        LIB/CleftDetector.cpp
        LIB/NATURaL/NATURaLDualAssembly.cpp
        LIB/NATURaL/RibosomeElongation.cpp
        LIB/NATURaL/TransloconInsertion.cpp
        LIB/hardware_dispatch.cpp
        LIB/hardware_detect.cpp
    )
    target_include_directories(test_binding_mode_statmech PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/tENCoM
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/ShannonThermoStack
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/NATURaL
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/CavityDetect
    )
    target_link_libraries(test_binding_mode_statmech PRIVATE GTest::gtest GTest::gtest_main)
    if(MSVC)
        target_compile_options(test_binding_mode_statmech PRIVATE /O2)
    else()
        target_compile_options(test_binding_mode_statmech PRIVATE -O2)
    endif()
    if(FLEXAIDS_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(test_binding_mode_statmech PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(FLEXAIDS_USE_EIGEN AND Eigen3_FOUND)
        target_link_libraries(test_binding_mode_statmech PRIVATE Eigen3::Eigen)
        target_compile_definitions(test_binding_mode_statmech PRIVATE FLEXAIDS_HAS_EIGEN)
    endif()
    flexaids_configure_msvc_test(test_binding_mode_statmech)
    add_test(NAME BindingModeStatMechTests COMMAND test_binding_mode_statmech)

    # test_binding_mode_advanced — BindingMode advanced methods (delta_G, WHAM, representative)
    add_executable(test_binding_mode_advanced
        tests/test_binding_mode_advanced.cpp
        tests/stubs.cpp
        LIB/gaboom.cpp
        LIB/Vcontacts.cpp
        LIB/statmech.cpp
        LIB/geometry.cpp
        LIB/tENCoM/tencm.cpp
        LIB/encom.cpp
        LIB/BindingMode.cpp
        LIB/fast_optics.cpp
        LIB/ShannonThermoStack/ShannonThermoStack.cpp
        LIB/CleftDetector.cpp
        LIB/NATURaL/NATURaLDualAssembly.cpp
        LIB/NATURaL/RibosomeElongation.cpp
        LIB/NATURaL/TransloconInsertion.cpp
        LIB/hardware_dispatch.cpp
        LIB/hardware_detect.cpp
    )
    target_include_directories(test_binding_mode_advanced PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/tENCoM
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/ShannonThermoStack
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/NATURaL
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/CavityDetect
    )
    target_link_libraries(test_binding_mode_advanced PRIVATE GTest::gtest GTest::gtest_main)
    if(MSVC)
        target_compile_options(test_binding_mode_advanced PRIVATE /O2)
    else()
        target_compile_options(test_binding_mode_advanced PRIVATE -O2)
    endif()
    if(FLEXAIDS_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(test_binding_mode_advanced PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(FLEXAIDS_USE_EIGEN AND Eigen3_FOUND)
        target_link_libraries(test_binding_mode_advanced PRIVATE Eigen3::Eigen)
        target_compile_definitions(test_binding_mode_advanced PRIVATE FLEXAIDS_HAS_EIGEN)
    endif()
    flexaids_configure_msvc_test(test_binding_mode_advanced)
    add_test(NAME BindingModeAdvancedTests COMMAND test_binding_mode_advanced)

    # test_ga_core — GA primitive operations (QuickSort, mutate, crossover, copy)
    add_executable(test_ga_core
        tests/test_ga_core.cpp
        tests/stubs.cpp
        LIB/gaboom.cpp
        LIB/Vcontacts.cpp
        LIB/statmech.cpp
        LIB/geometry.cpp
        LIB/tENCoM/tencm.cpp
        LIB/encom.cpp
        LIB/BindingMode.cpp
        LIB/fast_optics.cpp
        LIB/ShannonThermoStack/ShannonThermoStack.cpp
        LIB/CleftDetector.cpp
        LIB/NATURaL/NATURaLDualAssembly.cpp
        LIB/NATURaL/RibosomeElongation.cpp
        LIB/NATURaL/TransloconInsertion.cpp
        LIB/hardware_dispatch.cpp
        LIB/hardware_detect.cpp
    )
    target_include_directories(test_ga_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/tENCoM
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/ShannonThermoStack
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/NATURaL
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/CavityDetect
    )
    target_link_libraries(test_ga_core PRIVATE GTest::gtest GTest::gtest_main)
    if(MSVC)
        target_compile_options(test_ga_core PRIVATE /O2)
    else()
        target_compile_options(test_ga_core PRIVATE -O2)
    endif()
    if(FLEXAIDS_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(test_ga_core PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(FLEXAIDS_USE_EIGEN AND Eigen3_FOUND)
        target_link_libraries(test_ga_core PRIVATE Eigen3::Eigen)
        target_compile_definitions(test_ga_core PRIVATE FLEXAIDS_HAS_EIGEN)
    endif()
    flexaids_configure_msvc_test(test_ga_core)
    add_test(NAME GACoreTests COMMAND test_ga_core)

    # test_ga_validation — CF scoring functions, eval_span clamping, batch result structure
    add_executable(test_ga_validation
        tests/test_ga_validation.cpp
        tests/stubs.cpp
        LIB/gaboom.cpp
        LIB/Vcontacts.cpp
        LIB/statmech.cpp
        LIB/geometry.cpp
        LIB/tENCoM/tencm.cpp
        LIB/encom.cpp
        LIB/BindingMode.cpp
        LIB/fast_optics.cpp
        LIB/ShannonThermoStack/ShannonThermoStack.cpp
        LIB/CleftDetector.cpp
        LIB/NATURaL/NATURaLDualAssembly.cpp
        LIB/NATURaL/RibosomeElongation.cpp
        LIB/NATURaL/TransloconInsertion.cpp
        LIB/hardware_dispatch.cpp
        LIB/hardware_detect.cpp
    )
    target_include_directories(test_ga_validation PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/tENCoM
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/ShannonThermoStack
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/NATURaL
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB/CavityDetect
    )
    target_link_libraries(test_ga_validation PRIVATE GTest::gtest GTest::gtest_main)
    if(MSVC)
        target_compile_options(test_ga_validation PRIVATE /O2)
    else()
        target_compile_options(test_ga_validation PRIVATE -O2)
    endif()
    if(FLEXAIDS_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(test_ga_validation PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(FLEXAIDS_USE_EIGEN AND Eigen3_FOUND)
        target_link_libraries(test_ga_validation PRIVATE Eigen3::Eigen)
        target_compile_definitions(test_ga_validation PRIVATE FLEXAIDS_HAS_EIGEN)
    endif()
    flexaids_configure_msvc_test(test_ga_validation)
    add_test(NAME GAValidationTests COMMAND test_ga_validation)

    # test_fast_optics — lightweight FastOPTICS super-cluster extraction tests
    add_executable(test_fast_optics
        tests/test_fast_optics.cpp
        LIB/fast_optics.cpp
    )
    target_include_directories(test_fast_optics PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/LIB)
    target_link_libraries(test_fast_optics PRIVATE GTest::gtest GTest::gtest_main)
    if(MSVC)
        target_compile_options(test_fast_optics PRIVATE /O2)
    else()
        target_compile_options(test_fast_optics PRIVATE -O2)
    endif()
    flexaids_configure_msvc_test(test_fast_optics)
    add_test(NAME FastOPTICSTests COMMAND test_fast_optics)

    message(STATUS "Unit tests enabled — build targets: test_statmech, test_hardware_dispatch, test_hardware_detect_dispatch, test_tencom_diff, test_ptm_attachment, test_gaboom, test_vcontacts, test_cleft_cavity, test_encom, test_ring_conformer_library, test_sugar_pucker, test_chiral_center, test_cavity_detect, test_binding_mode_statmech, test_binding_mode_advanced, test_ga_core, test_ga_validation, test_fast_optics")
endif()

# ─── Swift Bridge (macOS only) ────────────────────────────────────────────
option(BUILD_SWIFT_BRIDGE "Build Swift bridge static library (macOS only)" OFF)
if(BUILD_SWIFT_BRIDGE)
    if(NOT APPLE)
        message(FATAL_ERROR "BUILD_SWIFT_BRIDGE requires macOS (Apple platform)")
    endif()

    enable_language(OBJCXX)

    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)

    add_library(FlexAIDCore STATIC
        swift/Sources/FlexAIDCore/FXStatMechEngine.mm
        swift/Sources/FlexAIDCore/FXENCoMEngine.mm
        swift/Sources/FlexAIDCore/FXBoltzmannLUT.mm
        swift/Sources/FlexAIDCore/FXGA.mm
        LIB/statmech.cpp
        LIB/encom.cpp
    )
    target_include_directories(FlexAIDCore PUBLIC
        swift/Sources/FlexAIDCore/include
        LIB
    )
    set_target_properties(FlexAIDCore PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
    set_source_files_properties(
        swift/Sources/FlexAIDCore/FXStatMechEngine.mm
        swift/Sources/FlexAIDCore/FXENCoMEngine.mm
        swift/Sources/FlexAIDCore/FXBoltzmannLUT.mm
        swift/Sources/FlexAIDCore/FXGA.mm
        PROPERTIES LANGUAGE OBJCXX
    )
    target_compile_options(FlexAIDCore PRIVATE -std=c++20 -O2)
    target_link_libraries(FlexAIDCore PUBLIC ${FOUNDATION_LIBRARY})

    message(STATUS "Swift bridge enabled — build target: FlexAIDCore (static library)")

    # test_unified_dispatch — runtime HardwareDispatcher tests
    add_executable(test_unified_dispatch
        tests/test_unified_dispatch.cpp
        LIB/HardwareDispatch.cpp
    )
    target_include_directories(test_unified_dispatch PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB)
    target_link_libraries(test_unified_dispatch PRIVATE GTest::gtest GTest::gtest_main)
    if(MSVC)
        target_compile_options(test_unified_dispatch PRIVATE /O2)
    else()
        target_compile_options(test_unified_dispatch PRIVATE -O2)
    endif()
    flexaids_configure_simd(test_unified_dispatch)
    if(FLEXAIDS_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(test_unified_dispatch PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(FLEXAIDS_USE_EIGEN AND Eigen3_FOUND)
        target_link_libraries(test_unified_dispatch PRIVATE Eigen3::Eigen)
        target_compile_definitions(test_unified_dispatch PRIVATE FLEXAIDS_HAS_EIGEN)
    elseif(FLEXAIDS_USE_EIGEN AND EIGEN3_FOUND)
        target_include_directories(test_unified_dispatch PRIVATE ${EIGEN3_INCLUDE_DIRS})
        target_compile_definitions(test_unified_dispatch PRIVATE FLEXAIDS_HAS_EIGEN)
    endif()
    add_test(NAME UnifiedDispatchTests COMMAND test_unified_dispatch)
endif()
