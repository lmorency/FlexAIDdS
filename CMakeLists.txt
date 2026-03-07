# On mac the easiest way to install boost is to use brew: brew install boost
# Then you can build the project using: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release and cmake --build build

# On Windows you can get boost by downloading the zip from the official website (install binaries) and running the executable
# If it is not detected automatically you can use something like -DBoost_DIR=C:\boost_1_86_0\stage\lib\cmake\Boost-1.86.0 or C:\boost_1_82_0\lib64-msvc-14.3\cmake\Boost-1.82.0 in the cmake command

cmake_minimum_required(VERSION 3.18)
project(FlexAID LANGUAGES CXX)

# ─── C++20 ────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

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
option(BUILD_PYTHON_BINDINGS "Build Python bindings via pybind11" OFF)

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
        LIB/BindingMode.cpp
        LIB/rna_structure.cpp
        LIB/shortest_path.cpp
        LIB/assign_shortflex.cpp
        LIB/slice_grid.cpp
        LIB/partition_grid.cpp
        LIB/generate_grid.cpp
        LIB/maps.cpp
        # ── new C++20 modules ──
        LIB/tencm.cpp
        LIB/statmech.cpp
        LIB/encom.cpp             # ← NEW: ENCoM vibrational entropy
        # ── v1.5 thermodynamic + hardware acceleration modules ──
        LIB/ShannonThermoStack/ShannonThermoStack.cpp
        LIB/LigandRingFlex/RingConformerLibrary.cpp
        LIB/LigandRingFlex/SugarPucker.cpp
        LIB/LigandRingFlex/LigandRingFlex.cpp
        LIB/NATURaL/NATURaLDualAssembly.cpp
        LIB/NATURaL/RibosomeElongation.cpp
        LIB/NATURaL/TransloconInsertion.cpp
        LIB/ChiralCenter/ChiralCenterGene.cpp
)

target_include_directories(FlexAID PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/LIB
    ${CMAKE_CURRENT_SOURCE_DIR}/LIB/ShannonThermoStack
    ${CMAKE_CURRENT_SOURCE_DIR}/LIB/LigandRingFlex
    ${CMAKE_CURRENT_SOURCE_DIR}/LIB/NATURaL
    ${CMAKE_CURRENT_SOURCE_DIR}/LIB/ChiralCenter
)

# ─── compiler flags ────────────────────────────────────────────────────
if(MSVC)
    target_compile_options(FlexAID PRIVATE /W4 /O2 /MT)
    if(FLEXAIDS_USE_AVX512)
        target_compile_options(FlexAID PRIVATE /arch:AVX512)
    elseif(FLEXAIDS_USE_AVX2)
        target_compile_options(FlexAID PRIVATE /arch:AVX2)
    endif()
else()
    target_compile_options(FlexAID PRIVATE -Wall -O3 -ffast-math)
    if(FLEXAIDS_USE_AVX512)
        # AVX-512 supersedes AVX2; include both for compatibility with mixed intrinsics
        target_compile_options(FlexAID PRIVATE -mavx512f -mavx512dq -mavx512bw -mavx2 -mfma)
        target_compile_definitions(FlexAID PRIVATE FLEXAIDS_USE_AVX512)
        message(STATUS "AVX-512 enabled (mavx512f + mavx512dq + mavx512bw)")
    elseif(FLEXAIDS_USE_AVX2)
        target_compile_options(FlexAID PRIVATE -mavx2 -mfma)
    endif()
endif()

# ─── Eigen3 link + definition ─────────────────────────────────────────────
if(FLEXAIDS_USE_EIGEN AND Eigen3_FOUND)
    target_link_libraries(FlexAID PRIVATE Eigen3::Eigen)
    target_compile_definitions(FlexAID PRIVATE FLEXAIDS_HAS_EIGEN)
elseif(FLEXAIDS_USE_EIGEN AND EIGEN3_FOUND)
    target_include_directories(FlexAID PRIVATE ${EIGEN3_INCLUDE_DIRS})
    target_compile_definitions(FlexAID PRIVATE FLEXAIDS_HAS_EIGEN)
endif()

# ─── OpenMP link ────────────────────────────────────────────────────────
if(FLEXAIDS_USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(FlexAID PRIVATE OpenMP::OpenMP_CXX)
endif()

# ─── CUDA sources ───────────────────────────────────────────────────────
if(FLEXAIDS_USE_CUDA)
    target_sources(FlexAID PRIVATE
        LIB/cuda_eval.cu                          # existing CF batch kernel
        LIB/ShannonThermoStack/shannon_cuda.cu    # Shannon entropy histogram kernel
    )
    target_compile_definitions(FlexAID PRIVATE FLEXAIDS_USE_CUDA)
    set_target_properties(FlexAID PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17)
    target_link_libraries(FlexAID PRIVATE CUDA::cudart)
    message(STATUS "CUDA enabled — CF batch kernel + Shannon histogram kernel")
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
    target_compile_definitions(FlexAID PRIVATE ENABLE_METAL_CORE)
    set_source_files_properties(LIB/ShannonThermoStack/ShannonMetalBridge.mm PROPERTIES
        LANGUAGE OBJCXX
        COMPILE_FLAGS "-std=c++20 -fno-objc-arc"
    )
endif()

# ─── Python bindings (pybind11 module) ─────────────────────────────────────────
if(BUILD_PYTHON_BINDINGS AND pybind11_FOUND)
    pybind11_add_module(_core
        python/bindings/core_bindings.cpp
        LIB/statmech.cpp
        LIB/encom.cpp
        # BindingMode requires full GA infrastructure
        # Partial exposure for now (thermodynamics only)
    )
    target_include_directories(_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/LIB
    )
    target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})
    
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
        LIB/benchmark_tencom.cpp
        LIB/tencm.cpp
        LIB/statmech.cpp
        LIB/encom.cpp
    )
    target_include_directories(benchmark_tencom PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/LIB)
    target_compile_options(benchmark_tencom PRIVATE -O3 -march=native -ffast-math)
    if(FLEXAIDS_USE_AVX512)
        target_compile_options(benchmark_tencom PRIVATE -mavx512f -mavx512dq -mavx2 -mfma)
        target_compile_definitions(benchmark_tencom PRIVATE FLEXAIDS_USE_AVX512)
    elseif(FLEXAIDS_USE_AVX2)
        target_compile_options(benchmark_tencom PRIVATE -mavx2 -mfma)
    endif()
    if(FLEXAIDS_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(benchmark_tencom PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(FLEXAIDS_USE_EIGEN AND Eigen3_FOUND)
        target_link_libraries(benchmark_tencom PRIVATE Eigen3::Eigen)
        target_compile_definitions(benchmark_tencom PRIVATE FLEXAIDS_HAS_EIGEN)
    endif()
    message(STATUS "TeNCoM benchmark enabled — build target: benchmark_tencom")
endif()
