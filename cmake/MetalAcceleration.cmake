# CMakeLists.txt - Metal GPU Acceleration Module
# Add this to your main FlexAIDdS CMakeLists.txt

# ============================================================================
# Metal GPU Acceleration (macOS only)
# ============================================================================

if(APPLE)
    message(STATUS "Detecting Metal GPU support...")
    
    # Find Metal frameworks
    find_library(METAL_LIBRARY Metal)
    find_library(METALKIT_LIBRARY MetalKit)
    find_library(FOUNDATION_LIBRARY Foundation)
    find_library(METAL_PERFORMANCE_SHADERS MetalPerformanceShaders)
    
    if(METAL_LIBRARY AND METALKIT_LIBRARY AND FOUNDATION_LIBRARY)
        set(ENABLE_METAL ON CACHE BOOL "Enable Metal GPU acceleration")
        message(STATUS "✓ Metal GPU acceleration: ENABLED")
        message(STATUS "  Metal framework: ${METAL_LIBRARY}")
        message(STATUS "  MetalKit framework: ${METALKIT_LIBRARY}")
        
        # Metal source files
        set(METAL_SOURCES
            src/acceleration/metal_scoring.h
            src/acceleration/metal_scoring.mm
        )
        
        # Configure Objective-C++ compilation for .mm files
        set_source_files_properties(
            src/acceleration/metal_scoring.mm
            PROPERTIES
            COMPILE_FLAGS "-x objective-c++ -fobjc-arc -Wno-deprecated-declarations"
            LANGUAGE CXX
        )
        
        # Add Metal sources to library
        target_sources(flexaid_lib PRIVATE ${METAL_SOURCES})
        
        # Link Metal frameworks
        target_link_libraries(flexaid_lib PRIVATE
            ${METAL_LIBRARY}
            ${METALKIT_LIBRARY}
            ${FOUNDATION_LIBRARY}
            ${METAL_PERFORMANCE_SHADERS}
        )
        
        # Define preprocessor macro
        target_compile_definitions(flexaid_lib PRIVATE FLEXAID_USE_METAL)
        target_compile_definitions(flexaid_lib PRIVATE METAL_AVAILABLE=1)
        
        # Metal test executable
        add_executable(test_metal_scoring
            tests/test_metal_scoring.cpp
            ${METAL_SOURCES}
        )
        
        set_source_files_properties(
            src/acceleration/metal_scoring.mm
            PROPERTIES
            COMPILE_FLAGS "-x objective-c++ -fobjc-arc"
        )
        
        target_link_libraries(test_metal_scoring PRIVATE
            flexaid_lib
            ${METAL_LIBRARY}
            ${METALKIT_LIBRARY}
            ${FOUNDATION_LIBRARY}
            ${METAL_PERFORMANCE_SHADERS}
        )
        
        target_include_directories(test_metal_scoring PRIVATE
            ${CMAKE_SOURCE_DIR}/src
        )
        
        # Add test to CTest
        enable_testing()
        add_test(NAME MetalGPUTests COMMAND test_metal_scoring)
        
        message(STATUS "✓ Metal test suite configured: test_metal_scoring")
        
    else()
        set(ENABLE_METAL OFF CACHE BOOL "Enable Metal GPU acceleration" FORCE)
        message(STATUS "✗ Metal GPU acceleration: DISABLED")
        message(STATUS "  Reason: Metal frameworks not found (requires macOS 11.0+)")
    endif()
    
    # Metal performance profiling tools
    if(ENABLE_METAL)
        add_executable(benchmark_metal
            benchmarks/benchmark_metal.cpp
            ${METAL_SOURCES}
        )
        
        target_link_libraries(benchmark_metal PRIVATE
            flexaid_lib
            ${METAL_LIBRARY}
            ${METALKIT_LIBRARY}
            ${FOUNDATION_LIBRARY}
        )
        
        target_include_directories(benchmark_metal PRIVATE
            ${CMAKE_SOURCE_DIR}/src
        )
        
        message(STATUS "✓ Metal benchmark tool configured: benchmark_metal")
    endif()
    
else()
    message(STATUS "✗ Metal GPU acceleration: UNAVAILABLE (not macOS)")
    set(ENABLE_METAL OFF CACHE BOOL "Enable Metal GPU acceleration" FORCE)
endif()

# ============================================================================
# Installation Rules for Metal Headers
# ============================================================================

if(ENABLE_METAL)
    install(FILES
        src/acceleration/metal_scoring.h
        DESTINATION include/flexaid/acceleration
        COMPONENT metal
    )
    
    install(FILES
        docs/Metal-Integration-Guide.md
        DESTINATION share/flexaid/docs
        COMPONENT metal
    )
endif()

# ============================================================================
# Platform Detection Helper
# ============================================================================

if(ENABLE_METAL)
    # Detect Apple Silicon vs Intel Mac
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE ARCH_NAME
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(ARCH_NAME STREQUAL "arm64")
        message(STATUS "  Platform: Apple Silicon (M1/M2/M3/M4)")
        set(METAL_PLATFORM "AppleSilicon")
    elseif(ARCH_NAME STREQUAL "x86_64")
        message(STATUS "  Platform: Intel Mac (discrete GPU required)")
        set(METAL_PLATFORM "IntelMac")
    else()
        message(STATUS "  Platform: Unknown (${ARCH_NAME})")
        set(METAL_PLATFORM "Unknown")
    endif()
    
    # Query GPU info at build time (optional, for diagnostics)
    execute_process(
        COMMAND system_profiler SPDisplaysDataType
        OUTPUT_VARIABLE GPU_INFO
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    
    string(REGEX MATCH "Chipset Model: ([^\n]+)" GPU_MODEL_MATCH "${GPU_INFO}")
    if(GPU_MODEL_MATCH)
        message(STATUS "  Detected GPU: ${CMAKE_MATCH_1}")
    endif()
endif()

# ============================================================================
# Build Options
# ============================================================================

option(METAL_ENABLE_VALIDATION "Enable Metal GPU validation layers" OFF)
option(METAL_ENABLE_PROFILING "Enable Metal performance profiling" OFF)
option(METAL_USE_HALF_PRECISION "Use FP16 for grid storage (experimental)" OFF)

if(METAL_ENABLE_VALIDATION)
    target_compile_definitions(flexaid_lib PRIVATE METAL_VALIDATION=1)
    message(STATUS "  Metal validation layers: ENABLED")
endif()

if(METAL_ENABLE_PROFILING)
    target_compile_definitions(flexaid_lib PRIVATE METAL_PROFILING=1)
    message(STATUS "  Metal GPU profiling: ENABLED")
endif()

if(METAL_USE_HALF_PRECISION)
    target_compile_definitions(flexaid_lib PRIVATE METAL_USE_HALF=1)
    message(STATUS "  Half-precision mode: ENABLED (experimental)")
endif()

# ============================================================================
# Example Usage in Parent CMakeLists.txt
# ============================================================================

# To integrate into main FlexAIDdS project:
#
# 1. Add this file as: cmake/MetalAcceleration.cmake
# 2. In main CMakeLists.txt, add:
#
#    include(cmake/MetalAcceleration.cmake)
#
# 3. Build with Metal support:
#
#    cmake -DENABLE_METAL=ON -DCMAKE_BUILD_TYPE=Release ..
#    make -j$(sysctl -n hw.ncpu)
#    ./test_metal_scoring
#
# 4. Disable Metal (fallback to CPU):
#
#    cmake -DENABLE_METAL=OFF ..
