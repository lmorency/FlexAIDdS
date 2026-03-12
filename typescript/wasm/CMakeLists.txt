# WASM build for FlexAIDdS TypeScript bindings
#
# Build: emcmake cmake -S . -B build && cmake --build build
# Output: flexaidds.wasm + flexaidds.js

cmake_minimum_required(VERSION 3.18)
project(FlexAIDdS_WASM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(flexaidds
    bindings.cpp
    ../../LIB/statmech.cpp
    ../../LIB/encom.cpp
)

target_include_directories(flexaidds PRIVATE ../../LIB)

# Emscripten-specific flags
set_target_properties(flexaidds PROPERTIES
    SUFFIX ".js"
    LINK_FLAGS "--bind -s MODULARIZE=1 -s EXPORT_NAME='FlexAIDdS' -s ALLOW_MEMORY_GROWTH=1 -s WASM=1 -O2"
)
