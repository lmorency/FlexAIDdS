// TurboQuantMetalBridge.h — C-callable bridge to Metal GPU TurboQuant dispatch
// Follows the same pattern as CavityDetect/CavityDetectMetalBridge.h.
//
// Copyright 2026 Le Bonhomme Pharma. Licensed under Apache-2.0.

#pragma once

#include <cstdint>
#include <vector>

// Returns true if Metal dispatch succeeded.
// Returns false if Metal is unavailable; caller falls back to CPU path.

/// Batch quantize N vectors of dimension d on Metal GPU.
/// @param pi_data      Row-major d×d rotation matrix (host pointer)
/// @param input_data   Row-major N×d input vectors (host pointer)
/// @param out_indices  Receives N×d uint8 centroid indices
/// @param out_norms    Receives N float L2 norms
/// @param boundaries   Sorted codebook boundary values (2^bit_width - 1 entries)
/// @param num_boundaries  Number of boundary values
/// @param N            Number of vectors
/// @param d            Dimension per vector
/// @param bit_width    Quantization bits (1–4)
bool turboquant_metal_batch_quantize(
    const float*  pi_data,
    const float*  input_data,
    uint8_t*      out_indices,
    float*        out_norms,
    const float*  boundaries,
    int           num_boundaries,
    int           N,
    int           d,
    int           bit_width);

/// Batch dequantize N vectors of dimension d on Metal GPU.
/// @param pit_data     Row-major d×d transpose rotation matrix (host pointer)
/// @param indices      N×d uint8 centroid indices (host pointer)
/// @param centroids    Codebook centroids (2^bit_width entries)
/// @param out_data     Receives N×d float reconstructed vectors
/// @param N            Number of vectors
/// @param d            Dimension per vector
/// @param bit_width    Quantization bits
bool turboquant_metal_batch_dequantize(
    const float*   pit_data,
    const uint8_t* indices,
    const float*   centroids,
    float*         out_data,
    int            N,
    int            d,
    int            bit_width);
