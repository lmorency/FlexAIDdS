// ShannonMetalBridge.h — C++ interface to Apple Metal Shannon GPU kernel
//
// Provides a C++ header-only interface. The implementation is in
// ShannonMetalBridge.mm (Objective-C++) and compiled only on APPLE targets.
// On non-Apple platforms this header is included but the implementation is
// never compiled; callers guard behind #ifdef FLEXAIDS_HAS_METAL_SHANNON.
#pragma once

#include <vector>

namespace ShannonMetalBridge {

// Compute Shannon entropy of `energies` using a Metal GPU histogram kernel.
// Falls back to CPU if no Metal device is available.
// Returns entropy in nats (natural logarithm).
double compute_shannon_entropy_metal(const std::vector<double>& energies,
                                     int num_bins = 20);

} // namespace ShannonMetalBridge
