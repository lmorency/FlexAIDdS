/// @file MPIMergeable.h
/// @brief C++20 concept for types that support serialize/merge transport.
///
/// Any accumulator that needs to be exchanged across MPI ranks (or any
/// other transport layer) should satisfy the MPIMergeable concept. This
/// enables generic exchange functions in MPITransport and future transport
/// backends (e.g., iCloud relay, WebSocket bridge).
///
/// The pattern:
///   1. Each rank accumulates data locally.
///   2. serialize() → byte buffer (snapshot of current state).
///   3. Transport the buffer to remote ranks.
///   4. deserialize_merge(buffer) → merge remote data into local state.
///
/// This same pattern appears across:
///   - SharedPosePool (C++, MPI): best-pose exchange between GA regions
///   - TargetKnowledgeBase (C++, MPI): cross-ligand knowledge accumulation
///   - FleetScheduler (Swift, iCloud): chunk result aggregation
///   - IntelligenceEngine (TypeScript, JSON): analysis history tracking
///
/// Copyright 2026 Le Bonhomme Pharma
/// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>
#include <concepts>
#include <cstddef>

namespace transport {

/// Concept for types that can be serialized to a byte buffer and merged
/// from a remote byte buffer. Enables generic MPI (or other transport)
/// exchange without type-specific boilerplate.
///
/// Required methods:
///   - serialize() const → std::vector<char>
///   - deserialize_merge(const char* data, size_t len) → void
template <typename T>
concept MPIMergeable = requires(T t, const T ct, const char* data, size_t len) {
    { ct.serialize() } -> std::convertible_to<std::vector<char>>;
    { t.deserialize_merge(data, len) } -> std::same_as<void>;
};

} // namespace transport
