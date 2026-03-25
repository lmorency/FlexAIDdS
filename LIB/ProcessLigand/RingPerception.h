// RingPerception.h — Smallest Set of Smallest Rings (SSSR) via Horton/Vismara
//
// Copyright 2026 Le Bonhomme Pharma
// SPDX-License-Identifier: Apache-2.0
//
// Algorithm overview:
//   1. Compute all simple cycle bases via DFS back-edge detection.
//   2. Build a candidate ring set using Horton's method:
//      for each bond (u,v), find the shortest path from u to v avoiding that bond,
//      yielding a candidate ring of size = path_length + 1.
//   3. Select |E| - |V| + C independent rings (where C is the number of connected
//      components) using Gaussian elimination over GF(2) on the edge-incidence
//      vectors of candidate rings, choosing the smallest candidates first.
//   4. Populate mol.rings with the resulting SSSR.
//
// Ring atoms are returned in ring-traversal order (adjacent atoms are consecutive).
// Rings up to size 12 are stored; larger rings are reported separately but not
// included in the SSSR to keep downstream algorithms tractable.

#pragma once

#include "BonMol.h"
#include <vector>
#include <cstdint>

namespace bonmol {
namespace ring_perception {

struct RingPerceptionResult {
    int num_rings;           // total rings found
    int circuit_rank;        // E - V + C (theoretical SSSR size)
    std::vector<int> large_ring_atoms; // first atom of any ring > size 12
};

/// Compute SSSR and populate mol.rings.
/// Also sets Bond::in_ring and Atom::ring_membership for all atoms.
/// Returns diagnostic result.
RingPerceptionResult perceive_rings(BonMol& mol);

/// Helper: return BFS shortest path from src to dst in mol,
/// optionally forbidding a specific bond (specified as bond index).
/// Returns empty vector if no path exists.
std::vector<int> bfs_shortest_path(const BonMol& mol, int src, int dst,
                                    int forbidden_bond = -1);

/// Helper: convert an ordered path [v0, v1, ..., vk] where v0==vk
/// into a Ring descriptor.
Ring path_to_ring(const std::vector<int>& path);

} // namespace ring_perception
} // namespace bonmol
