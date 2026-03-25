// RingPerception.cpp — SSSR implementation using Horton's algorithm
//
// Copyright 2026 Le Bonhomme Pharma
// SPDX-License-Identifier: Apache-2.0

#include "RingPerception.h"

#include <algorithm>
#include <cassert>
#include <deque>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace bonmol {
namespace ring_perception {

// ---------------------------------------------------------------------------
// BFS shortest path
// ---------------------------------------------------------------------------

std::vector<int> bfs_shortest_path(const BonMol& mol, int src, int dst,
                                    int forbidden_bond) {
    const int N = mol.num_atoms();
    if (src < 0 || dst < 0 || src >= N || dst >= N) return {};

    std::vector<int> parent(N, -1);
    std::vector<bool> visited(N, false);
    std::deque<int> queue;

    visited[src] = true;
    queue.push_back(src);

    bool found = false;
    while (!queue.empty() && !found) {
        int curr = queue.front();
        queue.pop_front();

        for (size_t k = 0; k < mol.adjacency[curr].size(); ++k) {
            int nb   = mol.adjacency[curr][k];
            int bidx = mol.bond_adj[curr][k];
            if (bidx == forbidden_bond) continue;
            if (visited[nb]) continue;
            visited[nb] = true;
            parent[nb]  = curr;
            if (nb == dst) { found = true; break; }
            queue.push_back(nb);
        }
    }

    if (!found) return {};

    // Reconstruct path dst → ... → src, then reverse
    std::vector<int> path;
    for (int v = dst; v != -1; v = parent[v]) path.push_back(v);
    std::reverse(path.begin(), path.end());
    return path; // path[0] == src, path.back() == dst
}

// ---------------------------------------------------------------------------
// Convert path to Ring
// ---------------------------------------------------------------------------

Ring path_to_ring(const std::vector<int>& path) {
    Ring r;
    r.atom_indices = path;
    r.size         = static_cast<int>(path.size());
    r.is_aromatic  = false; // filled in by Aromaticity module
    return r;
}

// ---------------------------------------------------------------------------
// Edge incidence vector (GF(2)) for Gaussian elimination
// Used to build independent cycle basis from candidate rings.
// ---------------------------------------------------------------------------

using GF2Vec = std::vector<uint8_t>; // one bit per edge (0 or 1)

static GF2Vec ring_to_gf2(const std::vector<int>& ring_atoms, const BonMol& mol) {
    const int E = mol.num_bonds();
    GF2Vec vec(E, 0);
    int sz = static_cast<int>(ring_atoms.size());
    for (int i = 0; i < sz; ++i) {
        int a = ring_atoms[i];
        int b = ring_atoms[(i + 1) % sz];
        int bidx = mol.find_bond(a, b);
        if (bidx >= 0 && bidx < E) vec[bidx] ^= 1;
    }
    return vec;
}

// GF(2) row-reduce a set of vectors. Returns the independent subset.
static std::vector<size_t> gf2_independent_indices(
        const std::vector<GF2Vec>& vecs, int expected_rank) {
    if (vecs.empty()) return {};
    const int E = static_cast<int>(vecs[0].size());

    // Echelon form matrix: each row is a GF2Vec
    std::vector<GF2Vec> basis;
    std::vector<int>    pivot_col; // pivot column for each basis row
    std::vector<size_t> chosen;

    for (size_t i = 0; i < vecs.size(); ++i) {
        if (static_cast<int>(chosen.size()) >= expected_rank) break;

        GF2Vec v = vecs[i];
        // Reduce v against existing basis
        for (size_t r = 0; r < basis.size(); ++r) {
            int pc = pivot_col[r];
            if (v[pc]) {
                for (int c = 0; c < E; ++c)
                    v[c] ^= basis[r][c];
            }
        }
        // Find first non-zero entry in v
        int pivot = -1;
        for (int c = 0; c < E; ++c) {
            if (v[c]) { pivot = c; break; }
        }
        if (pivot < 0) continue; // linearly dependent — skip
        basis.push_back(v);
        pivot_col.push_back(pivot);
        chosen.push_back(i);
    }
    return chosen;
}

// ---------------------------------------------------------------------------
// Connected components (for circuit rank = E - V + C)
// ---------------------------------------------------------------------------

static int connected_components(const BonMol& mol) {
    const int N = mol.num_atoms();
    if (N == 0) return 0;
    std::vector<bool> visited(N, false);
    int comps = 0;
    for (int start = 0; start < N; ++start) {
        if (visited[start]) continue;
        ++comps;
        std::deque<int> q;
        q.push_back(start);
        visited[start] = true;
        while (!q.empty()) {
            int v = q.front(); q.pop_front();
            for (int nb : mol.adjacency[v]) {
                if (!visited[nb]) { visited[nb] = true; q.push_back(nb); }
            }
        }
    }
    return comps;
}

// ---------------------------------------------------------------------------
// Main SSSR entry point
// ---------------------------------------------------------------------------

RingPerceptionResult perceive_rings(BonMol& mol) {
    RingPerceptionResult result{};
    mol.rings.clear();

    const int N = mol.num_atoms();
    const int E = mol.num_bonds();
    const int C = connected_components(mol);

    // Circuit rank = number of independent rings
    int circuit_rank = E - N + C;
    result.circuit_rank = circuit_rank;

    if (circuit_rank <= 0) {
        // Acyclic molecule: mark all bonds as non-ring
        for (auto& b : mol.bonds) b.in_ring = false;
        for (auto& a : mol.atoms) a.ring_membership = 0;
        result.num_rings = 0;
        return result;
    }

    // -----------------------------------------------------------------------
    // Step 1: Collect candidate rings using Horton's method.
    // For each bond (u,v), find shortest path from u to v NOT using that bond.
    // Ring = that path + the bond itself.
    // We limit to rings of size <= 12 for tractability.
    // -----------------------------------------------------------------------
    struct Candidate {
        std::vector<int> atoms; // ordered ring atoms (no duplication of first)
        int              size;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<size_t>(E) * 2);

    for (int bidx = 0; bidx < E; ++bidx) {
        const Bond& bond = mol.bonds[bidx];
        int u = bond.atom_i;
        int v = bond.atom_j;

        std::vector<int> path = bfs_shortest_path(mol, u, v, bidx);
        if (path.empty()) continue; // bond is a bridge, no ring through it

        // path goes from u ... v; ring = path (already u→...→v, close back with u)
        int sz = static_cast<int>(path.size()); // ring size
        if (sz > 12) {
            result.large_ring_atoms.push_back(u);
            continue;
        }
        Candidate cand;
        cand.atoms = path; // [u, ..., v], close ring by wrapping
        cand.size  = sz;
        candidates.push_back(std::move(cand));
    }

    // Sort candidates by size (smallest first) for greedy SSSR selection
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b){ return a.size < b.size; });

    // Deduplicate candidates (same set of atoms → same ring)
    {
        auto canonical = [](std::vector<int> v) -> std::vector<int> {
            auto it = std::min_element(v.begin(), v.end());
            std::rotate(v.begin(), it, v.end());
            if (v.size() > 1 && v.back() < v[1])
                std::reverse(v.begin() + 1, v.end());
            return v;
        };
        std::unordered_set<std::string> seen;
        std::vector<Candidate> unique;
        unique.reserve(candidates.size());
        for (auto& c : candidates) {
            auto cv = canonical(c.atoms);
            std::string key;
            key.reserve(cv.size() * 4);
            for (int a : cv) { key += std::to_string(a); key += ','; }
            if (seen.insert(key).second) unique.push_back(std::move(c));
        }
        candidates = std::move(unique);
    }

    // -----------------------------------------------------------------------
    // Step 2: GF(2) Gaussian elimination to select circuit_rank independent rings
    // -----------------------------------------------------------------------
    std::vector<GF2Vec> gf2_vecs;
    gf2_vecs.reserve(candidates.size());
    for (const auto& c : candidates) {
        gf2_vecs.push_back(ring_to_gf2(c.atoms, mol));
    }

    std::vector<size_t> chosen = gf2_independent_indices(gf2_vecs, circuit_rank);

    // -----------------------------------------------------------------------
    // Step 3: Build Ring objects from chosen candidates
    // -----------------------------------------------------------------------
    mol.rings.clear();
    mol.rings.reserve(chosen.size());

    for (size_t idx : chosen) {
        Ring r;
        r.atom_indices = candidates[idx].atoms;
        r.size         = candidates[idx].size;
        r.is_aromatic  = false; // set by Aromaticity module
        mol.rings.push_back(std::move(r));
    }

    // -----------------------------------------------------------------------
    // Step 4: Mark atoms and bonds that are in rings
    // -----------------------------------------------------------------------
    // Reset
    for (auto& a : mol.atoms) a.ring_membership = 0;
    for (auto& b : mol.bonds) b.in_ring = false;

    for (const Ring& ring : mol.rings) {
        int sz = ring.size;
        for (int i = 0; i < sz; ++i) {
            int ai = ring.atom_indices[i];
            int aj = ring.atom_indices[(i + 1) % sz];
            mol.atoms[ai].ring_membership++;
            int bidx = mol.find_bond(ai, aj);
            if (bidx >= 0) mol.bonds[bidx].in_ring = true;
        }
    }

    result.num_rings = static_cast<int>(mol.rings.size());
    return result;
}

} // namespace ring_perception
} // namespace bonmol
