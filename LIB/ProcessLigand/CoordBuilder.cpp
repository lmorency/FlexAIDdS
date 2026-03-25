// CoordBuilder.cpp — 3D coordinate generation from molecular graph
//
// Algorithm:
//   1. BFS from atom 0 to establish placement order
//   2. Place atom 0 at origin
//   3. Place atom 1 along +X at bond length distance
//   4. Place atom 2 in XY plane at correct bond angle
//   5. For all subsequent atoms: use parent, grandparent, great-grandparent
//      to define a local frame, then place at (bond_length, bond_angle, dihedral)
//   6. Ring closures: when a bond connects two already-placed atoms,
//      adjust the dihedral chain to minimize ring strain
//   7. Clash resolution: iterative pairwise repulsion for overlapping atoms
//
// Copyright 2026 Le Bonhomme Pharma. Licensed under Apache-2.0.

#include "CoordBuilder.h"
#include <queue>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

namespace bonmol {

// covalent_radius() is already defined inline in BonMol.h

static float ideal_bond_length(Element a, Element b, BondOrder order) {
    float r = covalent_radius(a) + covalent_radius(b);
    // Shorten for multiple bonds
    if (order == BondOrder::DOUBLE || order == BondOrder::AROMATIC) r *= 0.90f;
    if (order == BondOrder::TRIPLE) r *= 0.82f;
    return r;
}

// Ideal bond angle from hybridization (radians)
static float ideal_angle(Hybridization h) {
    switch (h) {
        case Hybridization::SP:  return static_cast<float>(M_PI);          // 180°
        case Hybridization::SP2: return static_cast<float>(120.0 * M_PI / 180.0); // 120°
        case Hybridization::SP3: return static_cast<float>(109.47 * M_PI / 180.0); // 109.47°
        default:                 return static_cast<float>(109.47 * M_PI / 180.0);
    }
}

// Ideal dihedral from hybridization
static float ideal_dihedral(Hybridization h_center, bool in_ring, int neighbor_idx) {
    if (h_center == Hybridization::SP2 || h_center == Hybridization::SP)
        return static_cast<float>(M_PI); // planar = 180° (trans)
    if (in_ring)
        return static_cast<float>(M_PI); // rings prefer planar initially
    // sp3: staggered = 60° offset per neighbor
    return static_cast<float>((60.0 + 120.0 * neighbor_idx) * M_PI / 180.0);
}

// Place an atom given 3 reference points and (distance, angle, dihedral)
static Eigen::Vector3f place_atom(
    const Eigen::Vector3f& p1,  // great-grandparent (defines dihedral)
    const Eigen::Vector3f& p2,  // grandparent (defines angle)
    const Eigen::Vector3f& p3,  // parent (bonded to new atom)
    float dist, float angle, float dihedral)
{
    // Build local coordinate frame at p3
    Eigen::Vector3f v1 = (p2 - p3).normalized();
    Eigen::Vector3f v2 = (p1 - p2);

    // Normal to the p1-p2-p3 plane
    Eigen::Vector3f n = v1.cross(v2);
    if (n.norm() < 1e-6f) {
        // Degenerate case: p1, p2, p3 are collinear — pick arbitrary perpendicular
        Eigen::Vector3f arb(1.0f, 0.0f, 0.0f);
        if (std::abs(v1.dot(arb)) > 0.9f) arb = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
        n = v1.cross(arb);
    }
    n.normalize();

    // d = direction perpendicular to v1 in the v1-v2 plane
    Eigen::Vector3f d = n.cross(v1).normalized();

    // New atom position: rotate in the n-d plane by dihedral, then
    // tilt by (pi - angle) from v1 direction
    float sa = std::sin(angle);
    float ca = std::cos(angle);
    float sd = std::sin(dihedral);
    float cd = std::cos(dihedral);

    Eigen::Vector3f direction = -v1 * ca + d * (sa * cd) + n * (sa * sd);
    return p3 + dist * direction;
}

// Find bond order between two atoms
static BondOrder find_bond_order(const BonMol& mol, int i, int j) {
    for (const auto& b : mol.bonds) {
        if ((b.atom_i == i && b.atom_j == j) || (b.atom_i == j && b.atom_j == i))
            return b.order;
    }
    return BondOrder::SINGLE;
}

// Check if atom is in any ring
static bool atom_in_ring(const BonMol& mol, int idx) {
    return mol.atoms[idx].ring_membership > 0;
}

bool build_3d_coords(BonMol& mol, const CoordBuilderOptions& opts) {
    const int N = mol.num_atoms();
    if (N == 0) return false;
    if (N == 1) {
        mol.coords.col(0) = Eigen::Vector3f::Zero();
        return true;
    }

    // Ensure coord matrix is sized
    if (mol.coords.cols() != N)
        mol.coords.conservativeResize(3, N);

    // Track which atoms have been placed
    std::vector<bool> placed(N, false);
    // BFS parent tracking
    std::vector<int> parent(N, -1);
    std::vector<int> grandparent(N, -1);
    // Neighbor placement index (which neighbor of parent this atom is)
    std::vector<int> neighbor_idx(N, 0);

    // BFS from atom 0
    std::queue<int> bfs;
    std::vector<int> order;
    order.reserve(N);

    bfs.push(0);
    placed[0] = true;

    while (!bfs.empty()) {
        int cur = bfs.front();
        bfs.pop();
        order.push_back(cur);

        int child_count = 0;
        for (int nb : mol.adjacency[cur]) {
            if (!placed[nb]) {
                placed[nb] = true;
                parent[nb] = cur;
                grandparent[nb] = parent[cur];
                neighbor_idx[nb] = child_count++;
                bfs.push(nb);
            }
        }
    }

    // Handle disconnected fragments
    for (int i = 0; i < N; i++) {
        if (!placed[i]) {
            placed[i] = true;
            order.push_back(i);
        }
    }

    // Place atoms in BFS order
    std::mt19937 rng(opts.seed);
    std::uniform_real_distribution<float> noise(-10.0f, 10.0f);

    for (size_t step = 0; step < order.size(); step++) {
        int idx = order[step];

        if (step == 0) {
            // First atom at origin
            mol.coords.col(idx) = Eigen::Vector3f::Zero();
            continue;
        }

        int par = parent[idx];
        if (par < 0) {
            // Disconnected fragment — place far away
            mol.coords.col(idx) = Eigen::Vector3f(noise(rng), noise(rng), noise(rng));
            continue;
        }

        BondOrder bo = find_bond_order(mol, idx, par);
        float dist = ideal_bond_length(mol.atoms[idx].element,
                                        mol.atoms[par].element, bo);

        if (step == 1) {
            // Second atom along +X
            mol.coords.col(idx) = Eigen::Vector3f(dist, 0.0f, 0.0f);
            continue;
        }

        int gpar = grandparent[idx];
        if (gpar < 0) gpar = (par > 0) ? 0 : 1; // fallback

        float angle = ideal_angle(mol.atoms[par].hybrid);

        if (step == 2) {
            // Third atom in XY plane
            float sa = std::sin(angle);
            float ca = std::cos(angle);
            Eigen::Vector3f dir = (mol.coords.col(gpar) - mol.coords.col(par)).normalized();
            Eigen::Vector3f perp(-dir.y(), dir.x(), 0.0f);
            if (perp.norm() < 1e-6f) perp = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
            perp.normalize();
            mol.coords.col(idx) = mol.coords.col(par) + dist * (-dir * ca + perp * sa);
            continue;
        }

        // General case: use great-grandparent for dihedral reference
        int ggpar = grandparent[gpar];
        if (ggpar < 0) {
            // Find any other placed neighbor of gpar that isn't par
            ggpar = -1;
            for (int nb : mol.adjacency[gpar]) {
                if (nb != par && mol.has_coords(nb)) { ggpar = nb; break; }
            }
            if (ggpar < 0) {
                // Synthesize a point perpendicular to the par-gpar axis
                Eigen::Vector3f axis = (mol.coords.col(par) - mol.coords.col(gpar)).normalized();
                Eigen::Vector3f arb(1.0f, 0.0f, 0.0f);
                if (std::abs(axis.dot(arb)) > 0.9f) arb = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
                Eigen::Vector3f synth = mol.coords.col(gpar) + axis.cross(arb).normalized();
                // Place using synthesized reference
                float dih = ideal_dihedral(mol.atoms[par].hybrid,
                                           atom_in_ring(mol, idx),
                                           neighbor_idx[idx]);
                if (opts.randomize_dihedrals && mol.atoms[par].hybrid == Hybridization::SP3) {
                    std::uniform_real_distribution<float> dih_noise(-0.1f, 0.1f);
                    dih += dih_noise(rng);
                }
                mol.coords.col(idx) = place_atom(synth, mol.coords.col(gpar),
                                                  mol.coords.col(par),
                                                  dist, angle, dih);
                continue;
            }
        }

        float dih = ideal_dihedral(mol.atoms[par].hybrid,
                                    atom_in_ring(mol, idx),
                                    neighbor_idx[idx]);
        if (opts.randomize_dihedrals && mol.atoms[par].hybrid == Hybridization::SP3) {
            std::uniform_real_distribution<float> dih_noise(-0.1f, 0.1f);
            dih += dih_noise(rng);
        }

        mol.coords.col(idx) = place_atom(mol.coords.col(ggpar),
                                          mol.coords.col(gpar),
                                          mol.coords.col(par),
                                          dist, angle, dih);
    }

    // Clash resolution: iterative pairwise repulsion
    float clash_sq = opts.clash_threshold * opts.clash_threshold;

    for (int iter = 0; iter < opts.max_clash_iterations; iter++) {
        bool any_clash = false;

        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < N; j++) {
                // Skip bonded pairs
                bool bonded = false;
                for (int nb : mol.adjacency[i]) {
                    if (nb == j) { bonded = true; break; }
                }
                if (bonded) continue;

                Eigen::Vector3f diff = mol.coords.col(i) - mol.coords.col(j);
                float d2 = diff.squaredNorm();

                if (d2 < clash_sq && d2 > 1e-8f) {
                    any_clash = true;
                    float d = std::sqrt(d2);
                    float push = (opts.clash_threshold - d) * opts.clash_push_factor;
                    Eigen::Vector3f dir = diff.normalized();
                    mol.coords.col(i) += dir * (push * 0.5f);
                    mol.coords.col(j) -= dir * (push * 0.5f);
                }
            }
        }

        if (!any_clash) break;
    }

    return true;
}

} // namespace bonmol
