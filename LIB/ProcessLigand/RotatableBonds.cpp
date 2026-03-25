// RotatableBonds.cpp — Rotatable bond identification
//
// Copyright 2026 Le Bonhomme Pharma
// SPDX-License-Identifier: Apache-2.0

#include "RotatableBonds.h"

namespace bonmol {
namespace rotatable_bonds {

// ---------------------------------------------------------------------------
// Amide bond detection: C(=O)-N
// ---------------------------------------------------------------------------

bool is_amide_bond(const BonMol& mol, int bidx) {
    const Bond& bond = mol.bonds[bidx];
    if (bond.order != BondOrder::SINGLE) return false;

    int i = bond.atom_i;
    int j = bond.atom_j;

    // Try both orientations: C–N
    auto check_amide = [&](int c_idx, int n_idx) -> bool {
        if (mol.atoms[c_idx].element != Element::C) return false;
        if (mol.atoms[n_idx].element != Element::N) return false;
        // c_idx must have a double bond to O
        for (int nb_bidx : mol.bond_adj[c_idx]) {
            const Bond& nb = mol.bonds[nb_bidx];
            if (nb.order == BondOrder::DOUBLE &&
                mol.atoms[(nb.atom_i == c_idx ? nb.atom_j : nb.atom_i)].element == Element::O)
                return true;
        }
        return false;
    };

    return check_amide(i, j) || check_amide(j, i);
}

// ---------------------------------------------------------------------------
// Disulfide bond detection: S-S
// ---------------------------------------------------------------------------

bool is_disulfide_bond(const BonMol& mol, int bidx) {
    const Bond& bond = mol.bonds[bidx];
    if (bond.order != BondOrder::SINGLE) return false;
    return mol.atoms[bond.atom_i].element == Element::S &&
           mol.atoms[bond.atom_j].element == Element::S;
}

// ---------------------------------------------------------------------------
// Check for adjacency to triple bond
// ---------------------------------------------------------------------------

static bool adjacent_to_triple(const BonMol& mol, int atom_idx) {
    for (int bidx : mol.bond_adj[atom_idx]) {
        if (mol.bonds[bidx].order == BondOrder::TRIPLE) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Check for aromatic C–N bond (partial double, should not rotate freely)
// ---------------------------------------------------------------------------

static bool is_aromatic_cn(const BonMol& mol, int bidx) {
    const Bond& bond = mol.bonds[bidx];
    if (bond.order != BondOrder::SINGLE) return false;
    int i = bond.atom_i;
    int j = bond.atom_j;
    bool ij_cn = (mol.atoms[i].element == Element::C && mol.atoms[j].element == Element::N);
    bool ji_cn = (mol.atoms[j].element == Element::C && mol.atoms[i].element == Element::N);
    if (!ij_cn && !ji_cn) return false;
    int c_idx = ij_cn ? i : j;
    // If the C atom is aromatic (part of aromatic ring), the C–N bond has
    // partial double-bond character (like aniline).
    return mol.atoms[c_idx].is_aromatic;
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

RotatableBondsResult identify_rotatable_bonds(BonMol& mol) {
    RotatableBondsResult result;
    result.count = 0;

    // Reset
    for (auto& b : mol.bonds) b.is_rotatable = false;

    for (int bidx = 0; bidx < mol.num_bonds(); ++bidx) {
        Bond& bond = mol.bonds[bidx];

        // Rule 1: must be single bond
        if (bond.order != BondOrder::SINGLE) continue;

        // Rule 2: not in ring
        if (bond.in_ring) continue;

        int i = bond.atom_i;
        int j = bond.atom_j;

        // Rule 3: both atoms must have degree >= 2
        if (mol.degree(i) < 2 || mol.degree(j) < 2) continue;

        // Rule 4: not amide
        if (is_amide_bond(mol, bidx)) continue;

        // Rule 5: not disulfide
        if (is_disulfide_bond(mol, bidx)) continue;

        // Rule 6: not adjacent to triple bond
        if (adjacent_to_triple(mol, i) || adjacent_to_triple(mol, j)) continue;

        // Rule 7: not aromatic C–N (partial double-bond character)
        if (is_aromatic_cn(mol, bidx)) continue;

        // Passed all rules: mark as rotatable
        bond.is_rotatable = true;
        result.bond_indices.push_back(bidx);
        ++result.count;
    }

    return result;
}

} // namespace rotatable_bonds
} // namespace bonmol
