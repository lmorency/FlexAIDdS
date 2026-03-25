// ValenceChecker.h — Molecular graph valence validation and implicit H assignment
//
// Copyright 2026 Le Bonhomme Pharma
// SPDX-License-Identifier: Apache-2.0
//
// Validates that each atom's bond-order sum is consistent with its element's
// expected valence(s), accounting for formal charge.
// Also computes implicit hydrogen count for atoms loaded from SDF/MOL2 files
// where implicit H is not explicitly represented.

#pragma once

#include "BonMol.h"
#include <vector>
#include <string>

namespace bonmol {
namespace valence {

struct ValenceError {
    int         atom_idx;
    Element     element;
    float       bond_order_sum;
    int         expected_valence;
    std::string message;
};

struct ValenceCheckResult {
    bool                    valid;
    std::vector<ValenceError> errors;
    std::vector<ValenceError> warnings; // over-valenced but tolerable (e.g. S(=O)(=O))
};

/// Check all atoms in mol for valence violations.
/// Also updates Atom::implicit_h_count for atoms where it is 0 and appropriate.
ValenceCheckResult check_valence(BonMol& mol);

/// Expected "normal" valences for an element, given its formal charge.
/// Returns the vector of acceptable total bond-order sums (integers).
std::vector<int> expected_valences(Element elem, int formal_charge);

/// Compute implicit H count for a single atom after all explicit bonds are placed.
/// Returns 0 if the element does not get implicit H (metals, noble gases, etc.)
int compute_implicit_h(const BonMol& mol, int atom_idx);

} // namespace valence
} // namespace bonmol
