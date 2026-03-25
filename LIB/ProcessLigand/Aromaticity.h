// Aromaticity.h — Hückel aromaticity detection and Kekulé assignment
//
// Copyright 2026 Le Bonhomme Pharma
// SPDX-License-Identifier: Apache-2.0
//
// Applies Hückel's 4n+2 rule to SSSR rings:
//   1. For each ring, check that all atoms are sp2 or have lone-pair π donors.
//   2. Count π electrons contributed by each atom.
//   3. If total π electrons satisfy 4n+2 (n=0,1,2,...), mark ring aromatic.
//   4. Mark aromatic atoms and bonds.
//   5. Kekulize: assign alternating single/double bonds within aromatic rings
//      via a matching algorithm so downstream valence checks are consistent.
//
// π electron donation rules:
//   C sp2     → 1  (p orbital)
//   N sp2     → 1  (N in ring with double bond exo; e.g. pyridine N)
//   N sp3/pl3 → 2  (lone pair; e.g. pyrrole N)
//   O         → 2  (lone pair; e.g. furan O)
//   S         → 2  (lone pair; e.g. thiophene S)
//   B sp2     → 0  (empty p orbital; counts as deficit)
//
// Must be called after RingPerception::perceive_rings() and after a preliminary
// hybridisation pass (done inside assign_aromaticity for simple cases).

#pragma once

#include "BonMol.h"

namespace bonmol {
namespace aromaticity {

struct AromaticityResult {
    int num_aromatic_rings;
    int num_aromatic_atoms;
    int num_aromatic_bonds;
    bool kekulized; // true if Kekulé assignment succeeded for all aromatic rings
};

/// Main entry point. Modifies mol in place:
///   - Sets Ring::is_aromatic
///   - Sets Atom::is_aromatic, Atom::hybrid for ring atoms
///   - Sets Bond::is_aromatic, and Kekulizes bond orders
/// Requires mol.rings to be populated (call perceive_rings first).
AromaticityResult assign_aromaticity(BonMol& mol);

/// Attempt Kekulé assignment for a single aromatic ring:
/// find an alternating single/double bond pattern consistent with atom valences.
/// Returns true on success. On failure, bonds remain AROMATIC.
bool kekulize_ring(BonMol& mol, const Ring& ring);

/// Count π electrons for an atom in a candidate aromatic ring.
/// Returns -1 if the atom cannot contribute (not sp2-capable).
int pi_electron_count(const BonMol& mol, int atom_idx, const Ring& ring);

/// Assign hybridisation for non-ring atoms as a prerequisite.
void assign_hybridisation(BonMol& mol);

} // namespace aromaticity
} // namespace bonmol
