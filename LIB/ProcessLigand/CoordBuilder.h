// CoordBuilder.h — 3D coordinate generation from molecular graph
//
// Builds 3D atom coordinates from topology (SMILES path) using:
//   1. BFS spanning tree traversal
//   2. Ideal bond lengths from covalent radii
//   3. Ideal bond angles from hybridization (109.5° sp3, 120° sp2, 180° sp)
//   4. Staggered dihedrals (sp3) or planar (sp2/aromatic)
//   5. Ring closure via dihedral adjustment
//   6. Clash resolution via simple repulsive displacement
//
// No external dependencies. Pure C++20 + Eigen.
//
// Copyright 2026 Le Bonhomme Pharma
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "BonMol.h"
#include <Eigen/Dense>

namespace bonmol {

struct CoordBuilderOptions {
    int   max_clash_iterations = 50;    // clash resolution passes
    float clash_threshold      = 0.8f;  // Å — atoms closer than this are clashing
    float clash_push_factor    = 0.5f;  // push displacement per iteration
    bool  randomize_dihedrals  = false; // add noise to sp3 dihedrals
    unsigned int seed          = 42;
};

/// Generate 3D coordinates for all atoms in mol that have NaN coords.
/// Modifies mol.coords in-place. Returns true if successful.
bool build_3d_coords(BonMol& mol, const CoordBuilderOptions& opts = {});

} // namespace bonmol
