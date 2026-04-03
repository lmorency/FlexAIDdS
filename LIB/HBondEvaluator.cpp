// HBondEvaluator.cpp — Directional hydrogen bonding potential (non-inline helpers)
//
// The core angular_multiplier functions are inline in HBondEvaluator.h.
// This file provides the heavy-atom H-bond scoring entry point that is
// called from vcfunction.cpp during the Voronoi contact loop.
//
// Apache-2.0 © 2026 Le Bonhomme Pharma

#include "HBondEvaluator.h"
#include "flexaid.h"

namespace hbond {

// Evaluate the H-bond angular correction for a contact pair.
//
// Called from vcfunction.cpp for each non-bonded contact where both atoms
// are H-bond capable.  Returns a multiplier in [0, 1] that scales the
// complementarity contribution.
//
// When explicit hydrogen coordinates are available (the donor atom has
// bonded hydrogen atoms in the atom array), the full D–H–A angle is used.
// Otherwise, the heavy-atom distance-based proxy is applied.

double evaluate_contact(const atom_struct* atomA, const atom_struct* atomB,
                        const atom_struct* atoms, double dist) {
    int typeA = atomA->type;
    int typeB = atomB->type;

    // Determine roles: which is donor, which is acceptor
    bool A_donor = is_donor_type(typeA);
    bool B_donor = is_donor_type(typeB);
    bool A_acceptor = is_acceptor_type(typeA);
    bool B_acceptor = is_acceptor_type(typeB);

    // Must have at least one donor and one acceptor
    if (!(A_donor && B_acceptor) && !(B_donor && A_acceptor))
        return 1.0;  // not an H-bond pair, no penalty

    bool salt_bridge = is_salt_bridge_pair(typeA, typeB);

    // Try to find explicit hydrogen on donor
    // Donor heavy atom has bonded hydrogens if bond[0] > 0
    const atom_struct* donor    = nullptr;
    const atom_struct* acceptor = nullptr;

    if (A_donor && B_acceptor) {
        donor    = atomA;
        acceptor = atomB;
    } else {
        donor    = atomB;
        acceptor = atomA;
    }

    // Search for hydrogen bonded to donor
    int n_bonds = donor->bond[0];
    const atom_struct* best_H = nullptr;
    double best_angle_mult = 0.0;

    for (int b = 1; b <= n_bonds && b <= 6; ++b) {
        int bonded_idx = donor->bond[b];
        if (bonded_idx < 0) continue;

        const atom_struct* bonded = &atoms[bonded_idx];

        // Check if bonded atom is hydrogen (type == 0 or name starts with 'H')
        if (bonded->name[0] != 'H' && bonded->name[0] != 'h')
            continue;

        // Compute full D–H–A angular multiplier
        double mult = angular_multiplier(
            static_cast<double>(donor->coor[0]),
            static_cast<double>(donor->coor[1]),
            static_cast<double>(donor->coor[2]),
            static_cast<double>(bonded->coor[0]),
            static_cast<double>(bonded->coor[1]),
            static_cast<double>(bonded->coor[2]),
            static_cast<double>(acceptor->coor[0]),
            static_cast<double>(acceptor->coor[1]),
            static_cast<double>(acceptor->coor[2]),
            salt_bridge);

        if (mult > best_angle_mult) {
            best_angle_mult = mult;
            best_H = bonded;
        }
    }

    if (best_H != nullptr) {
        return best_angle_mult;
    }

    // No explicit hydrogen found — use heavy-atom proxy
    return angular_multiplier_heavy_atom(
        static_cast<double>(donor->coor[0]),
        static_cast<double>(donor->coor[1]),
        static_cast<double>(donor->coor[2]),
        static_cast<double>(acceptor->coor[0]),
        static_cast<double>(acceptor->coor[1]),
        static_cast<double>(acceptor->coor[2]),
        dist, salt_bridge);
}

} // namespace hbond
