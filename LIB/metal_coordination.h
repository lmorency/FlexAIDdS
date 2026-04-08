// metal_coordination.h — Geometry-aware metal ion coordination potential
//
// Implements a Morse-like potential for metal-ligand coordination bonds,
// with per-metal/per-donor ideal distances and well depths derived from
// Cambridge Structural Database metalloprotein surveys (Harding 2001, 2006).
//
// Also provides coordination-number tracking and a gentle quadratic
// CN-deviation penalty applied after the pairwise loop completes.
//
// Integration: called from vcfunction.cpp during the pairwise contact loop.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <array>

// Need the full atom_struct definition for accessing .type member
#include "flexaid.h"

namespace metal_coord {

// ─── Coordination geometry types ────────────────────────────────────────────
enum class Geometry : uint8_t {
    TETRAHEDRAL,           // CN=4, ideal angle=109.5
    SQUARE_PLANAR,         // CN=4, ideal angle=90.0
    TRIGONAL_BIPYRAMIDAL,  // CN=5, ideal angle=90/120
    OCTAHEDRAL,            // CN=6, ideal angle=90.0
    PENTAGONAL_BIPYRAMIDAL,// CN=7, ideal angle=72.0/90.0
    SQUARE_ANTIPRISMATIC   // CN=8, ideal angle=70.5/99.6
};

// ─── Per-metal parameters ───────────────────────────────────────────────────
struct MetalParams {
    int      sybyl_type;    // SYBYL type (28=MG, 35=ZN, 36=CA, 37=FE)
    int      ideal_cn;      // preferred coordination number
    Geometry geometry;      // preferred geometry
    double   angle_primary; // primary ideal L-M-L angle (degrees)
};

// Lookup table: indexed by SYBYL type.  Only entries 28–38 are metals.
// Returns nullptr if the SYBYL type is not a recognized metal.
inline const MetalParams* get_metal_params(int sybyl_type) noexcept {
    // SYBYL: 28=MG, 29=SR, 30=CU, 31=MN, 32=HG, 33=CD, 34=NI, 35=ZN, 36=CA, 37=FE, 38=CO
    static constexpr MetalParams table[] = {
        {28, 6, Geometry::OCTAHEDRAL,            90.0},  // Mg2+
        {29, 0, Geometry::OCTAHEDRAL,             0.0},  // Sr — no params (placeholder)
        {30, 4, Geometry::TETRAHEDRAL,           109.5}, // Cu2+
        {31, 6, Geometry::OCTAHEDRAL,            90.0},  // Mn2+
        {32, 0, Geometry::TETRAHEDRAL,            0.0},  // Hg — placeholder
        {33, 0, Geometry::OCTAHEDRAL,             0.0},  // Cd — placeholder
        {34, 6, Geometry::OCTAHEDRAL,            90.0},  // Ni2+
        {35, 4, Geometry::TETRAHEDRAL,           109.5}, // Zn2+
        {36, 7, Geometry::PENTAGONAL_BIPYRAMIDAL, 72.0}, // Ca2+
        {37, 6, Geometry::OCTAHEDRAL,            90.0},  // Fe2+/3+
        {38, 6, Geometry::OCTAHEDRAL,            90.0},  // Co2+
    };
    if (sybyl_type < 28 || sybyl_type > 38) return nullptr;
    const MetalParams& p = table[sybyl_type - 28];
    if (p.ideal_cn == 0) return nullptr;  // placeholder entry
    return &p;
}

// ─── Per metal-donor pair: ideal distance and well depth ────────────────────
struct DonorAffinity {
    int    metal_sybyl;  // metal SYBYL type
    int    donor_sybyl;  // ligand atom SYBYL type (0 = wildcard for group)
    double ideal_dist;   // ideal M-L distance (Angstroms)
    double well_depth;   // energy well depth (kcal/mol, negative = favorable)
};

// Donor SYBYL types:
//  6-12: N types (N.1, N.2, N.3, N.4, N.AR, N.AM, N.PL3)
// 13: O.2 (carbonyl)  14: O.3 (hydroxyl/ether)  15: O.CO2 (carboxylate)
// 16: O.AR  17: S.2  18: S.3  19: S.O  20: S.O2  21: S.AR  22: P.3

// Returns the DonorAffinity for a (metal, donor) pair, or nullptr if unknown.
inline const DonorAffinity* get_donor_affinity(int metal_sybyl,
                                                int donor_sybyl) noexcept {
    // Ca2+ (36) — strongly prefers O donors
    static constexpr DonorAffinity ca_table[] = {
        {36, 13, 2.36, -12.0},  // Ca2+ – O.2  (carbonyl)
        {36, 14, 2.38, -10.0},  // Ca2+ – O.3  (hydroxyl/water)
        {36, 15, 2.36, -15.0},  // Ca2+ – O.CO2 (carboxylate) — strongest
        {36, 16, 2.38,  -8.0},  // Ca2+ – O.AR
        {36, 22, 2.55, -10.0},  // Ca2+ – P.3  (phosphate)
    };
    // Zn2+ (35) — coordinates N, O, S
    static constexpr DonorAffinity zn_table[] = {
        {35, 13, 2.05, -10.0},  // Zn2+ – O.2
        {35, 14, 2.10,  -8.0},  // Zn2+ – O.3
        {35, 15, 2.00, -12.0},  // Zn2+ – O.CO2
        {35, 18, 2.30, -16.0},  // Zn2+ – S.3  (Cys thiolate) — strongest
        {35, 17, 2.30, -14.0},  // Zn2+ – S.2
        {35, 22, 2.15,  -8.0},  // Zn2+ – P.3
    };
    // Mg2+ (28) — strict octahedral, prefers O
    static constexpr DonorAffinity mg_table[] = {
        {28, 13, 2.07, -12.0},  // Mg2+ – O.2
        {28, 14, 2.09, -10.0},  // Mg2+ – O.3
        {28, 15, 2.06, -14.0},  // Mg2+ – O.CO2
        {28, 22, 2.10, -12.0},  // Mg2+ – P.3
    };
    // Fe2+/3+ (37) — coordinates N, O, S
    static constexpr DonorAffinity fe_table[] = {
        {37, 13, 2.10, -12.0},  // Fe – O.2
        {37, 14, 2.12, -10.0},  // Fe – O.3
        {37, 15, 2.08, -14.0},  // Fe – O.CO2
        {37, 18, 2.30, -16.0},  // Fe – S.3
        {37, 17, 2.30, -14.0},  // Fe – S.2
    };
    // Cu2+ (30)
    static constexpr DonorAffinity cu_table[] = {
        {30, 13, 1.97, -10.0},  // Cu2+ – O.2
        {30, 14, 2.00,  -8.0},  // Cu2+ – O.3
        {30, 15, 1.97, -12.0},  // Cu2+ – O.CO2
        {30, 18, 2.15, -14.0},  // Cu2+ – S.3
    };
    // Mn2+ (31)
    static constexpr DonorAffinity mn_table[] = {
        {31, 13, 2.15, -10.0},  // Mn2+ – O.2
        {31, 14, 2.18,  -8.0},  // Mn2+ – O.3
        {31, 15, 2.14, -12.0},  // Mn2+ – O.CO2
    };
    // Ni2+ (34)
    static constexpr DonorAffinity ni_table[] = {
        {34, 13, 2.06, -10.0},  // Ni2+ – O.2
        {34, 14, 2.08,  -8.0},  // Ni2+ – O.3
        {34, 15, 2.04, -12.0},  // Ni2+ – O.CO2
        {34, 18, 2.20, -14.0},  // Ni2+ – S.3
    };
    // Co2+ (38)
    static constexpr DonorAffinity co_table[] = {
        {38, 13, 2.10, -10.0},  // Co2+ – O.2
        {38, 14, 2.12,  -8.0},  // Co2+ – O.3
        {38, 15, 2.08, -12.0},  // Co2+ – O.CO2
        {38, 18, 2.25, -14.0},  // Co2+ – S.3
    };

    // Helper macro-like lambda to search a table
    auto search = [](const DonorAffinity* tbl, int n,
                     int metal, int donor) -> const DonorAffinity* {
        // First try exact match
        for (int i = 0; i < n; ++i)
            if (tbl[i].metal_sybyl == metal && tbl[i].donor_sybyl == donor)
                return &tbl[i];
        return nullptr;
    };

    // Check for N-type donors: map all N types (6-12) to a generic N lookup
    // by trying exact first, then falling back to a representative N entry
    auto search_with_n_fallback = [&](const DonorAffinity* tbl, int n,
                                       int metal, int donor,
                                       double n_ideal_dist,
                                       double n_well_depth) -> const DonorAffinity* {
        const DonorAffinity* exact = search(tbl, n, metal, donor);
        if (exact) return exact;
        // Fallback for nitrogen types (6-12)
        if (donor >= 6 && donor <= 12) {
            // Return a static affinity for N donors of this metal
            // (stored in thread-local to return a pointer)
            thread_local DonorAffinity n_fallback;
            n_fallback = {metal, donor, n_ideal_dist, n_well_depth};
            return &n_fallback;
        }
        // Fallback for S types (17-21) not explicitly listed
        if (donor >= 17 && donor <= 21) {
            const DonorAffinity* s3 = search(tbl, n, metal, 18); // try S.3
            if (s3) {
                thread_local DonorAffinity s_fallback;
                s_fallback = {metal, donor, s3->ideal_dist, s3->well_depth * 0.8};
                return &s_fallback;
            }
        }
        return nullptr;
    };

#define METAL_SEARCH(tbl, n_dist, n_depth) \
    search_with_n_fallback(tbl, sizeof(tbl)/sizeof(tbl[0]), \
                           metal_sybyl, donor_sybyl, n_dist, n_depth)

    switch (metal_sybyl) {
        case 36: return METAL_SEARCH(ca_table, 2.50, -4.0);   // Ca2+: N weak
        case 35: return METAL_SEARCH(zn_table, 2.05, -14.0);  // Zn2+: N strong
        case 28: return METAL_SEARCH(mg_table, 2.20, -6.0);   // Mg2+: N moderate
        case 37: return METAL_SEARCH(fe_table, 2.15, -14.0);  // Fe: N strong
        case 30: return METAL_SEARCH(cu_table, 2.02, -12.0);  // Cu2+: N strong
        case 31: return METAL_SEARCH(mn_table, 2.22, -6.0);   // Mn2+: N moderate
        case 34: return METAL_SEARCH(ni_table, 2.10, -12.0);  // Ni2+: N strong
        case 38: return METAL_SEARCH(co_table, 2.15, -10.0);  // Co2+: N moderate
        default: return nullptr;
    }
#undef METAL_SEARCH
}

// ─── Query: is this SYBYL type a metal ion? ────────────────────────────────
inline bool is_metal_type(int sybyl_type) noexcept {
    return get_metal_params(sybyl_type) != nullptr;
}

// ─── Query: is this SYBYL type a potential coordination donor? ──────────────
inline bool is_coord_donor_type(int sybyl_type) noexcept {
    // N types (6-12), O types (13-16), S types (17-21), P.3 (22)
    return (sybyl_type >= 6 && sybyl_type <= 22);
}

// ─── Morse-like potential for metal-ligand coordination ─────────────────────
//
// E_morse(r) = D * { 2*exp[-a(r - r0)] - exp[-2a(r - r0)] }
//
// where D = well_depth (negative = favorable), r0 = ideal distance,
// a = Morse steepness parameter (controls well width, default 2.0 A^-1).
//
// Properties:
//   - Minimum at r = r0 with E = D (the well depth)
//   - E → 0 as r → infinity (bond breaks cleanly)
//   - Steep repulsion for r << r0 (positive energy)
//
inline double morse_potential(double r, double r0, double well_depth,
                              double alpha) noexcept {
    double x = alpha * (r - r0);
    double exp_neg_x = std::exp(-x);
    return well_depth * (2.0 * exp_neg_x - exp_neg_x * exp_neg_x);
}

// ─── Main scoring function ──────────────────────────────────────────────────
//
// Computes metal coordination energy for a pairwise contact between atoms
// idx_a and idx_b.  Automatically determines which (if any) is the metal
// and which is the donor.
//
// Returns 0.0 if neither atom is a metal, or if the metal-donor pair has
// no parameterized affinity.
//
// Parameters:
//   atoms:  full atom array
//   idx_a, idx_b: indices of the contacting atoms
//   dist:   distance between atoms (Angstroms)
//   weight: global weight multiplier (FA->metal_coord_weight)
//   alpha:  Morse steepness (FA->metal_coord_morse_a, default 2.0)
//
inline double compute_metal_coord_energy(
    const atom_struct* atoms,
    int idx_a, int idx_b,
    double dist,
    double weight,
    double alpha) noexcept
{
    int type_a = atoms[idx_a].type;
    int type_b = atoms[idx_b].type;

    // Identify metal and donor
    int metal_type = 0, donor_type = 0;
    if (is_metal_type(type_a) && is_coord_donor_type(type_b)) {
        metal_type = type_a;
        donor_type = type_b;
    } else if (is_metal_type(type_b) && is_coord_donor_type(type_a)) {
        metal_type = type_b;
        donor_type = type_a;
    } else {
        return 0.0;  // not a metal-donor pair
    }

    // Look up affinity parameters
    const DonorAffinity* aff = get_donor_affinity(metal_type, donor_type);
    if (!aff) return 0.0;

    // Compute Morse potential
    double e_morse = morse_potential(dist, aff->ideal_dist, aff->well_depth, alpha);

    return weight * e_morse;
}

// ─── Coordination number penalty ────────────────────────────────────────────
//
// Gentle quadratic penalty for deviating from the ideal coordination number.
// Applied per metal atom after the pairwise loop, NOT inside it.
//
// Returns penalty energy (kcal/mol).  Zero when actual_cn == ideal_cn.
//
inline double cn_penalty(int actual_cn, int ideal_cn) noexcept {
    int delta = actual_cn - ideal_cn;
    // -0.5 kcal/mol per unit of CN² deviation
    return -0.5 * static_cast<double>(delta * delta);
}

// ─── Coordination cutoff distance ───────────────────────────────────────────
// A contact is "coordinating" if dist < ideal_dist + COORD_CUTOFF_MARGIN
inline constexpr double COORD_CUTOFF_MARGIN = 0.8;  // Angstroms

inline bool is_coordinating(double dist, double ideal_dist) noexcept {
    return dist < (ideal_dist + COORD_CUTOFF_MARGIN);
}

} // namespace metal_coord
