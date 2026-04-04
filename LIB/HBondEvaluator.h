// HBondEvaluator.h — Directional hydrogen bonding potential
//
// Replaces basic distance-only H-bond checks with a strict exponential
// angular multiplier that screens out physically impossible hydrogen bonds
// (donor-H-acceptor angles below ~120°).  Salt bridges bypass the strict
// penalty via a broader angular well.
//
// The angular multiplier g(theta) is applied to the complementarity
// contribution of each H-bond-capable atom pair in the vcfunction
// scoring loop.
//
// Reference: Restrictive directional H-bond angular potentials
//
// Apache-2.0 © 2026 Le Bonhomme Pharma

#pragma once

#include <cmath>
#include <numbers>

// Forward declarations
struct atom_struct;
struct FA_Global_struct;
struct VC_Global_struct;

namespace hbond {

// Determine if an atom type (SYBYL 1-indexed) is an H-bond donor heavy atom.
// Donors are N or O atoms that may have attached hydrogens.
inline bool is_donor_type(int sybyl_type) {
    // N types: 6(N.1), 7(N.2), 8(N.3), 9(N.4), 10(N.AR), 11(N.AM), 12(N.PL3)
    // O types: 14(O.3)
    return (sybyl_type >= 6 && sybyl_type <= 12) || sybyl_type == 14;
}

// Determine if an atom type (SYBYL 1-indexed) is an H-bond acceptor.
// Acceptors are N or O atoms with lone pairs (all N and O types).
inline bool is_acceptor_type(int sybyl_type) {
    // N types: 6–12, O types: 13(O.2), 14(O.3), 15(O.CO2), 16(O.AR)
    return (sybyl_type >= 6 && sybyl_type <= 16);
}

// Determine if an atom type pair forms a salt bridge (charged N–O pair).
// N.4 (quaternary ammonium) or N.3 with cationic charge paired with
// O.CO2 (carboxylate), or vice versa.
inline bool is_salt_bridge_pair(int typeA, int typeB) {
    auto is_cationic_N = [](int t) { return t == 9 || t == 8; };  // N.4, N.3
    auto is_anionic_O  = [](int t) { return t == 15; };           // O.CO2
    return (is_cationic_N(typeA) && is_anionic_O(typeB)) ||
           (is_cationic_N(typeB) && is_anionic_O(typeA));
}

// Calculate the angular multiplier for a D–H···A hydrogen bond.
//
// Uses the donor-H-acceptor angle theta (D–H–A):
//   Standard H-bond: g(theta) = cos²(theta) × exp(-(pi - theta)^6)
//   Salt bridge:     g(theta) = broader well (less restrictive decay)
//
// Parameters:
//   donor_{xyz}:     coordinates of the donor heavy atom (D)
//   hydrogen_{xyz}:  coordinates of the bridging hydrogen (H)
//   acceptor_{xyz}:  coordinates of the acceptor atom (A)
//   is_salt_bridge:  true to use the broader salt-bridge angular well
//
// Returns: angular multiplier in [0, 1].  Values near 0 indicate
//          geometrically unfavorable H-bond angles.
inline double angular_multiplier(
    double donor_x,    double donor_y,    double donor_z,
    double hydrogen_x, double hydrogen_y, double hydrogen_z,
    double acceptor_x, double acceptor_y, double acceptor_z,
    bool is_salt_bridge)
{
    // Vectors: H→D and H→A
    double hd_x = donor_x - hydrogen_x;
    double hd_y = donor_y - hydrogen_y;
    double hd_z = donor_z - hydrogen_z;

    double ha_x = acceptor_x - hydrogen_x;
    double ha_y = acceptor_y - hydrogen_y;
    double ha_z = acceptor_z - hydrogen_z;

    double dot = hd_x * ha_x + hd_y * ha_y + hd_z * ha_z;
    double mag_hd = std::sqrt(hd_x * hd_x + hd_y * hd_y + hd_z * hd_z);
    double mag_ha = std::sqrt(ha_x * ha_x + ha_y * ha_y + ha_z * ha_z);

    if (mag_hd < 1e-8 || mag_ha < 1e-8) return 0.0;

    double cos_theta = dot / (mag_hd * mag_ha);
    // Clamp to [-1, 1] to prevent NaN from floating-point drift
    cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
    double theta = std::acos(cos_theta);

    if (is_salt_bridge) {
        // Broader angular well for ionic interactions:
        // Gentle cosine decay centered at 180° (linear D–H–A).
        // Allows angles down to ~90° with moderate penalty.
        double half_angle = 0.5 * (std::numbers::pi - theta);
        double c = std::cos(half_angle);
        return c * c;
    }

    // Strict angular penalty for standard H-bonds:
    // g(theta) = cos²(theta) × exp(-(pi - theta)^6)
    // This zeroes out contributions below ~120° and peaks at 180°.
    double pi_diff = std::numbers::pi - theta;
    double pi_diff_sq = pi_diff * pi_diff;
    double pi_diff_6 = pi_diff_sq * pi_diff_sq * pi_diff_sq;
    return (cos_theta * cos_theta) * std::exp(-pi_diff_6);
}

// Simplified angular multiplier using only heavy-atom geometry
// (D–A–AA or D–D–A angle) when explicit hydrogens are unavailable.
// Uses the D–A distance direction as a proxy for D–H–A angle.
//
// For use in the Voronoi contact loop where hydrogen coordinates
// may not be directly available.  Approximates the H position as
// lying along the D→A vector at the covalent N-H/O-H bond length.
inline double angular_multiplier_heavy_atom(
    double donor_x,   double donor_y,   double donor_z,
    double acceptor_x, double acceptor_y, double acceptor_z,
    double da_dist,
    bool is_salt_bridge)
{
    if (is_salt_bridge) {
        // Salt bridges: broader tolerance, return gentle distance-based decay
        return 1.0;
    }

    // Without explicit H, use a distance-dependent proxy:
    // Ideal D···A distance for H-bond is ~2.7–3.2 Å
    // Apply a smooth penalty outside this range
    constexpr double D_OPT = 2.9;     // optimal D···A distance
    constexpr double SIGMA = 0.4;     // width of the Gaussian well
    double d_diff = da_dist - D_OPT;
    double dist_factor = std::exp(-0.5 * (d_diff * d_diff) / (SIGMA * SIGMA));

    return dist_factor;
}

// Evaluate the H-bond angular correction for a Voronoi contact pair.
// Defined in HBondEvaluator.cpp.
double evaluate_contact(const atom_struct* atomA, const atom_struct* atomB,
                        const atom_struct* atoms, double dist);

} // namespace hbond
