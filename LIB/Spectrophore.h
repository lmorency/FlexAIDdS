// Spectrophore.h — 1D fingerprint encoding 3D property distributions
// 4 properties × 12 angular bins (icosahedral faces) × 3 radial shells = 144 floats
// Compatible between C++ (FlexAIDdS) and Python (NRGRank) implementations.
// Header-only, Apache-2.0 © 2026 Le Bonhomme Pharma

#pragma once

#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

namespace spectrophore {

// ── Constants ───────────────────────────────────────────────────────────────
constexpr int N_PROPERTIES = 4;     // shape, electrostatic, lipophilic, hbond
constexpr int N_ANGULAR    = 12;    // icosahedral face directions
constexpr int N_RADIAL     = 3;     // shells at 2Å, 4Å, 6Å
constexpr int DESCRIPTOR_SIZE = N_PROPERTIES * N_ANGULAR * N_RADIAL;  // 144

// Radial shell boundaries (Angstroms)
constexpr float RADIAL_SHELLS[N_RADIAL] = {2.0f, 4.0f, 6.0f};

// 12 icosahedral face normals (unit vectors pointing to face centers)
// These are the vertices of a regular icosahedron, normalized.
// Golden ratio φ = (1 + √5) / 2 ≈ 1.618034
// Vertices: permutations of (0, ±1, ±φ) normalized to unit length
constexpr float PHI = 1.6180339887f;
constexpr float ICO_NORM = 1.9021130326f;  // sqrt(1 + PHI*PHI)

// Precomputed icosahedral vertices (normalized)
// Using 12 of the 20 vertices of an icosahedron
inline const float (*ico_normals())[3] {
    static const float normals[12][3] = {
        { 0.0f / ICO_NORM,  1.0f / ICO_NORM,  PHI / ICO_NORM},
        { 0.0f / ICO_NORM, -1.0f / ICO_NORM,  PHI / ICO_NORM},
        { 0.0f / ICO_NORM,  1.0f / ICO_NORM, -PHI / ICO_NORM},
        { 0.0f / ICO_NORM, -1.0f / ICO_NORM, -PHI / ICO_NORM},
        { 1.0f / ICO_NORM,  PHI / ICO_NORM,  0.0f / ICO_NORM},
        {-1.0f / ICO_NORM,  PHI / ICO_NORM,  0.0f / ICO_NORM},
        { 1.0f / ICO_NORM, -PHI / ICO_NORM,  0.0f / ICO_NORM},
        {-1.0f / ICO_NORM, -PHI / ICO_NORM,  0.0f / ICO_NORM},
        { PHI / ICO_NORM,  0.0f / ICO_NORM,  1.0f / ICO_NORM},
        {-PHI / ICO_NORM,  0.0f / ICO_NORM,  1.0f / ICO_NORM},
        { PHI / ICO_NORM,  0.0f / ICO_NORM, -1.0f / ICO_NORM},
        {-PHI / ICO_NORM,  0.0f / ICO_NORM, -1.0f / ICO_NORM},
    };
    return normals;
}

// ── Spectrophore descriptor ─────────────────────────────────────────────────

struct Spectrophore {
    float values[DESCRIPTOR_SIZE];
    float centroid[3];

    Spectrophore() {
        memset(values, 0, sizeof(values));
        centroid[0] = centroid[1] = centroid[2] = 0.0f;
    }

    // Tanimoto similarity (treats descriptor as a continuous vector)
    float tanimoto(const Spectrophore& other) const {
        float dot_ab = 0.0f, dot_aa = 0.0f, dot_bb = 0.0f;
        for (int i = 0; i < DESCRIPTOR_SIZE; ++i) {
            dot_ab += values[i] * other.values[i];
            dot_aa += values[i] * values[i];
            dot_bb += other.values[i] * other.values[i];
        }
        float denom = dot_aa + dot_bb - dot_ab;
        if (denom < 1e-10f) return 1.0f;  // identical zero vectors
        return dot_ab / denom;
    }

    // Euclidean distance
    float euclidean(const Spectrophore& other) const {
        float sum = 0.0f;
        for (int i = 0; i < DESCRIPTOR_SIZE; ++i) {
            float d = values[i] - other.values[i];
            sum += d * d;
        }
        return std::sqrt(sum);
    }
};

// ── Helper: find angular bin (closest icosahedral normal) ───────────────────
inline int find_angular_bin(float dx, float dy, float dz) {
    float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (dist < 1e-8f) return 0;

    float nx = dx / dist, ny = dy / dist, nz = dz / dist;
    const auto& normals = ico_normals();

    int best_bin = 0;
    float best_dot = -2.0f;
    for (int i = 0; i < N_ANGULAR; ++i) {
        float dot = nx * normals[i][0] + ny * normals[i][1] + nz * normals[i][2];
        if (dot > best_dot) {
            best_dot = dot;
            best_bin = i;
        }
    }
    return best_bin;
}

// ── Helper: find radial shell ───────────────────────────────────────────────
inline int find_radial_bin(float dist) {
    for (int r = 0; r < N_RADIAL; ++r) {
        if (dist <= RADIAL_SHELLS[r]) return r;
    }
    return -1;  // beyond outer shell
}

// ── Descriptor index ────────────────────────────────────────────────────────
inline int desc_index(int property, int angular, int radial) {
    return property * N_ANGULAR * N_RADIAL + angular * N_RADIAL + radial;
}

// ── Compute spectrophore from MIF grid ──────────────────────────────────────
// Uses MIF energies as the electrostatic/vdW property.
// Grid points with their MIF energies encode the binding site properties.
//
// property 0: shape (count of grid points in each bin)
// property 1: electrostatic/vdW (sum of MIF energies)
// property 2: lipophilicity estimate (negative MIF = favorable = lipophilic pocket)
// property 3: hydrogen bond capacity estimate (from MIF gradient)
template<typename GridPointT>
Spectrophore compute_from_grid(
    const GridPointT* grid, int num_grd,
    const float* mif_energies,
    const float center[3])
{
    Spectrophore sp;
    sp.centroid[0] = center[0];
    sp.centroid[1] = center[1];
    sp.centroid[2] = center[2];

    float counts[DESCRIPTOR_SIZE];
    memset(counts, 0, sizeof(counts));

    for (int gp = 1; gp < num_grd; ++gp) {
        float dx = grid[gp].coor[0] - center[0];
        float dy = grid[gp].coor[1] - center[1];
        float dz = grid[gp].coor[2] - center[2];
        float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

        int radial_bin = find_radial_bin(dist);
        if (radial_bin < 0) continue;

        int angular_bin = find_angular_bin(dx, dy, dz);
        float e = mif_energies[gp];

        // Property 0: Shape — count occupied bins
        int idx0 = desc_index(0, angular_bin, radial_bin);
        sp.values[idx0] += 1.0f;

        // Property 1: Electrostatic/vdW — sum of MIF energies
        int idx1 = desc_index(1, angular_bin, radial_bin);
        sp.values[idx1] += e;

        // Property 2: Lipophilicity — favorable (negative) MIF indicates
        // hydrophobic pocket; use abs of negative contributions
        int idx2 = desc_index(2, angular_bin, radial_bin);
        if (e < 0.0f) sp.values[idx2] += -e;

        // Property 3: H-bond capacity — unfavorable (positive) MIF at close
        // range indicates polar/charged contacts (H-bond potential)
        int idx3 = desc_index(3, angular_bin, radial_bin);
        if (e > 0.0f && dist < 4.0f) sp.values[idx3] += e;

        counts[idx0] += 1.0f;
        counts[desc_index(1, angular_bin, radial_bin)] += 1.0f;
        counts[desc_index(2, angular_bin, radial_bin)] += 1.0f;
        counts[desc_index(3, angular_bin, radial_bin)] += 1.0f;
    }

    // Normalize: average per bin for properties 1-3
    for (int i = N_ANGULAR * N_RADIAL; i < DESCRIPTOR_SIZE; ++i) {
        if (counts[i] > 0.0f)
            sp.values[i] /= counts[i];
    }

    return sp;
}

// ── Compute spectrophore from atom coordinates ──────────────────────────────
// For ligands: uses atom positions and radii as properties.
// property 0: shape, property 1: vdW radius, property 2: hydrophobicity (crude),
// property 3: charge magnitude
struct SimpleAtom {
    float x, y, z;
    float radius;
    float charge;
};

inline Spectrophore compute_from_atoms(
    const SimpleAtom* atoms, int n_atoms,
    const float center[3])
{
    Spectrophore sp;
    sp.centroid[0] = center[0];
    sp.centroid[1] = center[1];
    sp.centroid[2] = center[2];

    float counts[DESCRIPTOR_SIZE];
    memset(counts, 0, sizeof(counts));

    for (int i = 0; i < n_atoms; ++i) {
        float dx = atoms[i].x - center[0];
        float dy = atoms[i].y - center[1];
        float dz = atoms[i].z - center[2];
        float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

        int radial_bin = find_radial_bin(dist);
        if (radial_bin < 0) continue;

        int angular_bin = find_angular_bin(dx, dy, dz);

        // Property 0: Shape
        sp.values[desc_index(0, angular_bin, radial_bin)] += 1.0f;
        // Property 1: Size (vdW radius)
        sp.values[desc_index(1, angular_bin, radial_bin)] += atoms[i].radius;
        // Property 2: Hydrophobicity (larger radius → more hydrophobic, crude estimate)
        sp.values[desc_index(2, angular_bin, radial_bin)] += atoms[i].radius > 1.5f ? 1.0f : 0.0f;
        // Property 3: Charge magnitude (H-bond capacity proxy)
        sp.values[desc_index(3, angular_bin, radial_bin)] += std::abs(atoms[i].charge);

        for (int p = 0; p < N_PROPERTIES; ++p)
            counts[desc_index(p, angular_bin, radial_bin)] += 1.0f;
    }

    // Normalize properties 1-3
    for (int i = N_ANGULAR * N_RADIAL; i < DESCRIPTOR_SIZE; ++i) {
        if (counts[i] > 0.0f)
            sp.values[i] /= counts[i];
    }

    return sp;
}

} // namespace spectrophore
