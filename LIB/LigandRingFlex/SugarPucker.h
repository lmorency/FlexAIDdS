// SugarPucker.h — Cremer-Pople pseudorotation gene for furanose rings
//
// Implements continuous pseudorotation phase sampling (0–360°) for
// five-membered furanose rings in nucleotides and sugar-containing ligands.
// Auto-detects ribose (O2' present) vs deoxyribose (O2' absent).
//
// Integrated into GA chromosome via sugarPuckerPhases[] vector.
// Contributes to Shannon entropy and hydration entropy via CF.
#pragma once

#include <vector>
#include <cmath>

// Forward declarations (avoid pulling in all of flexaid.h)
struct atom;

namespace sugar_pucker {

// ─── sugar type detection ─────────────────────────────────────────────────────
enum class SugarType { Ribose, Deoxyribose, Unknown };

// Detect sugar type from atom names in the ring.
// Returns Ribose if O2' is found, Deoxyribose otherwise.
SugarType detect_sugar_type(const atom* atoms, const int* ring_atom_indices, int ring_size);

// ─── Cremer-Pople pseudorotation ─────────────────────────────────────────────
// Parameters for a furanose ring:
//   P     — phase angle (degrees) [0, 360)
//   nu_max — amplitude (degrees), typically 30–45°
//
// Computes the 5 ring torsion angles from P and nu_max.
struct PuckerParams {
    float P;       // phase (degrees)
    float nu_max;  // amplitude (degrees)
};

// Compute ring torsions ν0..ν4 from Cremer-Pople parameters.
// Output: torsions[5] in degrees.
void compute_ring_torsions(const PuckerParams& params, float torsions[5]) noexcept;

// Energy penalty for pseudorotation (kcal/mol):
//   Ribose:      minimum at P≈18° (C3'-endo) and P≈162° (C2'-endo)
//   Deoxyribose: minimum at P≈162° (C2'-endo), secondary at P≈18°
double compute_pucker_energy(float phase_deg, SugarType stype) noexcept;

// ─── GA interface ─────────────────────────────────────────────────────────────
// Apply all sugar pucker phases from a chromosome to the atom array.
// `ring_indices[i]` lists the 5 ring-atom indices for the i-th sugar.
void apply_sugar_puckers(
    atom*                               atoms,
    const std::vector<std::vector<int>>& ring_indices,
    const std::vector<float>&           phases_deg,
    const std::vector<SugarType>&       sugar_types);

// Mutate one sugar pucker phase (small Gaussian step, σ ≈ 15°)
float mutate_phase(float current_phase_deg, float sigma_deg = 15.0f);

// ─── Shannon entropy contribution ────────────────────────────────────────────
// Average pucker-phase entropy contribution (bits) given a population of phases.
double compute_pucker_entropy(const std::vector<float>& phase_ensemble_deg);

} // namespace sugar_pucker
