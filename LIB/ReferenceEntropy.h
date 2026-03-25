// ReferenceEntropy.h — Reference state entropy correction for binding ΔG
//
// In ensemble docking (NMR/cryo-EM/MD conformers), the free energy must
// be corrected for the unbound receptor conformational entropy. Without
// this correction, flexible receptors appear artificially penalized.
//
// The reference entropy S_ref accounts for:
//   1. Receptor conformational entropy in the unbound (apo) state
//   2. Ligand conformational entropy in solution (free rotation)
//   3. Translational/rotational entropy loss upon binding (cratic term)
//
// ΔG_bind = ΔG_dock - T·ΔS_ref
//         = (H_bound - T·S_bound) - T·(S_apo_receptor + S_free_ligand - S_cratic)
//
// Methods:
//   - Receptor ensemble: Shannon entropy over multi-model RMSD distribution
//   - Ligand reference: Shannon entropy over rotatable bond torsion space
//   - Cratic correction: -T·R·ln(1/1660) ≈ +4.3 kcal/mol at 300K
//     (standard state 1M → 1/1660 M at MW~300 Da)
//
// References:
//   - Gilson & Zhou (2007) Annu. Rev. Biophys. Biomol. Struct. 36:21-42
//   - Killian et al. (2007) J. Chem. Theory Comput. 3:1024-1035
//
// Copyright 2026 Le Bonhomme Pharma
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "statmech.h"
#include <vector>
#include <cmath>

namespace reference_entropy {

inline constexpr double kB_kcal = 0.001987206;  // kcal/(mol·K)
inline constexpr double R_kcal  = kB_kcal;       // R = kB for molar quantities

// ─── Receptor ensemble entropy ───────────────────────────────────────────────
// Compute Shannon entropy over receptor conformer ensemble from energy spread.
// Input: energies of each receptor model (from scoring the apo receptor
//        against itself, or RMSD-based pseudo-energies).
struct ReceptorEntropyResult {
    double S_conf;           // conformational entropy (kcal/mol·K)
    double T_S_conf;         // -T·S_conf (kcal/mol) at given T
    int    n_models;         // number of conformers
    double energy_spread;    // max - min energy (kcal/mol)
};

ReceptorEntropyResult compute_receptor_entropy(
    const std::vector<double>& model_energies,
    double temperature_K = 300.0);

// ─── Ligand solution entropy ─────────────────────────────────────────────────
// Estimate the conformational entropy of the free ligand in solution
// from the number of rotatable bonds. Uses the empirical estimate:
//   S_rot ≈ n_rot × R × ln(3)    (3-fold barrier per rotatable bond)
struct LigandEntropyResult {
    double S_rot;            // rotational entropy (kcal/mol·K)
    double T_S_rot;          // -T·S_rot (kcal/mol)
    int    n_rotatable;
};

LigandEntropyResult compute_ligand_solution_entropy(
    int n_rotatable_bonds,
    double temperature_K = 300.0);

// ─── Cratic entropy ─────────────────────────────────────────────────────────
// Translational/rotational entropy lost upon binding.
// Standard state correction: ΔS_cratic = -R·ln(V_site/V_standard)
// ≈ -R·ln(1/1660) for a typical drug at 1M standard state.
// This is approximately +4.3 kcal/mol penalty at 300K.
struct CraticResult {
    double S_cratic;         // cratic entropy (kcal/mol·K)
    double T_S_cratic;       // -T·S_cratic (kcal/mol) — positive = penalty
};

CraticResult compute_cratic_entropy(double temperature_K = 300.0);

// ─── Combined reference correction ──────────────────────────────────────────
struct ReferenceEntropyCorrection {
    double T_dS_receptor;    // -T·ΔS_receptor (kcal/mol)
    double T_dS_ligand;      // -T·ΔS_ligand_solution (kcal/mol)
    double T_dS_cratic;      // -T·ΔS_cratic (kcal/mol)
    double T_dS_total;       // total correction to add to ΔG_dock
    double temperature;
};

ReferenceEntropyCorrection compute_reference_correction(
    const std::vector<double>& receptor_model_energies,
    int n_rotatable_bonds,
    double temperature_K = 300.0);

// ─── Ensemble consensus scoring ──────────────────────────────────────────────
// For multi-model receptor docking (NMR/cryo-EM), combine per-model
// ΔG values into a single Boltzmann-weighted consensus score.
//
// ΔG_consensus = -kT · ln[ Σ_i exp(-ΔG_i / kT) / N ]
//
// This naturally favors ligands that bind well to multiple conformers.
struct EnsembleConsensusResult {
    double dG_consensus;       // Boltzmann-weighted consensus ΔG (kcal/mol)
    double dG_best;            // best single-model ΔG
    double dG_worst;           // worst single-model ΔG
    double dG_mean;            // arithmetic mean
    double dG_stddev;          // standard deviation across models
    int    n_models;
    int    best_model_idx;     // 0-based index of best model
    std::vector<double> per_model_dG;
};

EnsembleConsensusResult compute_ensemble_consensus(
    const std::vector<double>& per_model_dG,
    double temperature_K = 300.0);

} // namespace reference_entropy
