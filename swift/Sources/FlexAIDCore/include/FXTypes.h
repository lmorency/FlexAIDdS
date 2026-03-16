// FXTypes.h — C-compatible data structures for FlexAIDdS Swift bridge
//
// Plain C structs mirroring C++ types from statmech.h, encom.h, BindingMode.h.
// No C++ dependencies — safe to import from Swift via the FlexAIDCore module.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ─── Thermodynamics (mirrors statmech::Thermodynamics) ─────────────────────

typedef struct {
    double temperature;       // K
    double log_Z;             // ln(Z) — stored for numerical stability
    double free_energy;       // Helmholtz F = -kT ln Z (kcal/mol)
    double mean_energy;       // <E> (kcal/mol)
    double mean_energy_sq;    // <E^2>
    double heat_capacity;     // C_v = (<E^2> - <E>^2) / (kT^2)
    double entropy;           // S = (<E> - F) / T (kcal mol^-1 K^-1)
    double std_energy;        // sigma_E = sqrt(C_v * kT^2) (kcal/mol)
} FXThermodynamics;

// ─── Vibrational entropy (mirrors encom::VibrationalEntropy) ────────────────

typedef struct {
    double S_vib_kcal_mol_K;  // Vibrational entropy (kcal mol^-1 K^-1)
    double S_vib_J_mol_K;     // Vibrational entropy (J mol^-1 K^-1)
    double omega_eff;         // Effective frequency (rad/s)
    int    n_modes;           // Number of non-zero modes (3N - 6)
    double temperature;       // K
} FXVibrationalEntropy;

// ─── WHAM bin (mirrors statmech::WHAMBin) ───────────────────────────────────

typedef struct {
    double coord_center;
    double count;
    double free_energy;       // kcal/mol
} FXWHAMBin;

// ─── TI point (mirrors statmech::TIPoint) ───────────────────────────────────

typedef struct {
    double lambda;            // coupling parameter [0,1]
    double dV_dlambda;        // <dV/dlambda>_lambda
} FXTIPoint;

// ─── Replica (mirrors statmech::Replica) ────────────────────────────────────

typedef struct {
    int    id;
    double temperature;
    double beta;              // 1/(kT)
    double current_energy;
} FXReplica;

// ─── Pose info (lightweight view of Pose for Swift consumption) ─────────────

typedef struct {
    int    chrom_index;       // Index in chromosome array
    int    order;             // OPTICS order
    float  reach_dist;        // Reachability distance
    double cf;                // Complementarity function score
    double boltzmann_weight;  // Boltzmann weight (deprecated, use StatMechEngine)
} FXPoseInfo;

// ─── Binding mode summary ───────────────────────────────────────────────────

typedef struct {
    int    size;              // Number of poses in this mode
    double free_energy;       // Helmholtz F (kcal/mol)
    double entropy;           // Configurational S (kcal mol^-1 K^-1)
    double enthalpy;          // Boltzmann-weighted <E> (kcal/mol)
    double heat_capacity;     // C_v
} FXBindingModeInfo;

// ─── Shannon ThermoStack result (mirrors shannon_thermo::FullThermoResult) ──

#define FX_SHANNON_MAX_BINS 256

typedef struct {
    double shannon_entropy;         // Configurational entropy (nats)
    double torsional_vib_entropy;   // Vibrational entropy (kcal/mol/K)
    double entropy_contribution;    // -T*S term (kcal/mol)
    double delta_G;                 // Total free energy (kcal/mol)

    // Convergence diagnostics
    int    is_converged;            // 1 if entropy plateau reached, 0 otherwise
    double convergence_rate;        // Relative change in last window

    // Histogram summary
    int    occupied_bins;           // Number of non-zero histogram bins
    int    total_bins;              // Total bins used
    int    num_histogram_bins;      // Actual number of bins in arrays below
    double histogram_centers[FX_SHANNON_MAX_BINS];  // Bin centers (energy, kcal/mol)
    double histogram_probs[FX_SHANNON_MAX_BINS];    // Normalized probabilities

    // Hardware backend string (null-terminated, max 32 chars)
    char   hardware_backend[32];
} FXShannonThermoResult;

// ─── Physical constants ────────────────────────────────────────────────────

static const double FX_KB_KCAL = 0.001987206;   // kcal mol^-1 K^-1
static const double FX_KB_SI   = 1.380649e-23;  // J K^-1

// ─── Memory management helpers ──────────────────────────────────────────────

void fx_free_doubles(double* ptr);
void fx_free_wham_bins(FXWHAMBin* ptr);
void fx_free_pose_infos(FXPoseInfo* ptr);

#ifdef __cplusplus
}
#endif
