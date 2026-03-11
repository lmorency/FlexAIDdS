// FXENCoMEngine.h — C interface to encom::ENCoMEngine
//
// Static-only API for vibrational entropy calculations.
// No instance state — all functions are stateless.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "FXTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

// Compute vibrational entropy from eigenvalues
// eigenvalues: array of normal mode eigenvalues (non-zero modes only)
// count: number of eigenvalues
// temperature_K: temperature in Kelvin
// eigenvalue_cutoff: skip modes below this threshold
FXVibrationalEntropy fx_encom_compute_vibrational_entropy(
    const double* eigenvalues, int count,
    double temperature_K, double eigenvalue_cutoff);

// Combine configurational + vibrational entropy
double fx_encom_total_entropy(double S_conf_kcal_mol_K, double S_vib_kcal_mol_K);

// Free energy with vibrational correction: F_total = F_elec - T*S_vib
double fx_encom_free_energy_with_vibrations(
    double F_electronic, double S_vib_kcal_mol_K, double temperature_K);

#ifdef __cplusplus
}
#endif
