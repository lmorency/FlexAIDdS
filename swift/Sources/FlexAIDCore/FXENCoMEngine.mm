// FXENCoMEngine.mm — Objective-C++ implementation of the ENCoMEngine C shim
//
// Bridges encom::ENCoMEngine (C++20) static methods to plain C for Swift.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

#include "FXENCoMEngine.h"
#include "encom.h"

#include <vector>

extern "C" FXVibrationalEntropy fx_encom_compute_vibrational_entropy(
    const double* eigenvalues, int count,
    double temperature_K, double eigenvalue_cutoff) {

    FXVibrationalEntropy result = {};
    if (!eigenvalues || count <= 0) return result;

    // Build NormalMode vector from eigenvalues
    // (eigenvectors not needed for entropy calculation — only eigenvalues matter)
    std::vector<encom::NormalMode> modes(count);
    for (int i = 0; i < count; ++i) {
        modes[i].index = i + 1;
        modes[i].eigenvalue = eigenvalues[i];
        modes[i].frequency = std::sqrt(std::abs(eigenvalues[i]));
    }

    auto vib = encom::ENCoMEngine::compute_vibrational_entropy(
        modes, temperature_K, eigenvalue_cutoff);

    result.S_vib_kcal_mol_K = vib.S_vib_kcal_mol_K;
    result.S_vib_J_mol_K    = vib.S_vib_J_mol_K;
    result.omega_eff         = vib.omega_eff;
    result.n_modes           = vib.n_modes;
    result.temperature       = vib.temperature;
    return result;
}

extern "C" double fx_encom_total_entropy(double S_conf_kcal_mol_K, double S_vib_kcal_mol_K) {
    return encom::ENCoMEngine::total_entropy(S_conf_kcal_mol_K, S_vib_kcal_mol_K);
}

extern "C" double fx_encom_free_energy_with_vibrations(
    double F_electronic, double S_vib_kcal_mol_K, double temperature_K) {
    return encom::ENCoMEngine::free_energy_with_vibrations(
        F_electronic, S_vib_kcal_mol_K, temperature_K);
}
