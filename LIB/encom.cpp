// encom.cpp — ENCoM vibrational entropy implementation

#define _USE_MATH_DEFINES
#include <cmath>

#include "encom.h"
#include <algorithm>
#include <sstream>
#include <iostream>

namespace encom {

// ──────────────────────────────────────────────────────────────────────────────
std::vector<NormalMode> ENCoMEngine::load_modes(
    const std::string& eigenvalue_file,
    const std::string& eigenvector_file)
{
    std::vector<NormalMode> modes;
    
    // Read eigenvalues
    std::ifstream eval_stream(eigenvalue_file);
    if (!eval_stream.is_open()) {
        throw std::runtime_error("Cannot open eigenvalue file: " + eigenvalue_file);
    }
    
    std::vector<double> eigenvalues;
    double val;
    while (eval_stream >> val) {
        eigenvalues.push_back(val);
    }
    eval_stream.close();
    
    if (eigenvalues.empty()) {
        throw std::runtime_error("No eigenvalues found in " + eigenvalue_file);
    }
    
    // Read eigenvectors (assume format: each row is a mode, columns are components)
    std::ifstream evec_stream(eigenvector_file);
    if (!evec_stream.is_open()) {
        throw std::runtime_error("Cannot open eigenvector file: " + eigenvector_file);
    }
    
    std::string line;
    int mode_index = 1;
    while (std::getline(evec_stream, line) && mode_index <= eigenvalues.size()) {
        std::istringstream iss(line);
        std::vector<double> components;
        double component;
        while (iss >> component) {
            components.push_back(component);
        }
        
        if (components.empty()) continue;  // skip empty lines
        
        NormalMode mode;
        mode.index = mode_index;
        mode.eigenvalue = eigenvalues[mode_index - 1];
        mode.frequency = std::sqrt(std::abs(mode.eigenvalue));  // ω = sqrt(λ)
        mode.eigenvector = std::move(components);
        
        modes.push_back(mode);
        ++mode_index;
    }
    evec_stream.close();
    
    if (modes.size() != eigenvalues.size()) {
        std::cerr << "Warning: eigenvalue count (" << eigenvalues.size() 
                  << ") != eigenvector count (" << modes.size() << ")\n";
    }
    
    return modes;
}

// ──────────────────────────────────────────────────────────────────────────────
VibrationalEntropy ENCoMEngine::compute_vibrational_entropy(
    const std::vector<NormalMode>& modes,
    double temperature_K,
    double eigenvalue_cutoff)
{
    VibrationalEntropy result;
    result.temperature = temperature_K;
    
    // Filter non-zero modes (exclude 6 rigid-body modes: 3 translation + 3 rotation)
    std::vector<double> nonzero_eigenvalues;
    for (const auto& mode : modes) {
        if (mode.eigenvalue > eigenvalue_cutoff) {
            nonzero_eigenvalues.push_back(mode.eigenvalue);
        }
    }
    
    result.n_modes = nonzero_eigenvalues.size();
    
    if (result.n_modes == 0) {
        // No vibrational modes → zero entropy
        result.S_vib_kcal_mol_K = 0.0;
        result.S_vib_J_mol_K = 0.0;
        result.omega_eff = 0.0;
        return result;
    }
    
    // Compute geometric mean of eigenvalues
    double geom_mean_eigenvalue = geometric_mean(nonzero_eigenvalues);
    
    // Convert to effective frequency (rad/s)
    // Note: ENCoM eigenvalues are in arbitrary units. For real calculations,
    // need proper force constant → frequency conversion with mass weighting.
    // Here we assume eigenvalues are already in frequency² units (rad/s)².
    result.omega_eff = std::sqrt(geom_mean_eigenvalue);
    
    // Quasi-harmonic entropy (Schlitter formula variant):
    // S_vib = (3N - 6) × k_B × [1 + ln(2π k_B T / (ħ ω_eff))]
    const double kBT = kB_SI * temperature_K;         // J
    const double arg = (2.0 * M_PI * kBT) / (hbar_SI * result.omega_eff);
    
    if (arg <= 0.0) {
        // Invalid frequency → zero entropy
        result.S_vib_kcal_mol_K = 0.0;
        result.S_vib_J_mol_K = 0.0;
        return result;
    }
    
    const double S_per_mode_SI = kB_SI * (1.0 + std::log(arg));  // J K⁻¹ per mode
    result.S_vib_J_mol_K = result.n_modes * S_per_mode_SI * NA;  // J mol⁻¹ K⁻¹
    
    // Convert to kcal mol⁻¹ K⁻¹
    const double J_to_kcal = 1.0 / 4184.0;
    result.S_vib_kcal_mol_K = result.S_vib_J_mol_K * J_to_kcal;
    
    return result;
}

// ──────────────────────────────────────────────────────────────────────────────
double ENCoMEngine::geometric_mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    
    // Compute product in log space for numerical stability
    double log_sum = 0.0;
    for (double val : values) {
        if (val <= 0.0) continue;  // skip non-positive
        log_sum += std::log(val);
    }
    
    return std::exp(log_sum / values.size());
}

}  // namespace encom
