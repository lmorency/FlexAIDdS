// ShannonThermoStack.h — Shannon Entropy + Torsional ENCoM thermodynamic stack
//
// Combines:
//   – Shannon configurational entropy over GA ensemble (binned into 256 mega-clusters)
//   – Torsional ENCoM vibrational entropy from NormalMode fluctuations (protein + nucleotide backbones)
//   – Hardware-accelerated histogram computation (Metal on Apple Silicon, OpenMP/Eigen on other platforms)
//
// Reuses StatMechEngine (statmech.h) and TorsionalENM (tencm.h) without modification.
// BindingPopulation is untouched; SugarPuckerGene flexibility is separate.
#pragma once

#include "../statmech.h"
#include "../tencm.h"
#include <vector>
#include <string>
#include <cmath>

namespace shannon_thermo {

// ─── constants ───────────────────────────────────────────────────────────────
inline constexpr int   SHANNON_BINS      = 256;    // mega-cluster discretisation
inline constexpr double kB_kcal          = 0.001987206; // kcal mol⁻¹ K⁻¹
inline constexpr double TEMPERATURE_K    = 298.15;
inline constexpr int   DEFAULT_HIST_BINS = 20;

// ─── result struct ───────────────────────────────────────────────────────────
struct FullThermoResult {
    double deltaG;              // total free energy (kcal/mol)
    double shannonEntropy;      // dimensionless bits (conformational)
    double torsionalVibEntropy; // kcal/mol·K (from ENCoM modes)
    double entropyContribution; // -T*S term (kcal/mol)
    std::string report;
};

// ─── Shannon entropy computation ──────────────────────────────────────────────
// Bins a vector of continuous values into numBins and computes Shannon entropy H.
// Uses OpenMP parallelism when available; Metal GPU on Apple Silicon.
double compute_shannon_entropy(const std::vector<double>& values,
                               int num_bins = DEFAULT_HIST_BINS);

// Same for integer state labels (discrete)
double compute_shannon_entropy_discrete(const std::vector<int>& states);

// ─── torsional vibrational entropy from ENCoM modes ─────────────────────────
// Sums harmonic oscillator entropy contribution for each normal mode:
//   S_vib = kB * [ hν/kBT / (exp(hν/kBT)-1) - ln(1-exp(-hν/kBT)) ]
// For low-frequency torsional modes approximated as: S ≈ kB * ln(kBT/hν)
double compute_torsional_vibrational_entropy(
    const std::vector<tencm::NormalMode>& modes,
    double temperature_K = TEMPERATURE_K);

// ─── full stack entry point ───────────────────────────────────────────────────
// Runs the complete ShannonThermoStack on a GA population ensemble.
//
// Parameters:
//   stat_engine    – populated StatMechEngine from the GA run
//   tencm_model    – built TorsionalENM (may be default-constructed if backbone
//                    flexibility is disabled)
//   base_deltaG    – enthalpy-dominated ΔG from scoring function (kcal/mol)
//   temperature_K  – simulation temperature
FullThermoResult run_shannon_thermo_stack(
    const statmech::StatMechEngine& stat_engine,
    const tencm::TorsionalENM&      tencm_model,
    double                          base_deltaG,
    double                          temperature_K = TEMPERATURE_K);

} // namespace shannon_thermo
