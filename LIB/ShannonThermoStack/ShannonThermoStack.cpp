// ShannonThermoStack.cpp — implementation
#include "ShannonThermoStack.h"

#ifdef ENABLE_METAL_CORE
#  include "ShannonMetalBridge.h"
#endif

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <stdexcept>

namespace shannon_thermo {

// ─── ShannonEnergyMatrix ─────────────────────────────────────────────────────

ShannonEnergyMatrix& ShannonEnergyMatrix::instance() {
    static ShannonEnergyMatrix inst;
    return inst;
}

void ShannonEnergyMatrix::initialise() {
    if (initialised_) return;

    matrix_.resize(SHANNON_BINS * SHANNON_BINS);

    // Generate uniform probabilities with Gaussian perturbation (seed 42)
    std::mt19937 rng(42);
    std::normal_distribution<double> perturb(0.0, 0.05);

    // Base probabilities: uniform over SHANNON_BINS
    std::vector<double> p_i(SHANNON_BINS), p_j(SHANNON_BINS);
    double base = 1.0 / SHANNON_BINS;
    for (int i = 0; i < SHANNON_BINS; ++i) {
        p_i[i] = std::max(1e-9, base + perturb(rng));
        p_j[i] = std::max(1e-9, base + perturb(rng));
    }

    // Normalise
    double sum_i = std::accumulate(p_i.begin(), p_i.end(), 0.0);
    double sum_j = std::accumulate(p_j.begin(), p_j.end(), 0.0);
    for (int k = 0; k < SHANNON_BINS; ++k) {
        p_i[k] /= sum_i;
        p_j[k] /= sum_j;
    }

    // Fill E[i][j] = -kT * p_i * log2(p_j)
    const double kT = kB_kcal * TEMPERATURE_K;
    const double log2_inv = 1.0 / std::log(2.0);
    for (int i = 0; i < SHANNON_BINS; ++i) {
        for (int j = 0; j < SHANNON_BINS; ++j) {
            matrix_[i * SHANNON_BINS + j] =
                -kT * p_i[i] * std::log(p_j[j]) * log2_inv;
        }
    }
    initialised_ = true;
}

// ─── compute_shannon_entropy ─────────────────────────────────────────────────

double compute_shannon_entropy(const std::vector<double>& values, int num_bins) {
    if (values.empty()) return 0.0;

#ifdef ENABLE_METAL_CORE
    // Delegate histogram computation to GPU on Apple Silicon
    return ShannonMetalBridge::compute_shannon_entropy_metal(values, num_bins);
#endif

    double min_v = *std::min_element(values.begin(), values.end());
    double max_v = *std::max_element(values.begin(), values.end());
    if (max_v - min_v < 1e-12) return 0.0;

    double bin_width = (max_v - min_v) / num_bins + 1e-10;
    std::vector<int> bins(num_bins, 0);

#ifdef _OPENMP
    std::vector<std::vector<int>> thread_bins;
    int n_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        { n_threads = omp_get_num_threads(); }
    }
    thread_bins.assign(n_threads, std::vector<int>(num_bins, 0));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& local = thread_bins[tid];
        #pragma omp for schedule(static)
        for (int i = 0; i < (int)values.size(); ++i) {
            int b = static_cast<int>((values[i] - min_v) / bin_width);
            b = std::min(std::max(b, 0), num_bins - 1);
            local[b]++;
        }
    }
    for (auto& tb : thread_bins)
        for (int b = 0; b < num_bins; ++b)
            bins[b] += tb[b];
#else
    for (double v : values) {
        int b = static_cast<int>((v - min_v) / bin_width);
        b = std::min(std::max(b, 0), num_bins - 1);
        bins[b]++;
    }
#endif

    return compute_shannon_entropy_discrete(bins);
}

double compute_shannon_entropy_discrete(const std::vector<int>& counts) {
    int total = std::accumulate(counts.begin(), counts.end(), 0);
    if (total == 0) return 0.0;

    double H = 0.0;
    const double log2_inv = 1.0 / std::log(2.0);
    for (int c : counts) {
        if (c > 0) {
            double p = static_cast<double>(c) / total;
            H -= p * std::log(p) * log2_inv;
        }
    }
    return H;
}

// ─── compute_torsional_vibrational_entropy ───────────────────────────────────

double compute_torsional_vibrational_entropy(
    const std::vector<tencm::NormalMode>& modes,
    double temperature_K)
{
    if (modes.empty()) return 0.0;

    const double kT = kB_kcal * temperature_K;
    double S_vib = 0.0;

    // Skip the first 6 modes (rigid body: 3 translations + 3 rotations)
    // For remaining low-frequency torsional modes: harmonic approximation
    //   S ≈ kB * [1 - ln(ν/kBT)]  (high-temperature / classical limit)
    for (size_t m = 6; m < modes.size(); ++m) {
        double eigenval = modes[m].eigenvalue; // kcal mol⁻¹ rad⁻²
        if (eigenval < 1e-6) continue;         // skip zero / imaginary modes

        // Angular frequency: ω² ≈ eigenvalue (in rad²/ps²; use reduced units)
        // Classical limit: S_mode = kB * (1 + ln(kBT/hω))
        // In kcal/mol units: hω ≈ eigenvalue (already in energy/rad² ≡ rad frequency)
        double ln_arg = kT / eigenval;
        if (ln_arg < 1e-6) continue;
        S_vib += kB_kcal * (1.0 + std::log(ln_arg));
    }
    return S_vib;
}

// ─── run_shannon_thermo_stack ────────────────────────────────────────────────

FullThermoResult run_shannon_thermo_stack(
    const statmech::StatMechEngine& stat_engine,
    const tencm::TorsionalENM&      tencm_model,
    double                          base_deltaG,
    double                          temperature_K)
{
    // Initialise the 256×256 lookup matrix (no-op if already done)
    ShannonEnergyMatrix::instance().initialise();

    // 1. Boltzmann weights → energies for Shannon histogram
    auto weights    = stat_engine.boltzmann_weights();
    auto thermo     = stat_engine.compute();

    // Convert weights to log-energies for binning (use negative log as proxy)
    std::vector<double> log_weights;
    log_weights.reserve(weights.size());
    for (double w : weights)
        if (w > 0.0)
            log_weights.push_back(-std::log(w));

    double S_conf_bits = compute_shannon_entropy(log_weights, DEFAULT_HIST_BINS);

    // 2. Torsional ENCoM vibrational entropy (protein backbone modes)
    double S_vib = 0.0;
    if (tencm_model.is_built()) {
        S_vib = compute_torsional_vibrational_entropy(
            tencm_model.modes(), temperature_K);
    }

    // 3. Combine: total entropy-weighted correction
    //    ΔG_corr = ΔG_base + (-T)(S_conf_physical + S_vib)
    //    S_conf_physical ≈ S_conf_bits * kB (convert bits to kcal/mol·K)
    double S_conf_physical = S_conf_bits * kB_kcal; // kcal/mol·K
    double total_entropy   = S_conf_physical * (1.0 + 0.5 * S_conf_bits) + S_vib;
    double entropy_contrib = -temperature_K * total_entropy;
    double final_deltaG    = base_deltaG + entropy_contrib;

    std::string report =
        "ShannonThermoStack: Shannon(conf)=" + std::to_string(S_conf_bits) +
        " bits, S_vib=" + std::to_string(S_vib) +
        " kcal/mol/K, ΔG=" + std::to_string(final_deltaG) + " kcal/mol"
        " [Metal=" +
#ifdef ENABLE_METAL_CORE
        "ON"
#else
        "OFF"
#endif
        + "]";

    return { final_deltaG, S_conf_bits, S_vib, entropy_contrib, report };
}

} // namespace shannon_thermo
