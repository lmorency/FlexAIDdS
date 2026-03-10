// statmech.cpp — Statistical Mechanics Engine implementation
//
// Notation:
//   β  = 1/(kB T)
//   Z  = Σ_i  n_i exp(−β E_i)        (canonical partition function)
//   F  = −kT ln Z                      (Helmholtz free energy)
//   ⟨E⟩ = (1/Z) Σ_i  n_i E_i exp(−β E_i)
//   C_v = (⟨E²⟩ − ⟨E⟩²) / (kT²)      (heat capacity)
//   S  = (⟨E⟩ − F) / T                 (entropy)
//
// All sums use log-sum-exp for numerical stability when energies span
// hundreds of kcal/mol (common in docking).

#include "statmech.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <stdexcept>

namespace statmech {

// ─── construction ────────────────────────────────────────────────────────────

StatMechEngine::StatMechEngine(double temperature_K)
    : T_(temperature_K)
    , beta_(1.0 / (kB_kcal * temperature_K))
{
    if (temperature_K <= 0.0)
        throw std::invalid_argument("StatMechEngine: temperature must be > 0");
}

// ─── add_sample ──────────────────────────────────────────────────────────────

void StatMechEngine::add_sample(double energy, int multiplicity) {
    ensemble_.push_back({energy, multiplicity});
}

// ─── log_sum_exp ─────────────────────────────────────────────────────────────

double StatMechEngine::log_sum_exp(std::span<const double> x) {
    // Handle empty array: return a value indicating "no valid data"
    if (x.empty()) return -1e308;  // Use large negative instead of -infinity for -ffast-math safety
    double x_max = *std::max_element(x.begin(), x.end());
    // If all values are -infinity or very small, return the max value
    // (This shouldn't happen in normal usage, but guard against it)
    if (x_max <= -1e308) return x_max;
    double sum = 0.0;
    for (double v : x)
        sum += std::exp(v - x_max);
    return x_max + std::log(sum);
}

// ─── compute ─────────────────────────────────────────────────────────────────

Thermodynamics StatMechEngine::compute() const {
    if (ensemble_.empty())
        throw std::runtime_error("StatMechEngine::compute: empty ensemble");

    const std::size_t N = ensemble_.size();

    // Build array of log-weights:  w_i = ln(n_i) − β E_i
    std::vector<double> log_w(N);
    for (std::size_t i = 0; i < N; ++i)
        log_w[i] = std::log(static_cast<double>(ensemble_[i].count)) -
                   beta_ * ensemble_[i].energy;

    double lnZ = log_sum_exp(log_w);

    // ⟨E⟩  = (1/Z) Σ n_i E_i exp(−β E_i)
    //       = exp(−lnZ) Σ E_i exp(log_w_i)
    // To keep stability: ⟨E⟩ = Σ E_i exp(log_w_i − lnZ)
    double E_avg  = 0.0;
    double E2_avg = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        double p_i = std::exp(log_w[i] - lnZ);
        double Ei  = ensemble_[i].energy;
        E_avg  += p_i * Ei;
        E2_avg += p_i * Ei * Ei;
    }

    double kT  = kB_kcal * T_;
    double var = E2_avg - E_avg * E_avg;

    Thermodynamics th;
    th.temperature    = T_;
    th.log_Z          = lnZ;
    th.free_energy    = -kT * lnZ;
    th.mean_energy    = E_avg;
    th.mean_energy_sq = E2_avg;
    th.heat_capacity  = var / (kT * kT);
    th.entropy        = (E_avg - th.free_energy) / T_;
    th.std_energy     = std::sqrt(std::max(0.0, var));
    return th;
}

// ─── boltzmann_weights ───────────────────────────────────────────────────────

std::vector<double> StatMechEngine::boltzmann_weights() const {
    if (ensemble_.empty()) return {};

    std::vector<double> log_w(ensemble_.size());
    for (std::size_t i = 0; i < ensemble_.size(); ++i)
        log_w[i] = std::log(static_cast<double>(ensemble_[i].count)) -
                   beta_ * ensemble_[i].energy;

    double lnZ = log_sum_exp(log_w);

    std::vector<double> w(ensemble_.size());
    for (std::size_t i = 0; i < ensemble_.size(); ++i)
        w[i] = std::exp(log_w[i] - lnZ);
    return w;
}

// ─── delta_G ─────────────────────────────────────────────────────────────────
// ΔG = F_this − F_ref = −kT (ln Z_this − ln Z_ref)

double StatMechEngine::delta_G(const StatMechEngine& reference) const {
    auto this_th = this->compute();
    auto ref_th  = reference.compute();
    double kT = kB_kcal * T_;
    return -kT * (this_th.log_Z - ref_th.log_Z);
}

// ─── Helmholtz convenience ───────────────────────────────────────────────────

double StatMechEngine::helmholtz(std::span<const double> energies, double T) {
    if (energies.empty())
        throw std::invalid_argument("helmholtz: empty energy list");
    double beta = 1.0 / (kB_kcal * T);
    std::vector<double> neg_beta_E(energies.size());
    for (std::size_t i = 0; i < energies.size(); ++i)
        neg_beta_E[i] = -beta * energies[i];
    double lnZ = log_sum_exp(neg_beta_E);
    return -(kB_kcal * T) * lnZ;
}

// ─── replica exchange ────────────────────────────────────────────────────────

std::vector<Replica>
StatMechEngine::init_replicas(std::span<const double> temperatures) {
    std::vector<Replica> reps;
    reps.reserve(temperatures.size());
    int id = 0;
    for (double T : temperatures) {
        Replica r;
        r.id             = id++;
        r.temperature    = T;
        r.beta           = 1.0 / (kB_kcal * T);
        r.current_energy = 0.0;
        reps.push_back(r);
    }
    return reps;
}

bool StatMechEngine::attempt_swap(Replica& a, Replica& b, std::mt19937& rng) {
    // Metropolis criterion:
    //   Δ = (β_a − β_b)(E_a − E_b)
    //   P_accept = min(1, exp(Δ))
    double delta = (a.beta - b.beta) * (a.current_energy - b.current_energy);
    if (delta >= 0.0) {
        std::swap(a.current_energy, b.current_energy);
        return true;
    }
    std::uniform_real_distribution<double> U(0.0, 1.0);
    if (U(rng) < std::exp(delta)) {
        std::swap(a.current_energy, b.current_energy);
        return true;
    }
    return false;
}

// ─── WHAM ────────────────────────────────────────────────────────────────────
// Weighted Histogram Analysis Method (Kumar et al. 1992)
// Simplified single-window version for post-hoc reweighting of GA ensemble.

std::vector<WHAMBin> StatMechEngine::wham(
    std::span<const double> energies,
    std::span<const double> coordinates,
    double temperature,
    int    n_bins,
    int    max_iter,
    double tolerance)
{
    if (energies.size() != coordinates.size())
        throw std::invalid_argument("wham: energies and coordinates size mismatch");
    if (energies.empty() || n_bins <= 0)
        throw std::invalid_argument("wham: invalid input");

    const std::size_t N = energies.size();
    double beta = 1.0 / (kB_kcal * temperature);

    // Find coordinate range
    double cmin = *std::min_element(coordinates.begin(), coordinates.end());
    double cmax = *std::max_element(coordinates.begin(), coordinates.end());
    double bin_w = (cmax - cmin) / n_bins;
    if (bin_w <= 0.0) bin_w = 1.0;

    // Histogram + Boltzmann-weighted histogram
    std::vector<double> raw_count(static_cast<std::size_t>(n_bins), 0.0);
    std::vector<double> boltz_sum(static_cast<std::size_t>(n_bins), 0.0);

    for (std::size_t i = 0; i < N; ++i) {
        int b = static_cast<int>((coordinates[i] - cmin) / bin_w);
        if (b < 0) b = 0;
        if (b >= n_bins) b = n_bins - 1;
        raw_count[static_cast<std::size_t>(b)] += 1.0;
        boltz_sum[static_cast<std::size_t>(b)] += std::exp(-beta * energies[i]);
    }

    // Free energy per bin: F_b = −kT ln( weighted_count_b / raw_count_b )
    // Iterative self-consistency (single-window simplification)
    std::vector<double> f_old(static_cast<std::size_t>(n_bins), 0.0);
    std::vector<double> f_new(static_cast<std::size_t>(n_bins), 0.0);

    for (int iter = 0; iter < max_iter; ++iter) {
        for (int b = 0; b < n_bins; ++b) {
            if (raw_count[static_cast<std::size_t>(b)] > 0.0) {
                f_new[static_cast<std::size_t>(b)] = -(kB_kcal * temperature) *
                    std::log(boltz_sum[static_cast<std::size_t>(b)] /
                             raw_count[static_cast<std::size_t>(b)]);
            } else {
                f_new[static_cast<std::size_t>(b)] = 0.0;
            }
        }
        // Shift so minimum = 0
        double fmin = *std::min_element(f_new.begin(), f_new.end());
        for (auto& f : f_new) f -= fmin;

        // Check convergence
        double maxdiff = 0.0;
        for (int b = 0; b < n_bins; ++b)
            maxdiff = std::max(maxdiff,
                std::abs(f_new[static_cast<std::size_t>(b)] -
                         f_old[static_cast<std::size_t>(b)]));
        f_old = f_new;
        if (maxdiff < tolerance) break;
    }

    // Build output
    std::vector<WHAMBin> result(static_cast<std::size_t>(n_bins));
    for (int b = 0; b < n_bins; ++b) {
        result[static_cast<std::size_t>(b)].coord_center = cmin + (b + 0.5) * bin_w;
        result[static_cast<std::size_t>(b)].count        = raw_count[static_cast<std::size_t>(b)];
        result[static_cast<std::size_t>(b)].free_energy  = f_new[static_cast<std::size_t>(b)];
    }
    return result;
}

// ─── thermodynamic integration ───────────────────────────────────────────────
// ΔG = ∫₀¹ ⟨∂V/∂λ⟩_λ dλ   (trapezoidal rule)

double StatMechEngine::thermodynamic_integration(std::span<const TIPoint> points) {
    if (points.size() < 2)
        throw std::invalid_argument("TI requires at least 2 points");

    double integral = 0.0;
    for (std::size_t i = 1; i < points.size(); ++i) {
        double dl = points[i].lambda - points[i-1].lambda;
        integral += 0.5 * dl * (points[i].dV_dlambda + points[i-1].dV_dlambda);
    }
    return integral;
}

// ─── BoltzmannLUT ────────────────────────────────────────────────────────────

BoltzmannLUT::BoltzmannLUT(double beta, double e_min, double e_max, int n_bins)
    : beta_(beta)
    , e_min_(e_min)
    , n_bins_(n_bins)
    , table_(static_cast<std::size_t>(n_bins))
{
    double range = e_max - e_min;
    if (range <= 0.0) range = 1.0;
    inv_bin_width_ = n_bins / range;

    for (int i = 0; i < n_bins; ++i) {
        double e = e_min + (static_cast<double>(i) + 0.5) * range / n_bins;
        table_[static_cast<std::size_t>(i)] = std::exp(-beta * e);
    }
}

double BoltzmannLUT::operator()(double energy) const noexcept {
    int idx = static_cast<int>((energy - e_min_) * inv_bin_width_);
    if (idx < 0) idx = 0;
    if (idx >= n_bins_) idx = n_bins_ - 1;
    return table_[static_cast<std::size_t>(idx)];
}

}  // namespace statmech
