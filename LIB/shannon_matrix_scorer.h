// shannon_matrix_scorer.h — Two-term scoring engine integrating the 256×256
// soft contact matrix with analytic LJ+Coulomb refinement.
//
// Pipeline:
//   1. Matrix pre-filter: fast O(1) lookup eliminates ~90% of poses
//   2. Analytic LJ+Coulomb refinement on survivors
//   3. Boltzmann-weighted Shannon entropy over ensemble
//   4. FastOPTICS super-cluster detection
//   5. Final ΔG proxy
//
// Both scoring paths (matrix and analytic) live in the same class, operating
// on the same pose data.
#pragma once

#include "soft_contact_matrix.h"
#include "atom_typing_256.h"
#include "ShannonThermoStack/ShannonThermoStack.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace scorer {

// ─── constants ──────────────────────────────────────────────────────────────
inline constexpr double kB_kcal   = 0.001987206;
inline constexpr double KCOULOMB  = 332.0637;   // kcal·Å/(mol·e²)
inline constexpr float  LJ_CUTOFF = 8.0f;       // Å, beyond this LJ ≈ 0
inline constexpr float  ELEC_CUTOFF = 12.0f;     // Å
inline constexpr float  DEFAULT_DIELECTRIC = 4.0f;

// ─── contact data ───────────────────────────────────────────────────────────
struct Contact {
    uint8_t type_a;        // 256-type of atom A
    uint8_t type_b;        // 256-type of atom B
    float   area;          // Voronoi contact area (Å²)
    float   distance;      // inter-atomic distance (Å)
    float   radius_a;      // vdW radius of atom A
    float   radius_b;      // vdW radius of atom B
    float   charge_a;      // partial charge of atom A (e)
    float   charge_b;      // partial charge of atom B (e)
};

// ─── pose scoring result ────────────────────────────────────────────────────
struct PoseScore {
    float matrix_score;    // fast matrix-based score
    float lj_score;        // Lennard-Jones analytic score
    float elec_score;      // Coulomb electrostatic score
    float total_score;     // combined final score
    bool  survived_filter; // true if passed matrix pre-filter
};

// ─── ensemble result ────────────────────────────────────────────────────────
struct EnsembleResult {
    double deltaG;              // free energy proxy (kcal/mol)
    double shannonEntropy;      // Shannon entropy (bits)
    double meanScore;           // Boltzmann-weighted mean score
    int    n_survivors;         // poses that passed matrix filter
    int    n_superclusters;     // FastOPTICS super-cluster count
    std::vector<float> scores;  // per-pose total scores (survivors only)
};

// ═══════════════════════════════════════════════════════════════════════════════
// ShannonMatrixScorer — unified two-term scoring engine
// ═══════════════════════════════════════════════════════════════════════════════

class ShannonMatrixScorer {
public:
    explicit ShannonMatrixScorer(const scm::SoftContactMatrix& matrix,
                                  double temperature_K = 298.15,
                                  float filter_threshold = 0.0f)
        : matrix_(matrix)
        , temperature_(temperature_K)
        , beta_(1.0 / (kB_kcal * temperature_K))
        , filter_threshold_(filter_threshold)
        , dielectric_(DEFAULT_DIELECTRIC)
    {}

    void set_temperature(double T) {
        temperature_ = T;
        beta_ = 1.0 / (kB_kcal * T);
    }

    void set_filter_threshold(float t) { filter_threshold_ = t; }
    void set_dielectric(float d) { dielectric_ = d; }

    // ── single-pose scoring ─────────────────────────────────────────────

    // Stage 1: Fast matrix pre-filter score
    float matrix_score(const std::vector<Contact>& contacts) const {
        float score = 0.0f;
        for (const auto& c : contacts)
            score += matrix_.lookup(c.type_a, c.type_b) * c.area;
        return score;
    }

    // Stage 2: Analytic LJ + Coulomb refinement
    float analytic_score(const std::vector<Contact>& contacts) const {
        float lj_total = 0.0f;
        float elec_total = 0.0f;

        for (const auto& c : contacts) {
            float r = c.distance;

            // Lennard-Jones 12-6
            if (r > 0.5f && r < LJ_CUTOFF) {
                float sigma = c.radius_a + c.radius_b;
                float sr6 = sigma / r;
                sr6 = sr6 * sr6 * sr6;  // (σ/r)³
                sr6 = sr6 * sr6;        // (σ/r)⁶
                float sr12 = sr6 * sr6;
                // Standard LJ: 4ε[(σ/r)¹² - (σ/r)⁶]
                // Use ε=0.1 kcal/mol as reference well depth
                lj_total += 0.4f * (sr12 - sr6);
            }

            // Coulomb with distance-dependent dielectric
            if (r > 0.5f && r < ELEC_CUTOFF &&
                c.charge_a != 0.0f && c.charge_b != 0.0f) {
                elec_total += static_cast<float>(KCOULOMB) *
                              c.charge_a * c.charge_b /
                              (dielectric_ * r * r);
            }
        }
        return lj_total + elec_total;
    }

    // Full two-stage scoring for a single pose
    PoseScore score_pose(const std::vector<Contact>& contacts) const {
        PoseScore result{};
        result.matrix_score = matrix_score(contacts);

        // Pre-filter: reject obviously bad poses
        if (result.matrix_score > filter_threshold_) {
            result.survived_filter = false;
            result.total_score = result.matrix_score;
            return result;
        }

        result.survived_filter = true;

        // Analytic refinement on survivors
        float analytic = analytic_score(contacts);
        result.lj_score = analytic;  // combined LJ+elec for simplicity
        result.elec_score = 0.0f;

        // Weighted combination: 60% matrix + 40% analytic
        result.total_score = 0.6f * result.matrix_score +
                             0.4f * analytic;
        return result;
    }

    // ── batch scoring with AVX2 ─────────────────────────────────────────

    float batch_matrix_score(const uint8_t* type_a, const uint8_t* type_b,
                              const float* areas, int n) const {
        return matrix_.score_contacts(type_a, type_b, areas, n);
    }

    // ── ensemble scoring with Shannon entropy ───────────────────────────

    EnsembleResult score_ensemble(
            const std::vector<std::vector<Contact>>& pose_contacts) const {
        EnsembleResult result{};
        int n_poses = static_cast<int>(pose_contacts.size());
        if (n_poses == 0) return result;

        // Stage 1: Score all poses, filter
        std::vector<float> all_scores(n_poses);
        std::vector<bool>  survived(n_poses, false);

        for (int i = 0; i < n_poses; ++i) {
            float ms = matrix_score(pose_contacts[i]);
            all_scores[i] = ms;
            if (ms <= filter_threshold_) {
                survived[i] = true;
                result.n_survivors++;
            }
        }

        // Stage 2: Analytic refinement on survivors
        std::vector<float> final_scores;
        final_scores.reserve(result.n_survivors);

        for (int i = 0; i < n_poses; ++i) {
            if (!survived[i]) continue;
            float analytic = analytic_score(pose_contacts[i]);
            float total = 0.6f * all_scores[i] + 0.4f * analytic;
            final_scores.push_back(total);
        }

        if (final_scores.empty()) {
            result.deltaG = 1e6;
            return result;
        }

        // Stage 3: Boltzmann weighting + Shannon entropy
        double e_min = *std::min_element(final_scores.begin(),
                                          final_scores.end());
        std::vector<double> boltz_weights(final_scores.size());
        double Z = 0.0;
        for (size_t i = 0; i < final_scores.size(); ++i) {
            boltz_weights[i] = std::exp(-beta_ * (final_scores[i] - e_min));
            Z += boltz_weights[i];
        }
        for (auto& w : boltz_weights) w /= Z;

        // Shannon entropy (bits)
        double H = 0.0;
        for (double w : boltz_weights) {
            if (w > 1e-15) H -= w * std::log2(w);
        }

        // Mean score
        double mean = 0.0;
        for (size_t i = 0; i < final_scores.size(); ++i)
            mean += boltz_weights[i] * final_scores[i];

        // Free energy: F = -kT ln Z + e_min
        double F = -kB_kcal * temperature_ * std::log(Z) + e_min;

        // Stage 4: Super-cluster detection on the matrix
        auto clusters = scm::find_super_clusters(matrix_);
        result.n_superclusters = clusters.n_clusters;

        result.deltaG = F;
        result.shannonEntropy = H;
        result.meanScore = mean;
        result.scores = std::move(final_scores);
        return result;
    }

private:
    scm::SoftContactMatrix matrix_;
    double temperature_;
    double beta_;
    float  filter_threshold_;
    float  dielectric_;
};

} // namespace scorer
