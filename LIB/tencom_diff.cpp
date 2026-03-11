// tencom_diff.cpp — Eigenvalue/eigenvector differential engine implementation

#include "tencom_diff.h"

#include <algorithm>
#include <numeric>
#include <cmath>

namespace tencom_diff {

// ─── Convert tencm modes to encom modes ─────────────────────────────────────

std::vector<encom::NormalMode> to_encom_modes(
    const std::vector<tencm::NormalMode>& tencm_modes)
{
    std::vector<encom::NormalMode> result;
    result.reserve(tencm_modes.size());

    for (int i = 0; i < static_cast<int>(tencm_modes.size()); ++i) {
        encom::NormalMode em;
        em.index = i + 1;
        em.eigenvalue = tencm_modes[i].eigenvalue;
        em.frequency = std::sqrt(std::abs(em.eigenvalue));
        em.eigenvector = tencm_modes[i].eigenvector;
        result.push_back(std::move(em));
    }

    return result;
}

// ─── Compute differential ───────────────────────────────────────────────────

DifferentialResult compute_differential(
    const tencm::TorsionalENM& ref_enm,
    const tencm::TorsionalENM& tgt_enm,
    const std::string& ref_name,
    const std::string& tgt_name,
    double temperature_K,
    double eigenvalue_cutoff)
{
    DifferentialResult result;
    result.ref_name = ref_name;
    result.tgt_name = tgt_name;
    result.temperature = temperature_K;

    const auto& ref_modes = ref_enm.modes();
    const auto& tgt_modes = tgt_enm.modes();

    // ── Vibrational entropy for each ────────────────────────────────────────
    auto ref_encom = to_encom_modes(ref_modes);
    auto tgt_encom = to_encom_modes(tgt_modes);

    result.svib_ref = encom::ENCoMEngine::compute_vibrational_entropy(
        ref_encom, temperature_K, eigenvalue_cutoff);
    result.svib_tgt = encom::ENCoMEngine::compute_vibrational_entropy(
        tgt_encom, temperature_K, eigenvalue_cutoff);

    result.delta_S_vib = result.svib_tgt.S_vib_kcal_mol_K
                       - result.svib_ref.S_vib_kcal_mol_K;
    result.delta_F_vib = -temperature_K * result.delta_S_vib;

    // ── Per-mode comparisons ────────────────────────────────────────────────
    int n_compare = std::min(static_cast<int>(ref_modes.size()),
                             static_cast<int>(tgt_modes.size()));
    bool dims_match = (ref_enm.n_bonds() == tgt_enm.n_bonds());

    result.mode_comparisons.reserve(n_compare);
    for (int i = 0; i < n_compare; ++i) {
        ModeComparison mc;
        mc.mode_index = i;
        mc.eigenvalue_ref = ref_modes[i].eigenvalue;
        mc.eigenvalue_tgt = tgt_modes[i].eigenvalue;
        mc.delta_eigenvalue = mc.eigenvalue_tgt - mc.eigenvalue_ref;

        // Mode overlap (dot product of eigenvectors)
        if (dims_match && !ref_modes[i].eigenvector.empty()
                       && !tgt_modes[i].eigenvector.empty()
                       && ref_modes[i].eigenvector.size() == tgt_modes[i].eigenvector.size()) {
            double dot = 0.0;
            for (size_t j = 0; j < ref_modes[i].eigenvector.size(); ++j) {
                dot += ref_modes[i].eigenvector[j] * tgt_modes[i].eigenvector[j];
            }
            mc.overlap = std::abs(dot);
        } else {
            mc.overlap = std::numeric_limits<double>::quiet_NaN();
        }

        result.mode_comparisons.push_back(mc);
    }

    // ── B-factor differentials ──────────────────────────────────────────────
    float T = static_cast<float>(temperature_K);
    result.bfactors_ref = ref_enm.bfactors(T);
    result.bfactors_tgt = tgt_enm.bfactors(T);

    int n_bf = std::min(result.bfactors_ref.size(), result.bfactors_tgt.size());
    result.delta_bfactors.resize(n_bf);
    for (int i = 0; i < n_bf; ++i) {
        result.delta_bfactors[i] = result.bfactors_tgt[i] - result.bfactors_ref[i];
    }

    return result;
}

}  // namespace tencom_diff
