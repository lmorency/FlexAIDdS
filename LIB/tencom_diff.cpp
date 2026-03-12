// tencom_diff.cpp — Eigenvalue/eigenvector differential engine implementation

#include "tencom_diff.h"

#include <algorithm>
#include <numeric>
#include <iostream>
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

    // ── Dimension mismatch warnings ─────────────────────────────────────────
    int ref_nres = ref_enm.n_residues();
    int tgt_nres = tgt_enm.n_residues();
    if (ref_nres != tgt_nres) {
        std::cerr << "  Warning: residue count mismatch (ref=" << ref_nres
                  << ", tgt=" << tgt_nres << ") — mode overlaps will be NaN.\n";
    }
    int ref_nmodes = static_cast<int>(ref_modes.size());
    int tgt_nmodes = static_cast<int>(tgt_modes.size());
    if (ref_nmodes != tgt_nmodes) {
        int diff_pct = std::abs(ref_nmodes - tgt_nmodes) * 100
                       / std::max(ref_nmodes, tgt_nmodes);
        if (diff_pct > 20) {
            std::cerr << "  Warning: mode count differs by " << diff_pct
                      << "% (ref=" << ref_nmodes << ", tgt=" << tgt_nmodes << ").\n";
        }
    }

    // ── Per-mode comparisons ────────────────────────────────────────────────
    int n_compare = std::min(ref_nmodes, tgt_nmodes);
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

    // ── Per-residue vibrational entropy decomposition ────────────────────────
    // Distribute global S_vib across residues proportional to their B-factors.
    // B_i ∝ <Δr_i²> ∝ Σ_k (v_ki² / λ_k), so B_i/Σ(B_j) gives each
    // residue's fractional contribution to the total thermal fluctuation.
    auto decompose_svib = [](const std::vector<float>& bfactors, double total_svib) {
        std::vector<double> per_res(bfactors.size(), 0.0);
        double sum_bf = 0.0;
        for (float bf : bfactors) sum_bf += bf;
        if (sum_bf > 0.0) {
            for (size_t i = 0; i < bfactors.size(); ++i) {
                per_res[i] = total_svib * (bfactors[i] / sum_bf);
            }
        }
        return per_res;
    };

    result.per_residue_svib_ref = decompose_svib(
        result.bfactors_ref, result.svib_ref.S_vib_kcal_mol_K);
    result.per_residue_svib_tgt = decompose_svib(
        result.bfactors_tgt, result.svib_tgt.S_vib_kcal_mol_K);

    int n_pr = std::min(result.per_residue_svib_ref.size(),
                        result.per_residue_svib_tgt.size());
    result.per_residue_delta_svib.resize(n_pr);
    for (int i = 0; i < n_pr; ++i) {
        result.per_residue_delta_svib[i] =
            result.per_residue_svib_tgt[i] - result.per_residue_svib_ref[i];
    }

    return result;
}

}  // namespace tencom_diff
