// NATURaLDualAssembly.h — Native Assembly of co-Transcriptionally/co-Translationally
//                          Unified Receptor–Ligand (NATURaL) module
//
// Auto-detects when the ligand is nucleotide/sugar-containing or the receptor
// is a nucleic acid chain, then activates co-translational DualAssembly mode.
//
// In DualAssembly, the receptor chain grows residue-by-residue while the ligand
// is present from the start, computing incremental CF and Shannon entropy at each
// growth step to capture co-translational stereochemical selection.
//
// Fully integrated with:
//   – ShannonThermoStack (entropy per growth step)
//   – SugarPuckerGene (activated for nucleotide ligands)
//   – AlphaShape Contact Function (incremental ΔSASA)
#pragma once

#include "../flexaid.h"
#include "../statmech.h"
#include <string>
#include <vector>
#include <functional>

namespace natural {

// ─── detection ───────────────────────────────────────────────────────────────

// Returns true if any ligand atom has an O2' or ribose-pattern atom name,
// indicating a nucleotide or nucleoside ligand.
bool is_nucleotide_ligand(const atom* atoms, int n_lig_atoms);

// Returns true if the receptor contains nucleic acid residues
// (deoxyribose/ribose backbone: DA, DG, DC, DT, A, G, C, U, etc.)
bool is_nucleic_acid_receptor(const resid* residues, int n_residues);

// ─── NATURaL configuration ───────────────────────────────────────────────────
struct NATURaLConfig {
    bool   enabled                = false;
    bool   co_translational_growth = false;
    bool   sugar_pucker_auto       = true;
    double temperature_K           = 298.15;
    int    max_growth_steps        = -1; // -1 = full sequence length
};

// Auto-configure from receptor and ligand properties.
NATURaLConfig auto_configure(const atom*  atoms,
                              int          n_lig_atoms,
                              const resid* residues,
                              int          n_residues);

// ─── DualAssembly engine ─────────────────────────────────────────────────────
class DualAssemblyEngine {
public:
    explicit DualAssemblyEngine(const NATURaLConfig& cfg,
                                 FA_Global* FA, VC_Global* VC,
                                 atom* atoms, resid* residues,
                                 int n_residues);

    // Run the incremental co-translational growth simulation.
    // At each step: grow one residue, compute CF + Shannon entropy.
    // Returns the thermodynamic score trajectory.
    struct GrowthStep {
        int    residue_idx;
        double cf_score;        // Contact Function (kcal/mol)
        double shannon_entropy; // bits
        double cumulative_deltaG; // kcal/mol
    };
    std::vector<GrowthStep> run();

    // Final ΔG after full co-translational folding
    double final_deltaG() const noexcept { return final_deltaG_; }

    // Whether NATURaL mode was active
    bool is_active() const noexcept { return config_.enabled; }

private:
    NATURaLConfig config_;
    FA_Global*    FA_;
    VC_Global*    VC_;
    atom*         atoms_;
    resid*        residues_;
    int           n_residues_;
    double        final_deltaG_ = 0.0;

    // Compute a lightweight CF score for the current partial complex
    double compute_partial_cf(int n_grown_residues) const;

    // Compute Shannon entropy over accumulated growth ensemble
    double compute_growth_entropy(const std::vector<double>& cf_trajectory) const;
};

} // namespace natural
