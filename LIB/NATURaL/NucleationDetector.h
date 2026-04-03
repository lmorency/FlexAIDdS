// NucleationDetector.h — Cotranslational nucleation seed detection
//
// Detects sequence features that nucleate co-translational folding:
//
//   RNA receptors (riboswitches, ribozymes, rRNA):
//     • Hairpin seeds   — Watson-Crick stem ≥ 4 bp + loop 3–8 nt
//                         scored by simplified Turner 2004 nearest-neighbour ΔG
//     • G-quadruplex    — (G3+N1-7)×3 G3+ motif; stabilised by Mg²⁺ / K⁺
//
//   Protein receptors:
//     • Helix propensity — Chou-Fasman P_α window (6 residues, mean > 1.03)
//     • Hydrophobic cluster — ≥4 consecutive ILVFMW residues
//
// Each detected seed returns a folding_rate_boost multiplier (≥1.0) applied on
// top of the baseline k_fold in DualAssemblyEngine::run().  This allows the
// growth loop to model nucleation-competent windows that arise specifically from
// the sequence context, independent of whether the position is a pause site.
//
// References:
//   Turner 2004  Nucleic Acids Res 32:D135 (RNA nearest-neighbour params)
//   Zuker  2003  Nucleic Acids Res 31:3406 (mfold ΔG conventions)
//   Sen   2002  Curr. Opinion Struct. Biol. 12:327 (G-quadruplex biology)
//   Chou & Fasman 1978 Annu Rev Biochem 47:251 (helix propensity)
//   Dill  1990  Biochemistry 29:7133 (hydrophobic collapse nuclei)
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>

namespace natural {

// ─── NucleationSeed ──────────────────────────────────────────────────────────
struct NucleationSeed {
    enum class Type {
        RNA_HAIRPIN,         // palindromic stem-loop in RNA sequence
        RNA_GQUADRUPLEX,     // G3+N1-7 repeats forming G-quartet
        PROTEIN_HELIX,       // Chou-Fasman α-helix propensity window
        PROTEIN_HYDROPHOBIC  // hydrophobic cluster collapse nucleus
    };

    Type        type;
    int         start_pos;           // 0-based position of seed in sequence
    int         end_pos;             // inclusive
    double      folding_rate_boost;  // multiplier on k_fold (≥1.0)
    double      estimated_dG;        // kcal/mol (negative = thermodynamically favorable)
    std::string motif;               // human-readable description (e.g. "stem:GCGCGC")
};

// ─── NucleationSeedDetector ──────────────────────────────────────────────────
class NucleationSeedDetector {
public:
    // ── Primary entry point ───────────────────────────────────────────────────
    // Dispatches to RNA or protein detection based on is_rna flag.
    static std::vector<NucleationSeed> detect(
        const std::string& seq,
        bool               is_rna,
        double             temperature_K = 310.0);

    // ── RNA detectors ─────────────────────────────────────────────────────────

    // Hairpin stem-loop: O(n²) scan for complementary runs (Watson-Crick + GU wobble).
    // min_stem_bp ≥ 4; loop_min/max = 3/8 nt (Groebe & Uhlenbeck 1988).
    // folding_rate_boost = 1 + max(0, -ΔG_stem) / kT
    static std::vector<NucleationSeed> detect_rna_hairpins(
        const std::string& seq,
        int    min_stem_bp    = 4,
        int    loop_min_nt    = 3,
        int    loop_max_nt    = 8,
        double temperature_K  = 310.0);

    // G-quadruplex: (G3+N1-7)×3 G3+ pattern; folding_rate_boost = 5.0 (strong Mg²⁺ stabilisation)
    static std::vector<NucleationSeed> detect_rna_gquads(
        const std::string& seq,
        int min_g_run = 3);

    // ── Protein detectors ─────────────────────────────────────────────────────

    // Chou-Fasman α-helix propensity: 6-residue sliding window.
    // Seed when mean P_α > helix_thresh (default 1.03 = nucleation-competent).
    // folding_rate_boost = 1 + (mean_Palpha - helix_thresh) × 5
    static std::vector<NucleationSeed> detect_protein_helix(
        const std::string& seq,
        int    window       = 6,
        double helix_thresh = 1.03);

    // Hydrophobic cluster: ≥min_run consecutive ILVFMW residues.
    // folding_rate_boost = 2.0 + 0.3 × (run_len − min_run)
    static std::vector<NucleationSeed> detect_protein_hydrophobic(
        const std::string& seq,
        int min_run = 4);

    // ── Utility ───────────────────────────────────────────────────────────────

    // Build a position → boost factor vector (1.0 where no seed).
    // Overlapping seeds take the maximum boost at each position.
    static std::vector<double> position_boost_map(
        const std::vector<NucleationSeed>& seeds,
        int seq_len);

private:
    // ── RNA helpers ───────────────────────────────────────────────────────────

    // Watson-Crick complement (A-U, G-C) + GU wobble
    static bool are_complementary(char a, char b);

    // Simplified Turner 2004 nearest-neighbour ΔG for a stem of given length.
    // GC: -1.0, AU: -0.4, GU wobble: -0.1 kcal/mol (per pair, no stacking for brevity)
    // Loop penalty: +3.0 + 1.7*ln(loop_len/3) kcal/mol (Mathews 1999)
    static double hairpin_dG(const std::string& seq,
                              int stem_start, int stem_len,
                              int loop_start, int loop_len);

    // ── Protein helpers ───────────────────────────────────────────────────────

    // Chou-Fasman P_α table (Chou & Fasman 1978, updated by Levitt 1978)
    static double cf_palpha(char aa);

    // Is this amino acid part of the ILVFMW hydrophobic core set?
    static bool is_core_hydrophobic(char aa);
};

} // namespace natural
