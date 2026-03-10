// TransloconInsertion.h — Sec translocon lateral gating model
//
// Models co-translational insertion of transmembrane (TM) helices via the
// Sec61 translocon using the empirical apparent free energy scale of:
//
//   Hessa et al. (2007) Nature 450:1026–1030
//   "Molecular code for transmembrane-helix recognition by the Sec61 translocon"
//
// For each 19-residue sliding window, ΔG_insert is computed as the sum of
// per-residue contributions from the Hessa 2007 scale plus a position-weighted
// helix-dipole correction (Wimley-White 1996).  Insertion probability:
//
//   P_insert = 1 / (1 + exp(ΔG_insert / k_B T))
//
// Integration with NATURaL DualAssembly:
//   – Called at each GrowthStep once the emerging chain exceeds TM_WINDOW_LEN
//   – Sets GrowthStep::tm_inserted / tm_insertion_dG
//   – Accounts for ribosome tunnel: only residues beyond tunnel exit are screened
//
// Hardware acceleration:
//   – AVX-512 / AVX2: 8 (or 4) residue score accumulation per cycle
//   – Eigen: ArrayXf for vectorized per-position scoring
//   – OpenMP: parallel window scanning for long sequences
#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

#include <vector>
#include <string>
#include <array>

#ifdef FLEXAIDS_HAS_EIGEN
#  include <Eigen/Dense>
#  include <Eigen/Core>
#endif

#ifdef FLEXAIDS_USE_AVX512
#  include <immintrin.h>
#elif defined(__AVX2__)
#  include <immintrin.h>
#endif

namespace translocon {

// ─── constants ────────────────────────────────────────────────────────────────
static constexpr int    TM_WINDOW_LEN    = 19;   // canonical TM helix length (aa)
static constexpr double KT_310K         = 0.616; // kcal/mol at 310 K (body temp)
static constexpr double HELIX_DIPOLE_CORR = -0.12; // kcal/mol per helical position

// ─── per-residue apparent ΔG (Hessa 2007 Table 1, kcal/mol) ──────────────────
// Positive = hydrophilic (penalty), negative = hydrophobic (favorable)
static constexpr std::array<double, 256> make_hessa_scale() noexcept {
    std::array<double, 256> s{};
    for (auto& v : s) v = 0.5;  // default: neutral residue
    // 1-letter codes → apparent ΔG (kcal/mol), from Hessa et al. 2007:
    s['A'] = s['a'] =  0.11;
    s['R'] = s['r'] =  2.58;
    s['N'] = s['n'] =  2.05;
    s['D'] = s['d'] =  3.49;
    s['C'] = s['c'] = -0.13;
    s['Q'] = s['q'] =  0.77;
    s['E'] = s['e'] =  3.22;
    s['G'] = s['g'] =  0.74;
    s['H'] = s['h'] =  1.74;
    s['I'] = s['i'] = -0.31;
    s['L'] = s['l'] = -0.56;
    s['K'] = s['k'] =  2.15;
    s['M'] = s['m'] = -0.23;
    s['F'] = s['f'] = -1.13;
    s['P'] = s['p'] =  1.56;   // helix-breaking: penalty
    s['S'] = s['s'] =  0.84;
    s['T'] = s['t'] =  0.45;
    s['W'] = s['w'] = -0.74;
    s['Y'] = s['y'] = -0.08;
    s['V'] = s['v'] = -0.07;
    s['X'] = s['x'] =  0.50;   // unknown
    return s;
}
static constexpr auto HESSA_SCALE = make_hessa_scale();

// ─── per-position weight (helix-dipole, Wimley-White 1996) ───────────────────
// Cosine envelope: positions at helix center contribute more than termini.
inline double position_weight(int pos_in_window, int window_len = TM_WINDOW_LEN) {
    double x = (2.0 * pos_in_window / (window_len - 1)) - 1.0; // [-1, 1]
    return 0.5 * (1.0 + std::cos(M_PI * x));                    // raised cosine
}

// ─── result types ─────────────────────────────────────────────────────────────
struct TMWindow {
    int    start_residue;       // index into receptor sequence
    int    length;              // always TM_WINDOW_LEN unless near terminus
    double deltaG_insert;       // kcal/mol; negative = spontaneous insertion
    double p_insert;            // P = 1/(1 + exp(ΔG / kT))
    bool   is_inserted;         // true if p_insert > insertion_threshold
};

// ─── TransloconInsertion ─────────────────────────────────────────────────────
class TransloconInsertion {
public:
    // insertion_threshold: P_insert > this → laterally gated
    explicit TransloconInsertion(double temperature_K       = 310.0,
                                  double insertion_threshold = 0.5,
                                  int    tunnel_length_aa    = 34);

    // Scan the sequence for insertable TM windows (parallelized).
    // sequence: 1-letter amino acid sequence.
    std::vector<TMWindow> scan(const std::string& sequence) const;

    // Check a single window starting at start_res in sequence.
    TMWindow check_window(const std::string& sequence, int start_res) const;

    // Score a single TM window (Hessa 2007 + position weights).
    // Returns ΔG_insert in kcal/mol.
    double score_window(const std::string& sequence,
                        int                start_res,
                        int                len) const;

    // Accessors
    double temperature_K()       const noexcept { return T_K_; }
    double insertion_threshold() const noexcept { return threshold_; }
    int    tunnel_length()       const noexcept { return tunnel_len_; }

private:
    double T_K_;
    double kT_;
    double threshold_;
    int    tunnel_len_;

    // AVX-512 accelerated window scoring (8 residues/cycle)
    double score_window_avx512(const char* seq, int len) const noexcept;
    // AVX2 accelerated window scoring (4 residues/cycle)
    double score_window_avx2(const char* seq, int len) const noexcept;
    // Eigen-vectorised position weights + dot product
    double score_window_eigen(const char* seq, int len) const noexcept;
    // Scalar fallback
    double score_window_scalar(const char* seq, int len) const noexcept;
};

} // namespace translocon
