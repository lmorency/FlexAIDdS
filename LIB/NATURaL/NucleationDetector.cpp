// NucleationDetector.cpp — Cotranslational nucleation seed detection
#include "NucleationDetector.h"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <cassert>

namespace natural {

// ─── Primary dispatcher ──────────────────────────────────────────────────────
std::vector<NucleationSeed> NucleationSeedDetector::detect(
    const std::string& seq, bool is_rna, double temperature_K)
{
    std::vector<NucleationSeed> all;
    if (seq.empty()) return all;

    if (is_rna) {
        auto hairpins = detect_rna_hairpins(seq, 4, 3, 8, temperature_K);
        auto gquads   = detect_rna_gquads(seq, 3);
        all.insert(all.end(), hairpins.begin(), hairpins.end());
        all.insert(all.end(), gquads.begin(),   gquads.end());
    } else {
        auto helices  = detect_protein_helix(seq, 6, 1.03);
        auto hydro    = detect_protein_hydrophobic(seq, 4);
        all.insert(all.end(), helices.begin(), helices.end());
        all.insert(all.end(), hydro.begin(),   hydro.end());
    }
    // Sort by start position for deterministic output
    std::sort(all.begin(), all.end(),
              [](const NucleationSeed& a, const NucleationSeed& b){
                  return a.start_pos < b.start_pos;
              });
    return all;
}

// ─── RNA complement check ────────────────────────────────────────────────────
bool NucleationSeedDetector::are_complementary(char a, char b)
{
    // Canonicalise to uppercase; treat T as U
    auto canon = [](char c) -> char {
        c = (char)std::toupper((unsigned char)c);
        return (c == 'T') ? 'U' : c;
    };
    char ca = canon(a), cb = canon(b);
    if ((ca=='A' && cb=='U') || (ca=='U' && cb=='A')) return true;
    if ((ca=='G' && cb=='C') || (ca=='C' && cb=='G')) return true;
    if ((ca=='G' && cb=='U') || (ca=='U' && cb=='G')) return true; // GU wobble
    return false;
}

// ─── Hairpin ΔG (simplified Turner 2004) ────────────────────────────────────
// Stem contribution: -1.0 kcal per GC pair, -0.4 per AU, -0.1 per GU
// Loop penalty (Mathews 1999 loop entropy): +3.0 + 1.7*ln(loop_len/3)
double NucleationSeedDetector::hairpin_dG(
    const std::string& seq,
    int stem_start, int stem_len,
    int loop_start, int loop_len)
{
    double dG = 0.0;
    auto canon = [](char c) -> char {
        c = (char)std::toupper((unsigned char)c);
        return (c == 'T') ? 'U' : c;
    };

    // Stem stacking energy
    for (int i = 0; i < stem_len; ++i) {
        if (stem_start + i >= (int)seq.size()) break;
        int j_idx = loop_start + loop_len + (stem_len - 1 - i);
        if (j_idx >= (int)seq.size()) break;

        char a = canon(seq[stem_start + i]);
        char b = canon(seq[j_idx]);
        if ((a=='G' && b=='C') || (a=='C' && b=='G'))      dG -= 1.0;
        else if ((a=='A' && b=='U') || (a=='U' && b=='A')) dG -= 0.4;
        else if ((a=='G' && b=='U') || (a=='U' && b=='G')) dG -= 0.1;
    }

    // Loop entropy penalty
    if (loop_len >= 3) {
        dG += 3.0 + 1.7 * std::log(static_cast<double>(loop_len) / 3.0);
    }
    return dG;
}

// ─── RNA hairpin seed detection (O(n²)) ──────────────────────────────────────
std::vector<NucleationSeed> NucleationSeedDetector::detect_rna_hairpins(
    const std::string& seq,
    int    min_stem_bp,
    int    loop_min_nt,
    int    loop_max_nt,
    double temperature_K)
{
    std::vector<NucleationSeed> seeds;
    const double kT = 0.001987206 * temperature_K; // kcal/mol
    const int n = (int)seq.size();

    // For each possible stem start i, try increasing stem lengths
    for (int i = 0; i + 2 * min_stem_bp + loop_min_nt <= n; ++i) {
        for (int stem_len = min_stem_bp; ; ++stem_len) {
            // Check stem pairing: seq[i..i+stem_len-1] pairs with the reverse of
            // seq[j..j+stem_len-1] where j = loop_end + 1
            // Structure: 5' [stem_start..i+stem_len-1] [loop] [j..j+stem_len-1] 3'
            // We scan all loop sizes in [loop_min_nt, loop_max_nt]

            bool any_loop_found = false;
            for (int loop_len = loop_min_nt; loop_len <= loop_max_nt; ++loop_len) {
                int loop_start = i + stem_len;
                int j_start    = loop_start + loop_len; // start of complementary stem
                int j_end      = j_start + stem_len - 1; // end of complementary stem

                if (j_end >= n) break;

                // Check all base pairs of the stem
                bool valid_stem = true;
                int n_pairs = 0;
                for (int p = 0; p < stem_len; ++p) {
                    char a = seq[i + p];
                    char b = seq[j_end - p]; // antiparallel
                    if (!are_complementary(a, b)) { valid_stem = false; break; }
                    ++n_pairs;
                }

                if (valid_stem && n_pairs >= min_stem_bp) {
                    double dG = hairpin_dG(seq, i, stem_len, loop_start, loop_len);
                    if (dG < 0.0) { // only thermodynamically stable hairpins
                        double boost = 1.0 + std::max(0.0, -dG) / kT;
                        boost = std::min(boost, 20.0); // cap boost

                        std::ostringstream motif;
                        motif << "hairpin:stem=" << stem_len << "bp,loop=" << loop_len
                              << "nt,dG=" << std::fixed << std::setprecision(1) << dG
                              << "kcal/mol,"
                              << seq.substr(i, stem_len) << ".." << seq.substr(j_start, stem_len);

                        seeds.push_back({
                            NucleationSeed::Type::RNA_HAIRPIN,
                            i,                       // start_pos (5' stem start)
                            j_end,                   // end_pos   (3' stem end)
                            boost,
                            dG,
                            motif.str()
                        });
                        any_loop_found = true;
                    }
                }
            }

            // Stop extending stem if no valid loop found at this length,
            // or if we'd exceed the sequence
            int min_j_end = i + stem_len + loop_min_nt + stem_len - 1;
            if (min_j_end >= n) break;

            // Also break if the stem itself is invalid for all loop sizes
            // (this avoids exponential cost — we already checked all loops above)
            (void)any_loop_found; // continue trying longer stems
        }
    }

    // De-duplicate: keep highest-boost seed for overlapping positions
    // (greedy: remove seeds fully contained in a higher-boost seed)
    std::sort(seeds.begin(), seeds.end(),
              [](const NucleationSeed& a, const NucleationSeed& b){
                  return a.folding_rate_boost > b.folding_rate_boost;
              });
    std::vector<NucleationSeed> deduped;
    std::vector<bool> used(seeds.size(), false);
    for (int i = 0; i < (int)seeds.size(); ++i) {
        if (used[i]) continue;
        deduped.push_back(seeds[i]);
        // Mark seeds fully overlapping the current one as used
        for (int j = i + 1; j < (int)seeds.size(); ++j) {
            if (!used[j] &&
                seeds[j].start_pos >= seeds[i].start_pos &&
                seeds[j].end_pos   <= seeds[i].end_pos)
                used[j] = true;
        }
    }
    return deduped;
}

// ─── G-quadruplex seed detection ─────────────────────────────────────────────
// Pattern: G{min_g_run,}[ACGU]{1,7} repeated 3 times, followed by G{min_g_run,}
// Strongly stabilised by Mg²⁺ and K⁺; high k_fold boost.
std::vector<NucleationSeed> NucleationSeedDetector::detect_rna_gquads(
    const std::string& seq,
    int min_g_run)
{
    std::vector<NucleationSeed> seeds;
    const int n = (int)seq.size();

    // Find all G-run start positions and lengths
    struct GRun { int start, len; };
    std::vector<GRun> gruns;
    int i = 0;
    while (i < n) {
        if (seq[i] == 'G' || seq[i] == 'g') {
            int start = i;
            while (i < n && (seq[i] == 'G' || seq[i] == 'g')) ++i;
            int len = i - start;
            if (len >= min_g_run) gruns.push_back({start, len});
        } else {
            ++i;
        }
    }

    // Need at least 4 G-runs for a G-quadruplex
    if ((int)gruns.size() < 4) return seeds;

    // Try all combinations of 4 G-runs (greedy left-to-right)
    for (int a = 0; a + 3 < (int)gruns.size(); ++a) {
        for (int b = a + 1; b + 2 < (int)gruns.size(); ++b) {
            int gap_ab = gruns[b].start - (gruns[a].start + gruns[a].len);
            if (gap_ab < 1 || gap_ab > 7) continue;

            for (int c = b + 1; c + 1 < (int)gruns.size(); ++c) {
                int gap_bc = gruns[c].start - (gruns[b].start + gruns[b].len);
                if (gap_bc < 1 || gap_bc > 7) continue;

                for (int d = c + 1; d < (int)gruns.size(); ++d) {
                    int gap_cd = gruns[d].start - (gruns[c].start + gruns[c].len);
                    if (gap_cd < 1 || gap_cd > 7) continue;

                    int tier = std::min({gruns[a].len, gruns[b].len,
                                         gruns[c].len, gruns[d].len});
                    // ΔG: each G-quartet stabilises by ~-3.8 kcal/mol (Sen 2002)
                    double dG = -3.8 * tier;

                    std::ostringstream motif;
                    motif << "G4:tiers=" << tier
                          << ",pos=" << gruns[a].start << "+" << gap_ab
                          << "+" << gruns[b].start << "+" << gap_bc
                          << "+" << gruns[c].start << "+" << gap_cd
                          << "+" << gruns[d].start;

                    seeds.push_back({
                        NucleationSeed::Type::RNA_GQUADRUPLEX,
                        gruns[a].start,
                        gruns[d].start + gruns[d].len - 1,
                        5.0,   // strong boost — G4 is highly cooperative
                        dG,
                        motif.str()
                    });
                    break; // take first valid d per (a,b,c)
                }
            }
        }
    }
    return seeds;
}

// ─── Chou-Fasman P_α values ───────────────────────────────────────────────────
// Source: Chou & Fasman 1978; updated by Levitt 1978 survey.
double NucleationSeedDetector::cf_palpha(char aa)
{
    switch ((char)std::toupper((unsigned char)aa)) {
        case 'A': return 1.45;
        case 'L': return 1.34;
        case 'M': return 1.30;
        case 'F': return 1.16;
        case 'W': return 1.08;
        case 'K': return 1.23;
        case 'Q': return 1.17;
        case 'E': return 1.53;
        case 'H': return 1.24;
        case 'I': return 1.00;
        case 'V': return 0.97;
        case 'R': return 0.97;
        case 'D': return 0.98;
        case 'T': return 0.77;
        case 'S': return 0.79;
        case 'C': return 0.77;
        case 'N': return 0.73;
        case 'Y': return 0.69;
        case 'P': return 0.57; // Pro breaks helices
        case 'G': return 0.53; // Gly too flexible
        default:  return 0.90; // unknown AA — conservative estimate
    }
}

// ─── Protein helix propensity seeds ─────────────────────────────────────────
std::vector<NucleationSeed> NucleationSeedDetector::detect_protein_helix(
    const std::string& seq,
    int window, double helix_thresh)
{
    std::vector<NucleationSeed> seeds;
    const int n = (int)seq.size();
    if (n < window) return seeds;

    for (int i = 0; i + window <= n; ++i) {
        double sum = 0.0;
        for (int j = i; j < i + window; ++j)
            sum += cf_palpha(seq[j]);
        double mean_pa = sum / window;

        if (mean_pa > helix_thresh) {
            double boost = 1.0 + (mean_pa - helix_thresh) * 5.0;
            boost = std::min(boost, 10.0);

            // Merge with previous seed if adjacent or overlapping
            if (!seeds.empty() &&
                seeds.back().type == NucleationSeed::Type::PROTEIN_HELIX &&
                i <= seeds.back().end_pos + 1)
            {
                // Extend: update end and take max boost
                seeds.back().end_pos = i + window - 1;
                seeds.back().folding_rate_boost =
                    std::max(seeds.back().folding_rate_boost, boost);
            } else {
                std::ostringstream motif;
                motif << "helix:P_alpha=" << std::fixed << std::setprecision(2)
                      << mean_pa << "," << seq.substr(i, window);
                seeds.push_back({
                    NucleationSeed::Type::PROTEIN_HELIX,
                    i, i + window - 1,
                    boost,
                    -0.5 * (mean_pa - helix_thresh) * window, // rough ΔG estimate
                    motif.str()
                });
            }
        }
    }
    return seeds;
}

// ─── Protein hydrophobic cluster seeds ───────────────────────────────────────
bool NucleationSeedDetector::is_core_hydrophobic(char aa)
{
    switch ((char)std::toupper((unsigned char)aa)) {
        case 'I': case 'L': case 'V': case 'F': case 'M': case 'W': return true;
        default: return false;
    }
}

std::vector<NucleationSeed> NucleationSeedDetector::detect_protein_hydrophobic(
    const std::string& seq, int min_run)
{
    std::vector<NucleationSeed> seeds;
    const int n = (int)seq.size();
    int i = 0;
    while (i < n) {
        if (is_core_hydrophobic(seq[i])) {
            int start = i;
            while (i < n && is_core_hydrophobic(seq[i])) ++i;
            int run_len = i - start;
            if (run_len >= min_run) {
                double boost = 2.0 + 0.3 * (run_len - min_run);
                boost = std::min(boost, 8.0);
                std::ostringstream motif;
                motif << "hydrophobic_cluster:len=" << run_len
                      << "," << seq.substr(start, run_len);
                seeds.push_back({
                    NucleationSeed::Type::PROTEIN_HYDROPHOBIC,
                    start, i - 1,
                    boost,
                    -0.4 * run_len, // rough hydrophobic collapse ΔG
                    motif.str()
                });
            }
        } else {
            ++i;
        }
    }
    return seeds;
}

// ─── position_boost_map ───────────────────────────────────────────────────────
std::vector<double> NucleationSeedDetector::position_boost_map(
    const std::vector<NucleationSeed>& seeds, int seq_len)
{
    std::vector<double> map(seq_len, 1.0);
    for (const auto& s : seeds) {
        int end = std::min(s.end_pos, seq_len - 1);
        for (int p = std::max(0, s.start_pos); p <= end; ++p)
            map[p] = std::max(map[p], s.folding_rate_boost);
    }
    return map;
}

} // namespace natural
