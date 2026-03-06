// NATURaLDualAssembly.cpp — co-translational / co-transcriptional folding
//
// The same Zhao 2011 master-equation rationale applies to both polypeptide
// and nucleotide chains:
//
//   Polypeptide (ribosome):
//     dP_n/dt = k_{n-1} · P_{n-1} − k_n · P_n
//     k_n  = codon-specific aa elongation rate (s⁻¹)  [Dong 1996 / Ingolia 2011]
//     Tunnel: 34 aa occluded (Goldman 2010 Cell 143:92)
//
//   Nucleotide chain (RNA polymerase):
//     Same master equation, k_n = nucleotide-specific NTP incorporation rate
//     k_n  from Uptain 1997 (E. coli RNAP) / Jonkers 2014 (Human RNAP II)
//     Tunnel: 8 nt (RNAP RNA:DNA hybrid; Nudler 2012 Cell 149:1438)
//
// Both modes are handled by the same RibosomeElongation class (same interface,
// different CodonRateTable) — "same rational for nucleotides than polypeptides".
//
// TransloconInsertion (TM helix lateral gating via Sec61 translocon) is also
// evaluated at each growth step for polypeptide chains, using the Hessa 2007
// empirical ΔG scale.
//
// Hardware acceleration in this file:
//   – AVX-512 / AVX2 / Eigen / OpenMP propagated via TransloconInsertion
//   – RibosomeElongation uses Eigen VectorXd for the tridiagonal ODE
//   – ShannonThermoStack / compute_growth_entropy uses AVX-512 histogramming
#include "NATURaLDualAssembly.h"
#include "RibosomeElongation.h"
#include "TransloconInsertion.h"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <sstream>
#include <vector>

#ifdef FLEXAIDS_HAS_EIGEN
#  include <Eigen/Dense>
#endif

namespace natural {

// ─── nucleic acid residue names ───────────────────────────────────────────────
static const char* NUCLEIC_ACID_NAMES[] = {
    // RNA
    "A", "G", "C", "U", "ADE", "GUA", "CYT", "URA",
    // DNA
    "DA", "DG", "DC", "DT", "DA3", "DG3", "DC3", "DT3",
    "DA5", "DG5", "DC5", "DT5",
    nullptr
};

// ─── is_nucleotide_ligand ────────────────────────────────────────────────────
bool is_nucleotide_ligand(const atom* atoms, int n_lig_atoms) {
    if (!atoms || n_lig_atoms <= 0) return false;
    for (int i = 0; i < n_lig_atoms; ++i) {
        const char* name = atoms[i].name;
        if (!name) continue;
        // O2' or O2* indicates ribose (RNA/nucleoside)
        if (strstr(name, "O2'") || strstr(name, "O2*") || strstr(name, "O2P"))
            return true;
        // Common nucleotide atom names
        if (strstr(name, "N9") || strstr(name, "N1") || strstr(name, "C5'"))
            return true;
    }
    return false;
}

// ─── is_nucleic_acid_receptor ────────────────────────────────────────────────
bool is_nucleic_acid_receptor(const resid* residues, int n_residues) {
    if (!residues || n_residues <= 0) return false;
    for (int i = 0; i < n_residues; ++i) {
        const char* name = residues[i].name;
        if (!name) continue;
        for (int k = 0; NUCLEIC_ACID_NAMES[k]; ++k) {
            if (strncmp(name, NUCLEIC_ACID_NAMES[k],
                        strlen(NUCLEIC_ACID_NAMES[k])) == 0)
                return true;
        }
    }
    return false;
}

// ─── auto_configure ──────────────────────────────────────────────────────────
NATURaLConfig auto_configure(const atom*  atoms,
                              int          n_lig_atoms,
                              const resid* residues,
                              int          n_residues)
{
    NATURaLConfig cfg;

    bool nucl_lig = is_nucleotide_ligand(atoms, n_lig_atoms);
    bool nucl_rec = is_nucleic_acid_receptor(residues, n_residues);

    if (nucl_lig || nucl_rec) {
        cfg.enabled                = true;
        cfg.co_translational_growth = true;
        cfg.sugar_pucker_auto       = nucl_lig;
        // Nucleotide context: use RNAP rates (same master equation, different table)
        if (nucl_rec && !nucl_lig) {
            // Pure nucleic acid receptor — co-transcriptional assembly
            cfg.use_ribosome_speed = false;  // will use RNAP rates
        }

        std::cout << "[NATURaL] Auto-detected "
                  << (nucl_lig ? "nucleotide ligand" : "")
                  << (nucl_lig && nucl_rec ? " + " : "")
                  << (nucl_rec ? "nucleic acid receptor" : "")
                  << " → enabling co-"
                  << (cfg.use_ribosome_speed ? "translational" : "transcriptional")
                  << " DualAssembly ["
                  << (cfg.organism == ribosome::Organism::EcoliK12 ? "E.coli" : "Human")
                  << "]\n";
    }
    return cfg;
}

// ─── build elongation model from config ──────────────────────────────────────
// Returns a RibosomeElongation built from the receptor sequence using either
// ribosome (polypeptide) or RNAP (nucleotide) rates depending on cfg.
// The same Zhao 2011 master equation is used in both cases.
static ribosome::RibosomeElongation build_elongation_model(
    const NATURaLConfig& cfg,
    const resid*         residues,
    int                  n_residues)
{
    using namespace ribosome;

    // Build appropriate rate table
    CodonRateTable table =
        cfg.use_ribosome_speed
            ? (cfg.organism == Organism::EcoliK12
                   ? CodonRateTable::build_ecoli()
                   : CodonRateTable::build_human())
            : (cfg.organism == Organism::EcoliK12
                   ? CodonRateTable::build_rnap_ecoli()
                   : CodonRateTable::build_rnap_human());

    // Extract 1-letter sequence and codon identifiers from residue array.
    // For nucleotide chains, residue names map to single-nucleotide "codons".
    std::string seq;
    std::vector<std::string> codons;
    seq.reserve(n_residues);
    codons.reserve(n_residues);

    for (int i = 0; i < n_residues; ++i) {
        const char* rname = residues[i].name;
        if (!rname) { seq += 'X'; codons.emplace_back(); continue; }

        if (!cfg.use_ribosome_speed) {
            // Nucleotide mode: 1-letter nucleotide code for RNAP rate lookup
            char nt = 'N';
            if      (rname[0] == 'A' || strncmp(rname, "ADE", 3) == 0) nt = 'A';
            else if (rname[0] == 'G' || strncmp(rname, "GUA", 3) == 0) nt = 'G';
            else if (rname[0] == 'C' || strncmp(rname, "CYT", 3) == 0) nt = 'C';
            else if (rname[0] == 'U' || strncmp(rname, "URA", 3) == 0) nt = 'U';
            else if (rname[0] == 'T' || rname[1] == 'T')               nt = 'T';
            seq  += nt;
            codons.emplace_back(1, nt);  // single-character codon
        } else {
            // Polypeptide mode: use 1-letter AA code from residue name
            // (fallback to 'A' for unknown residues)
            seq  += residues[i].name[0];
            codons.emplace_back(); // mean rate used when codon is empty
        }
    }

    double k_ini = cfg.use_ribosome_speed ? K_INI_DEFAULT  : K_RNAP_INI_DEFAULT;
    double k_ter = cfg.use_ribosome_speed ? K_TERM_DEFAULT  : K_RNAP_TERM_DEFAULT;

    return RibosomeElongation(seq, codons, table, k_ini, k_ter);
}

// ─── DualAssemblyEngine ──────────────────────────────────────────────────────

DualAssemblyEngine::DualAssemblyEngine(const NATURaLConfig& cfg,
                                         FA_Global* FA, VC_Global* VC,
                                         atom* atoms, resid* residues,
                                         int n_residues)
    : config_(cfg), FA_(FA), VC_(VC),
      atoms_(atoms), residues_(residues), n_residues_(n_residues)
{}

std::vector<DualAssemblyEngine::GrowthStep> DualAssemblyEngine::run() {
    std::vector<GrowthStep> trajectory;

    if (!config_.enabled || !config_.co_translational_growth) {
        return trajectory;
    }

    int max_steps = (config_.max_growth_steps > 0)
                    ? config_.max_growth_steps
                    : n_residues_;
    trajectory.reserve(max_steps);

    // ── Build elongation model (ribosome or RNAP, same master equation) ──────
    auto elong_model = build_elongation_model(config_, residues_, n_residues_);
    const auto& k_el = elong_model.elongation_rates();

    // Pre-compute analytical arrival times for each residue
    // <T_n> = 1/k_ini + Σ_{i<n} 1/k_i  (Zhao 2011 Eq. 7)
    std::vector<double> t_arrival(max_steps + 1);
    t_arrival[0] = 1.0 / elong_model.k_ini();
    for (int n = 1; n <= max_steps; ++n) {
        double k = (n - 1 < (int)k_el.size()) ? k_el[n - 1]
                                                : elong_model.elongation_rates().back();
        t_arrival[n] = t_arrival[n - 1] + 1.0 / k;
    }

    // Tunnel length depends on mode
    const double tunnel_len = config_.use_ribosome_speed
                              ? ribosome::TUNNEL_LENGTH_AA
                              : ribosome::RNAP_TUNNEL_NT;

    // Pause threshold
    const double pause_threshold = config_.use_ribosome_speed ? 0.30 : ribosome::RNAP_PAUSE_THRESHOLD;

    // Harmonic mean rate for pause detection
    double hmean_rate = elong_model.elongation_rates().empty()
                        ? 16.5
                        : [&](){
                            double inv_sum = 0; int cnt = 0;
                            for (double k : k_el) { if (k > 1e-9) { inv_sum += 1.0/k; ++cnt; } }
                            return cnt > 0 ? cnt / inv_sum : 16.5;
                          }();

    // ── TransloconInsertion (polypeptide chains only) ─────────────────────────
    std::unique_ptr<translocon::TransloconInsertion> tm_model;
    std::string aa_seq_for_tm;
    if (config_.model_tm_insertion && config_.use_ribosome_speed) {
        tm_model = std::make_unique<translocon::TransloconInsertion>(
            config_.temperature_K, 0.5,
            static_cast<int>(ribosome::TUNNEL_LENGTH_AA));
        // Build 1-letter sequence for TransloconInsertion window scanning
        aa_seq_for_tm.reserve(n_residues_);
        for (int r = 0; r < n_residues_; ++r) {
            const char* rname = residues_[r].name;
            aa_seq_for_tm += (rname && rname[0]) ? rname[0] : 'X';
        }
    }

    // ── Folding windows from master equation ─────────────────────────────────
    auto fw = elong_model.folding_windows(ribosome::K_FOLD_DEFAULT);
    // Build a quick lookup: residue_idx → folding window info
    // (fw is already sorted by residue index; use binary search or map)

    // ── Main growth loop ──────────────────────────────────────────────────────
    std::vector<double> cf_trajectory;
    cf_trajectory.reserve(max_steps);
    double cumulative_dG = 0.0;

    for (int step = 0; step < max_steps; ++step) {
        // Elongation kinetics at this residue
        double k_n = (step < (int)k_el.size()) ? k_el[step] : hmean_rate;
        double dwell_time  = 1.0 / k_n;
        double t_arr       = (step < (int)t_arrival.size()) ? t_arrival[step] : 0.0;
        bool   in_tunnel   = (step < static_cast<int>(tunnel_len));
        bool   is_pause    = (!in_tunnel) && (k_n < pause_threshold * hmean_rate);

        // Co-translational folding probability: P_fold = k_fold/(k_fold + k_el)
        double k_fold_here = ribosome::K_FOLD_DEFAULT;
        if (is_pause) k_fold_here *= 3.0;   // Pechmann 2013 pause-site boost
        double p_cotrans = k_fold_here / (k_fold_here + k_n);

        // CF score for partial complex
        double cf = compute_partial_cf(step + 1);
        cf_trajectory.push_back(cf);

        // Shannon entropy over the growing CF ensemble
        double S_growth = compute_growth_entropy(cf_trajectory);

        // Incremental ΔG: ΔH from CF + (−T·ΔS) entropy term, time-weighted
        const double kT = 0.001987206 * config_.temperature_K;
        double delta_dG = cf - kT * S_growth;
        // Weight by dwell time: intermediates sampled longer contribute more
        delta_dG *= dwell_time;
        cumulative_dG += delta_dG;

        // ── TM translocon insertion check ─────────────────────────────────────
        bool   tm_inserted   = false;
        double tm_insert_dG  = 0.0;
        if (tm_model && !in_tunnel && step >= translocon::TM_WINDOW_LEN - 1) {
            // Check if the most recently emerged window is a TM segment
            int win_start = std::max(0, step - translocon::TM_WINDOW_LEN + 1);
            auto win = tm_model->check_window(aa_seq_for_tm, win_start);
            tm_inserted  = win.is_inserted;
            tm_insert_dG = win.deltaG_insert;
            if (tm_inserted) {
                // Include translocon ΔG contribution (time-weighted)
                cumulative_dG += tm_insert_dG * dwell_time;
            }
        }

        trajectory.push_back({
            step,
            t_arr,
            k_n,
            dwell_time,
            is_pause,
            in_tunnel,
            cf,
            S_growth,
            p_cotrans,
            cumulative_dG,
            tm_inserted,
            tm_insert_dG
        });
    }

    final_deltaG_ = cumulative_dG;

    // Optional: validate master equation against analytic solution
    if (n_residues_ >= 20) {
        auto vr = ribosome::validate_master_equation(
            std::min(n_residues_, 60),
            config_.organism);
        if (!vr.passed) {
            std::cerr << "[NATURaL] WARNING: " << vr.message << "\n";
        }
    }

    return trajectory;
}

// ─── compute_partial_cf ───────────────────────────────────────────────────────
double DualAssemblyEngine::compute_partial_cf(int n_grown_residues) const {
    // Simplified CF estimate: each grown residue contributes an attractive
    // contact energy. Production code calls the full cffunction()/vcfunction()
    // pipeline (ic2cf.cpp) with the current partially-grown complex.
    if (!FA_ || !atoms_ || !residues_) return 0.0;

    // Scale CF by accessible surface area proxy:
    // more residues → more surface buried → stronger (negative) CF.
    // Using -0.1 kcal/mol/residue as a placeholder; production uses the
    // actual FlexAID Contact Function grid evaluation.
    double count = std::min(n_grown_residues, n_residues_);
    return -0.1 * count; // kcal/mol (attractive)
}

// ─── compute_growth_entropy ───────────────────────────────────────────────────
// Shannon entropy of the CF distribution accumulated along the growth trajectory.
// Dispatches to AVX-512 → AVX2 → Eigen → scalar for the histogram fill.
double DualAssemblyEngine::compute_growth_entropy(
    const std::vector<double>& cf_trajectory) const
{
    if (cf_trajectory.empty()) return 0.0;

    double min_cf = *std::min_element(cf_trajectory.begin(), cf_trajectory.end());
    double max_cf = *std::max_element(cf_trajectory.begin(), cf_trajectory.end());
    if (max_cf - min_cf < 1e-8) return 0.0;

    constexpr int BINS = 32;
    double bw  = (max_cf - min_cf) / BINS + 1e-10;
    double inv_bw = 1.0 / bw;

    std::array<int, BINS> counts{};
    int total = static_cast<int>(cf_trajectory.size());

#if defined(FLEXAIDS_USE_AVX512) && defined(__AVX512F__)
    // AVX-512: process 8 doubles per cycle
    {
        __m512d v_min  = _mm512_set1_pd(min_cf);
        __m512d v_ibw  = _mm512_set1_pd(inv_bw);
        __m512d v_bmax = _mm512_set1_pd(static_cast<double>(BINS - 1));
        int i = 0, n = total;
        for (; i + 8 <= n; i += 8) {
            __m512d v    = _mm512_loadu_pd(cf_trajectory.data() + i);
            __m512d off  = _mm512_sub_pd(v, v_min);
            __m512d bins = _mm512_mul_pd(off, v_ibw);
            bins = _mm512_max_pd(_mm512_setzero_pd(),
                                 _mm512_min_pd(bins, v_bmax));
            __m256i bi = _mm512_cvttpd_epi32(bins);
            alignas(32) int bvals[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(bvals), bi);
            for (int j = 0; j < 8; ++j) counts[bvals[j]]++;
        }
        for (; i < n; ++i) {
            int b = static_cast<int>((cf_trajectory[i] - min_cf) * inv_bw);
            b = std::clamp(b, 0, BINS - 1);
            counts[b]++;
        }
    }
#elif defined(__AVX2__)
    // AVX2: process 4 doubles per cycle
    {
        __m256d v_min  = _mm256_set1_pd(min_cf);
        __m256d v_ibw  = _mm256_set1_pd(inv_bw);
        int i = 0, n = total;
        for (; i + 4 <= n; i += 4) {
            __m256d v   = _mm256_loadu_pd(cf_trajectory.data() + i);
            __m256d off = _mm256_sub_pd(v, v_min);
            __m256d bv  = _mm256_mul_pd(off, v_ibw);
            alignas(32) double bvals[4];
            _mm256_store_pd(bvals, bv);
            for (int j = 0; j < 4; ++j) {
                int b = static_cast<int>(bvals[j]);
                counts[std::clamp(b, 0, BINS - 1)]++;
            }
        }
        for (; i < n; ++i) {
            int b = static_cast<int>((cf_trajectory[i] - min_cf) * inv_bw);
            counts[std::clamp(b, 0, BINS - 1)]++;
        }
    }
#elif defined(FLEXAIDS_HAS_EIGEN)
    // Eigen: vectorised offset and scale; scalar bin accumulation
    {
        Eigen::Map<const Eigen::ArrayXd> vals(cf_trajectory.data(), total);
        Eigen::ArrayXd bins = ((vals - min_cf) * inv_bw).floor().cwiseMax(0).cwiseMin(BINS - 1);
        for (int i = 0; i < total; ++i)
            counts[static_cast<int>(bins(i))]++;
    }
#else
    for (int i = 0; i < total; ++i) {
        int b = static_cast<int>((cf_trajectory[i] - min_cf) * inv_bw);
        counts[std::clamp(b, 0, BINS - 1)]++;
    }
#endif

    // Shannon entropy: H = -Σ p_i log2(p_i)
    double H = 0.0;
    const double log2_inv = 1.0 / std::log(2.0);

#ifdef FLEXAIDS_HAS_EIGEN
    Eigen::ArrayXd prob(BINS);
    for (int i = 0; i < BINS; ++i) prob(i) = (double)counts[i] / total;
    Eigen::ArrayXd safe_p   = (prob > 1e-15).select(prob, Eigen::ArrayXd::Ones(BINS));
    Eigen::ArrayXd log_p    = (prob > 1e-15).select(safe_p.log(), Eigen::ArrayXd::Zero(BINS));
    H = -(prob * log_p).sum() * log2_inv;
#else
    for (int c : counts) {
        if (c > 0) {
            double p = (double)c / total;
            H -= p * std::log(p) * log2_inv;
        }
    }
#endif

    return H;
}

} // namespace natural
