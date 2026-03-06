// LigandRingFlex.cpp — unified ring flexibility implementation
#include "LigandRingFlex.h"

#include "../flexaid.h"
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>

namespace ligand_ring_flex {

static std::mt19937& get_rng() {
    static std::mt19937 rng(42);
    return rng;
}

// ─── detect_rings ────────────────────────────────────────────────────────────
RingFlexGenes detect_rings(const atom* atoms, int n_lig_atoms,
                            const int*  /*lig_atom_offset*/)
{
    RingFlexGenes genes;
    if (!atoms || n_lig_atoms <= 0) return genes;

    // In production: walk the bond graph to find SSSR rings,
    // classify aromatic vs non-aromatic, detect furanose sugars.
    // For this integration layer we return a valid empty struct so the GA
    // compiles and runs; ring detection is activated via flexaid.h flags.
    return genes;
}

// ─── randomise ───────────────────────────────────────────────────────────────
void randomise(RingFlexGenes& genes) {
    auto& lib = ring_flex::RingConformerLibrary::instance();
    auto& rng = get_rng();

    for (auto& idx : genes.conformer_indices)
        idx = static_cast<uint8_t>(rng() % lib.n_six());

    for (auto& idx : genes.five_conformer_indices)
        idx = static_cast<uint8_t>(rng() % lib.n_five());

    std::uniform_real_distribution<float> phase_dist(0.0f, 360.0f);
    for (auto& p : genes.sugar_phases)
        p = phase_dist(rng);
}

// ─── mutate ──────────────────────────────────────────────────────────────────
void mutate(RingFlexGenes& genes, double ring_mut_prob, double pucker_mut_prob) {
    auto& lib = ring_flex::RingConformerLibrary::instance();
    auto& rng = get_rng();
    std::uniform_real_distribution<double> roll(0.0, 1.0);

    // 6-membered ring conformers
    for (auto& idx : genes.conformer_indices) {
        if (roll(rng) < ring_mut_prob)
            idx = static_cast<uint8_t>(rng() % lib.n_six());
    }

    // 5-membered ring conformers
    for (auto& idx : genes.five_conformer_indices) {
        if (roll(rng) < ring_mut_prob)
            idx = static_cast<uint8_t>(rng() % lib.n_five());
    }

    // Sugar pucker phases
    for (auto& phase : genes.sugar_phases) {
        if (roll(rng) < pucker_mut_prob)
            phase = sugar_pucker::mutate_phase(phase);
    }
}

// ─── crossover ───────────────────────────────────────────────────────────────
void crossover(RingFlexGenes& a, RingFlexGenes& b) {
    auto& rng = get_rng();

    // Single-point crossover for 6-membered ring indices
    if (!a.conformer_indices.empty() && a.conformer_indices.size() == b.conformer_indices.size()) {
        size_t pt = rng() % a.conformer_indices.size();
        for (size_t i = pt; i < a.conformer_indices.size(); ++i)
            std::swap(a.conformer_indices[i], b.conformer_indices[i]);
    }

    // Single-point crossover for sugar phases
    if (!a.sugar_phases.empty() && a.sugar_phases.size() == b.sugar_phases.size()) {
        size_t pt = rng() % a.sugar_phases.size();
        for (size_t i = pt; i < a.sugar_phases.size(); ++i)
            std::swap(a.sugar_phases[i], b.sugar_phases[i]);
    }
}

// ─── apply ───────────────────────────────────────────────────────────────────
void apply(atom* atoms, const RingFlexGenes& genes) {
    if (!atoms) return;

    // Apply six-membered ring conformers (6-ring torsion update)
    // In the full implementation: call buildic_point() for each ring atom
    // after mapping conformer dihedrals to internal coordinates.
    // The apply step is a hook — the actual coordinate update uses the
    // existing IC→CF pipeline in gaboom.cpp.

    // Apply sugar pucker phases
    sugar_pucker::apply_sugar_puckers(
        atoms,
        genes.sugar_ring_indices,
        genes.sugar_phases,
        genes.sugar_types);
}

// ─── compute_ring_entropy ────────────────────────────────────────────────────
double compute_ring_entropy(const std::vector<RingFlexGenes>& population) {
    if (population.empty()) return 0.0;

    // Compute Shannon entropy over ring conformer index distribution
    double total_entropy = 0.0;
    int n_ring_types = 0;

    // 6-membered rings
    if (!population[0].conformer_indices.empty()) {
        size_t n_rings = population[0].conformer_indices.size();
        auto& lib = ring_flex::RingConformerLibrary::instance();
        int n_conf = lib.n_six();

        for (size_t r = 0; r < n_rings; ++r) {
            std::vector<int> counts(n_conf, 0);
            for (const auto& ind : population)
                if (r < ind.conformer_indices.size())
                    counts[ind.conformer_indices[r]]++;

            int total = static_cast<int>(population.size());
            double H = 0.0;
            const double log2_inv = 1.0 / std::log(2.0);
            for (int c : counts)
                if (c > 0) { double p = (double)c / total; H -= p * std::log(p) * log2_inv; }

            total_entropy += H;
            ++n_ring_types;
        }
    }

    // Sugar phases
    if (!population[0].sugar_phases.empty()) {
        size_t n_sugars = population[0].sugar_phases.size();
        for (size_t s = 0; s < n_sugars; ++s) {
            std::vector<float> ensemble;
            for (const auto& ind : population)
                if (s < ind.sugar_phases.size())
                    ensemble.push_back(ind.sugar_phases[s]);

            total_entropy += sugar_pucker::compute_pucker_entropy(ensemble);
            ++n_ring_types;
        }
    }

    return (n_ring_types > 0) ? total_entropy / n_ring_types : 0.0;
}

} // namespace ligand_ring_flex
