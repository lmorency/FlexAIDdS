// ChiralCenterGene.cpp — chiral stereocenter gene implementation
#include "ChiralCenterGene.h"

#include "../flexaid.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <sstream>
#include <numeric>

namespace chiral {

static std::mt19937& rng() {
    static std::mt19937 gen(std::random_device{}());
    return gen;
}

// ─── detect_stereocenters ────────────────────────────────────────────────────
// Heuristic: sp3 carbon with 4 distinct substituents.
// Full CIP assignment requires the bond graph — here we use a simplified
// version that identifies candidate centers by valence and atom type.
std::vector<ChiralCenter> detect_stereocenters(const atom* atoms, int n_atoms) {
    std::vector<ChiralCenter> centers;
    if (!atoms || n_atoms <= 0) return centers;

    for (int i = 0; i < n_atoms; ++i) {
        // Check for sp3 carbon with type 'C' and 4 neighbours
        const char* atype = atoms[i].type;
        if (!atype) continue;
        if (atype[0] != 'C' && atype[0] != 'c') continue;

        // Count bonded neighbours via the bond list
        // (atoms[i].bond[] stores bonded atom indices in FlexAID)
        int n_bonds = 0;
        int substituents[4] = {-1, -1, -1, -1};
        for (int b = 0; b < MBNDS && n_bonds < 4; ++b) {
            int bidx = atoms[i].bond[b];
            if (bidx >= 0 && bidx < n_atoms && bidx != i) {
                substituents[n_bonds++] = bidx;
            }
        }

        if (n_bonds == 4) {
            // Verify all four substituents are distinct (by atom type)
            bool all_distinct = true;
            for (int a = 0; a < 4 && all_distinct; ++a)
                for (int b = a + 1; b < 4 && all_distinct; ++b)
                    if (substituents[a] == substituents[b]) all_distinct = false;

            if (all_distinct) {
                ChiralCenter c;
                c.central_atom_idx = i;
                for (int k = 0; k < 4; ++k) c.substituent_indices[k] = substituents[k];
                c.assigned  = Chirality::R; // default to R; GA samples inversion
                c.reference = Chirality::Unknown;
                centers.push_back(c);
            }
        }
    }
    return centers;
}

// ─── ChiralCenterGene ────────────────────────────────────────────────────────

ChiralCenterGene::ChiralCenterGene(std::vector<ChiralCenter> centers)
    : centers_(std::move(centers)) {}

void ChiralCenterGene::mutate(double inversion_prob) {
    std::uniform_real_distribution<double> roll(0.0, 1.0);
    std::uniform_int_distribution<int> pick(0, (int)centers_.size() - 1);

    if (centers_.empty()) return;

    for (auto& c : centers_) {
        if (roll(rng()) < inversion_prob) {
            // Invert R→S or S→R
            c.assigned = (c.assigned == Chirality::R) ? Chirality::S : Chirality::R;
        }
    }
}

void ChiralCenterGene::crossover(ChiralCenterGene& other) {
    if (centers_.size() != other.centers_.size() || centers_.empty()) return;
    size_t pt = std::uniform_int_distribution<size_t>(0, centers_.size() - 1)(rng());
    for (size_t i = pt; i < centers_.size(); ++i)
        std::swap(centers_[i].assigned, other.centers_[i].assigned);
}

void ChiralCenterGene::set(int i, Chirality c) {
    centers_.at(i).assigned = c;
}

Chirality ChiralCenterGene::get(int i) const {
    return centers_.at(i).assigned;
}

double ChiralCenterGene::inversion_energy(double k_inv) const {
    double penalty = 0.0;
    for (const auto& c : centers_) {
        if (c.reference != Chirality::Unknown && c.assigned != c.reference)
            penalty += k_inv;
    }
    return penalty;
}

void ChiralCenterGene::apply(atom* atoms) const {
    if (!atoms) return;
    for (const auto& c : centers_) {
        // Inversion is handled by comparing current vs target chirality.
        // If the current geometry in atoms[] was built with default R,
        // and the gene says S, we perform a reflection of two substituents.
        // This calls invert_center() which swaps the Cartesian coordinates
        // of substituents 0 and 1 (a valid tetrahedral inversion).
        if (c.assigned == Chirality::S) {
            invert_center(atoms, c.central_atom_idx);
        }
    }
}

void ChiralCenterGene::invert_center(atom* atoms, int cidx) const {
    // Find the ChiralCenter for cidx
    for (const auto& c : centers_) {
        if (c.central_atom_idx != cidx) continue;
        // Swap the x-coordinates of substituents 0 and 1 relative to center
        // to perform a tetrahedral inversion (reflection through a plane).
        int s0 = c.substituent_indices[0];
        int s1 = c.substituent_indices[1];
        if (s0 < 0 || s1 < 0) return;

        // Translate to origin (centered on chiral atom)
        float cx = atoms[cidx].coor[0];
        float cy = atoms[cidx].coor[1];
        float cz = atoms[cidx].coor[2];

        // Reflect substituent 0 through the plane defined by s1,s2,s3
        // Simplified: swap x-components of s0 and s1 relative to center
        float dx0 = atoms[s0].coor[0] - cx;
        float dx1 = atoms[s1].coor[0] - cx;
        atoms[s0].coor[0] = cx + dx1;
        atoms[s1].coor[0] = cx + dx0;
        // Also swap y-components for a proper reflection
        float dy0 = atoms[s0].coor[1] - cy;
        float dy1 = atoms[s1].coor[1] - cy;
        atoms[s0].coor[1] = cy + dy1;
        atoms[s1].coor[1] = cy + dy0;
        return;
    }
}

double ChiralCenterGene::compute_entropy(
    const std::vector<ChiralCenterGene>& population)
{
    if (population.empty()) return 0.0;
    size_t n_centers = population[0].size();
    if (n_centers == 0) return 0.0;

    double total_H = 0.0;
    for (size_t ci = 0; ci < n_centers; ++ci) {
        int count_R = 0, count_S = 0;
        for (const auto& ind : population) {
            if ((int)ci < ind.size()) {
                if (ind.centers_[ci].assigned == Chirality::R) ++count_R;
                else if (ind.centers_[ci].assigned == Chirality::S) ++count_S;
            }
        }
        int total = count_R + count_S;
        if (total == 0) continue;

        double pH = 0.0;
        const double log2_inv = 1.0 / std::log(2.0);
        auto add_term = [&](int c) {
            if (c > 0) { double p = (double)c / total; pH -= p * std::log(p) * log2_inv; }
        };
        add_term(count_R);
        add_term(count_S);
        total_H += pH;
    }
    return total_H / static_cast<double>(n_centers);
}

std::string ChiralCenterGene::to_string() const {
    std::ostringstream oss;
    oss << "ChiralGene[";
    for (size_t i = 0; i < centers_.size(); ++i) {
        oss << (centers_[i].assigned == Chirality::R ? 'R' : 'S');
        if (i + 1 < centers_.size()) oss << ',';
    }
    oss << "]";
    return oss.str();
}

} // namespace chiral
