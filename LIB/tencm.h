// tencm.h — Torsional Elastic Network Contact Model (TENCM) for backbone flexibility
//
// Implements the torsional ENM of Delarue & Sanejouand (J. Mol. Biol. 2002) and
// Yang, Song & Cui (Biophys J. 2009):
//   – Spring network over Cα contacts within r_cutoff
//   – Torsional DOFs: one pseudo-torsion per Cα–Cα bond (i → i+1)
//   – Hessian H_kl = Σ_{contacts(i,j)} k_ij (ΔJ_ki · ΔJ_li)
//   – Normal modes via symmetric Jacobi diagonalisation
//   – Mode-weighted Boltzmann sampling for backbone perturbation during GA
//
// Used by FlexAIDdS to generate protein backbone flexibility without rebuilding
// the full rotamer library every GA generation.
#pragma once

#include <vector>
#include <array>
#include <span>
#include <concepts>
#include <cmath>
#include <numbers>
#include <memory>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cstring>

#include "flexaid.h"

namespace tencm {

// ─── constants ───────────────────────────────────────────────────────────────
inline constexpr float kB_kcal    = 0.001987206f; // kcal mol⁻¹ K⁻¹
inline constexpr float DEFAULT_RC = 9.0f;          // Å contact cutoff
inline constexpr float DEFAULT_K0 = 1.0f;          // spring constant (kcal mol⁻¹ Å⁻²)
inline constexpr int   N_MODES    = 20;            // low-frequency modes kept

// ─── data structures ─────────────────────────────────────────────────────────

// Elastic contact between Cα i and Cα j (i < j)
struct Contact {
    int   i, j;
    float k;     // spring constant k_ij = k0 * (rc/r0)^6
    float r0;    // equilibrium Cα–Cα distance
};

// Pseudo-torsion DOF: rotation about bond between Cα_k and Cα_{k+1}
struct PseudoBond {
    int   k;          // 0-based index: bond connects residue k to k+1
    float axis[3];    // unit vector (r_{k+1} - r_k) / |...|
    float pivot[3];   // midpoint (r_k + r_{k+1}) / 2
};

// One normal mode: eigenvalue (stiffness) + eigenvector over torsion DOFs
struct NormalMode {
    double eigenvalue;                // kcal mol⁻¹ rad⁻²
    std::vector<double> eigenvector; // length = n_bonds
};

// A perturbed backbone conformation
struct Conformer {
    std::vector<float> delta_theta;           // torsion perturbations (rad)
    std::vector<std::array<float,3>> ca;      // perturbed Cα positions
    float strain_energy;                      // ½ δθᵀ H δθ (kcal/mol)
};

// ─── concepts ────────────────────────────────────────────────────────────────
template<typename T>
concept FloatLike = std::floating_point<T>;

// ─── main engine ─────────────────────────────────────────────────────────────
class TorsionalENM {
public:
    // Build network from parsed atom/residue arrays.
    // Selects Cα atoms from protein residues.
    void build(const atom*  atoms,
               const resid* residue,
               int          res_cnt,
               float        cutoff = DEFAULT_RC,
               float        k0     = DEFAULT_K0);

    // Sample one perturbed backbone conformation.
    // temperature: Kelvin; rng: seeded generator passed in for reproducibility.
    Conformer sample(float temperature, std::mt19937& rng) const;

    // Apply a Conformer back to the FA atoms array (in-place coordinate update).
    void apply(const Conformer& conf,
               atom*            atoms,
               const resid*     residue) const;

    // Predicted Cα B-factors at given T (Å²)
    std::vector<float> bfactors(float temperature) const;

    // Build directly from Cα coordinates (no FA atom/resid dependency).
    // ca_coords: sequential Cα positions (x,y,z) along the chain.
    void build_from_ca(const std::vector<std::array<float,3>>& ca_coords,
                       float cutoff = DEFAULT_RC,
                       float k0     = DEFAULT_K0);

    // Getters
    int n_residues() const noexcept { return static_cast<int>(ca_.size()); }
    int n_bonds()    const noexcept { return static_cast<int>(bonds_.size()); }
    const std::vector<NormalMode>& modes() const noexcept { return modes_; }
    bool is_built()  const noexcept { return built_; }
    const std::vector<std::array<float,3>>& ca_positions() const noexcept { return ca_; }

private:
    // Internal Cα coordinate store (row-major, index = sequential residue idx)
    std::vector<std::array<float,3>> ca_;
    // Map: sequential residue index → first atom index of Cα in FA atoms[]
    std::vector<int> ca_atom_idx_;
    // For each Cα, the corresponding res_cnt index (1-based FA convention)
    std::vector<int> res_idx_;

    std::vector<Contact>    contacts_;
    std::vector<PseudoBond> bonds_;
    std::vector<NormalMode> modes_;

    // Hessian stored as dense symmetric matrix (n_bonds × n_bonds)
    std::vector<double> H_;

    bool  built_  = false;
    float cutoff_ = DEFAULT_RC;
    float k0_     = DEFAULT_K0;

    // Build steps
    void extract_ca(const atom* atoms, const resid* residue, int res_cnt);
    void build_contacts();
    void build_bonds();
    void assemble_hessian();
    void diagonalize();     // Jacobi iteration on H_

    // Jacobian: ∂r_{atom_i}/∂θ_{bond_k}.
    // Returns {0,0,0} when atom_i is upstream of bond_k.
    std::array<float,3> jac(int bond_k, int atom_i) const noexcept;

    // Jacobi sweep helper
    static void jacobi_rotate(std::vector<double>& A,
                               std::vector<double>& V,
                               int n, int p, int q) noexcept;
};

// ─── free function: quick fluctuation amplitude at given T ───────────────────
// Returns the rms displacement (Å) of residue i predicted by the model.
float residue_rms_fluctuation(const TorsionalENM& tencm,
                              int residue_idx,
                              float temperature);

}  // namespace tencm
