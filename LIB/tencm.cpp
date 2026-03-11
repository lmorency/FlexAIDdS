// tencm.cpp — Torsional Elastic Network Contact Model implementation
//
// Mathematical basis (Delarue & Sanejouand 2002; Yang, Song & Cui 2009):
//
//  Spring potential over Cα contacts within r_cutoff:
//    V = ½ Σ_{(i,j)∈contacts} k_ij |δr_i - δr_j|²
//
//  Torsion-space Jacobian (pseudo-bond k connects Cα_k → Cα_{k+1}):
//    J_k(i) = e_k × (r_i - p_k)    if i > k  (downstream)
//             {0,0,0}               if i ≤ k  (upstream)
//    where e_k = unit bond axis, p_k = bond midpoint
//
//  Torsional Hessian:
//    H_kl = Σ_{(i,j)∈contacts} k_ij [(J_k(i)-J_k(j)) · (J_l(i)-J_l(j))]
//
//  Normal modes: symmetric Jacobi diagonalisation  H V = V Λ
//
//  Boltzmann sampling at temperature T:
//    σ_m² = kB T / λ_m  (equipartition per mode, skip m=0..5 ≈ rigid-body)
//    δθ   = Σ_{m≥6} σ_m * z_m * v_m,   z_m ~ N(0,1)
//
//  Perturbed Cα: r_i' = r_i + Σ_k J_k(i) δθ_k
//
// Hardware acceleration:
//   - OpenMP: parallel contact search, B-factor computation, Cα update
//   - Eigen:  Hessian eigendecomposition, strain energy GEMV

#include "tencm.h"
#include "simd_distance.h"

#include <cstring>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <stdexcept>
#include <numbers>
#include <random>

#ifdef _OPENMP
#  include <omp.h>
#endif

#ifdef FLEXAIDS_HAS_EIGEN
#  include <Eigen/Dense>
#endif

namespace tencm {

// ─── local utility helpers ───────────────────────────────────────────────────

static void cross3f(const float* a, const float* b, float* c) noexcept {
    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];
}
// ─── TorsionalENM::build ─────────────────────────────────────────────────────
void TorsionalENM::build(const atom*  atoms,
                          const resid* residue,
                          int          res_cnt,
                          float        cutoff,
                          float        k0)
{
    cutoff_ = cutoff;
    k0_     = k0;
    built_  = false;

    extract_ca(atoms, residue, res_cnt);

    if (static_cast<int>(ca_.size()) < 3) {
        std::cerr << "TENCM: fewer than 3 Cα atoms found; skipping.\n";
        return;
    }

    build_contacts();
    build_bonds();
    assemble_hessian();
    diagonalize();

    built_ = true;
}

// ─── extract_ca ──────────────────────────────────────────────────────────────
void TorsionalENM::extract_ca(const atom*  atoms,
                               const resid* residue,
                               int          res_cnt)
{
    ca_.clear();
    ca_atom_idx_.clear();
    res_idx_.clear();

    for (int ri = 1; ri <= res_cnt; ++ri) {
        if (residue[ri].type != 0) continue;   // protein residues only

        int first = residue[ri].fatm[0];
        int last  = residue[ri].latm[0];
        for (int ai = first; ai <= last; ++ai) {
            const char* nm = atoms[ai].name;
            // Match " CA " or "CA  " PDB atom name
            bool is_ca = (nm[0]==' '&&nm[1]=='C'&&nm[2]=='A'&&nm[3]==' ') ||
                         (nm[0]=='C'&&nm[1]=='A'&&nm[2]==' '&&nm[3]==' ');
            if (is_ca) {
                ca_.push_back({ atoms[ai].coor[0],
                                atoms[ai].coor[1],
                                atoms[ai].coor[2] });
                ca_atom_idx_.push_back(ai);
                res_idx_.push_back(ri);
                break;
            }
        }
    }
}

// ─── build_contacts ──────────────────────────────────────────────────────────
void TorsionalENM::build_contacts()
{
    contacts_.clear();
    const int N   = static_cast<int>(ca_.size());
    const float rc2 = cutoff_ * cutoff_;

#ifdef _OPENMP
    // OpenMP: each thread builds a private contact list, then merge
    int n_threads = omp_get_max_threads();
    std::vector<std::vector<Contact>> thread_contacts(n_threads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& local = thread_contacts[tid];

        #pragma omp for schedule(dynamic, 4)
        for (int i = 0; i < N - 1; ++i) {
            for (int j = i + 2; j < N; ++j) {
                float dx = ca_[i][0] - ca_[j][0];
                float dy = ca_[i][1] - ca_[j][1];
                float dz = ca_[i][2] - ca_[j][2];
                float r2 = dx*dx + dy*dy + dz*dz;
                if (r2 <= rc2) {
                    float r0    = std::sqrt(r2);
                    float ratio = cutoff_ / r0;
                    float k = k0_ * (ratio*ratio*ratio*ratio*ratio*ratio);
                    local.push_back({i, j, k, r0});
                }
            }
        }
    }
    // Merge thread-private contact lists
    for (auto& tc : thread_contacts)
        contacts_.insert(contacts_.end(), tc.begin(), tc.end());
#else
    for (int i = 0; i < N - 1; ++i) {
        for (int j = i + 2; j < N; ++j) {   // skip direct bonded neighbour (i+1)
            float dx = ca_[i][0] - ca_[j][0];
            float dy = ca_[i][1] - ca_[j][1];
            float dz = ca_[i][2] - ca_[j][2];
            float r2 = dx*dx + dy*dy + dz*dz;
            if (r2 <= rc2) {
                float r0    = std::sqrt(r2);
                float ratio = cutoff_ / r0;
                // Distance-dependent spring constant (Yang et al. 2009 eq. 4)
                float k = k0_ * (ratio*ratio*ratio*ratio*ratio*ratio);
                contacts_.push_back({i, j, k, r0});
            }
        }
    }
#endif
}

// ─── build_bonds ─────────────────────────────────────────────────────────────
void TorsionalENM::build_bonds()
{
    bonds_.clear();
    const int N = static_cast<int>(ca_.size());

    for (int k = 0; k < N - 1; ++k) {
        PseudoBond pb;
        pb.k = k;

        float ax = ca_[k+1][0] - ca_[k][0];
        float ay = ca_[k+1][1] - ca_[k][1];
        float az = ca_[k+1][2] - ca_[k][2];
        float inv = 1.0f / std::sqrt(ax*ax + ay*ay + az*az);
        pb.axis[0] = ax * inv;
        pb.axis[1] = ay * inv;
        pb.axis[2] = az * inv;

        pb.pivot[0] = 0.5f * (ca_[k][0] + ca_[k+1][0]);
        pb.pivot[1] = 0.5f * (ca_[k][1] + ca_[k+1][1]);
        pb.pivot[2] = 0.5f * (ca_[k][2] + ca_[k+1][2]);

        bonds_.push_back(pb);
    }
}

// ─── jac ─────────────────────────────────────────────────────────────────────
std::array<float,3>
TorsionalENM::jac(int bond_k, int atom_i) const noexcept
{
    // Upstream atoms are unaffected by this torsion rotation
    if (atom_i <= bond_k) return {0.0f, 0.0f, 0.0f};

    const PseudoBond& pb = bonds_[static_cast<std::size_t>(bond_k)];
    float d[3] = { ca_[atom_i][0] - pb.pivot[0],
                   ca_[atom_i][1] - pb.pivot[1],
                   ca_[atom_i][2] - pb.pivot[2] };
    float j[3];
    cross3f(pb.axis, d, j);
    return {j[0], j[1], j[2]};
}

// ─── assemble_hessian ────────────────────────────────────────────────────────
void TorsionalENM::assemble_hessian()
{
    const int M = static_cast<int>(bonds_.size());
    H_.assign(static_cast<std::size_t>(M * M), 0.0);

    // Pre-compute Jacobians for all (bond, atom) pairs
    const int N = static_cast<int>(ca_.size());
    std::vector<std::array<float,3>> J(static_cast<std::size_t>(M * N));

#ifdef _OPENMP
    // Jacobian precomputation is embarrassingly parallel over bonds
    #pragma omp parallel for schedule(static)
#endif
    for (int k = 0; k < M; ++k)
        for (int i = 0; i < N; ++i)
            J[static_cast<std::size_t>(k * N + i)] = jac(k, i);

    // Hessian assembly: accumulate contact contributions
    for (const auto& c : contacts_) {
        const int  ci  = c.i;
        const int  cj  = c.j;
        const float kij = c.k;

        for (int k = 0; k < M; ++k) {
            const auto& jki = J[static_cast<std::size_t>(k * N + ci)];
            const auto& jkj = J[static_cast<std::size_t>(k * N + cj)];
            float djk[3] = { jki[0]-jkj[0], jki[1]-jkj[1], jki[2]-jkj[2] };

            // Exploit symmetry: iterate l ≥ k only, mirror afterwards
            for (int l = k; l < M; ++l) {
                const auto& jli = J[static_cast<std::size_t>(l * N + ci)];
                const auto& jlj = J[static_cast<std::size_t>(l * N + cj)];
                float djl[3] = { jli[0]-jlj[0], jli[1]-jlj[1], jli[2]-jlj[2] };

                double contrib = kij * static_cast<double>(
                    djk[0]*djl[0] + djk[1]*djl[1] + djk[2]*djl[2]);

                H_[static_cast<std::size_t>(k * M + l)] += contrib;
                if (l != k)
                    H_[static_cast<std::size_t>(l * M + k)] += contrib;
            }
        }
    }
}

// ─── Jacobi eigenvalue decomposition ─────────────────────────────────────────
// Classic off-diagonal Jacobi sweeps for a symmetric n×n matrix A.
// On exit: A holds eigenvalues on diagonal; V holds eigenvectors in columns.
void TorsionalENM::jacobi_rotate(std::vector<double>& A,
                                  std::vector<double>& V,
                                  int n, int p, int q) noexcept
{
    if (std::abs(A[static_cast<std::size_t>(p*n+q)]) < 1e-15) return;

    double tau = (A[static_cast<std::size_t>(q*n+q)] -
                  A[static_cast<std::size_t>(p*n+p)]) /
                 (2.0 * A[static_cast<std::size_t>(p*n+q)]);
    double t = (tau >= 0.0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau*tau));
    double c = 1.0 / std::sqrt(1.0 + t*t);
    double s = t * c;

    // Update A
    double app = A[static_cast<std::size_t>(p*n+p)];
    double aqq = A[static_cast<std::size_t>(q*n+q)];
    double apq = A[static_cast<std::size_t>(p*n+q)];

    A[static_cast<std::size_t>(p*n+p)] = c*c*app - 2.0*s*c*apq + s*s*aqq;
    A[static_cast<std::size_t>(q*n+q)] = s*s*app + 2.0*s*c*apq + c*c*aqq;
    A[static_cast<std::size_t>(p*n+q)] = A[static_cast<std::size_t>(q*n+p)] = 0.0;

    for (int r = 0; r < n; ++r) {
        if (r == p || r == q) continue;
        double arp = A[static_cast<std::size_t>(r*n+p)];
        double arq = A[static_cast<std::size_t>(r*n+q)];
        A[static_cast<std::size_t>(r*n+p)] = A[static_cast<std::size_t>(p*n+r)] = c*arp - s*arq;
        A[static_cast<std::size_t>(r*n+q)] = A[static_cast<std::size_t>(q*n+r)] = s*arp + c*arq;
    }

    // Update eigenvector matrix V
    for (int r = 0; r < n; ++r) {
        double vrp = V[static_cast<std::size_t>(r*n+p)];
        double vrq = V[static_cast<std::size_t>(r*n+q)];
        V[static_cast<std::size_t>(r*n+p)] = c*vrp - s*vrq;
        V[static_cast<std::size_t>(r*n+q)] = s*vrp + c*vrq;
    }
}

// ─── diagonalize ─────────────────────────────────────────────────────────────
void TorsionalENM::diagonalize()
{
    const int M = static_cast<int>(bonds_.size());
    if (M == 0) return;

#ifdef FLEXAIDS_HAS_EIGEN
    // Eigen: use SelfAdjointEigenSolver for robust, optimized eigendecomposition
    {
        Eigen::Map<const Eigen::MatrixXd> Hmap(H_.data(), M, M);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Hmap);

        const auto& eigenvalues  = solver.eigenvalues();   // ascending order
        const auto& eigenvectors = solver.eigenvectors();

        modes_.clear();
        modes_.reserve(static_cast<std::size_t>(M));
        for (int i = 0; i < M; ++i) {
            NormalMode nm;
            nm.eigenvalue = eigenvalues(i);
            nm.eigenvector.resize(static_cast<std::size_t>(M));
            for (int j = 0; j < M; ++j)
                nm.eigenvector[static_cast<std::size_t>(j)] = eigenvectors(j, i);
            modes_.push_back(std::move(nm));
        }
    }
#else
    // Manual Jacobi iteration fallback
    std::vector<double> A = H_;
    std::vector<double> V(static_cast<std::size_t>(M * M), 0.0);
    for (int i = 0; i < M; ++i)
        V[static_cast<std::size_t>(i*M+i)] = 1.0;

    const int MAX_SWEEPS = 50;
    for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep) {
        double max_off = 0.0;
        int pp = 0, qq = 1;
        for (int p = 0; p < M - 1; ++p)
            for (int q = p + 1; q < M; ++q) {
                double v = std::abs(A[static_cast<std::size_t>(p*M+q)]);
                if (v > max_off) { max_off = v; pp = p; qq = q; }
            }
        if (max_off < 1e-12) break;
        jacobi_rotate(A, V, M, pp, qq);
    }

    modes_.clear();
    modes_.reserve(static_cast<std::size_t>(M));
    for (int i = 0; i < M; ++i) {
        NormalMode nm;
        nm.eigenvalue = A[static_cast<std::size_t>(i*M+i)];
        nm.eigenvector.resize(static_cast<std::size_t>(M));
        for (int j = 0; j < M; ++j)
            nm.eigenvector[static_cast<std::size_t>(j)] = V[static_cast<std::size_t>(j*M+i)];
        modes_.push_back(std::move(nm));
    }

    // Sort by ascending eigenvalue (lowest = softest = most flexible)
    std::sort(modes_.begin(), modes_.end(),
              [](const NormalMode& a, const NormalMode& b){
                  return a.eigenvalue < b.eigenvalue; });
#endif
}

// ─── sample ──────────────────────────────────────────────────────────────────
Conformer TorsionalENM::sample(float temperature, std::mt19937& rng) const
{
    if (!built_) throw std::runtime_error("TENCM: model not built");

    const int M = static_cast<int>(bonds_.size());
    const int N = static_cast<int>(ca_.size());

    std::normal_distribution<float> gauss(0.0f, 1.0f);

    Conformer conf;
    conf.delta_theta.assign(static_cast<std::size_t>(M), 0.0f);

    // Skip first 6 modes (rigid-body; eigenvalue ≈ 0)
    const int SKIP = std::min(6, M);
    const double kBT = static_cast<double>(kB_kcal) * temperature;

    for (int m = SKIP; m < M && m < SKIP + N_MODES; ++m) {
        double lam = modes_[static_cast<std::size_t>(m)].eigenvalue;
        if (lam < 1e-8) continue;   // nearly zero stiffness → skip
        double sigma = std::sqrt(kBT / lam);
        float  z     = gauss(rng);

        for (int k = 0; k < M; ++k)
            conf.delta_theta[static_cast<std::size_t>(k)] +=
                static_cast<float>(sigma * z *
                modes_[static_cast<std::size_t>(m)].eigenvector[static_cast<std::size_t>(k)]);
    }

    // Build perturbed Cα coordinates:  r_i' = r_i + Σ_k J_k(i) δθ_k
    conf.ca.resize(static_cast<std::size_t>(N));

#ifdef _OPENMP
    // Each atom's displacement is independent — parallelize over atoms
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < N; ++i) {
        float disp[3] = {0.0f, 0.0f, 0.0f};
        for (int k = 0; k < M; ++k) {
            float dth = conf.delta_theta[static_cast<std::size_t>(k)];
            if (dth == 0.0f || i <= k) continue;
            auto jki = jac(k, i);
            disp[0] += jki[0] * dth;
            disp[1] += jki[1] * dth;
            disp[2] += jki[2] * dth;
        }
        conf.ca[static_cast<std::size_t>(i)] = {
            ca_[static_cast<std::size_t>(i)][0] + disp[0],
            ca_[static_cast<std::size_t>(i)][1] + disp[1],
            ca_[static_cast<std::size_t>(i)][2] + disp[2]
        };
    }

    // Elastic strain energy: ½ δθᵀ H δθ
#ifdef FLEXAIDS_HAS_EIGEN
    {
        Eigen::Map<const Eigen::VectorXd> dth(
            reinterpret_cast<const double*>(nullptr), 0);  // placeholder
        // Convert float delta_theta to double for Eigen GEMV
        Eigen::VectorXd dth_d(M);
        for (int k = 0; k < M; ++k)
            dth_d(k) = static_cast<double>(conf.delta_theta[static_cast<std::size_t>(k)]);
        Eigen::Map<const Eigen::MatrixXd> Hmap(H_.data(), M, M);
        conf.strain_energy = static_cast<float>(0.5 * dth_d.dot(Hmap * dth_d));
    }
#else
    conf.strain_energy = 0.0f;
    for (int k = 0; k < M; ++k) {
        double row = 0.0;
        for (int l = 0; l < M; ++l)
            row += H_[static_cast<std::size_t>(k*M+l)] *
                   static_cast<double>(conf.delta_theta[static_cast<std::size_t>(l)]);
        conf.strain_energy += static_cast<float>(
            0.5 * static_cast<double>(conf.delta_theta[static_cast<std::size_t>(k)]) * row);
    }
#endif

    return conf;
}

// ─── apply ───────────────────────────────────────────────────────────────────
// Translate Cα displacements to all heavy atoms in the residue by rigid-body
// shift (centroid translation only — backbone torsion rebuild is handled by
// the existing buildcc/buildic pipeline called afterwards).
void TorsionalENM::apply(const Conformer& conf,
                          atom*            atoms,
                          const resid*     residue) const
{
    if (!built_) return;
    const int N = static_cast<int>(ca_.size());

    for (int seq = 0; seq < N; ++seq) {
        int ri = res_idx_[static_cast<std::size_t>(seq)];
        // Displacement of this residue's Cα
        int   ai = ca_atom_idx_[static_cast<std::size_t>(seq)];
        float dx = conf.ca[static_cast<std::size_t>(seq)][0] - atoms[ai].coor[0];
        float dy = conf.ca[static_cast<std::size_t>(seq)][1] - atoms[ai].coor[1];
        float dz = conf.ca[static_cast<std::size_t>(seq)][2] - atoms[ai].coor[2];

        // Apply the same rigid shift to all atoms of this residue (rotamer 0)
        int first = residue[ri].fatm[0];
        int last  = residue[ri].latm[0];
        for (int a = first; a <= last; ++a) {
            atoms[a].coor[0] += dx;
            atoms[a].coor[1] += dy;
            atoms[a].coor[2] += dz;
        }
    }
}

// ─── bfactors ────────────────────────────────────────────────────────────────
// B_i = (8π²/3) <Δr_i²>  where <Δr_i²> = kBT Σ_m (σ_m² |J_m(i)|²)
std::vector<float> TorsionalENM::bfactors(float temperature) const
{
    const int N  = static_cast<int>(ca_.size());
    const int M  = static_cast<int>(bonds_.size());
    const double kBT = static_cast<double>(kB_kcal) * temperature;
    const int SKIP = std::min(6, M);

    std::vector<float> bf(static_cast<std::size_t>(N), 0.0f);

#ifdef _OPENMP
    // B-factor for each atom is completely independent — parallelize
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < N; ++i) {
        double msf = 0.0;  // mean-square fluctuation
        for (int m = SKIP; m < M && m < SKIP + N_MODES; ++m) {
            double lam = modes_[static_cast<std::size_t>(m)].eigenvalue;
            if (lam < 1e-8) continue;
            // σ_m² = kBT/λ_m; contribution to msf: σ_m² * |J_m(i)|²
            // J_m(i) = Σ_k v_mk * J_k(i)   (linear combination of bond Jacobians)
            float jmi[3] = {0.0f, 0.0f, 0.0f};
            for (int k = 0; k < M; ++k) {
                if (i <= k) continue;
                auto jki = jac(k, i);
                float vmk = static_cast<float>(
                    modes_[static_cast<std::size_t>(m)].eigenvector[static_cast<std::size_t>(k)]);
                jmi[0] += vmk * jki[0];
                jmi[1] += vmk * jki[1];
                jmi[2] += vmk * jki[2];
            }
            double jmag2 = static_cast<double>(
                jmi[0]*jmi[0] + jmi[1]*jmi[1] + jmi[2]*jmi[2]);
            msf += (kBT / lam) * jmag2;
        }
        // B-factor = (8π²/3) * MSF  (isotropic approximation)
        bf[static_cast<std::size_t>(i)] = static_cast<float>(
            (8.0 * std::numbers::pi * std::numbers::pi / 3.0) * msf);
    }
    return bf;
}

// ─── residue_rms_fluctuation ─────────────────────────────────────────────────
float residue_rms_fluctuation(const TorsionalENM& tencm,
                              int  residue_idx,
                              float temperature)
{
    auto bf = tencm.bfactors(temperature);
    if (residue_idx < 0 || residue_idx >= static_cast<int>(bf.size()))
        return 0.0f;
    // MSF = B / (8π²/3)
    float msf = bf[static_cast<std::size_t>(residue_idx)] /
                static_cast<float>(8.0 * std::numbers::pi * std::numbers::pi / 3.0);
    return std::sqrt(msf);
}

}  // namespace tencm
