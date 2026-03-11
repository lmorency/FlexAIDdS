// tencom_output.cpp — BindingMode-like output for tENCoM vibrational analysis

#include "tencom_output.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cstdio>
#include <cstring>

namespace tencom_output {

// ─── FlexPopulation::sort_by_free_energy ────────────────────────────────────

void FlexPopulation::sort_by_free_energy() {
    // Keep mode 0 (reference) first, then sort remaining by delta_F_vib
    if (modes.size() <= 2) return;

    std::sort(modes.begin() + 1, modes.end(),
              [](const FlexMode& a, const FlexMode& b) {
                  return a.delta_F_vib < b.delta_F_vib;
              });
}

// ─── FlexPopulation::write_mode_pdb ─────────────────────────────────────────

void FlexPopulation::write_mode_pdb(const FlexMode& mode,
                                     const tencom_pdb::CalphaStructure& structure) const
{
    // Build output filename: prefix_mode_N.pdb
    std::string outfile = output_prefix + "_mode_" + std::to_string(mode.mode_id) + ".pdb";

    std::ofstream ofs(outfile);
    if (!ofs.is_open()) {
        std::cerr << "Error: cannot write to " << outfile << "\n";
        return;
    }

    // ── REMARK section (mirrors BindingMode::output_BindingMode pattern) ────
    ofs << "REMARK FlexAIDdS tENCoM vibrational entropy analysis\n";

    if (mode.mode_id == 0) {
        ofs << "REMARK Mode:0 (reference) Source:" << mode.pdb_path << "\n";
    } else {
        ofs << "REMARK Mode:" << mode.mode_id
            << " Source:" << mode.pdb_path << "\n";
    }

    ofs << std::fixed << std::setprecision(6);
    ofs << "REMARK S_vib = " << mode.S_vib << " kcal/mol/K\n";
    ofs << "REMARK Delta_S_vib = " << mode.delta_S_vib << " kcal/mol/K\n";

    ofs << std::setprecision(4);
    ofs << "REMARK Delta_F_vib = " << mode.delta_F_vib << " kcal/mol\n";
    ofs << "REMARK N_modes = " << mode.n_modes
        << "  N_residues = " << mode.n_residues
        << "  Temperature = " << temperature << " K\n";
    ofs << "REMARK Full_flexibility = ON\n";

    // Eigenvalue summary (first few modes)
    if (!mode.mode_data.empty()) {
        ofs << "REMARK Eigenvalue differentials (mode delta_eigenvalue overlap):\n";
        int n_show = std::min(static_cast<int>(mode.mode_data.size()), 10);
        for (int i = 0; i < n_show; ++i) {
            const auto& mc = mode.mode_data[i];
            ofs << "REMARK   mode_" << mc.mode_index
                << " delta_eig=" << std::setprecision(6) << mc.delta_eigenvalue;
            if (!std::isnan(mc.overlap)) {
                ofs << " overlap=" << std::setprecision(4) << mc.overlap;
            }
            ofs << "\n";
        }
        if (static_cast<int>(mode.mode_data.size()) > 10) {
            ofs << "REMARK   ... (" << mode.mode_data.size() - 10 << " more modes)\n";
        }
    }

    // B-factor summary (per-residue)
    if (!mode.bfactors.empty()) {
        ofs << "REMARK B-factors (per-residue, Angstrom^2):\n";
        ofs << "REMARK  ";
        for (size_t i = 0; i < mode.bfactors.size(); ++i) {
            ofs << " " << std::setprecision(2) << mode.bfactors[i];
            if ((i + 1) % 15 == 0 && i + 1 < mode.bfactors.size()) {
                ofs << "\nREMARK  ";
            }
        }
        ofs << "\n";
    }

    if (!mode.delta_bfactors.empty()) {
        ofs << "REMARK Delta_B-factors (vs reference):\n";
        ofs << "REMARK  ";
        for (size_t i = 0; i < mode.delta_bfactors.size(); ++i) {
            ofs << " " << std::setprecision(2) << mode.delta_bfactors[i];
            if ((i + 1) % 15 == 0 && i + 1 < mode.delta_bfactors.size()) {
                ofs << "\nREMARK  ";
            }
        }
        ofs << "\n";
    }

    // ── Composition summary ────────────────────────────────────────────────
    if (structure.n_protein > 0 || structure.n_dna > 0 || structure.n_rna > 0) {
        ofs << "REMARK Composition:";
        if (structure.n_protein > 0) ofs << " protein=" << structure.n_protein;
        if (structure.n_dna > 0)     ofs << " DNA=" << structure.n_dna;
        if (structure.n_rna > 0)     ofs << " RNA=" << structure.n_rna;
        ofs << "\n";
    }

    // ── ATOM records (backbone representatives) ─────────────────────────────
    for (int ai = 1; ai <= structure.res_cnt; ++ai) {
        const atom& a = structure.atoms[ai];
        const resid& r = structure.residues[ai];

        // B-factor column: use computed B-factor if available, else 0
        float bf = 0.0f;
        if (ai - 1 < static_cast<int>(mode.bfactors.size())) {
            bf = mode.bfactors[ai - 1];
        }

        // Use correct atom name based on residue type
        const char* atom_label = " CA ";
        if (ai < static_cast<int>(structure.residue_types.size())) {
            auto rt = structure.residue_types[ai];
            if (rt == tencom_pdb::ResidueType::DNA ||
                rt == tencom_pdb::ResidueType::RNA) {
                atom_label = " C4'";
            }
        }

        char line[128];
        std::snprintf(line, sizeof(line),
            "ATOM  %5d %4s %3s %c%4d    %8.3f%8.3f%8.3f%6.2f%6.2f           C",
            ai, atom_label, r.name, r.chn, r.number,
            a.coor[0], a.coor[1], a.coor[2],
            1.00f, bf);
        ofs << line << "\n";
    }

    ofs << "END\n";
    ofs.close();

    std::cout << "  Wrote: " << outfile << "\n";
}

// ─── FlexPopulation::print_summary ──────────────────────────────────────────

void FlexPopulation::print_summary(std::ostream& os) const {
    os << "\n=== FlexAIDdS tENCoM Vibrational Entropy Differential ===\n";
    os << "Temperature: " << temperature << " K\n";
    os << "Full flexibility: ON (all modes)\n\n";

    os << std::left
       << std::setw(6)  << "Mode"
       << std::setw(40) << "Source"
       << std::setw(16) << "S_vib"
       << std::setw(16) << "Delta_S_vib"
       << std::setw(16) << "Delta_F_vib"
       << std::setw(10) << "N_modes"
       << std::setw(10) << "N_res"
       << "\n";
    os << std::string(114, '-') << "\n";

    for (const auto& m : modes) {
        // Truncate source path for display
        std::string display_path = m.pdb_path;
        if (display_path.size() > 38) {
            display_path = "..." + display_path.substr(display_path.size() - 35);
        }

        os << std::left << std::fixed
           << std::setw(6)  << m.mode_id
           << std::setw(40) << display_path
           << std::setw(16) << std::setprecision(6) << m.S_vib
           << std::setw(16) << std::setprecision(6) << m.delta_S_vib
           << std::setw(16) << std::setprecision(4) << m.delta_F_vib
           << std::setw(10) << m.n_modes
           << std::setw(10) << m.n_residues
           << "\n";
    }

    // Mode overlap summary for targets
    for (const auto& m : modes) {
        if (m.mode_id == 0 || m.mode_data.empty()) continue;

        os << "\n  Mode " << m.mode_id << " (" << m.label << ") — top eigenvalue differentials:\n";
        int n_show = std::min(static_cast<int>(m.mode_data.size()), 5);
        for (int i = 0; i < n_show; ++i) {
            const auto& mc = m.mode_data[i];
            os << "    mode_" << mc.mode_index
               << "  eig_ref=" << std::setprecision(4) << mc.eigenvalue_ref
               << "  eig_tgt=" << mc.eigenvalue_tgt
               << "  delta=" << mc.delta_eigenvalue;
            if (!std::isnan(mc.overlap)) {
                os << "  overlap=" << mc.overlap;
            }
            os << "\n";
        }
    }

    os << "\n";
}

// ─── FlexPopulation::output_all ─────────────────────────────────────────────

void FlexPopulation::output_all(
    const std::vector<tencom_pdb::CalphaStructure>& structures) const
{
    print_summary();

    if (structures.size() != modes.size()) {
        std::cerr << "Warning: structure count (" << structures.size()
                  << ") != mode count (" << modes.size() << ")\n";
    }

    int n = std::min(structures.size(), modes.size());
    for (int i = 0; i < n; ++i) {
        write_mode_pdb(modes[i], structures[i]);
    }

    std::cout << "\ntENCoM analysis complete. "
              << modes.size() << " mode(s) written.\n";
}

}  // namespace tencom_output
