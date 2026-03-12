// test_tencom_diff.cpp — Unit tests for the tENCoM differential engine
//
// Tests: PDB Cα reader, differential computation, mode overlap, B-factor diffs.
//
// NOTE: gtest must be included before flexaid.h to avoid the #define E macro
// conflicting with GoogleTest template parameters.

#include <gtest/gtest.h>

// flexaid.h defines E as a macro which clashes with GoogleTest templates.
// Include our headers after gtest, then the macro is already expanded in gtest.
#include "pdb_calpha.h"
#include "tencom_diff.h"
#include "tencm.h"
#include "encom.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <cmath>
#include <string>
#include <filesystem>

// ─── Helper: write a synthetic PDB to a temp file ───────────────────────────

static std::string write_synthetic_pdb(const std::string& prefix,
                                        int n_residues,
                                        float radius = 2.3f,
                                        float rise = 1.5f,
                                        float perturb = 0.0f)
{
    std::string path = std::filesystem::temp_directory_path().string()
                       + "/" + prefix + ".pdb";

    std::ofstream ofs(path);
    const float turn = 100.0f * 3.14159265f / 180.0f;

    for (int r = 0; r < n_residues; ++r) {
        float x = radius * std::cos(r * turn) + perturb * (r % 3 == 0 ? 0.5f : 0.0f);
        float y = radius * std::sin(r * turn) + perturb * (r % 3 == 1 ? 0.3f : 0.0f);
        float z = r * rise + perturb * (r % 3 == 2 ? 0.2f : 0.0f);

        char line[82];
        std::snprintf(line, sizeof(line),
            "ATOM  %5d  CA  ALA A%4d    %8.3f%8.3f%8.3f  1.00  0.00           C",
            r + 1, r + 1, x, y, z);
        ofs << line << "\n";
    }
    ofs << "END\n";
    ofs.close();

    return path;
}

// ─── PDB Reader Tests ───────────────────────────────────────────────────────

TEST(PDBCalphaReader, ReadsCorrectResidueCount) {
    auto path = write_synthetic_pdb("test_reader_20", 20);
    auto structure = tencom_pdb::read_pdb_calpha(path);
    EXPECT_EQ(structure.res_cnt, 20);
    std::remove(path.c_str());
}

TEST(PDBCalphaReader, AtomCoordinatesAreCorrect) {
    auto path = write_synthetic_pdb("test_reader_coords", 5, 2.3f, 1.5f, 0.0f);
    auto structure = tencom_pdb::read_pdb_calpha(path);

    ASSERT_EQ(structure.res_cnt, 5);

    // First residue: r=0, x = 2.3*cos(0) = 2.3, y = 2.3*sin(0) = 0, z = 0
    EXPECT_NEAR(structure.atoms[1].coor[0], 2.3f, 0.01f);
    EXPECT_NEAR(structure.atoms[1].coor[1], 0.0f, 0.01f);
    EXPECT_NEAR(structure.atoms[1].coor[2], 0.0f, 0.01f);

    std::remove(path.c_str());
}

TEST(PDBCalphaReader, ResidueFieldsPopulated) {
    auto path = write_synthetic_pdb("test_reader_fields", 10);
    auto structure = tencom_pdb::read_pdb_calpha(path);

    ASSERT_GE(structure.res_cnt, 1);
    EXPECT_EQ(structure.residues[1].type, 0);  // protein
    EXPECT_EQ(std::string(structure.residues[1].name, 3), "ALA");
    EXPECT_EQ(structure.residues[1].chn, 'A');
    ASSERT_NE(structure.residues[1].fatm, nullptr);
    ASSERT_NE(structure.residues[1].latm, nullptr);

    std::remove(path.c_str());
}

TEST(PDBCalphaReader, ThrowsOnMissingFile) {
    EXPECT_THROW(
        tencom_pdb::read_pdb_calpha("/nonexistent/path.pdb"),
        std::runtime_error
    );
}

TEST(PDBCalphaReader, CompatibleWithTorsionalENM) {
    auto path = write_synthetic_pdb("test_enm_compat", 30);
    auto structure = tencom_pdb::read_pdb_calpha(path);

    tencm::TorsionalENM enm;
    enm.build(structure.atoms.data(), structure.residues.data(), structure.res_cnt);

    EXPECT_TRUE(enm.is_built());
    EXPECT_EQ(enm.n_residues(), 30);
    EXPECT_GT(enm.modes().size(), 0u);

    std::remove(path.c_str());
}

// ─── Differential Engine Tests ──────────────────────────────────────────────

TEST(DifferentialEngine, IdenticalStructuresZeroDelta) {
    auto path = write_synthetic_pdb("test_diff_ident", 30);
    auto s1 = tencom_pdb::read_pdb_calpha(path);
    auto s2 = tencom_pdb::read_pdb_calpha(path);

    tencm::TorsionalENM enm1, enm2;
    enm1.build(s1.atoms.data(), s1.residues.data(), s1.res_cnt);
    enm2.build(s2.atoms.data(), s2.residues.data(), s2.res_cnt);

    auto diff = tencom_diff::compute_differential(enm1, enm2, "ref", "tgt", 300.0);

    // Delta S_vib should be exactly 0 for identical structures
    EXPECT_NEAR(diff.delta_S_vib, 0.0, 1e-10);
    EXPECT_NEAR(diff.delta_F_vib, 0.0, 1e-10);

    // All eigenvalue deltas should be ~0 (allow floating-point rounding)
    for (const auto& mc : diff.mode_comparisons) {
        EXPECT_NEAR(mc.delta_eigenvalue, 0.0, 1e-8);
    }

    // B-factor deltas should be 0
    for (float db : diff.delta_bfactors) {
        EXPECT_NEAR(db, 0.0f, 1e-6f);
    }

    std::remove(path.c_str());
}

TEST(DifferentialEngine, IdenticalStructuresFullOverlap) {
    auto path = write_synthetic_pdb("test_diff_overlap", 30);
    auto s1 = tencom_pdb::read_pdb_calpha(path);
    auto s2 = tencom_pdb::read_pdb_calpha(path);

    tencm::TorsionalENM enm1, enm2;
    enm1.build(s1.atoms.data(), s1.residues.data(), s1.res_cnt);
    enm2.build(s2.atoms.data(), s2.residues.data(), s2.res_cnt);

    auto diff = tencom_diff::compute_differential(enm1, enm2);

    // Mode overlap should be 1.0 for identical structures
    for (const auto& mc : diff.mode_comparisons) {
        if (!std::isnan(mc.overlap)) {
            EXPECT_NEAR(mc.overlap, 1.0, 1e-6);
        }
    }

    std::remove(path.c_str());
}

TEST(DifferentialEngine, PerturbedStructureNonzeroDelta) {
    auto path_ref = write_synthetic_pdb("test_diff_ref", 30, 2.3f, 1.5f, 0.0f);
    auto path_tgt = write_synthetic_pdb("test_diff_tgt", 30, 2.3f, 1.5f, 0.5f);

    auto s_ref = tencom_pdb::read_pdb_calpha(path_ref);
    auto s_tgt = tencom_pdb::read_pdb_calpha(path_tgt);

    tencm::TorsionalENM enm_ref, enm_tgt;
    enm_ref.build(s_ref.atoms.data(), s_ref.residues.data(), s_ref.res_cnt);
    enm_tgt.build(s_tgt.atoms.data(), s_tgt.residues.data(), s_tgt.res_cnt);

    auto diff = tencom_diff::compute_differential(
        enm_ref, enm_tgt, "ref", "tgt", 300.0);

    // Delta S_vib should be nonzero for different structures
    EXPECT_NE(diff.delta_S_vib, 0.0);
    // delta_F_vib = -T * delta_S_vib
    EXPECT_NEAR(diff.delta_F_vib, -300.0 * diff.delta_S_vib, 1e-10);

    // At least some eigenvalue deltas should be nonzero
    bool any_nonzero = false;
    for (const auto& mc : diff.mode_comparisons) {
        if (std::abs(mc.delta_eigenvalue) > 1e-10) {
            any_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(any_nonzero);

    std::remove(path_ref.c_str());
    std::remove(path_tgt.c_str());
}

TEST(DifferentialEngine, DifferentSizeStructures) {
    auto path_ref = write_synthetic_pdb("test_diff_size_ref", 30);
    auto path_tgt = write_synthetic_pdb("test_diff_size_tgt", 25);

    auto s_ref = tencom_pdb::read_pdb_calpha(path_ref);
    auto s_tgt = tencom_pdb::read_pdb_calpha(path_tgt);

    tencm::TorsionalENM enm_ref, enm_tgt;
    enm_ref.build(s_ref.atoms.data(), s_ref.residues.data(), s_ref.res_cnt);
    enm_tgt.build(s_tgt.atoms.data(), s_tgt.residues.data(), s_tgt.res_cnt);

    auto diff = tencom_diff::compute_differential(enm_ref, enm_tgt);

    // Should still produce a result
    EXPECT_GT(diff.mode_comparisons.size(), 0u);

    // Mode overlap should be NaN (different dimensionality)
    for (const auto& mc : diff.mode_comparisons) {
        EXPECT_TRUE(std::isnan(mc.overlap));
    }

    // B-factor differential should compare up to min size
    EXPECT_EQ(diff.delta_bfactors.size(),
              std::min(diff.bfactors_ref.size(), diff.bfactors_tgt.size()));

    std::remove(path_ref.c_str());
    std::remove(path_tgt.c_str());
}

// ─── Mode Conversion Tests ─────────────────────────────────────────────────

TEST(ModeConversion, TencmToEncomPreservesEigenvalues) {
    auto path = write_synthetic_pdb("test_convert", 30);
    auto structure = tencom_pdb::read_pdb_calpha(path);

    tencm::TorsionalENM enm;
    enm.build(structure.atoms.data(), structure.residues.data(), structure.res_cnt);

    auto tencm_modes = enm.modes();
    auto encom_modes = tencom_diff::to_encom_modes(tencm_modes);

    ASSERT_EQ(encom_modes.size(), tencm_modes.size());
    for (size_t i = 0; i < tencm_modes.size(); ++i) {
        EXPECT_DOUBLE_EQ(encom_modes[i].eigenvalue, tencm_modes[i].eigenvalue);
        EXPECT_EQ(encom_modes[i].index, static_cast<int>(i) + 1);
    }

    std::remove(path.c_str());
}

// ─── Nucleic Acid Tests ─────────────────────────────────────────────────────

static std::string write_synthetic_dna_pdb(const std::string& prefix,
                                            int n_nucleotides,
                                            float radius = 4.5f,
                                            float rise = 3.4f)
{
    std::string path = std::filesystem::temp_directory_path().string()
                       + "/" + prefix + ".pdb";

    std::ofstream ofs(path);
    const float turn = 36.0f * 3.14159265f / 180.0f;  // ~10 bp per turn
    const char* bases[] = {" DA", " DT", " DC", " DG"};

    for (int r = 0; r < n_nucleotides; ++r) {
        float x = radius * std::cos(r * turn);
        float y = radius * std::sin(r * turn);
        float z = r * rise;

        // Write C4' atom (backbone representative for nucleic acids)
        char line[82];
        std::snprintf(line, sizeof(line),
            "ATOM  %5d  C4' %3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00           C",
            r + 1, bases[r % 4], r + 1, x, y, z);
        ofs << line << "\n";
    }
    ofs << "END\n";
    ofs.close();

    return path;
}

static std::string write_mixed_pdb(const std::string& prefix,
                                    int n_protein, int n_dna)
{
    std::string path = std::filesystem::temp_directory_path().string()
                       + "/" + prefix + ".pdb";

    std::ofstream ofs(path);
    int atom_num = 0;
    int res_num = 0;

    // Protein chain A
    const float turn_p = 100.0f * 3.14159265f / 180.0f;
    for (int r = 0; r < n_protein; ++r) {
        ++atom_num; ++res_num;
        float x = 2.3f * std::cos(r * turn_p);
        float y = 2.3f * std::sin(r * turn_p);
        float z = r * 1.5f;
        char line[82];
        std::snprintf(line, sizeof(line),
            "ATOM  %5d  CA  ALA A%4d    %8.3f%8.3f%8.3f  1.00  0.00           C",
            atom_num, res_num, x, y, z);
        ofs << line << "\n";
    }

    // DNA chain B
    const float turn_d = 36.0f * 3.14159265f / 180.0f;
    const char* bases[] = {" DA", " DT", " DC", " DG"};
    for (int r = 0; r < n_dna; ++r) {
        ++atom_num; ++res_num;
        float x = 4.5f * std::cos(r * turn_d) + 20.0f;
        float y = 4.5f * std::sin(r * turn_d);
        float z = r * 3.4f;
        char line[82];
        std::snprintf(line, sizeof(line),
            "ATOM  %5d  C4' %3s B%4d    %8.3f%8.3f%8.3f  1.00  0.00           C",
            atom_num, bases[r % 4], res_num, x, y, z);
        ofs << line << "\n";
    }

    ofs << "END\n";
    ofs.close();

    return path;
}

TEST(NucleicAcidReader, ReadsDNAResidues) {
    auto path = write_synthetic_dna_pdb("test_dna_20", 20);
    auto structure = tencom_pdb::read_pdb_calpha(path);

    EXPECT_EQ(structure.res_cnt, 20);
    EXPECT_EQ(structure.n_dna, 20);
    EXPECT_EQ(structure.n_protein, 0);
    EXPECT_EQ(structure.n_rna, 0);

    std::remove(path.c_str());
}

TEST(NucleicAcidReader, DNACompatibleWithENM) {
    auto path = write_synthetic_dna_pdb("test_dna_enm", 30);
    auto structure = tencom_pdb::read_pdb_calpha(path);

    tencm::TorsionalENM enm;
    enm.build(structure.atoms.data(), structure.residues.data(), structure.res_cnt);

    EXPECT_TRUE(enm.is_built());
    EXPECT_EQ(enm.n_residues(), 30);
    EXPECT_GT(enm.modes().size(), 0u);

    std::remove(path.c_str());
}

TEST(NucleicAcidReader, MixedProteinDNA) {
    auto path = write_mixed_pdb("test_mixed", 20, 15);
    auto structure = tencom_pdb::read_pdb_calpha(path);

    EXPECT_EQ(structure.res_cnt, 35);
    EXPECT_EQ(structure.n_protein, 20);
    EXPECT_EQ(structure.n_dna, 15);

    // Should build ENM across the whole complex
    tencm::TorsionalENM enm;
    enm.build(structure.atoms.data(), structure.residues.data(), structure.res_cnt);
    EXPECT_TRUE(enm.is_built());
    EXPECT_EQ(enm.n_residues(), 35);

    std::remove(path.c_str());
}

TEST(NucleicAcidReader, ResidueTypeTracking) {
    auto path = write_mixed_pdb("test_types", 10, 10);
    auto structure = tencom_pdb::read_pdb_calpha(path);

    // First 10 should be protein, next 10 DNA
    for (int i = 1; i <= 10; ++i) {
        EXPECT_EQ(structure.residue_types[i], tencom_pdb::ResidueType::PROTEIN);
    }
    for (int i = 11; i <= 20; ++i) {
        EXPECT_EQ(structure.residue_types[i], tencom_pdb::ResidueType::DNA);
    }

    std::remove(path.c_str());
}

TEST(NucleicAcidReader, DNADifferentialWorks) {
    auto path_ref = write_synthetic_dna_pdb("test_dna_diff_ref", 25);
    auto path_tgt = write_synthetic_dna_pdb("test_dna_diff_tgt", 25);

    auto s_ref = tencom_pdb::read_pdb_calpha(path_ref);
    auto s_tgt = tencom_pdb::read_pdb_calpha(path_tgt);

    tencm::TorsionalENM enm_ref, enm_tgt;
    enm_ref.build(s_ref.atoms.data(), s_ref.residues.data(), s_ref.res_cnt);
    enm_tgt.build(s_tgt.atoms.data(), s_tgt.residues.data(), s_tgt.res_cnt);

    auto diff = tencom_diff::compute_differential(enm_ref, enm_tgt, "dna_ref", "dna_tgt");

    // Identical structures: delta should be zero
    EXPECT_NEAR(diff.delta_S_vib, 0.0, 1e-10);

    std::remove(path_ref.c_str());
    std::remove(path_tgt.c_str());
}

TEST(NucleicAcidReader, ClassifiesResidueTypes) {
    EXPECT_EQ(tencom_pdb::classify_residue("ALA"), tencom_pdb::ResidueType::PROTEIN);
    EXPECT_EQ(tencom_pdb::classify_residue("GLY"), tencom_pdb::ResidueType::PROTEIN);
    EXPECT_EQ(tencom_pdb::classify_residue(" DA"), tencom_pdb::ResidueType::DNA);
    EXPECT_EQ(tencom_pdb::classify_residue(" DT"), tencom_pdb::ResidueType::DNA);
    EXPECT_EQ(tencom_pdb::classify_residue("  A"), tencom_pdb::ResidueType::RNA);
    EXPECT_EQ(tencom_pdb::classify_residue("  U"), tencom_pdb::ResidueType::RNA);
    EXPECT_EQ(tencom_pdb::classify_residue("HOH"), tencom_pdb::ResidueType::UNKNOWN);
    EXPECT_EQ(tencom_pdb::classify_residue("ZZZ"), tencom_pdb::ResidueType::UNKNOWN);
}

// ─── Robustness Tests ──────────────────────────────────────────────────────

TEST(PDBCalphaReader, SkipsShortLines) {
    // PDB file with lines too short for ATOM records — should be skipped
    std::string path = std::filesystem::temp_directory_path().string()
                       + "/test_short_lines.pdb";
    {
        std::ofstream ofs(path);
        ofs << "ATOM  short line\n";  // too short (< 54 chars)
        ofs << "HETATM also short\n";
        ofs << "REMARK this is fine\n";
        // Add one valid line so it doesn't throw "no atoms found"
        char line[82];
        std::snprintf(line, sizeof(line),
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C");
        ofs << line << "\n";
        ofs << "END\n";
    }
    auto structure = tencom_pdb::read_pdb_calpha(path);
    EXPECT_EQ(structure.res_cnt, 1);
    std::remove(path.c_str());
}

TEST(PDBCalphaReader, ThrowsOnInvalidCoordinates) {
    std::string path = std::filesystem::temp_directory_path().string()
                       + "/test_bad_coords.pdb";
    {
        std::ofstream ofs(path);
        // Valid columns except coordinates are garbage
        ofs << "ATOM      1  CA  ALA A   1       XXXXXXXX  2.000   3.000  1.00  0.00           C\n";
        ofs << "END\n";
    }
    EXPECT_THROW(tencom_pdb::read_pdb_calpha(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(PDBCalphaReader, SkipsInvalidResidueNumber) {
    std::string path = std::filesystem::temp_directory_path().string()
                       + "/test_bad_resnum.pdb";
    {
        std::ofstream ofs(path);
        // Residue number field is "XXXX" — should be skipped with warning
        ofs << "ATOM      1  CA  ALA AXXXX       1.000   2.000   3.000  1.00  0.00           C\n";
        // Add one valid line
        char line[82];
        std::snprintf(line, sizeof(line),
            "ATOM      2  CA  ALA A   2       4.000   5.000   6.000  1.00  0.00           C");
        ofs << line << "\n";
        ofs << "END\n";
    }
    auto structure = tencom_pdb::read_pdb_calpha(path);
    EXPECT_EQ(structure.res_cnt, 1);
    EXPECT_NEAR(structure.atoms[1].coor[0], 4.0f, 0.01f);
    std::remove(path.c_str());
}

TEST(PDBCalphaReader, ThrowsOnAllNonBackboneAtoms) {
    std::string path = std::filesystem::temp_directory_path().string()
                       + "/test_no_backbone.pdb";
    {
        std::ofstream ofs(path);
        // CB atom — not a backbone representative
        char line[82];
        std::snprintf(line, sizeof(line),
            "ATOM      1  CB  ALA A   1       1.000   2.000   3.000  1.00  0.00           C");
        ofs << line << "\n";
        ofs << "END\n";
    }
    EXPECT_THROW(tencom_pdb::read_pdb_calpha(path), std::runtime_error);
    std::remove(path.c_str());
}

// ─── Per-Residue Decomposition Tests ─────────────────────────────────────

TEST(PerResidueDecomposition, SumsToTotalSvib) {
    auto path_ref = write_synthetic_pdb("test_prdecomp_ref", 30);
    auto path_tgt = write_synthetic_pdb("test_prdecomp_tgt", 30, 2.3f, 1.5f, 0.3f);

    auto s_ref = tencom_pdb::read_pdb_calpha(path_ref);
    auto s_tgt = tencom_pdb::read_pdb_calpha(path_tgt);

    tencm::TorsionalENM enm_ref, enm_tgt;
    enm_ref.build(s_ref.atoms.data(), s_ref.residues.data(), s_ref.res_cnt);
    enm_tgt.build(s_tgt.atoms.data(), s_tgt.residues.data(), s_tgt.res_cnt);

    auto diff = tencom_diff::compute_differential(enm_ref, enm_tgt, "ref", "tgt", 300.0);

    // Per-residue S_vib should sum to total S_vib
    ASSERT_FALSE(diff.per_residue_svib_ref.empty());
    ASSERT_FALSE(diff.per_residue_svib_tgt.empty());

    double sum_ref = 0.0, sum_tgt = 0.0;
    for (double sv : diff.per_residue_svib_ref) sum_ref += sv;
    for (double sv : diff.per_residue_svib_tgt) sum_tgt += sv;

    EXPECT_NEAR(sum_ref, diff.svib_ref.S_vib_kcal_mol_K, 1e-8);
    EXPECT_NEAR(sum_tgt, diff.svib_tgt.S_vib_kcal_mol_K, 1e-8);

    // Delta per-residue should sum to total delta
    double sum_delta = 0.0;
    for (double dsv : diff.per_residue_delta_svib) sum_delta += dsv;
    EXPECT_NEAR(sum_delta, diff.delta_S_vib, 1e-6);

    std::remove(path_ref.c_str());
    std::remove(path_tgt.c_str());
}

// ─── Vibrational Entropy Tests ──────────────────────────────────────────────

TEST(VibrationalEntropy, ReferenceEntropyPositive) {
    auto path = write_synthetic_pdb("test_svib", 50);
    auto structure = tencom_pdb::read_pdb_calpha(path);

    tencm::TorsionalENM enm;
    enm.build(structure.atoms.data(), structure.residues.data(), structure.res_cnt);

    auto encom_modes = tencom_diff::to_encom_modes(enm.modes());
    auto svib = encom::ENCoMEngine::compute_vibrational_entropy(encom_modes, 300.0);

    // Vibrational entropy at 300K should be positive
    EXPECT_GT(svib.S_vib_kcal_mol_K, 0.0);
    EXPECT_GT(svib.n_modes, 0);

    std::remove(path.c_str());
}
