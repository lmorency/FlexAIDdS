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

    // All eigenvalue deltas should be 0
    for (const auto& mc : diff.mode_comparisons) {
        EXPECT_NEAR(mc.delta_eigenvalue, 0.0, 1e-10);
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
