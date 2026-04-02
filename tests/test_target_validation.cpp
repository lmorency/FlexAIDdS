// test_target_validation.cpp — Unit tests for TargetValidation
//
// Tests structural validation checks against synthetic atom/residue arrays.
//
// Copyright 2026 Le Bonhomme Pharma
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "TargetValidation.h"

#include <cstring>
#include <vector>
#include <cmath>

using namespace target::validation;

// ── Helper: create a minimal FA_Global for testing ─────────────────────

static FA_Global make_test_fa(int atm_cnt_real, int res_cnt, int ntypes = 10,
                               bool multi_model = false, int n_models = 1)
{
    FA_Global fa{};
    fa.atm_cnt_real = atm_cnt_real;
    fa.atm_cnt = atm_cnt_real;
    fa.res_cnt = res_cnt;
    fa.ntypes = ntypes;
    fa.multi_model = multi_model;
    fa.n_models = n_models;
    return fa;
}

// ── Helper: create synthetic atom array ────────────────────────────────

static std::vector<atom> make_atoms(int count, float spacing = 3.8f)
{
    std::vector<atom> atoms(count);
    for (int i = 0; i < count; ++i) {
        std::memset(&atoms[i], 0, sizeof(atom));
        atoms[i].coor[0] = i * spacing;
        atoms[i].coor[1] = 0.0f;
        atoms[i].coor[2] = 0.0f;
        atoms[i].type = 1;  // valid type
        atoms[i].ofres = i; // link atom to residue i (1:1 for Cα-only)
        std::strncpy(atoms[i].name, " CA ", 4);
        atoms[i].name[4] = '\0';
    }
    return atoms;
}

static std::vector<resid> make_residues(int count)
{
    std::vector<resid> residues(count);
    for (int i = 0; i < count; ++i) {
        std::memset(&residues[i], 0, sizeof(resid));
        std::strncpy(residues[i].name, "ALA", 3);
        residues[i].chn = 'A';
        residues[i].number = i + 1;
    }
    return residues;
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

TEST(TargetValidation, AtomCountPass) {
    auto atoms = make_atoms(50);
    auto r = check_atom_count(atoms.data(), 50, 10);
    EXPECT_TRUE(r.pass);
}

TEST(TargetValidation, AtomCountFail) {
    auto atoms = make_atoms(5);
    auto r = check_atom_count(atoms.data(), 5, 10);
    EXPECT_FALSE(r.pass);
    EXPECT_NE(r.message.find("below minimum"), std::string::npos);
}

TEST(TargetValidation, AtomCountNullptr) {
    auto r = check_atom_count(nullptr, 0, 10);
    EXPECT_FALSE(r.pass);
}

TEST(TargetValidation, ResidueCountPass) {
    auto res = make_residues(10);
    auto r = check_residue_count(res.data(), 10, 2);
    EXPECT_TRUE(r.pass);
}

TEST(TargetValidation, ResidueCountFail) {
    auto r = check_residue_count(nullptr, 0, 2);
    EXPECT_FALSE(r.pass);
}

TEST(TargetValidation, ChainBreakNone) {
    // Atoms spaced 3.8 Å apart — normal Cα-Cα distance
    auto atoms = make_atoms(10, 3.8f);
    auto res = make_residues(10);
    auto r = check_chain_breaks(atoms.data(), res.data(), 10, 10, 4.5f);
    EXPECT_TRUE(r.pass);
}

TEST(TargetValidation, ChainBreakDetected) {
    auto atoms = make_atoms(10, 3.8f);
    // Create a gap between atoms 4 and 5
    atoms[5].coor[0] = atoms[4].coor[0] + 10.0f;
    for (int i = 6; i < 10; ++i)
        atoms[i].coor[0] = atoms[5].coor[0] + (i - 5) * 3.8f;

    auto res = make_residues(10);
    auto r = check_chain_breaks(atoms.data(), res.data(), 10, 10, 4.5f);
    EXPECT_FALSE(r.pass);
    EXPECT_NE(r.message.find("chain break"), std::string::npos);
}

TEST(TargetValidation, ChainBreakDifferentChains) {
    // Gap between chains should NOT be flagged
    auto atoms = make_atoms(10, 3.8f);
    atoms[5].coor[0] = 100.0f;  // big gap
    for (int i = 6; i < 10; ++i) {
        atoms[i].coor[0] = atoms[5].coor[0] + (i - 5) * 3.8f;
    }
    auto res = make_residues(10);
    // Mark residues 5-9 as chain B (different from A)
    for (int i = 5; i < 10; ++i) {
        res[i].chn = 'B';
    }
    auto r = check_chain_breaks(atoms.data(), res.data(), 10, 10, 4.5f);
    EXPECT_TRUE(r.pass);
}

TEST(TargetValidation, StericClashNone) {
    auto atoms = make_atoms(20, 3.8f);
    auto r = check_steric_clashes(atoms.data(), 20, 1.5f, 50);
    EXPECT_TRUE(r.pass);
}

TEST(TargetValidation, StericClashDetected) {
    auto atoms = make_atoms(20, 3.8f);
    // Place atom 5 very close to atom 4
    atoms[5].coor[0] = atoms[4].coor[0] + 0.5f;
    auto r = check_steric_clashes(atoms.data(), 20, 1.5f, 0);
    EXPECT_FALSE(r.pass);
    EXPECT_NE(r.message.find("steric clash"), std::string::npos);
}

TEST(TargetValidation, GridPopulatedPass) {
    auto r = check_grid_populated(100, 10);
    EXPECT_TRUE(r.pass);
}

TEST(TargetValidation, GridPopulatedFail) {
    auto r = check_grid_populated(5, 10);
    EXPECT_FALSE(r.pass);
}

TEST(TargetValidation, TypesAssignedAllGood) {
    auto atoms = make_atoms(10);
    // type = 1, ntypes = 10 → dummy type is 9
    auto r = check_types_assigned(atoms.data(), 10, 10);
    EXPECT_TRUE(r.pass);
    EXPECT_TRUE(r.message.empty());
}

TEST(TargetValidation, TypesAssignedSomeUntyped) {
    auto atoms = make_atoms(10);
    atoms[3].type = 9;  // dummy type (ntypes - 1)
    atoms[7].type = 9;
    auto r = check_types_assigned(atoms.data(), 10, 10);
    EXPECT_TRUE(r.pass);  // warning, not error
    EXPECT_NE(r.message.find("unresolved"), std::string::npos);
}

TEST(TargetValidation, MultiModelConsistencyPass) {
    std::vector<std::vector<float>> coords(3, std::vector<float>(30, 0.0f));
    auto r = check_multimodel_consistency(true, 3, coords, 10);
    EXPECT_TRUE(r.pass);
}

TEST(TargetValidation, MultiModelConsistencySizeMismatch) {
    std::vector<std::vector<float>> coords(2, std::vector<float>(30, 0.0f));
    auto r = check_multimodel_consistency(true, 3, coords, 10);
    EXPECT_FALSE(r.pass);
    EXPECT_NE(r.message.find("does not match"), std::string::npos);
}

TEST(TargetValidation, MultiModelConsistencyCoordMismatch) {
    std::vector<std::vector<float>> coords(3, std::vector<float>(30, 0.0f));
    coords[1].resize(27);  // wrong number of coordinates
    auto r = check_multimodel_consistency(true, 3, coords, 10);
    EXPECT_FALSE(r.pass);
}

TEST(TargetValidation, MultiModelSingleModelSkipped) {
    std::vector<std::vector<float>> coords;
    auto r = check_multimodel_consistency(false, 1, coords, 10);
    EXPECT_TRUE(r.pass);
}

TEST(TargetValidation, RunAllChecksValid) {
    auto atoms = make_atoms(50, 3.8f);
    auto residues = make_residues(50);
    FA_Global fa = make_test_fa(50, 50);

    auto result = run_all_checks(&fa, atoms.data(), residues.data(), 100);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.atom_count, 50);
    EXPECT_EQ(result.residue_count, 50);
    EXPECT_EQ(result.grid_point_count, 100);
    EXPECT_TRUE(result.errors.empty());
}

TEST(TargetValidation, RunAllChecksNullFA) {
    auto result = run_all_checks(nullptr, nullptr, nullptr, 0);
    EXPECT_FALSE(result.valid);
    EXPECT_FALSE(result.errors.empty());
}

TEST(TargetValidation, RunAllChecksTooFewAtoms) {
    auto atoms = make_atoms(3, 3.8f);
    auto residues = make_residues(3);
    FA_Global fa = make_test_fa(3, 3);

    auto result = run_all_checks(&fa, atoms.data(), residues.data(), 100);
    EXPECT_FALSE(result.valid);
}
