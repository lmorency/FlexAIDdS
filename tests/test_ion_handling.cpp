// tests/test_ion_handling.cpp
// Unit tests for HETATM / ion handling in assign_radii and assign_types.
// Verifies that metal ions and organic cofactors receive correct VdW radii
// and SYBYL atom types after the HETATM else-block was filled in.
// Apache-2.0 © 2026 Le Bonhomme Pharma

#include <gtest/gtest.h>
#include "../LIB/flexaid.h"
#include <cstring>
#include <cstdlib>

// Declarations of the functions under test
void assign_radii(atom* atoms, resid* residue, int atm_cnt);
void assign_types(FA_Global* FA, atom* atoms, resid* residue, char aminofile[]);

// ===========================================================================
// Helpers
// ===========================================================================

static FA_Global make_fa(int ntypes = 40) {
    FA_Global fa;
    memset(&fa, 0, sizeof(fa));
    fa.ntypes = ntypes;
    return fa;
}

// Build a minimal single-atom HETATM residue (type=1).
// Returns atom index 1 and residue index 1 inside the provided arrays.
static void make_ion_residue(atom* atoms, resid* residue,
                              const char* resname, const char* atomname)
{
    memset(&atoms[1], 0, sizeof(atom));
    memset(&residue[1], 0, sizeof(resid));

    strncpy(residue[1].name, resname, 3);  residue[1].name[3] = '\0';
    residue[1].type = 1;  // HETATM
    residue[1].fatm  = (int*)calloc(2, sizeof(int));
    residue[1].latm  = (int*)calloc(2, sizeof(int));
    residue[1].fatm[0] = 1;
    residue[1].latm[0] = 1;

    strncpy(atoms[1].name, atomname, 4);  atoms[1].name[4] = '\0';
    atoms[1].ofres  = 1;
    atoms[1].radius = 0.0f;
    atoms[1].type   = 39;  // dummy default (ntypes-1 for ntypes=40)
    atoms[1].optres = nullptr;
}

static void free_residue_arrays(resid* residue) {
    free(residue[1].fatm);
    free(residue[1].latm);
}

// ===========================================================================
// assign_radii — ion table
// ===========================================================================

struct IonRadiusCase { const char* resname; const char* atomname; float expected_r; };

class IonRadiusTest : public ::testing::TestWithParam<IonRadiusCase> {};

TEST_P(IonRadiusTest, CorrectRadius) {
    auto [resname, atomname, expected_r] = GetParam();

    atom   atoms[2];
    resid  residue[2];
    make_ion_residue(atoms, residue, resname, atomname);

    assign_radii(atoms, residue, /*atm_cnt=*/1);

    EXPECT_NEAR(atoms[1].radius, expected_r, 0.01f)
        << "Residue " << resname << " got wrong radius";

    free_residue_arrays(residue);
}

INSTANTIATE_TEST_SUITE_P(Ions, IonRadiusTest, ::testing::Values(
    IonRadiusCase{"MG ", " MG ", 1.73f},
    IonRadiusCase{"ZN ", " ZN ", 1.39f},
    IonRadiusCase{"CA ", " CA ", 1.74f},
    IonRadiusCase{"NA ", " NA ", 2.27f},
    IonRadiusCase{"K  ", " K  ", 2.75f},
    IonRadiusCase{"FE ", " FE ", 1.47f},
    IonRadiusCase{"FE2", " FE ", 1.47f},
    IonRadiusCase{"CU ", " CU ", 1.40f},
    IonRadiusCase{"MN ", " MN ", 1.61f},
    IonRadiusCase{"CL ", " CL ", 1.75f},
    IonRadiusCase{"BR ", " BR ", 1.85f}
));

// ===========================================================================
// assign_radii — organic cofactor element fallback
// ===========================================================================

TEST(OrganicCofactorRadius, CarbonAtom) {
    atom  atoms[2]; resid residue[2];
    make_ion_residue(atoms, residue, "HEM", " C1 ");
    assign_radii(atoms, residue, 1);
    EXPECT_NEAR(atoms[1].radius, 1.88f, 0.01f);
    free_residue_arrays(residue);
}

TEST(OrganicCofactorRadius, NitrogenAtom) {
    atom  atoms[2]; resid residue[2];
    make_ion_residue(atoms, residue, "FAD", " N1 ");
    assign_radii(atoms, residue, 1);
    EXPECT_NEAR(atoms[1].radius, 1.64f, 0.01f);
    free_residue_arrays(residue);
}

TEST(OrganicCofactorRadius, OxygenAtom) {
    atom  atoms[2]; resid residue[2];
    make_ion_residue(atoms, residue, "ATP", " O1 ");
    assign_radii(atoms, residue, 1);
    EXPECT_NEAR(atoms[1].radius, 1.42f, 0.01f);
    free_residue_arrays(residue);
}

TEST(OrganicCofactorRadius, SulfurAtom) {
    atom  atoms[2]; resid residue[2];
    make_ion_residue(atoms, residue, "COA", " S1 ");
    assign_radii(atoms, residue, 1);
    EXPECT_NEAR(atoms[1].radius, 1.77f, 0.01f);
    free_residue_arrays(residue);
}

TEST(OrganicCofactorRadius, PhosphorusAtom) {
    atom  atoms[2]; resid residue[2];
    make_ion_residue(atoms, residue, "ATP", " P  ");
    assign_radii(atoms, residue, 1);
    EXPECT_NEAR(atoms[1].radius, 1.80f, 0.01f);
    free_residue_arrays(residue);
}

TEST(OrganicCofactorRadius, UnknownFallback) {
    atom  atoms[2]; resid residue[2];
    // 'X' in position 1 of name → default 1.50 Å
    make_ion_residue(atoms, residue, "UNK", " X1 ");
    assign_radii(atoms, residue, 1);
    EXPECT_NEAR(atoms[1].radius, 1.50f, 0.01f);
    free_residue_arrays(residue);
}

// ===========================================================================
// assign_radii — ATOM record (protein) is unaffected
// ===========================================================================

TEST(ProteinRadiusUnchanged, CarbonAlpha) {
    atom  atoms[2]; resid residue[2];
    memset(&atoms[1], 0, sizeof(atom));
    memset(&residue[1], 0, sizeof(resid));
    strncpy(residue[1].name, "ALA", 3);  residue[1].name[3] = '\0';
    residue[1].type = 0;  // ATOM (protein)
    residue[1].fatm = (int*)calloc(2, sizeof(int));
    residue[1].latm = (int*)calloc(2, sizeof(int));
    residue[1].fatm[0] = 1; residue[1].latm[0] = 1;
    strncpy(atoms[1].name, " CA ", 4);  atoms[1].name[4] = '\0';
    atoms[1].ofres = 1; atoms[1].radius = 0.0f;

    assign_radii(atoms, residue, 1);

    // Cα is an aliphatic carbon → radius[2] = 1.88 Å (C4)
    EXPECT_NEAR(atoms[1].radius, 1.88f, 0.01f);

    free(residue[1].fatm); free(residue[1].latm);
}

// ===========================================================================
// assign_types — SYBYL ion type assignment
// ===========================================================================

// assign_types reads from AMINO.def — pass an empty (but valid) file
// so the AMINO.def loop is a no-op and only the ion-type loop runs.
#include <cstdio>

static std::string write_empty_aminodef() {
    char tmp[] = "/tmp/test_amino_XXXXXX.def";
    // mkstemps for .def suffix
    int fd = mkstemps(tmp, 4);
    if (fd >= 0) { close(fd); }
    return std::string(tmp);
}

struct IonTypeCase { const char* resname; const char* atomname; int expected_sybyl; };

class IonTypeTest : public ::testing::TestWithParam<IonTypeCase> {};

TEST_P(IonTypeTest, CorrectSybylType) {
    auto [resname, atomname, expected_sybyl] = GetParam();

    FA_Global fa = make_fa(/*ntypes=*/40);
    atom   atoms[2];
    resid  residue[2];
    make_ion_residue(atoms, residue, resname, atomname);

    std::string aminofile = write_empty_aminodef();
    assign_types(&fa, atoms, residue, const_cast<char*>(aminofile.c_str()));
    std::remove(aminofile.c_str());

    EXPECT_EQ(atoms[1].type, expected_sybyl)
        << "Residue " << resname << " got wrong SYBYL type";

    free_residue_arrays(residue);
}

INSTANTIATE_TEST_SUITE_P(IonTypes, IonTypeTest, ::testing::Values(
    IonTypeCase{"MG ", " MG ", 28},
    IonTypeCase{"ZN ", " ZN ", 35},
    IonTypeCase{"CA ", " CA ", 36},
    IonTypeCase{"FE ", " FE ", 37},
    IonTypeCase{"FE2", " FE ", 37},
    IonTypeCase{"MN ", " MN ", 31},
    IonTypeCase{"CU ", " CU ", 30},
    IonTypeCase{"NI ", " NI ", 34}
));

TEST(IonTypeTest, FallbackWhenNtypesSmall) {
    // If FA->ntypes < required SYBYL type, type should NOT be overwritten
    FA_Global fa = make_fa(/*ntypes=*/20);
    atom  atoms[2]; resid residue[2];
    make_ion_residue(atoms, residue, "ZN ", " ZN ");
    atoms[1].type = 19;  // dummy = ntypes-1

    std::string aminofile = write_empty_aminodef();
    assign_types(&fa, atoms, residue, const_cast<char*>(aminofile.c_str()));
    std::remove(aminofile.c_str());

    // SYBYL 35 > ntypes=20, so the ion block should skip assignment
    EXPECT_EQ(atoms[1].type, 19);  // unchanged
    free_residue_arrays(residue);
}

TEST(IonTypeTest, WaterGetsHydrophilic) {
    FA_Global fa = make_fa(40);
    atom  atoms[2]; resid residue[2];
    make_ion_residue(atoms, residue, "HOH", " O  ");
    atoms[1].type = 39;

    std::string aminofile = write_empty_aminodef();
    assign_types(&fa, atoms, residue, const_cast<char*>(aminofile.c_str()));
    std::remove(aminofile.c_str());

    EXPECT_EQ(atoms[1].type, 1);  // hydrophilic
    free_residue_arrays(residue);
}
