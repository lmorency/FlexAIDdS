// ion_utils.h — Shared metal-ion residue name lookup table
//
// Used by modify_pdb.cpp (HETATM filtering) and tencm.cpp (spring network)
// to consistently identify metal ion residues by their 3-char PDB residue name.
//
// Apache-2.0 © 2026 Le Bonhomme Pharma
#pragma once
#include <cstring>

/// Return true if the 3-char PDB residue name (may have trailing space) is a metal ion.
/// Matches the same set as assign_radii.cpp's ion table.
inline bool is_ion_resname(const char* r3) {
    static const char* t[] = {
        "MG ","ZN ","CA ","NA ","K  ","FE ","FE2","FE3",
        "CU ","CU1","CU2","MN ","CO ","NI ","CL ","BR ",
        "IOD","LI ","CD ","HG ","PB ", nullptr
    };
    for (int i = 0; t[i]; ++i)
        if (!strncmp(r3, t[i], 3)) return true;
    return false;
}
