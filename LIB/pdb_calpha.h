// pdb_calpha.h — Lightweight PDB Cα reader for standalone tENCoM tool
//
// Parses PDB ATOM records and populates FlexAID's atom/resid structs with
// only the fields needed by TorsionalENM::extract_ca():
//   atom: name, coor[3], ofres
//   resid: name, number, chn, type (0=protein), fatm[0], latm[0]
//
// Uses 1-based indexing matching FlexAID convention.
#pragma once

#include "flexaid.h"

#include <string>
#include <vector>
#include <stdexcept>

namespace tencom_pdb {

// Standard amino acid 3-letter codes
inline bool is_standard_amino_acid(const char* name) {
    static const char* aa[] = {
        "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
        "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
        // common variants
        "HID","HIE","HIP","CYX","MSE",
        nullptr
    };
    for (const char** p = aa; *p; ++p) {
        if (strncmp(name, *p, 3) == 0) return true;
    }
    return false;
}

// Holds a parsed PDB structure with only Cα-relevant data.
// Arrays are 1-based: index 0 is a placeholder.
struct CalphaStructure {
    std::vector<atom>  atoms;     // 1-based (atoms[0] unused)
    std::vector<resid> residues;  // 1-based (residues[0] unused)
    int res_cnt = 0;              // number of residues (max valid index)
    std::string filename;

    CalphaStructure() = default;
    ~CalphaStructure();

    // Move-only (manages fatm/latm allocations)
    CalphaStructure(CalphaStructure&& o) noexcept;
    CalphaStructure& operator=(CalphaStructure&& o) noexcept;
    CalphaStructure(const CalphaStructure&) = delete;
    CalphaStructure& operator=(const CalphaStructure&) = delete;

private:
    void free_residue_memory();
};

// Parse PDB file, extract only CA atoms of standard amino acids.
// Returns CalphaStructure ready for TorsionalENM::build().
// Throws std::runtime_error on file I/O failure.
CalphaStructure read_pdb_calpha(const std::string& pdb_path);

}  // namespace tencom_pdb
