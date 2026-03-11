// pdb_calpha.h — Lightweight PDB backbone reader for standalone tENCoM tool
//
// Parses PDB ATOM records and populates FlexAID's atom/resid structs with
// only the fields needed by TorsionalENM::extract_ca().
//
// Supports:
//   - Proteins: extracts Cα atoms
//   - DNA/RNA:  extracts C4' atoms (sugar ring backbone representative)
//   - Mixed protein + nucleic acid complexes
//
// Internally all backbone representatives are stored with atom name " CA "
// so that TorsionalENM::extract_ca() finds them without modification.
//
// Uses 1-based indexing matching FlexAID convention.
#pragma once

#include "flexaid.h"

#include <string>
#include <vector>
#include <stdexcept>

namespace tencom_pdb {

// ─── Residue type classification ────────────────────────────────────────────

enum class ResidueType {
    PROTEIN,
    DNA,
    RNA,
    UNKNOWN
};

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

// Standard DNA nucleotide codes (PDB 3-letter and 1+2 space variants)
inline bool is_dna_nucleotide(const char* name) {
    static const char* dna[] = {
        " DA"," DT"," DC"," DG",     // PDB standard (space-prefixed)
        "DA ","DT ","DC ","DG ",     // alternative formatting
        " DI",                        // inosine
        nullptr
    };
    for (const char** p = dna; *p; ++p) {
        if (strncmp(name, *p, 3) == 0) return true;
    }
    return false;
}

// Standard RNA nucleotide codes
inline bool is_rna_nucleotide(const char* name) {
    static const char* rna[] = {
        "  A","  U","  C","  G",     // PDB standard (2-space-prefixed)
        "  I",                        // inosine
        " +A"," +U"," +C"," +G",    // modified
        "A  ","U  ","C  ","G  ",     // alternative formatting
        nullptr
    };
    for (const char** p = rna; *p; ++p) {
        if (strncmp(name, *p, 3) == 0) return true;
    }
    return false;
}

// Classify residue type
inline ResidueType classify_residue(const char* name) {
    if (is_standard_amino_acid(name)) return ResidueType::PROTEIN;
    if (is_dna_nucleotide(name))      return ResidueType::DNA;
    if (is_rna_nucleotide(name))      return ResidueType::RNA;
    return ResidueType::UNKNOWN;
}

// ─── Structure container ────────────────────────────────────────────────────

// Holds a parsed PDB structure with backbone representative atoms.
// For proteins: Cα. For DNA/RNA: C4' (sugar ring).
// Arrays are 1-based: index 0 is a placeholder.
struct CalphaStructure {
    std::vector<atom>  atoms;     // 1-based (atoms[0] unused)
    std::vector<resid> residues;  // 1-based (residues[0] unused)
    int res_cnt = 0;              // number of residues (max valid index)
    std::string filename;

    // Per-residue type tracking (1-based, parallel to residues[])
    std::vector<ResidueType> residue_types;

    // Summary counts
    int n_protein = 0;
    int n_dna     = 0;
    int n_rna     = 0;

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

// Parse PDB file, extract backbone representative atoms:
//   - Protein residues: Cα atom
//   - DNA/RNA residues: C4' atom (sugar ring backbone)
// Returns CalphaStructure ready for TorsionalENM::build().
// Throws std::runtime_error on file I/O failure or no atoms found.
CalphaStructure read_pdb_calpha(const std::string& pdb_path);

}  // namespace tencom_pdb
