// pdb_calpha.cpp — Lightweight PDB backbone reader implementation
//
// Extracts backbone representative atoms from proteins (Cα) and
// nucleic acids (C4') for use with TorsionalENM.

#include "pdb_calpha.h"

#include <fstream>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <map>

namespace tencom_pdb {

// ─── CalphaStructure memory management ──────────────────────────────────────

void CalphaStructure::free_residue_memory() {
    for (int i = 1; i <= res_cnt; ++i) {
        if (residues[i].fatm) { delete[] residues[i].fatm; residues[i].fatm = nullptr; }
        if (residues[i].latm) { delete[] residues[i].latm; residues[i].latm = nullptr; }
    }
}

CalphaStructure::~CalphaStructure() {
    free_residue_memory();
}

CalphaStructure::CalphaStructure(CalphaStructure&& o) noexcept
    : atoms(std::move(o.atoms))
    , residues(std::move(o.residues))
    , res_cnt(o.res_cnt)
    , filename(std::move(o.filename))
    , residue_types(std::move(o.residue_types))
    , n_protein(o.n_protein)
    , n_dna(o.n_dna)
    , n_rna(o.n_rna)
{
    o.res_cnt = 0;
}

CalphaStructure& CalphaStructure::operator=(CalphaStructure&& o) noexcept {
    if (this != &o) {
        free_residue_memory();
        atoms         = std::move(o.atoms);
        residues      = std::move(o.residues);
        res_cnt       = o.res_cnt;
        filename      = std::move(o.filename);
        residue_types = std::move(o.residue_types);
        n_protein     = o.n_protein;
        n_dna         = o.n_dna;
        n_rna         = o.n_rna;
        o.res_cnt = 0;
    }
    return *this;
}

// ─── Helper: check if atom name matches a backbone representative ───────────

// For proteins: " CA " or "CA  "
static bool is_ca_atom(const char* nm) {
    return (nm[0]==' ' && nm[1]=='C' && nm[2]=='A' && nm[3]==' ') ||
           (nm[0]=='C' && nm[1]=='A' && nm[2]==' ' && nm[3]==' ');
}

// For DNA/RNA: " C4'" or "C4' " — the sugar ring C4' atom
// This is the standard backbone representative for nucleic acid ENMs.
static bool is_c4prime_atom(const char* nm) {
    return (nm[0]==' ' && nm[1]=='C' && nm[2]=='4' && nm[3]=='\'') ||
           (nm[0]=='C' && nm[1]=='4' && nm[2]=='\'' && nm[3]==' ');
}

// ─── PDB parser ─────────────────────────────────────────────────────────────

CalphaStructure read_pdb_calpha(const std::string& pdb_path) {
    std::ifstream ifs(pdb_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open PDB file: " + pdb_path);
    }

    // Temporary storage per residue
    struct ResEntry {
        char        res_name[4];
        char        chain;
        int         res_number;
        float       bb_x, bb_y, bb_z;  // backbone representative coordinates
        bool        has_backbone = false;
        ResidueType type = ResidueType::UNKNOWN;
    };

    std::vector<ResEntry> res_entries;
    std::map<std::string, int> res_key_map;

    std::string line;
    while (std::getline(ifs, line)) {
        if (line.size() < 54) continue;
        // Accept both ATOM and HETATM (some modified nucleotides are HETATM)
        bool is_atom   = (line.substr(0, 6) == "ATOM  ");
        bool is_hetatm = (line.substr(0, 6) == "HETATM");
        if (!is_atom && !is_hetatm) continue;

        // Atom name (columns 13-16, 0-indexed 12-15)
        char atom_name[5] = {};
        std::strncpy(atom_name, line.c_str() + 12, 4);
        atom_name[4] = '\0';

        // Alternate location indicator (column 17)
        char altloc = (line.size() > 16) ? line[16] : ' ';
        if (altloc != ' ' && altloc != 'A') continue;

        // Residue name (columns 18-20)
        char res_name[4] = {};
        std::strncpy(res_name, line.c_str() + 17, 3);
        res_name[3] = '\0';

        // Classify residue
        ResidueType rtype = classify_residue(res_name);
        if (rtype == ResidueType::UNKNOWN) continue;

        // Check if this atom is the backbone representative for its residue type
        bool is_backbone_rep = false;
        if (rtype == ResidueType::PROTEIN) {
            is_backbone_rep = is_ca_atom(atom_name);
        } else {
            // DNA or RNA: use C4' as backbone representative
            is_backbone_rep = is_c4prime_atom(atom_name);
        }
        if (!is_backbone_rep) continue;

        // Chain (column 22)
        char chain = line[21];

        // Residue number (columns 23-26)
        int res_number = std::atoi(line.substr(22, 4).c_str());

        // Insertion code (column 27)
        char ins = (line.size() > 26) ? line[26] : ' ';

        // Coordinates (columns 31-54)
        float x = static_cast<float>(std::atof(line.substr(30, 8).c_str()));
        float y = static_cast<float>(std::atof(line.substr(38, 8).c_str()));
        float z = static_cast<float>(std::atof(line.substr(46, 8).c_str()));

        // Unique key: chain + resnum + insertion
        std::string key = std::string(1, chain) + std::to_string(res_number) + ins;

        auto it = res_key_map.find(key);
        if (it == res_key_map.end()) {
            ResEntry entry{};
            std::strncpy(entry.res_name, res_name, 3);
            entry.chain = chain;
            entry.res_number = res_number;
            entry.bb_x = x;
            entry.bb_y = y;
            entry.bb_z = z;
            entry.has_backbone = true;
            entry.type = rtype;
            res_key_map[key] = static_cast<int>(res_entries.size());
            res_entries.push_back(entry);
        }
    }

    if (res_entries.empty()) {
        throw std::runtime_error(
            "No backbone atoms found in PDB file: " + pdb_path +
            "\n  (looked for protein Cα and nucleic acid C4' atoms)");
    }

    // Build CalphaStructure with 1-based indexing
    CalphaStructure result;
    result.filename = pdb_path;
    result.res_cnt = static_cast<int>(res_entries.size());

    // Allocate: index 0 is placeholder
    result.atoms.resize(result.res_cnt + 1);
    result.residues.resize(result.res_cnt + 1);
    result.residue_types.resize(result.res_cnt + 1, ResidueType::UNKNOWN);

    // Zero-init placeholder at index 0
    std::memset(&result.atoms[0], 0, sizeof(atom));
    result.atoms[0].par = nullptr;
    result.atoms[0].cons = nullptr;
    result.atoms[0].optres = nullptr;
    result.atoms[0].eigen = nullptr;

    std::memset(&result.residues[0], 0, sizeof(resid));
    result.residues[0].fatm = nullptr;
    result.residues[0].latm = nullptr;
    result.residues[0].bonded = nullptr;
    result.residues[0].shortpath = nullptr;
    result.residues[0].shortflex = nullptr;
    result.residues[0].bond = nullptr;
    result.residues[0].gpa = nullptr;

    for (int i = 0; i < result.res_cnt; ++i) {
        const auto& entry = res_entries[i];
        int ri = i + 1;
        int ai = i + 1;

        // Track residue type
        result.residue_types[ri] = entry.type;
        switch (entry.type) {
            case ResidueType::PROTEIN: ++result.n_protein; break;
            case ResidueType::DNA:     ++result.n_dna;     break;
            case ResidueType::RNA:     ++result.n_rna;     break;
            default: break;
        }

        // Populate atom
        atom& a = result.atoms[ai];
        std::memset(&a, 0, sizeof(atom));
        a.coor[0] = entry.bb_x;
        a.coor[1] = entry.bb_y;
        a.coor[2] = entry.bb_z;
        a.coor_ori[0] = entry.bb_x;
        a.coor_ori[1] = entry.bb_y;
        a.coor_ori[2] = entry.bb_z;
        a.coor_ref = nullptr;
        a.number = ai;
        a.ofres = ri;
        a.isbb = 1;
        // Store as " CA " so TorsionalENM::extract_ca() matches it.
        // For nucleic acids this represents the C4' position.
        std::strncpy(a.name, " CA ", 4);
        a.name[4] = '\0';
        std::strncpy(a.element, "C", 2);
        a.element[2] = '\0';
        a.par = nullptr;
        a.cons = nullptr;
        a.optres = nullptr;
        a.eigen = nullptr;

        // Populate residue
        resid& r = result.residues[ri];
        std::memset(&r, 0, sizeof(resid));
        std::strncpy(r.name, entry.res_name, 3);
        r.name[3] = '\0';
        r.chn = entry.chain;
        r.number = entry.res_number;
        r.type = 0;  // must be 0 for extract_ca() to accept it
        r.trot = 1;
        r.rot = 0;

        r.fatm = new int[1];
        r.latm = new int[1];
        r.fatm[0] = ai;
        r.latm[0] = ai;

        r.bonded = nullptr;
        r.shortpath = nullptr;
        r.shortflex = nullptr;
        r.bond = nullptr;
        r.gpa = nullptr;
    }

    // Print summary
    bool has_types = result.n_protein > 0 || result.n_dna > 0 || result.n_rna > 0;
    std::cout << "  Parsed " << result.res_cnt << " residues";
    if (has_types) {
        std::cout << " (";
        bool first = true;
        if (result.n_protein > 0) { std::cout << result.n_protein << " protein"; first = false; }
        if (result.n_dna > 0)     { if (!first) std::cout << ", "; std::cout << result.n_dna << " DNA"; first = false; }
        if (result.n_rna > 0)     { if (!first) std::cout << ", "; std::cout << result.n_rna << " RNA"; }
        std::cout << ")";
    }
    std::cout << "\n";

    return result;
}

}  // namespace tencom_pdb
