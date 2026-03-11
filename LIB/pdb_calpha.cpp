// pdb_calpha.cpp — Lightweight PDB Cα reader implementation

#include "pdb_calpha.h"

#include <fstream>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <map>

namespace tencom_pdb {

// ─── CalphaStructure memory management ──────────────────────────────────────

void CalphaStructure::free_residue_memory() {
    // Free fatm/latm for all 1-based residues
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
{
    o.res_cnt = 0;  // prevent double-free
}

CalphaStructure& CalphaStructure::operator=(CalphaStructure&& o) noexcept {
    if (this != &o) {
        free_residue_memory();
        atoms    = std::move(o.atoms);
        residues = std::move(o.residues);
        res_cnt  = o.res_cnt;
        filename = std::move(o.filename);
        o.res_cnt = 0;
    }
    return *this;
}

// ─── PDB parser ─────────────────────────────────────────────────────────────

CalphaStructure read_pdb_calpha(const std::string& pdb_path) {
    std::ifstream ifs(pdb_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open PDB file: " + pdb_path);
    }

    // Temporary storage: residue key → (residue info, CA atom)
    struct ResEntry {
        char   res_name[4];
        char   chain;
        int    res_number;
        float  ca_x, ca_y, ca_z;
        bool   has_ca = false;
    };

    // Use insertion order via vector + map for dedup
    std::vector<ResEntry> res_entries;
    std::map<std::string, int> res_key_map;  // key → index in res_entries

    std::string line;
    while (std::getline(ifs, line)) {
        // Only process ATOM records
        if (line.size() < 54) continue;
        if (line.substr(0, 6) != "ATOM  ") continue;

        // Extract atom name (columns 13-16, 0-indexed 12-15)
        char atom_name[5] = {};
        std::strncpy(atom_name, line.c_str() + 12, 4);
        atom_name[4] = '\0';

        // Only keep CA atoms
        bool is_ca = (atom_name[0]==' ' && atom_name[1]=='C' && atom_name[2]=='A' && atom_name[3]==' ') ||
                     (atom_name[0]=='C' && atom_name[1]=='A' && atom_name[2]==' ' && atom_name[3]==' ');
        if (!is_ca) continue;

        // Alternate location indicator (column 17, 0-indexed 16)
        char altloc = (line.size() > 16) ? line[16] : ' ';
        if (altloc != ' ' && altloc != 'A') continue;  // take first altLoc only

        // Residue name (columns 18-20, 0-indexed 17-19)
        char res_name[4] = {};
        std::strncpy(res_name, line.c_str() + 17, 3);
        res_name[3] = '\0';

        if (!is_standard_amino_acid(res_name)) continue;

        // Chain (column 22, 0-indexed 21)
        char chain = line[21];

        // Residue number (columns 23-26, 0-indexed 22-25)
        int res_number = std::atoi(line.substr(22, 4).c_str());

        // Insertion code (column 27, 0-indexed 26)
        char ins = (line.size() > 26) ? line[26] : ' ';

        // Coordinates (columns 31-54, 0-indexed 30-53)
        float x = static_cast<float>(std::atof(line.substr(30, 8).c_str()));
        float y = static_cast<float>(std::atof(line.substr(38, 8).c_str()));
        float z = static_cast<float>(std::atof(line.substr(46, 8).c_str()));

        // Build unique key: chain + resnum + insertion
        std::string key = std::string(1, chain) + std::to_string(res_number) + ins;

        auto it = res_key_map.find(key);
        if (it == res_key_map.end()) {
            // New residue
            ResEntry entry{};
            std::strncpy(entry.res_name, res_name, 3);
            entry.chain = chain;
            entry.res_number = res_number;
            entry.ca_x = x;
            entry.ca_y = y;
            entry.ca_z = z;
            entry.has_ca = true;
            res_key_map[key] = static_cast<int>(res_entries.size());
            res_entries.push_back(entry);
        }
        // If duplicate, skip (first altLoc wins)
    }

    if (res_entries.empty()) {
        throw std::runtime_error("No Cα atoms found in PDB file: " + pdb_path);
    }

    // Build CalphaStructure with 1-based indexing
    CalphaStructure result;
    result.filename = pdb_path;
    result.res_cnt = static_cast<int>(res_entries.size());

    // Allocate: index 0 is placeholder
    result.atoms.resize(result.res_cnt + 1);
    result.residues.resize(result.res_cnt + 1);

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
        int ri = i + 1;  // 1-based residue index
        int ai = i + 1;  // 1-based atom index (one CA per residue)

        // Populate atom
        atom& a = result.atoms[ai];
        std::memset(&a, 0, sizeof(atom));
        a.coor[0] = entry.ca_x;
        a.coor[1] = entry.ca_y;
        a.coor[2] = entry.ca_z;
        a.coor_ori[0] = entry.ca_x;
        a.coor_ori[1] = entry.ca_y;
        a.coor_ori[2] = entry.ca_z;
        a.coor_ref = nullptr;
        a.number = ai;
        a.ofres = ri;
        a.isbb = 1;
        // Set atom name to " CA " (PDB convention)
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
        r.type = 0;  // protein
        r.trot = 1;
        r.rot = 0;

        // Allocate fatm/latm for rotamer 0
        r.fatm = new int[1];
        r.latm = new int[1];
        r.fatm[0] = ai;  // first atom = this CA
        r.latm[0] = ai;  // last atom = this CA (only one atom per residue)

        // Null out unused pointers
        r.bonded = nullptr;
        r.shortpath = nullptr;
        r.shortflex = nullptr;
        r.bond = nullptr;
        r.gpa = nullptr;
    }

    return result;
}

}  // namespace tencom_pdb
