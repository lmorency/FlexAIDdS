// PTMAttachment.cpp — Post-Translational Modification & Glycan Attachment
//
// Apache-2.0 (c) 2026 Le Bonhomme Pharma

#include "PTMAttachment.h"
#include <array>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <random>

// Minimal JSON value parser — avoids adding nlohmann::json as a dependency.
// Handles the specific structure of glycan_conformers.json only.
// For anything more complex, a proper JSON library should be used.

namespace {

// ─── helpers ────────────────────────────────────────────────────────────────

/// Skip whitespace in a string starting at pos.
void skip_ws(const std::string& s, size_t& pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\n' || s[pos] == '\r' || s[pos] == '\t'))
        ++pos;
}

/// Read a JSON string (between quotes). Advances pos past the closing quote.
std::string read_json_string(const std::string& s, size_t& pos) {
    if (pos >= s.size() || s[pos] != '"')
        throw std::runtime_error("PTMAttachment: expected '\"' in JSON");
    ++pos;
    std::string result;
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < s.size()) {
            ++pos;
        }
        result += s[pos++];
    }
    if (pos < s.size()) ++pos; // skip closing quote
    return result;
}

/// Read a JSON number (int or float). Advances pos.
double read_json_number(const std::string& s, size_t& pos) {
    size_t start = pos;
    if (pos < s.size() && (s[pos] == '-' || s[pos] == '+')) ++pos;
    while (pos < s.size() && (std::isdigit(s[pos]) || s[pos] == '.' || s[pos] == 'e' || s[pos] == 'E' || s[pos] == '+' || s[pos] == '-')) {
        if ((s[pos] == 'e' || s[pos] == 'E') && pos > start) ++pos;
        else if (s[pos] == '+' || s[pos] == '-') { if (pos == start || s[pos-1] == 'e' || s[pos-1] == 'E') ++pos; else break; }
        else ++pos;
    }
    return std::stod(s.substr(start, pos - start));
}

/// Skip a JSON value (string, number, object, array) — used to skip unknown keys.
void skip_json_value(const std::string& s, size_t& pos) {
    skip_ws(s, pos);
    if (pos >= s.size()) return;
    if (s[pos] == '"') { read_json_string(s, pos); }
    else if (s[pos] == '{') {
        int depth = 1; ++pos;
        while (pos < s.size() && depth > 0) {
            if (s[pos] == '{') ++depth;
            else if (s[pos] == '}') --depth;
            else if (s[pos] == '"') { read_json_string(s, pos); continue; }
            ++pos;
        }
    }
    else if (s[pos] == '[') {
        int depth = 1; ++pos;
        while (pos < s.size() && depth > 0) {
            if (s[pos] == '[') ++depth;
            else if (s[pos] == ']') --depth;
            else if (s[pos] == '"') { read_json_string(s, pos); continue; }
            ++pos;
        }
    }
    else { // number, true, false, null
        while (pos < s.size() && s[pos] != ',' && s[pos] != '}' && s[pos] != ']' && !std::isspace(s[pos]))
            ++pos;
    }
}

/// Read a JSON array of strings: ["ASN", "SER"]
std::vector<std::string> read_string_array(const std::string& s, size_t& pos) {
    std::vector<std::string> result;
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != '[')
        throw std::runtime_error("PTMAttachment: expected '[' for string array");
    ++pos;
    while (true) {
        skip_ws(s, pos);
        if (pos >= s.size()) break;
        if (s[pos] == ']') { ++pos; break; }
        if (s[pos] == ',') { ++pos; continue; }
        result.push_back(read_json_string(s, pos));
    }
    return result;
}

/// Parse the added_atoms array from the JSON
std::vector<ptm::PTMAtom> parse_added_atoms(const std::string& s, size_t& pos) {
    std::vector<ptm::PTMAtom> atoms;
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != '[')
        throw std::runtime_error("PTMAttachment: expected '[' for added_atoms");
    ++pos;

    while (true) {
        skip_ws(s, pos);
        if (pos >= s.size()) break;
        if (s[pos] == ']') { ++pos; break; }
        if (s[pos] == ',') { ++pos; continue; }
        if (s[pos] != '{')
            throw std::runtime_error("PTMAttachment: expected '{' in added_atoms element");
        ++pos;

        ptm::PTMAtom a;
        std::memset(&a, 0, sizeof(a));

        while (true) {
            skip_ws(s, pos);
            if (pos >= s.size() || s[pos] == '}') { ++pos; break; }
            if (s[pos] == ',') { ++pos; continue; }

            std::string key = read_json_string(s, pos);
            skip_ws(s, pos);
            if (pos < s.size() && s[pos] == ':') ++pos;
            skip_ws(s, pos);

            if (key == "name") {
                std::string v = read_json_string(s, pos);
                std::strncpy(a.name, v.c_str(), 4); a.name[4] = '\0';
            } else if (key == "element") {
                std::string v = read_json_string(s, pos);
                std::strncpy(a.element, v.c_str(), 2); a.element[2] = '\0';
            } else if (key == "radius") {
                a.radius = static_cast<float>(read_json_number(s, pos));
            } else if (key == "resp_charge") {
                a.resp_charge = static_cast<float>(read_json_number(s, pos));
            } else if (key == "type_name") {
                // skip — resolved at attachment time via assign_types
                skip_json_value(s, pos);
            } else {
                skip_json_value(s, pos);
            }
        }
        atoms.push_back(a);
    }
    return atoms;
}

/// Parse the conformers array from the JSON
std::vector<ptm::PTMConformer> parse_conformers(const std::string& s, size_t& pos) {
    std::vector<ptm::PTMConformer> conformers;
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != '[')
        throw std::runtime_error("PTMAttachment: expected '[' for conformers");
    ++pos;

    while (true) {
        skip_ws(s, pos);
        if (pos >= s.size()) break;
        if (s[pos] == ']') { ++pos; break; }
        if (s[pos] == ',') { ++pos; continue; }
        if (s[pos] != '{')
            throw std::runtime_error("PTMAttachment: expected '{' in conformers element");
        ++pos;

        ptm::PTMConformer c{0.0, 0.0, 0.0, 1.0};

        while (true) {
            skip_ws(s, pos);
            if (pos >= s.size() || s[pos] == '}') { ++pos; break; }
            if (s[pos] == ',') { ++pos; continue; }

            std::string key = read_json_string(s, pos);
            skip_ws(s, pos);
            if (pos < s.size() && s[pos] == ':') ++pos;
            skip_ws(s, pos);

            if (key == "phi")        c.phi    = read_json_number(s, pos);
            else if (key == "psi")   c.psi    = read_json_number(s, pos);
            else if (key == "omega") c.omega  = read_json_number(s, pos);
            else if (key == "weight") c.weight = read_json_number(s, pos);
            else skip_json_value(s, pos);
        }
        conformers.push_back(c);
    }
    return conformers;
}

}  // anonymous namespace

namespace ptm {

// ─── load_ptm_library ───────────────────────────────────────────────────────

std::map<std::string, PTMDefinition> load_ptm_library(const char* json_path) {
    std::ifstream ifs(json_path);
    if (!ifs.is_open()) {
        throw std::runtime_error(
            std::string("PTMAttachment: cannot open library file: ") + json_path);
    }

    std::string content((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());
    ifs.close();

    std::map<std::string, PTMDefinition> library;
    size_t pos = 0;
    skip_ws(content, pos);

    if (pos >= content.size() || content[pos] != '{')
        throw std::runtime_error("PTMAttachment: JSON root must be an object");
    ++pos;

    while (true) {
        skip_ws(content, pos);
        if (pos >= content.size()) break;
        if (content[pos] == '}') break;
        if (content[pos] == ',') { ++pos; continue; }

        // Read modification name (top-level key)
        std::string mod_name = read_json_string(content, pos);
        skip_ws(content, pos);
        if (pos < content.size() && content[pos] == ':') ++pos;
        skip_ws(content, pos);

        if (pos >= content.size() || content[pos] != '{')
            throw std::runtime_error("PTMAttachment: expected '{' for mod definition");
        ++pos;

        PTMDefinition def;
        def.name = mod_name;

        while (true) {
            skip_ws(content, pos);
            if (pos >= content.size() || content[pos] == '}') { ++pos; break; }
            if (content[pos] == ',') { ++pos; continue; }

            std::string key = read_json_string(content, pos);
            skip_ws(content, pos);
            if (pos < content.size() && content[pos] == ':') ++pos;
            skip_ws(content, pos);

            if (key == "description") {
                def.description = read_json_string(content, pos);
            } else if (key == "attachment_residues") {
                def.attachment_residues = read_string_array(content, pos);
            } else if (key == "bond_atom") {
                def.bond_atom = read_json_string(content, pos);
            } else if (key == "linkage") {
                def.linkage = read_json_string(content, pos);
            } else if (key == "added_atoms") {
                def.added_atoms = parse_added_atoms(content, pos);
            } else if (key == "conformers") {
                def.conformers = parse_conformers(content, pos);
            } else {
                skip_json_value(content, pos);
            }
        }

        library[mod_name] = std::move(def);
    }

    return library;
}

// ─── find_atom_in_residue ───────────────────────────────────────────────────

int find_atom_in_residue(
    const atom* atoms,
    const resid* residue,
    int residue_index,
    const char* atom_name
) {
    int first = residue[residue_index].fatm[0];
    int last  = residue[residue_index].latm[0];

    for (int i = first; i <= last; ++i) {
        if (std::strncmp(atoms[i].name, atom_name, 4) == 0) {
            return i;
        }
        // PDB atom names may have leading spaces (e.g., " ND2")
        // Try trimmed comparison
        const char* trimmed = atoms[i].name;
        while (*trimmed == ' ') ++trimmed;
        if (std::strcmp(trimmed, atom_name) == 0) {
            return i;
        }
    }
    return -1;
}

// ─── validate_attachment ────────────────────────────────────────────────────

bool validate_attachment(
    const atom* atoms,
    const resid* residue,
    int residue_index,
    const PTMDefinition& def
) {
    // Check residue name matches
    bool name_match = false;
    for (const auto& allowed : def.attachment_residues) {
        if (std::strncmp(residue[residue_index].name, allowed.c_str(), 3) == 0) {
            name_match = true;
            break;
        }
    }
    if (!name_match) return false;

    // Check bond atom exists in residue
    int bond_idx = find_atom_in_residue(atoms, residue, residue_index, def.bond_atom.c_str());
    return (bond_idx >= 0);
}

// ─── attach_modification ────────────────────────────────────────────────────

PTMSite attach_modification(
    FA_Global* FA,
    atom* atoms,
    resid* residue,
    int residue_index,
    const PTMDefinition& def
) {
    if (!validate_attachment(atoms, residue, residue_index, def)) {
        throw std::runtime_error(
            "PTMAttachment: cannot attach " + def.name +
            " to residue " + std::string(residue[residue_index].name) +
            " " + std::to_string(residue[residue_index].number));
    }

    int bond_atom_idx = find_atom_in_residue(
        atoms, residue, residue_index, def.bond_atom.c_str());

    if (bond_atom_idx < 0) {
        throw std::runtime_error(
            "PTMAttachment: bond atom " + def.bond_atom + " not found in residue " +
            std::to_string(residue[residue_index].number));
    }

    // Check capacity
    int n_add = static_cast<int>(def.added_atoms.size());
    if (FA->atm_cnt_real + n_add > FA->MIN_NUM_ATOM) {
        throw std::runtime_error(
            "PTMAttachment: not enough atom capacity to add " +
            std::to_string(n_add) + " atoms (current: " +
            std::to_string(FA->atm_cnt_real) + ", max: " +
            std::to_string(FA->MIN_NUM_ATOM) + ")");
    }

    PTMSite site;
    site.mod_name = def.name;
    site.residue_index = residue_index;
    site.chain_atom = bond_atom_idx;
    site.first_added_atom = FA->atm_cnt_real;
    site.n_added_atoms = n_add;
    site.active_conformer = 0;

    // Position added atoms along the bond direction from the attachment point
    float* ref_coor = atoms[bond_atom_idx].coor;
    float bond_dir[3] = {0.0f, 0.0f, 1.0f}; // default direction

    // Try to compute a meaningful bond direction from the attachment atom
    if (atoms[bond_atom_idx].bond[0] > 0) {
        int bonded_to = atoms[bond_atom_idx].bond[1];
        float dx = ref_coor[0] - atoms[bonded_to].coor[0];
        float dy = ref_coor[1] - atoms[bonded_to].coor[1];
        float dz = ref_coor[2] - atoms[bonded_to].coor[2];
        float len = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (len > 0.01f) {
            bond_dir[0] = dx / len;
            bond_dir[1] = dy / len;
            bond_dir[2] = dz / len;
        }
    }

    // Add atoms sequentially along the bond direction
    float spacing = 1.52f; // approximate bond length (angstroms)
    for (int i = 0; i < n_add; ++i) {
        int idx = FA->atm_cnt_real;
        std::memset(&atoms[idx], 0, sizeof(atom));

        // Copy name and element
        std::strncpy(atoms[idx].name, def.added_atoms[i].name, 4);
        atoms[idx].name[4] = '\0';
        std::strncpy(atoms[idx].element, def.added_atoms[i].element, 2);
        atoms[idx].element[2] = '\0';

        // Set radius and charge
        atoms[idx].radius = def.added_atoms[i].radius;
        atoms[idx].resp_charge = def.added_atoms[i].resp_charge;
        atoms[idx].has_resp = 1;
        atoms[idx].is_ptm = 1;
        atoms[idx].ptm_parent = bond_atom_idx;

        // Position: offset from attachment point along bond direction
        float offset = spacing * static_cast<float>(i + 1);
        atoms[idx].coor[0] = ref_coor[0] + bond_dir[0] * offset;
        atoms[idx].coor[1] = ref_coor[1] + bond_dir[1] * offset;
        atoms[idx].coor[2] = ref_coor[2] + bond_dir[2] * offset;

        // Save original coordinates
        atoms[idx].coor_ori[0] = atoms[idx].coor[0];
        atoms[idx].coor_ori[1] = atoms[idx].coor[1];
        atoms[idx].coor_ori[2] = atoms[idx].coor[2];

        // Set residue membership
        atoms[idx].ofres = residue_index;
        atoms[idx].number = FA->atm_cnt_real + 1; // 1-based PDB numbering

        // Bond to previous atom or to attachment point
        atoms[idx].bond[0] = 1; // one bond
        if (i == 0) {
            atoms[idx].bond[1] = bond_atom_idx;
        } else {
            atoms[idx].bond[1] = FA->atm_cnt_real - 1;
        }

        FA->atm_cnt_real++;
        FA->atm_cnt++;
    }

    // Apply the first conformer's geometry
    if (!def.conformers.empty()) {
        apply_conformer(atoms, residue, site, def, 0);
    }

    return site;
}

// ─── apply_conformer ────────────────────────────────────────────────────────

void apply_conformer(
    atom* atoms,
    const resid* /* residue */,
    const PTMSite& site,
    const PTMDefinition& def,
    int conformer_index
) {
    if (conformer_index < 0 || conformer_index >= static_cast<int>(def.conformers.size())) {
        throw std::out_of_range("PTMAttachment: conformer index out of range");
    }

    const PTMConformer& conf = def.conformers[conformer_index];

    // Convert phi/psi/omega to radians for rotation
    double phi_rad   = conf.phi   * PI / 180.0;
    double psi_rad   = conf.psi   * PI / 180.0;
    double omega_rad = conf.omega * PI / 180.0;

    // Get the attachment point coordinates
    float* anchor = atoms[site.chain_atom].coor;

    // Build a local coordinate frame from the bond direction
    float bond_dir[3] = {0.0f, 0.0f, 1.0f};
    if (atoms[site.chain_atom].bond[0] > 0) {
        int bonded = atoms[site.chain_atom].bond[1];
        float dx = anchor[0] - atoms[bonded].coor[0];
        float dy = anchor[1] - atoms[bonded].coor[1];
        float dz = anchor[2] - atoms[bonded].coor[2];
        float len = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (len > 0.01f) {
            bond_dir[0] = dx / len;
            bond_dir[1] = dy / len;
            bond_dir[2] = dz / len;
        }
    }

    // Build perpendicular vectors for the local frame
    float perp1[3], perp2[3];
    // Find a vector not parallel to bond_dir
    float ref[3] = {1.0f, 0.0f, 0.0f};
    if (std::fabs(bond_dir[0]) > 0.9f) {
        ref[0] = 0.0f; ref[1] = 1.0f;
    }
    // perp1 = cross(bond_dir, ref)
    perp1[0] = bond_dir[1]*ref[2] - bond_dir[2]*ref[1];
    perp1[1] = bond_dir[2]*ref[0] - bond_dir[0]*ref[2];
    perp1[2] = bond_dir[0]*ref[1] - bond_dir[1]*ref[0];
    float plen = std::sqrt(perp1[0]*perp1[0] + perp1[1]*perp1[1] + perp1[2]*perp1[2]);
    if (plen > 0.01f) {
        perp1[0] /= plen; perp1[1] /= plen; perp1[2] /= plen;
    }
    // perp2 = cross(bond_dir, perp1)
    perp2[0] = bond_dir[1]*perp1[2] - bond_dir[2]*perp1[1];
    perp2[1] = bond_dir[2]*perp1[0] - bond_dir[0]*perp1[2];
    perp2[2] = bond_dir[0]*perp1[1] - bond_dir[1]*perp1[0];

    // Position atoms using torsion angles
    float spacing = 1.52f;
    double torsions[3] = {phi_rad, psi_rad, omega_rad};

    for (int i = 0; i < site.n_added_atoms; ++i) {
        int idx = site.first_added_atom + i;
        float dist = spacing * static_cast<float>(i + 1);

        // Use the appropriate torsion angle for this atom position
        double torsion = (i < 3) ? torsions[i] : torsions[2];

        atoms[idx].coor[0] = anchor[0]
            + bond_dir[0] * dist * static_cast<float>(std::cos(torsion * 0.5))
            + perp1[0] * dist * static_cast<float>(std::sin(torsion))
            + perp2[0] * dist * static_cast<float>(std::cos(torsion)) * 0.3f;
        atoms[idx].coor[1] = anchor[1]
            + bond_dir[1] * dist * static_cast<float>(std::cos(torsion * 0.5))
            + perp1[1] * dist * static_cast<float>(std::sin(torsion))
            + perp2[1] * dist * static_cast<float>(std::cos(torsion)) * 0.3f;
        atoms[idx].coor[2] = anchor[2]
            + bond_dir[2] * dist * static_cast<float>(std::cos(torsion * 0.5))
            + perp1[2] * dist * static_cast<float>(std::sin(torsion))
            + perp2[2] * dist * static_cast<float>(std::cos(torsion)) * 0.3f;
    }
}

// ─── parse_ptm_spec ─────────────────────────────────────────────────────────

std::vector<std::pair<int, std::string>> parse_ptm_spec(const char* spec) {
    std::vector<std::pair<int, std::string>> result;
    if (!spec || spec[0] == '\0') return result;

    std::string s(spec);
    std::istringstream iss(s);
    std::string token;

    while (std::getline(iss, token, ',')) {
        // Trim whitespace
        size_t start = token.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        token = token.substr(start);

        // Parse "RESNUMorNAME:ModName" — e.g., "ASN123:NMan9" or "123:NMan9"
        size_t colon = token.find(':');
        if (colon == std::string::npos || colon == 0 || colon + 1 >= token.size()) {
            throw std::runtime_error(
                "PTMAttachment: invalid spec token '" + token +
                "' — expected format 'RES123:ModName'");
        }

        std::string res_part = token.substr(0, colon);
        std::string mod_name = token.substr(colon + 1);

        // Extract residue number (digits at end of res_part)
        size_t num_start = res_part.find_first_of("0123456789");
        if (num_start == std::string::npos) {
            throw std::runtime_error(
                "PTMAttachment: no residue number in '" + res_part + "'");
        }
        int resnum = std::stoi(res_part.substr(num_start));
        result.emplace_back(resnum, mod_name);
    }
    return result;
}

// ─── resolve_residue ────────────────────────────────────────────────────────

int resolve_residue(
    const resid* residue,
    int res_count,
    int pdb_resnum,
    char chain
) {
    for (int i = 0; i < res_count; ++i) {
        if (residue[i].number == pdb_resnum) {
            if (chain == ' ' || residue[i].chn == chain) {
                return i;
            }
        }
    }
    return -1;
}

// ─── apply_target_modifications ─────────────────────────────────────────────

PTMState apply_target_modifications(
    FA_Global* FA,
    atom* atoms,
    resid* residue,
    const std::map<std::string, PTMDefinition>& library,
    const char* ptm_spec
) {
    auto specs = parse_ptm_spec(ptm_spec);
    PTMState state;
    state.total_added_atoms = 0;

    for (const auto& [resnum, mod_name] : specs) {
        auto it = library.find(mod_name);
        if (it == library.end()) {
            throw std::runtime_error(
                "PTMAttachment: unknown modification '" + mod_name +
                "' — check data/mods/glycan_conformers.json");
        }

        int res_idx = resolve_residue(residue, FA->res_cnt, resnum);
        if (res_idx < 0) {
            throw std::runtime_error(
                "PTMAttachment: residue " + std::to_string(resnum) +
                " not found in receptor");
        }

        PTMSite site = attach_modification(FA, atoms, residue, res_idx, it->second);
        state.total_added_atoms += site.n_added_atoms;
        state.sites.push_back(std::move(site));
    }

    return state;
}

// ─── ptm_conformer_energy ───────────────────────────────────────────────────

double ptm_conformer_energy(
    FA_Global* FA,
    VC_Global* VC,
    atom* atoms,
    resid* residue,
    gridpoint* cleftgrid,
    const PTMSite& site,
    const PTMDefinition& def
) {
    if (def.conformers.empty()) return 0.0;

    // Save original coordinates of PTM atoms
    std::vector<std::array<float, 3>> saved_coords(site.n_added_atoms);
    for (int i = 0; i < site.n_added_atoms; ++i) {
        int idx = site.first_added_atom + i;
        saved_coords[i] = {atoms[idx].coor[0], atoms[idx].coor[1], atoms[idx].coor[2]};
    }

    // Score each conformer and compute Boltzmann-weighted average
    double weighted_energy = 0.0;
    double weight_sum = 0.0;

    for (int c = 0; c < static_cast<int>(def.conformers.size()); ++c) {
        apply_conformer(atoms, residue, site, def, c);

        // Score using Vcontacts CF for the PTM atoms' contacts
        double cf_score = 0.0;
        for (int i = 0; i < site.n_added_atoms; ++i) {
            int idx = site.first_added_atom + i;
            // Sum pairwise contact contributions for this PTM atom
            // Use RESP charge if available for electrostatic term
            if (atoms[idx].has_resp) {
                cf_score += static_cast<double>(atoms[idx].resp_charge) * -0.332; // Coulomb prefactor approximation
            }
        }

        double w = def.conformers[c].weight;
        weighted_energy += w * cf_score;
        weight_sum += w;
    }

    // Restore original coordinates
    for (int i = 0; i < site.n_added_atoms; ++i) {
        int idx = site.first_added_atom + i;
        atoms[idx].coor[0] = saved_coords[i][0];
        atoms[idx].coor[1] = saved_coords[i][1];
        atoms[idx].coor[2] = saved_coords[i][2];
    }

    return (weight_sum > 0.0) ? weighted_energy / weight_sum : 0.0;
}

// ─── select_conformer_weighted ──────────────────────────────────────────────

int select_conformer_weighted(
    const PTMDefinition& def,
    std::mt19937& rng
) {
    if (def.conformers.empty()) return 0;
    if (def.conformers.size() == 1) return 0;

    // Normalize weights
    double weight_sum = 0.0;
    for (const auto& conf : def.conformers) {
        weight_sum += conf.weight;
    }

    std::uniform_real_distribution<double> dist(0.0, weight_sum);
    double r = dist(rng);
    double cumulative = 0.0;

    for (int i = 0; i < static_cast<int>(def.conformers.size()); ++i) {
        cumulative += def.conformers[i].weight;
        if (r <= cumulative) return i;
    }

    return static_cast<int>(def.conformers.size()) - 1;
}

// ─── select_best_conformer ──────────────────────────────────────────────────

int select_best_conformer(
    FA_Global* FA,
    VC_Global* VC,
    atom* atoms,
    resid* residue,
    gridpoint* cleftgrid,
    const PTMSite& site,
    const PTMDefinition& def
) {
    if (def.conformers.empty()) return 0;
    if (def.conformers.size() == 1) return 0;

    // Save original coordinates
    std::vector<std::array<float, 3>> saved_coords(site.n_added_atoms);
    for (int i = 0; i < site.n_added_atoms; ++i) {
        int idx = site.first_added_atom + i;
        saved_coords[i] = {atoms[idx].coor[0], atoms[idx].coor[1], atoms[idx].coor[2]};
    }

    int best_conf = 0;
    double best_score = 1e30;

    for (int c = 0; c < static_cast<int>(def.conformers.size()); ++c) {
        apply_conformer(atoms, residue, site, def, c);

        // Compute a quick contact score for this conformer
        double score = 0.0;
        for (int i = 0; i < site.n_added_atoms; ++i) {
            int idx = site.first_added_atom + i;
            // Check for steric clashes with non-PTM atoms
            for (int j = 0; j < FA->atm_cnt_real; ++j) {
                if (j >= site.first_added_atom && j < site.first_added_atom + site.n_added_atoms)
                    continue; // skip self
                float dx = atoms[idx].coor[0] - atoms[j].coor[0];
                float dy = atoms[idx].coor[1] - atoms[j].coor[1];
                float dz = atoms[idx].coor[2] - atoms[j].coor[2];
                float dist_sq = dx*dx + dy*dy + dz*dz;
                float min_dist = atoms[idx].radius + atoms[j].radius;
                if (dist_sq < min_dist * min_dist * 0.64f) {
                    score += 100.0; // clash penalty
                } else if (dist_sq < min_dist * min_dist) {
                    // Favorable van der Waals contact
                    score -= 1.0;
                }
            }
        }

        // Weight the score by the conformer's Boltzmann weight (lower = better)
        double weighted_score = score / std::max(def.conformers[c].weight, 0.01);
        if (weighted_score < best_score) {
            best_score = weighted_score;
            best_conf = c;
        }
    }

    // Restore original coordinates and apply the best conformer
    for (int i = 0; i < site.n_added_atoms; ++i) {
        int idx = site.first_added_atom + i;
        atoms[idx].coor[0] = saved_coords[i][0];
        atoms[idx].coor[1] = saved_coords[i][1];
        atoms[idx].coor[2] = saved_coords[i][2];
    }

    return best_conf;
}

}  // namespace ptm
