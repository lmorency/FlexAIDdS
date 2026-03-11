// PTMAttachment.h — Post-Translational Modification & Glycan Attachment Module
//
// Attaches PTMs (phosphorylation, sulfation, acetylation) and glycans
// (N-glycan, O-glycan) to receptor residues on the target chain.
// Each modification:
//   1. Adds atoms with GLYCAM-validated RESP partial charges
//   2. Generates conformers with phi/psi/omega torsion angles
//   3. Triggers BindingPopulation recalculation so the full ensemble
//      sees the modified target
//
// Apache-2.0 (c) 2026 Le Bonhomme Pharma
#pragma once

#include "../flexaid.h"
#include <string>
#include <vector>
#include <map>

namespace ptm {

// ─── data structures ────────────────────────────────────────────────────────

/// A single added atom from a PTM/glycan modification
struct PTMAtom {
    char   name[5];
    char   element[3];
    float  radius;
    float  resp_charge;
    int    type_hint;       // SYBYL-like type index (resolved at attachment time)
};

/// A single torsion angle conformer for a PTM/glycan
struct PTMConformer {
    double phi;             // degrees
    double psi;             // degrees
    double omega;           // degrees (0 if not applicable, e.g., phosphate)
    double weight;          // Boltzmann weight (normalised to sum=1 within mod)
};

/// Description of one PTM or glycan modification type
struct PTMDefinition {
    std::string name;
    std::string description;
    std::string linkage;              // "N-glycosidic", "O-glycosidic", "phosphoester", etc.
    std::string bond_atom;            // target atom name to bond to (e.g., "ND2", "OG", "OH")
    std::vector<std::string> attachment_residues;  // e.g., {"ASN"}, {"SER","THR"}, {"TYR"}
    std::vector<PTMAtom> added_atoms;
    std::vector<PTMConformer> conformers;
};

/// A single attachment site on the receptor
struct PTMSite {
    std::string mod_name;   // key into the definition library (e.g., "NMan9", "Phosphate")
    int   residue_index;    // internal residue index in resid[]
    int   chain_atom;       // internal atom index of the bond attachment point
    int   first_added_atom; // first atom index of added PTM atoms in atoms[]
    int   n_added_atoms;    // number of atoms added
    int   active_conformer; // which conformer is currently applied
};

/// Tracks all modifications applied to a receptor
struct PTMState {
    std::vector<PTMSite> sites;
    int total_added_atoms;
};

// ─── library ────────────────────────────────────────────────────────────────

/// Load PTM/glycan definitions from a JSON file (data/mods/glycan_conformers.json)
/// Returns a map keyed by modification name.
std::map<std::string, PTMDefinition> load_ptm_library(const char* json_path);

// ─── attachment ─────────────────────────────────────────────────────────────

/// Find the internal atom index of a named atom (e.g., "ND2") within a residue.
/// Returns -1 if not found.
int find_atom_in_residue(
    const atom* atoms,
    const resid* residue,
    int residue_index,
    const char* atom_name
);

/// Validate that a modification can be applied to the given residue.
/// Checks: residue name matches attachment_residues, bond_atom exists.
bool validate_attachment(
    const atom* atoms,
    const resid* residue,
    int residue_index,
    const PTMDefinition& def
);

/// Attach a PTM/glycan to a specific receptor residue.
/// Adds atoms to the atoms array, assigns RESP charges, and positions them
/// using the first conformer's torsion angles.
/// Returns the PTMSite descriptor (or throws on failure).
///
/// IMPORTANT: The caller must ensure atoms[] has capacity for additional atoms.
/// Use FA->MIN_NUM_ATOM headroom (FlexAID pre-allocates with margin).
PTMSite attach_modification(
    FA_Global* FA,
    atom* atoms,
    resid* residue,
    int residue_index,
    const PTMDefinition& def
);

/// Apply a specific conformer to an already-attached PTM site.
/// Rotates the added atoms using the conformer's phi/psi/omega torsions
/// relative to the attachment bond.
void apply_conformer(
    atom* atoms,
    const resid* residue,
    const PTMSite& site,
    const PTMDefinition& def,
    int conformer_index
);

// ─── integration with BindingPopulation ─────────────────────────────────────

/// Parse a PTM specification string like "ASN123:NMan9,THR45:OGalNAc"
/// Returns pairs of (residue_PDB_number, modification_name).
std::vector<std::pair<int, std::string>> parse_ptm_spec(const char* spec);

/// Resolve a PDB residue number + chain to an internal residue index.
/// Returns -1 if not found.
int resolve_residue(
    const resid* residue,
    int res_count,
    int pdb_resnum,
    char chain = ' '
);

/// Apply all PTM modifications specified in the spec string to the receptor.
/// This is the high-level entry point:
///   1. Parses spec
///   2. Loads definitions from library
///   3. Validates each site
///   4. Attaches modifications
///   5. Returns the full PTMState for downstream use
PTMState apply_target_modifications(
    FA_Global* FA,
    atom* atoms,
    resid* residue,
    const std::map<std::string, PTMDefinition>& library,
    const char* ptm_spec
);

/// Compute the conformer-weighted energy contribution of a PTM site.
/// Cycles through all conformers, evaluates each via the scoring function,
/// and returns the Boltzmann-weighted average CF shift.
double ptm_conformer_energy(
    FA_Global* FA,
    VC_Global* VC,
    atom* atoms,
    resid* residue,
    gridpoint* cleftgrid,
    const PTMSite& site,
    const PTMDefinition& def
);

/// Select a conformer index using weighted random sampling.
/// Weights are taken from def.conformers[i].weight (normalized internally).
/// Returns the selected conformer index.
int select_conformer_weighted(
    const PTMDefinition& def,
    std::mt19937& rng
);

/// Enumerate all conformers for a PTM site, score each via Vcontacts,
/// and return the Boltzmann-optimal conformer index (lowest weighted energy).
int select_best_conformer(
    FA_Global* FA,
    VC_Global* VC,
    atom* atoms,
    resid* residue,
    gridpoint* cleftgrid,
    const PTMSite& site,
    const PTMDefinition& def
);

}  // namespace ptm
