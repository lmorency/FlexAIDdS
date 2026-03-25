// SmilesParser.h — OpenSMILES-compliant parser producing a BonMol
//
// Copyright 2026 Le Bonhomme Pharma
// SPDX-License-Identifier: Apache-2.0
//
// Handles:
//   - Organic-subset atoms: B C N O P S F Cl Br I (and lowercase aromatic c n o s p)
//   - Bracket atoms: [13CH4+], [NH4+], [nH], [O-], etc.
//   - Bond types: - = # : (aromatic) . (disconnect) and implicit single
//   - Branches:  ( and )
//   - Ring closures: digits 0-9, %10-%99
//   - Chirality: @ @@ — parsed but stereochemistry not enforced
//   - Isotopes: [13C]
//   - Hydrogen counts: [CH3], [NH2], [nH]
//   - Formal charge: [NH4+], [O-], [Ca++] / [Ca2+]
//   - Atom-map numbers: [C:1]
//
// Coordinates are left as NaN (SMILES is topology only).
// Aromaticity finalisation and typing are deferred to the pipeline.

#pragma once

#include "BonMol.h"
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <optional>
#include <vector>

namespace bonmol {

/// Exception thrown on parse error with position information.
class SmilesParseError : public std::runtime_error {
public:
    int position; ///< character position in SMILES where error occurred
    SmilesParseError(const std::string& msg, int pos)
        : std::runtime_error("SMILES parse error at position " +
                             std::to_string(pos) + ": " + msg),
          position(pos) {}
};

/// Parse result wrapping the molecule and any non-fatal warnings.
struct SmilesParseResult {
    BonMol mol;
    std::vector<std::string> warnings;
};

class SmilesParser {
public:
    SmilesParser() = default;

    /// Parse a SMILES string into a BonMol.
    /// Throws SmilesParseError on fatal errors.
    SmilesParseResult parse(const std::string& smiles);

private:
    // -----------------------------------------------------------------------
    // Internal state
    // -----------------------------------------------------------------------
    std::string      smiles_;
    int              pos_ = 0;
    BonMol           mol_;
    std::vector<std::string> warnings_;

    // Ring-closure bookkeeping: ring_open_[digit] = {atom_idx, bond_order}
    struct RingOpen {
        int       atom_idx = -1;
        BondOrder order    = BondOrder::SINGLE;
        bool      aromatic = false;
    };
    std::unordered_map<int, RingOpen> ring_open_;

    // Branch stack: each level stores the atom index to return to
    std::vector<int> branch_stack_;

    // Index of the most recently added atom (for implicit bond formation)
    int prev_atom_ = -1;

    // Pending bond between prev_atom_ and next atom (set by bond token)
    std::optional<BondOrder> pending_order_;
    bool pending_aromatic_ = false;

    // -----------------------------------------------------------------------
    // Parsing helpers
    // -----------------------------------------------------------------------
    char peek(int offset = 0) const noexcept;
    char consume() noexcept;
    bool at_end() const noexcept;

    // Parse one SMILES token. Returns false when the string is exhausted.
    bool parse_token();

    // Parse an organic-subset atom (uppercase or lowercase letter)
    int parse_organic_atom();

    // Parse a bracket atom [...]
    int parse_bracket_atom();

    // Parse an element symbol starting at current position (1 or 2 chars).
    // Advances pos_ accordingly.
    Element parse_element(bool& is_aromatic_out);

    // Parse an integer starting at current position.
    int parse_int();

    // Connect atom new_atom to prev_atom_ using pending bond info, then
    // clear pending state.
    void connect(int new_atom);

    // Handle ring closure digit (or %NN)
    void handle_ring_closure(int ring_num, BondOrder explicit_order, bool aromatic);

    // Determine default bond order between two atoms (single or aromatic).
    BondOrder default_bond_order(int atom_i, int atom_j) const noexcept;

    // Determine valence-based implicit H count for a given atom.
    int compute_implicit_h(int atom_idx) const noexcept;

    void reset();
};

} // namespace bonmol
