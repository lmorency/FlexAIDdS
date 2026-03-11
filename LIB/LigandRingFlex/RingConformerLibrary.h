// RingConformerLibrary.h — Non-aromatic ring conformer library for GA
//
// Implements discrete conformer sampling for:
//   6-membered rings: chair (C), half-chair (H), boat (B), twist-boat (TB)
//   5-membered rings: envelope (E) with 5 pseudo-rotational phases
//
// Each conformer is stored as a set of ring-internal dihedral angles (degrees).
// The GA selects a conformer index; ApplyConformer() updates ring-atom coordinates.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include "flexaid.h"

namespace ring_flex {

// ─── ring types ──────────────────────────────────────────────────────────────
enum class RingSize : uint8_t { Five = 5, Six = 6 };

// Conformer category for 6-membered rings
enum class SixConformerType : uint8_t {
    Chair1 = 0,   // 4C1 (most stable)
    Chair2,       // 1C4
    BoatA,        // 2,5B
    BoatB,        // B2,5
    TwistBoatA,   // 2SO
    TwistBoatB,   // OS2
    HalfChairA,   // 3H4
    HalfChairB,   // 4H3
    COUNT
};

// Conformer category for 5-membered rings (envelope forms E0–E4)
enum class FiveConformerType : uint8_t {
    E0 = 0, E1, E2, E3, E4,
    COUNT
};

// ─── ring dihedral table ─────────────────────────────────────────────────────
// Dihedral angles (deg) for each ring atom for 6-membered ring conformers.
// Indices 0–5 correspond to ring atoms in sequence.
struct SixConformer {
    SixConformerType type;
    std::string      name;
    std::array<float, 6> dihedrals; // ν0..ν5 (degrees)
};

struct FiveConformer {
    FiveConformerType type;
    std::string       name;
    std::array<float, 5> dihedrals; // ν0..ν4 (degrees)
    float phase_P;   // Cremer-Pople phase angle (degrees)
};

// ─── library singleton ───────────────────────────────────────────────────────
class RingConformerLibrary {
public:
    static RingConformerLibrary& instance();

    const std::vector<SixConformer>&  six_conformers()  const noexcept { return six_; }
    const std::vector<FiveConformer>& five_conformers() const noexcept { return five_; }

    int n_six()  const noexcept { return static_cast<int>(six_.size()); }
    int n_five() const noexcept { return static_cast<int>(five_.size()); }

    // Random index selection (uniform)
    int random_six_index()  const;
    int random_five_index() const;

    // Retrieve by index
    const SixConformer&  six_at(int i)  const { return six_.at(i); }
    const FiveConformer& five_at(int i) const { return five_.at(i); }

private:
    RingConformerLibrary();
    std::vector<SixConformer>  six_;
    std::vector<FiveConformer> five_;

    void build_six_conformers();
    void build_five_conformers();
};

// ─── ring atom descriptor ────────────────────────────────────────────────────
// Identifies a ring in the ligand by atom indices and ring size.
struct RingDescriptor {
    std::vector<int> atom_indices; // ordered ring-atom indices in the FA atoms[] array
    RingSize         size;
    bool             is_aromatic;  // if true, skip (not sampled)
};

// Detect all non-aromatic rings in the ligand atom array.
// `atom_indices`: ordered ligand atom indices into the global atoms[] array.
// `n_atoms`: number of ligand atoms.
// `atoms`: global atom array (needed to walk bond[0..bond[0]] adjacency).
// Uses the global ::atom_struct (typedef'd as atom) from flexaid.h.
std::vector<RingDescriptor> detect_non_aromatic_rings(
    const int* atom_indices, int n_atoms,
    const ::atom* atoms);

} // namespace ring_flex
