// atom_typing_256.h — 8-bit atom type encoding for 256×256 soft contact matrix
//
// Extends FlexAID's 40-type SYBYL system to 256 types:
//   Bits 0–4: base type (32 classes, superset of SYBYL)
//   Bits 5–6: AM1-BCC charge bin (4 levels: anionic, weak-neg, weak-pos, cationic)
//   Bit    7: H-bond donor/acceptor flag
//
// Includes sybyl_to_base() bridge from FlexAID's 40-type world, context-aware
// refinements for C_ar_hetadj and C_pi_bridging (NATURaL-critical for indole/
// tryptamine π-systems), and base_to_sybyl_parent() for the 256→40 projection.
#pragma once

#include <cstdint>
#include <cstring>
#include <array>
#include <cmath>

namespace atom256 {

// ─── base types (bits 0–4, 32 classes) ──────────────────────────────────────
enum BaseType : uint8_t {
    C_sp         =  0,   // C.1  — triple bond carbon
    C_sp2        =  1,   // C.2  — double bond carbon
    C_sp3        =  2,   // C.3  — alkane carbon
    C_ar         =  3,   // C.AR — aromatic carbon
    C_cat        =  4,   // C.CAT — cationic carbon
    N_sp         =  5,   // N.1
    N_sp2        =  6,   // N.2
    N_sp3        =  7,   // N.3
    N_quat       =  8,   // N.4  — quaternary ammonium
    N_ar         =  9,   // N.AR
    N_am         = 10,   // N.AM — amide nitrogen
    N_pl3        = 11,   // N.PL3 — planar sp3 nitrogen
    O_sp2        = 12,   // O.2  — carbonyl oxygen
    O_sp3        = 13,   // O.3  — hydroxyl / ether
    O_co2        = 14,   // O.CO2 — carboxylate
    O_ar         = 15,   // O.AR — aromatic oxygen
    S_sp2        = 16,   // S.2
    S_sp3        = 17,   // S.3
    S_oxide      = 18,   // S.O
    S_dioxide    = 19,   // S.O2
    S_ar         = 20,   // S.AR — aromatic sulfur
    P_sp3        = 21,   // P.3
    HAL_F        = 22,   // F
    HAL_Cl       = 23,   // CL
    HAL_Br       = 24,   // BR
    HAL_I        = 25,   // I
    C_ar_hetadj  = 26,   // aromatic C adjacent to heteroatom (indole C3a, C7a)
    C_pi_bridge  = 27,   // π-bridging carbon (tryptamine/indole bridge)
    Metal_Zn     = 28,   // ZN
    Metal_Ca     = 29,   // CA
    Metal_Fe     = 30,   // FE
    Solvent      = 31,   // SOL / DUMMY / other metals
    BASE_TYPE_COUNT = 32
};

// ─── charge bins (bits 5–6, 4 levels) ───────────────────────────────────────
enum ChargeBin : uint8_t {
    Q_ANIONIC  = 0,   // charge < -0.25
    Q_WEAK_NEG = 1,   // -0.25 <= charge < 0.0
    Q_WEAK_POS = 2,   //  0.0  <= charge < 0.25
    Q_CATIONIC = 3,   // charge >= 0.25
};

// ─── encoding / decoding ────────────────────────────────────────────────────

inline constexpr uint8_t encode(uint8_t base_type, uint8_t charge_bin,
                                 bool hbond) noexcept {
    return (static_cast<uint8_t>(hbond) << 7) |
           ((charge_bin & 0x03) << 5) |
           (base_type & 0x1F);
}

inline constexpr uint8_t get_base(uint8_t code) noexcept { return code & 0x1F; }
inline constexpr uint8_t get_charge_bin(uint8_t code) noexcept { return (code >> 5) & 0x03; }
inline constexpr bool    get_hbond(uint8_t code) noexcept { return (code >> 7) & 0x01; }

// ─── charge quantisation ────────────────────────────────────────────────────

inline ChargeBin quantise_charge(float partial_charge) noexcept {
    if (partial_charge < -0.25f) return Q_ANIONIC;
    if (partial_charge <  0.00f) return Q_WEAK_NEG;
    if (partial_charge <  0.25f) return Q_WEAK_POS;
    return Q_CATIONIC;
}

// ─── H-bond classification ──────────────────────────────────────────────────
// Donor: N-H, O-H, S-H bonds present
// Acceptor: N (lone pair), O (lone pair), F
// We mark both donors and acceptors with the same flag.

inline bool is_hbond_capable(uint8_t base_type, float partial_charge,
                              int n_hydrogens) noexcept {
    switch (base_type) {
        // Nitrogen types with H → donor; all N → acceptor
        case N_sp: case N_sp2: case N_sp3: case N_quat:
        case N_ar: case N_am: case N_pl3:
            return true;
        // Oxygen types — always H-bond capable
        case O_sp2: case O_sp3: case O_co2: case O_ar:
            return true;
        // Sulfur — only with H attached or strong charge
        case S_sp2: case S_sp3: case S_oxide: case S_ar:
            return n_hydrogens > 0 || std::fabs(partial_charge) > 0.3f;
        // Fluorine — acceptor only
        case HAL_F:
            return true;
        default:
            return false;
    }
}

// ─── SYBYL (1–40) ↔ base type (0–31) mapping ───────────────────────────────

// Forward mapping: SYBYL type → canonical base type (without context refinement)
inline uint8_t sybyl_to_base(int sybyl_type) noexcept {
    // SYBYL types are 1-indexed (1–40)
    static constexpr uint8_t table[41] = {
        Solvent,     // 0: unused (placeholder)
        C_sp,        // 1: C.1
        C_sp2,       // 2: C.2
        C_sp3,       // 3: C.3
        C_ar,        // 4: C.AR
        C_cat,       // 5: C.CAT
        N_sp,        // 6: N.1
        N_sp2,       // 7: N.2
        N_sp3,       // 8: N.3
        N_quat,      // 9: N.4
        N_ar,        // 10: N.AR
        N_am,        // 11: N.AM
        N_pl3,       // 12: N.PL3
        O_sp2,       // 13: O.2
        O_sp3,       // 14: O.3
        O_co2,       // 15: O.CO2
        O_ar,        // 16: O.AR
        S_sp2,       // 17: S.2
        S_sp3,       // 18: S.3
        S_oxide,     // 19: S.O
        S_dioxide,   // 20: S.O2
        S_ar,        // 21: S.AR
        P_sp3,       // 22: P.3
        HAL_F,       // 23: F
        HAL_Cl,      // 24: CL
        HAL_Br,      // 25: BR
        HAL_I,       // 26: I
        Solvent,     // 27: SE  → Solvent (rare)
        Solvent,     // 28: MG  → Solvent (grouped metals)
        Solvent,     // 29: SR  → Solvent
        Solvent,     // 30: CU  → Solvent
        Solvent,     // 31: MN  → Solvent
        Solvent,     // 32: HG  → Solvent
        Solvent,     // 33: CD  → Solvent
        Solvent,     // 34: NI  → Solvent
        Metal_Zn,    // 35: ZN
        Metal_Ca,    // 36: CA
        Metal_Fe,    // 37: FE
        Solvent,     // 38: CO.OH → Solvent
        Solvent,     // 39: DUMMY
        Solvent,     // 40: SOLVENT
    };
    if (sybyl_type < 0 || sybyl_type > 40) return Solvent;
    return table[sybyl_type];
}

// Reverse mapping: base type → SYBYL parent (1-indexed)
inline int base_to_sybyl_parent(uint8_t base_type) noexcept {
    static constexpr int table[32] = {
         1,  2,  3,  4,  5,        // C types
         6,  7,  8,  9, 10, 11, 12, // N types
        13, 14, 15, 16,            // O types
        17, 18, 19, 20, 21,        // S types
        22,                        // P.3
        23, 24, 25, 26,            // halogens
         4,                        // C_ar_hetadj → C.AR
         2,                        // C_pi_bridge → C.2
        35, 36, 37,                // metals
        40,                        // Solvent
    };
    if (base_type >= 32) return 40;
    return table[base_type];
}

// ─── context-aware refinement ───────────────────────────────────────────────
// Promotes C_ar to C_ar_hetadj or C_pi_bridge based on bonding environment.
// Call after initial sybyl_to_base() assignment when neighbor information is
// available.
//
// Parameters:
//   base:                initial base type from sybyl_to_base()
//   is_aromatic_carbon:  true if base == C_ar
//   has_heteroatom_neighbor: true if any bonded atom is N, O, or S
//   is_bridgehead:       true if atom is at ring junction (shared between two
//                        fused aromatic rings, e.g., indole C3a/C7a)

inline uint8_t refine_base_type(uint8_t base, bool is_aromatic_carbon,
                                 bool has_heteroatom_neighbor,
                                 bool is_bridgehead) noexcept {
    if (!is_aromatic_carbon || base != C_ar) return base;
    if (is_bridgehead) return C_pi_bridge;
    if (has_heteroatom_neighbor) return C_ar_hetadj;
    return base;
}

// ─── full encoding from SYBYL type + charge + H-bond info ──────────────────

inline uint8_t encode_from_sybyl(int sybyl_type, float partial_charge,
                                  int n_hydrogens,
                                  bool has_heteroatom_neighbor = false,
                                  bool is_bridgehead = false) noexcept {
    uint8_t base = sybyl_to_base(sybyl_type);
    bool aromatic_c = (sybyl_type == 4);  // C.AR
    base = refine_base_type(base, aromatic_c, has_heteroatom_neighbor,
                            is_bridgehead);
    ChargeBin qbin = quantise_charge(partial_charge);
    bool hb = is_hbond_capable(base, partial_charge, n_hydrogens);
    return encode(base, qbin, hb);
}

// ─── name table for debugging ───────────────────────────────────────────────

inline const char* base_type_name(uint8_t base) noexcept {
    static const char* names[32] = {
        "C.sp", "C.sp2", "C.sp3", "C.ar", "C.cat",
        "N.sp", "N.sp2", "N.sp3", "N.4", "N.ar", "N.am", "N.pl3",
        "O.sp2", "O.sp3", "O.co2", "O.ar",
        "S.sp2", "S.sp3", "S.O", "S.O2", "S.ar",
        "P.3",
        "F", "Cl", "Br", "I",
        "C.ar.het", "C.pi.br",
        "Zn", "Ca", "Fe",
        "SOL"
    };
    return (base < 32) ? names[base] : "???";
}

inline const char* charge_bin_name(uint8_t qbin) noexcept {
    static const char* names[4] = {"anion", "w-neg", "w-pos", "cation"};
    return (qbin < 4) ? names[qbin] : "???";
}

} // namespace atom256
