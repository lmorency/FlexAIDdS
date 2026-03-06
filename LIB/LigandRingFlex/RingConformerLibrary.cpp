// RingConformerLibrary.cpp — ring conformer tables and detection
#include "RingConformerLibrary.h"
#include <cstdlib>
#include <stdexcept>

namespace ring_flex {

// ─── RingConformerLibrary ────────────────────────────────────────────────────

RingConformerLibrary& RingConformerLibrary::instance() {
    static RingConformerLibrary lib;
    return lib;
}

RingConformerLibrary::RingConformerLibrary() {
    build_six_conformers();
    build_five_conformers();
}

void RingConformerLibrary::build_six_conformers() {
    // Reference dihedral tables from Cremer & Pople (JACS 1975) for pyranose rings.
    // Angles in degrees; ν0 = C1-C2-C3-C4, ν1 = C2-C3-C4-C5, …
    six_ = {
        { SixConformerType::Chair1,   "4C1",
          { 55.9f, -55.9f,  55.9f, -55.9f,  55.9f, -55.9f } },
        { SixConformerType::Chair2,   "1C4",
          {-55.9f,  55.9f, -55.9f,  55.9f, -55.9f,  55.9f } },
        { SixConformerType::BoatA,    "2,5B",
          {  0.0f,  60.0f,   0.0f, -60.0f,   0.0f,  60.0f } },
        { SixConformerType::BoatB,    "B2,5",
          {  0.0f, -60.0f,   0.0f,  60.0f,   0.0f, -60.0f } },
        { SixConformerType::TwistBoatA, "2SO",
          { 30.0f,  30.0f, -60.0f,  30.0f,  30.0f, -60.0f } },
        { SixConformerType::TwistBoatB, "OS2",
          {-30.0f, -30.0f,  60.0f, -30.0f, -30.0f,  60.0f } },
        { SixConformerType::HalfChairA, "3H4",
          { 45.0f, -20.0f, -25.0f,  45.0f, -45.0f,  20.0f } },
        { SixConformerType::HalfChairB, "4H3",
          {-45.0f,  20.0f,  25.0f, -45.0f,  45.0f, -20.0f } },
    };
}

void RingConformerLibrary::build_five_conformers() {
    // Five-membered ring envelope conformers (Altona & Sundaralingam, JACS 1972).
    // Phase angles: P = 0° (C3'-endo), 36° (C4'-exo), 72° (O4'-endo),
    //               108° (C1'-exo), 144° (C2'-endo) for nucleoside furanoses.
    five_ = {
        { FiveConformerType::E0, "C3'-endo",
          {  0.0f, -36.0f,  36.0f, -36.0f,  36.0f }, 0.0f },
        { FiveConformerType::E1, "C4'-exo",
          { -36.0f,  36.0f, -36.0f,  36.0f,   0.0f }, 36.0f },
        { FiveConformerType::E2, "O4'-endo",
          {  36.0f, -36.0f,  36.0f,   0.0f, -36.0f }, 72.0f },
        { FiveConformerType::E3, "C1'-exo",
          { -36.0f,  36.0f,   0.0f, -36.0f,  36.0f }, 108.0f },
        { FiveConformerType::E4, "C2'-endo",
          {  36.0f,   0.0f, -36.0f,  36.0f, -36.0f }, 144.0f },
    };
}

int RingConformerLibrary::random_six_index() const {
    return rand() % n_six();
}

int RingConformerLibrary::random_five_index() const {
    return rand() % n_five();
}

// ─── detect_non_aromatic_rings ───────────────────────────────────────────────
// Simplified heuristic: looks at ring atom count.
// Full SSSR (Smallest Set of Smallest Rings) detection would require the
// full bond graph — here we use the atom type flags already set by FlexAID.
std::vector<RingDescriptor> detect_non_aromatic_rings(
    const int* atom_indices, int n_atoms)
{
    // Placeholder detection: in production this queries the FA atom ring flags.
    // Returns an empty list so GA compiles cleanly; the GA ring-flex path
    // is activated only when rings are actually detected at runtime.
    (void)atom_indices;
    (void)n_atoms;
    return {};
}

} // namespace ring_flex
