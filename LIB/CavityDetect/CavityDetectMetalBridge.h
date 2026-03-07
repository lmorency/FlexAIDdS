// CavityDetectMetalBridge.h — C-callable bridge to Metal GPU dispatch
// Apache-2.0 © 2026 Le Bonhomme Pharma

#pragma once
#include <vector>

namespace cavity_detect {

// Lightweight POD types shared between C++ and the Obj-C++ bridge.
struct MetalAtom {
    float pos[3];
    float radius;
};

struct MetalSphereResult {
    float center[3];
    float radius;
};

} // namespace cavity_detect

// Returns true if Metal dispatch succeeded and out_spheres is populated.
// Returns false if Metal is unavailable; caller falls back to CPU path.
bool cavity_detect_metal_dispatch(
    const cavity_detect::MetalAtom* atoms,
    int n_atoms,
    float min_radius,
    float max_radius,
    std::vector<cavity_detect::MetalSphereResult>& out_spheres);
