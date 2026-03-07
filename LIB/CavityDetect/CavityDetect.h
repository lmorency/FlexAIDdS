// CavityDetect.h
// Native C++20 Surfnet cavity detection for FlexAIDΔS + FreeNRG
// Apache-2.0 © 2026 Le Bonhomme Pharma
// Port of LP's Get_Cleft (zero GPL contact)

#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include "flexaid.h"   // reuses atom_struct / resid_struct

namespace cavity_detect {

struct DetectedSphere {
    float center[3];
    float radius;
    int cleft_id;
};

struct DetectedCleft {
    int id;
    int label;
    std::vector<DetectedSphere> spheres;
    float volume;
    float center[3];
    float effrad;  // effective radius
};

class CavityDetector {
public:
    CavityDetector() = default;

    // Load from FlexAIDdS in-memory structures (main path)
    void load_from_fa(const atom* atoms, const resid* residues, int res_cnt);

    // Standalone PDB parser (for FreeNRG CLI / tests)
    void load_from_pdb(const std::string& pdb_file);

    // Core Surfnet detection (min_radius = probe, max_radius = upper bound)
    void detect(float min_radius = 1.5f, float max_radius = 4.0f);

    // Post-processing
    void merge_clefts();
    void sort_clefts();                    // rank by num_spheres descending
    void assign_atoms_to_clefts(float contact_threshold = 5.0f);

    // Anchor residue filtering (port of -a RESNUMCA logic)
    void filter_anchor_residues(const std::string& anchor_residues);

    // Accessors
    const std::vector<DetectedCleft>& clefts() const { return m_clefts; }

    // FlexAIDdS compatibility
    sphere* to_flexaid_spheres(int cleft_id = 1) const;   // returns linked list for generate_grid()

    // Backward compat output
    void write_sphere_pdb(const std::string& filename, int cleft_id = 1) const;

private:
    std::vector<atom> m_atoms;               // internal copy
    std::vector<DetectedCleft> m_clefts;
    float m_sphere_lwb = 1.5f;
    float m_sphere_upb = 4.0f;

    // Helper (will be OpenMP-parallel in .cpp)
    float distance(const float* a, const float* b) const;

    // Cluster a flat list of accepted probe spheres into DetectedCleft objects,
    // compute per-cleft geometry, and finalize m_clefts.
    void _cluster_and_finalize(std::vector<DetectedSphere> spheres);
};

} // namespace cavity_detect
