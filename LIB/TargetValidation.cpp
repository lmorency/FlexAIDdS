// TargetValidation.cpp — Structural validation checks for receptor targets
//
// Copyright 2026 Le Bonhomme Pharma
// SPDX-License-Identifier: Apache-2.0

#include "TargetValidation.h"

#include <cmath>
#include <cstring>
#include <sstream>

namespace target {
namespace validation {

// ────────────────────────────────────────────────────────────────────────
// Structural completeness
// ────────────────────────────────────────────────────────────────────────

CheckResult check_atom_count(const atom* atoms, int atm_cnt, int min_atoms)
{
    if (!atoms || atm_cnt < min_atoms) {
        std::ostringstream oss;
        oss << "Atom count " << atm_cnt << " below minimum " << min_atoms;
        return {false, oss.str()};
    }
    return {true, ""};
}

CheckResult check_residue_count(const resid* residue, int res_cnt, int min_residues)
{
    if (!residue || res_cnt < min_residues) {
        std::ostringstream oss;
        oss << "Residue count " << res_cnt << " below minimum " << min_residues;
        return {false, oss.str()};
    }
    return {true, ""};
}

CheckResult check_chain_breaks(const atom* atoms, const resid* residue,
                                int res_cnt, int atm_cnt,
                                float max_ca_dist)
{
    // Find Cα atoms and check sequential Cα-Cα distances
    std::vector<int> ca_indices;
    ca_indices.reserve(res_cnt);

    for (int i = 0; i < atm_cnt; ++i) {
        // Match " CA " (PDB standard Cα naming)
        if (std::strncmp(atoms[i].name, " CA ", 4) == 0) {
            ca_indices.push_back(i);
        }
    }

    if (ca_indices.size() < 2) {
        return {true, ""};  // not enough Cα atoms to check
    }

    int breaks = 0;
    for (size_t k = 1; k < ca_indices.size(); ++k) {
        int i = ca_indices[k - 1];
        int j = ca_indices[k];

        // Only check within same chain (chain stored on residue, not atom)
        if (residue && atoms[i].ofres >= 0 && atoms[j].ofres >= 0 &&
            atoms[i].ofres < res_cnt && atoms[j].ofres < res_cnt) {
            if (residue[atoms[i].ofres].chn != residue[atoms[j].ofres].chn) continue;
        }

        float dx = atoms[j].coor[0] - atoms[i].coor[0];
        float dy = atoms[j].coor[1] - atoms[i].coor[1];
        float dz = atoms[j].coor[2] - atoms[i].coor[2];
        float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

        if (dist > max_ca_dist) {
            ++breaks;
        }
    }

    if (breaks > 0) {
        std::ostringstream oss;
        oss << breaks << " chain break(s) detected (CA-CA distance > "
            << max_ca_dist << " A)";
        return {false, oss.str()};
    }
    return {true, ""};
}

// ────────────────────────────────────────────────────────────────────────
// Steric health
// ────────────────────────────────────────────────────────────────────────

CheckResult check_steric_clashes(const atom* atoms, int atm_cnt,
                                  float clash_threshold, int max_allowed)
{
    int clashes = 0;
    float thresh_sq = clash_threshold * clash_threshold;

    // O(n²) check — acceptable for validation (not inner loop)
    for (int i = 0; i < atm_cnt && clashes <= max_allowed; ++i) {
        for (int j = i + 1; j < atm_cnt && clashes <= max_allowed; ++j) {
            float dx = atoms[j].coor[0] - atoms[i].coor[0];
            float dy = atoms[j].coor[1] - atoms[i].coor[1];
            float dz = atoms[j].coor[2] - atoms[i].coor[2];
            float dsq = dx * dx + dy * dy + dz * dz;

            if (dsq < thresh_sq && dsq > 0.01f) {  // skip self-overlap
                ++clashes;
            }
        }
    }

    if (clashes > max_allowed) {
        std::ostringstream oss;
        oss << clashes << " steric clashes (< " << clash_threshold
            << " A), exceeds maximum " << max_allowed;
        return {false, oss.str()};
    }
    if (clashes > 0) {
        std::ostringstream oss;
        oss << clashes << " steric clashes (< " << clash_threshold
            << " A) within tolerance";
        return {true, oss.str()};  // pass with warning
    }
    return {true, ""};
}

// ────────────────────────────────────────────────────────────────────────
// Grid validity
// ────────────────────────────────────────────────────────────────────────

CheckResult check_grid_populated(int num_grd, int min_points)
{
    if (num_grd < min_points) {
        std::ostringstream oss;
        oss << "Grid has " << num_grd << " points, minimum is " << min_points;
        return {false, oss.str()};
    }
    return {true, ""};
}

// ────────────────────────────────────────────────────────────────────────
// Type assignment
// ────────────────────────────────────────────────────────────────────────

CheckResult check_types_assigned(const atom* atoms, int atm_cnt, int ntypes)
{
    int untyped = 0;
    int dummy_type = ntypes - 1;

    for (int i = 0; i < atm_cnt; ++i) {
        if (atoms[i].type == dummy_type) {
            ++untyped;
        }
    }

    if (untyped > 0) {
        std::ostringstream oss;
        oss << untyped << " atoms have unresolved type assignment (dummy type "
            << dummy_type << ")";
        // Warning, not error — some HETATM may legitimately have dummy type
        return {true, oss.str()};
    }
    return {true, ""};
}

// ────────────────────────────────────────────────────────────────────────
// Multi-model consistency
// ────────────────────────────────────────────────────────────────────────

CheckResult check_multimodel_consistency(bool multi_model, int n_models,
                                          const std::vector<std::vector<float>>& model_coords,
                                          int atm_cnt)
{
    if (!multi_model || n_models <= 1) {
        return {true, ""};  // single model, nothing to check
    }

    if (static_cast<int>(model_coords.size()) != n_models) {
        std::ostringstream oss;
        oss << "Model coordinate array size (" << model_coords.size()
            << ") does not match n_models (" << n_models << ")";
        return {false, oss.str()};
    }

    int expected_size = atm_cnt * 3;
    for (int m = 0; m < n_models; ++m) {
        if (static_cast<int>(model_coords[m].size()) != expected_size) {
            std::ostringstream oss;
            oss << "Model " << m << " has " << model_coords[m].size()
                << " coordinates, expected " << expected_size;
            return {false, oss.str()};
        }
    }

    return {true, ""};
}

// ────────────────────────────────────────────────────────────────────────
// Aggregate
// ────────────────────────────────────────────────────────────────────────

ValidationResult run_all_checks(const FA_Global* FA,
                                 const atom* atoms,
                                 const resid* residue,
                                 int num_grd)
{
    ValidationResult result{};
    result.valid = true;
    result.atom_count = FA ? FA->atm_cnt_real : 0;
    result.residue_count = FA ? FA->res_cnt : 0;
    result.grid_point_count = num_grd;
    result.clash_count = 0;
    result.untyped_atom_count = 0;
    result.has_chain_breaks = false;

    auto record = [&](const CheckResult& cr, bool is_error) {
        if (!cr.pass) {
            if (is_error) {
                result.errors.push_back(cr.message);
                result.valid = false;
            } else {
                result.warnings.push_back(cr.message);
            }
        } else if (!cr.message.empty()) {
            result.warnings.push_back(cr.message);
        }
    };

    if (!FA || !atoms) {
        result.valid = false;
        result.errors.push_back("NULL FA_Global or atoms pointer");
        return result;
    }

    // Structural completeness (errors)
    record(check_atom_count(atoms, FA->atm_cnt_real), true);
    record(check_residue_count(residue, FA->res_cnt), true);

    // Chain breaks (warning — common in PDB structures)
    auto cb = check_chain_breaks(atoms, residue, FA->res_cnt, FA->atm_cnt_real, 4.5f);
    result.has_chain_breaks = !cb.pass;
    record(cb, false);  // warning only

    // Steric clashes (error if > 50, warning otherwise)
    auto sc = check_steric_clashes(atoms, FA->atm_cnt_real, 1.5f, 50);
    record(sc, !sc.pass);

    // Grid (error if too small)
    record(check_grid_populated(num_grd), true);

    // Type assignment (warning)
    record(check_types_assigned(atoms, FA->atm_cnt_real, FA->ntypes), false);

    // Multi-model consistency (error if mismatched)
    record(check_multimodel_consistency(
        FA->multi_model, FA->n_models, FA->model_coords, FA->atm_cnt_real), true);

    return result;
}

} // namespace validation
} // namespace target
