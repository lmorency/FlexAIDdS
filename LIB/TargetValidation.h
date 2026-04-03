// TargetValidation.h — Structural validation checks for receptor targets
//
// Separated from TargetServer for independent testability.
// Each check function returns a pass/fail with a human-readable message.
// run_all_checks() aggregates all checks into a ValidationResult.
//
// Copyright 2026 Le Bonhomme Pharma
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "flexaid.h"
#include <string>
#include <vector>

namespace target {
namespace validation {

// ── Individual check result ────────────────────────────────────────────
struct CheckResult {
    bool        pass;
    std::string message;
};

// ── Aggregated validation result ───────────────────────────────────────
struct ValidationResult {
    bool valid;                         // true if no errors (warnings OK)
    int  atom_count;
    int  residue_count;
    int  grid_point_count;
    int  clash_count;                   // atom pairs closer than threshold
    int  untyped_atom_count;            // atoms that failed type assignment
    bool has_chain_breaks;
    std::vector<std::string> warnings;  // non-fatal issues
    std::vector<std::string> errors;    // fatal issues
};

// ── Structural completeness ────────────────────────────────────────────

/// Check minimum atom count (reject trivially small inputs)
CheckResult check_atom_count(const atom* atoms, int atm_cnt, int min_atoms = 10);

/// Check minimum residue count
CheckResult check_residue_count(const resid* residue, int res_cnt, int min_residues = 2);

/// Check for Cα chain breaks (Cα-Cα distance > max_ca_dist)
CheckResult check_chain_breaks(const atom* atoms, const resid* residue,
                                int res_cnt, int atm_cnt,
                                float max_ca_dist = 4.5f);

// ── Steric health ──────────────────────────────────────────────────────

/// Count atom pairs closer than clash_threshold (Å)
/// Returns number of clashes found.
CheckResult check_steric_clashes(const atom* atoms, int atm_cnt,
                                  float clash_threshold = 1.5f,
                                  int max_allowed = 50);

// ── Grid validity ──────────────────────────────────────────────────────

/// Check that the binding site grid has enough points
CheckResult check_grid_populated(int num_grd, int min_points = 10);

// ── Type assignment ────────────────────────────────────────────────────

/// Check that all real atoms have been assigned a type (not ntypes-1 dummy)
CheckResult check_types_assigned(const atom* atoms, int atm_cnt, int ntypes);

// ── Multi-model consistency ────────────────────────────────────────────

/// If multi-model, check all models have same atom count
CheckResult check_multimodel_consistency(bool multi_model, int n_models,
                                          const std::vector<std::vector<float>>& model_coords,
                                          int atm_cnt);

// ── Aggregate ──────────────────────────────────────────────────────────

/// Run all checks and return aggregated result.
/// The 'valid' field is true only if there are zero errors.
ValidationResult run_all_checks(const FA_Global* FA,
                                 const atom* atoms,
                                 const resid* residue,
                                 int num_grd);

} // namespace validation
} // namespace target
