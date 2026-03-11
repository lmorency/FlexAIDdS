#pragma once

#include <nlohmann/json.hpp>

// Single source of truth for all FlexAIDdS parameters.
// Every key here has a sensible default — the user only needs to override
// what they want to change via a JSON config file (-c flag).
//
// Design principle: FULL FLEXIBILITY ON by default.
// Use --rigid to disable all flexibility for fast screening.

inline nlohmann::json flexaid_default_config() {
    return {
        // ── Scoring ──────────────────────────────────────────────
        {"scoring", {
            {"function", "VCT"},              // Voronoi contact scoring (best accuracy)
            {"self_consistency", "MAX"},       // A→B and B→A contact self-consistency
            {"plane_definition", "X"},         // plane definition mode
            {"normalize_area", false},
            {"accessible_surface", false},     // ACS weighting
            {"acs_weight", 1.0},
            {"solvent_penalty", 0.0f}
        }},

        // ── Optimization step sizes ──────────────────────────────
        {"optimization", {
            {"translation_step", 0.25},       // Angstroms
            {"angle_step", 5.0},              // degrees
            {"dihedral_step", 5.0},           // degrees
            {"flexible_step", 10.0},          // degrees for ligand flex bonds
            {"grid_spacing", 0.375}           // grid spacer length
        }},

        // ── Flexibility (FULL by default) ────────────────────────
        {"flexibility", {
            {"ligand_torsions", true},         // DEEFLX: dead-end elimination for ligand flex
            {"intramolecular", true},          // consider intramolecular forces
            {"intramolecular_fraction", 1.0},
            {"permeability", 1.0},             // atom permeability
            {"rotamer_permeability", 0.8},     // rotamer acceptance VDW permeability
            {"ring_conformers", true},         // LigandRingFlex sampling
            {"chirality", true},               // ChiralCenter R/S discrimination
            {"binding_site_conformations", 1}, // pbloops
            {"bonded_loops", 2},               // bloops: exclude interactions n bonds away
            {"use_flexdee", false},            // dead-end elimination for sidechains
            {"dee_clash", 0.5}
        }},

        // ── Thermodynamics ───────────────────────────────────────
        {"thermodynamics", {
            {"temperature", 300},              // Kelvin (0 = entropy off)
            {"clustering_algorithm", "CF"},    // CF, DP, or FO
            {"cluster_rmsd", 2.0}              // RMSD threshold for clustering
        }},

        // ── Genetic Algorithm ────────────────────────────────────
        {"ga", {
            {"num_chromosomes", 1000},
            {"num_generations", 500},
            {"crossover_rate", 0.8},
            {"mutation_rate", 0.03},
            {"fitness_model", "PSHARE"},
            {"reproduction_model", "BOOM"},
            {"boom_fraction", 1.0},
            {"population_init", "RANDOM"},
            {"seed", 0},                       // 0 = time-based
            {"adaptive", false},
            {"adaptive_k", {1.0, 0.5, 1.0, 0.5}},
            {"sharing_alpha", 1.0},
            {"sharing_peaks", 5.0},
            {"sharing_scale", 10.0},
            {"intragenes", false},
            {"duplicates", false},
            {"initial_mutation_prob", 0.0},
            {"end_mutation_prob", 0.0},
            {"steady_state_num", 0}
        }},

        // ── Output ───────────────────────────────────────────────
        {"output", {
            {"max_results", 10},
            {"scored_only", false},
            {"score_ligand_only", false},
            {"htp_mode", false},
            {"print_chromosomes", 10},
            {"print_interval", 1},
            {"rrg_skip", 0},
            {"output_generations", false},
            {"output_range", false},
            {"rotamer_output", false}
        }},

        // ── Protein ──────────────────────────────────────────────
        {"protein", {
            {"is_protein", true},
            {"exclude_het", false},
            {"remove_water", true},
            {"omit_buried", false}
        }},

        // ── Advanced ─────────────────────────────────────────────
        {"advanced", {
            {"vcontacts_index", false},
            {"supernode", false},
            {"force_interaction", false},
            {"interaction_factor", 5.0}
        }}
    };
}
