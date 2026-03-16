// config_defaults.h — Single source of truth for all FlexAIDdS parameters
//
// Every key has a sensible default — the user only needs to override
// what they want to change via a JSON config file (-c flag).
//
// Design principle: FULL FLEXIBILITY ON by default.
// Use --rigid to disable all flexibility for fast screening.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "json_value.h"

inline json::Value flexaid_default_config() {
    using V = json::Value;
    using O = json::Object;
    using A = json::Array;

    return V(O{
        // ── Scoring ──────────────────────────────────────────────
        {"scoring", V(O{
            {"function",         V("VCT")},     // Voronoi contact scoring (best accuracy)
            {"self_consistency", V("MAX")},      // A→B and B→A contact self-consistency
            {"plane_definition", V("X")},        // plane definition mode
            {"normalize_area",   V(false)},
            {"accessible_surface", V(false)},    // ACS weighting
            {"acs_weight",       V(1.0)},
            {"solvent_penalty",  V(0.0)},
        })},

        // ── Optimization step sizes ──────────────────────────────
        {"optimization", V(O{
            {"translation_step", V(0.25)},       // Angstroms
            {"angle_step",       V(5.0)},        // degrees
            {"dihedral_step",    V(5.0)},        // degrees
            {"flexible_step",    V(10.0)},       // degrees for ligand flex bonds
            {"grid_spacing",     V(0.375)},      // grid spacer length
        })},

        // ── Flexibility (FULL by default) ────────────────────────
        {"flexibility", V(O{
            {"ligand_torsions",             V(true)},   // DEEFLX: dead-end elimination for ligand flex
            {"intramolecular",              V(true)},   // consider intramolecular forces
            {"intramolecular_fraction",     V(1.0)},
            {"permeability",                V(1.0)},    // atom permeability
            {"rotamer_permeability",        V(0.8)},    // rotamer acceptance VDW permeability
            {"ring_conformers",             V(true)},   // LigandRingFlex sampling
            {"chirality",                   V(true)},   // ChiralCenter R/S discrimination
            {"binding_site_conformations",  V(1)},      // pbloops
            {"bonded_loops",                V(2)},      // bloops: exclude interactions n bonds away
            {"use_flexdee",                 V(false)},  // dead-end elimination for sidechains
            {"dee_clash",                   V(0.5)},
        })},

        // ── Thermodynamics ───────────────────────────────────────
        {"thermodynamics", V(O{
            {"temperature",           V(300)},   // Kelvin (0 = entropy off)
            {"clustering_algorithm",  V("CF")},  // CF, DP, or FO
            {"cluster_rmsd",          V(2.0)},   // RMSD threshold for clustering
        })},

        // ── Genetic Algorithm ────────────────────────────────────
        {"ga", V(O{
            {"num_chromosomes",      V(1000)},
            {"num_generations",      V(500)},
            {"crossover_rate",       V(0.8)},
            {"mutation_rate",        V(0.03)},
            {"fitness_model",        V("SMFREE")},  // SMFREE = entropy-aware (StatMech Free energy + sharing)
            {"reproduction_model",   V("BOOM")},
            {"boom_fraction",        V(1.0)},
            {"population_init",      V("RANDOM")},
            {"seed",                 V(0)},          // 0 = time-based
            {"adaptive",             V(false)},
            {"adaptive_k",           V(A{V(1.0), V(0.5), V(1.0), V(0.5)})},
            {"sharing_alpha",        V(1.0)},
            {"sharing_peaks",        V(5.0)},
            {"sharing_scale",        V(10.0)},
            {"intragenes",           V(false)},
            {"duplicates",           V(false)},
            {"initial_mutation_prob", V(0.0)},
            {"end_mutation_prob",    V(0.0)},
            {"steady_state_num",     V(0)},
            {"entropy_weight",       V(0.5)},   // SMFREE blending: 0=rank-only, 1=pure Boltzmann
            {"entropy_interval",     V(0)},     // log ensemble thermo every N gens (0=auto)
            {"use_shannon",          V(false)},  // include Shannon configurational entropy
        })},

        // ── Output ───────────────────────────────────────────────
        {"output", V(O{
            {"max_results",        V(10)},
            {"scored_only",        V(false)},
            {"score_ligand_only",  V(false)},
            {"htp_mode",           V(false)},
            {"print_chromosomes",  V(10)},
            {"print_interval",     V(1)},
            {"rrg_skip",           V(0)},
            {"output_generations", V(false)},
            {"output_range",       V(false)},
            {"rotamer_output",     V(false)},
        })},

        // ── Protein ──────────────────────────────────────────────
        {"protein", V(O{
            {"is_protein",   V(true)},
            {"exclude_het",  V(false)},
            {"remove_water", V(true)},
            {"omit_buried",  V(false)},
        })},

        // ── Advanced ─────────────────────────────────────────────
        {"advanced", V(O{
            {"vcontacts_index",    V(false)},
            {"supernode",          V(false)},
            {"force_interaction",  V(false)},
            {"interaction_factor", V(5.0)},
            {"assume_folded",      V(false)},  // skip NATURaL co-translational chain growth
        })},
    });
}

inline json::Value flexaid_rigid_overrides() {
    using V = json::Value;
    using O = json::Object;
    return V(O{
        {"flexibility", V(O{
            {"ligand_torsions",      V(false)},
            {"intramolecular",       V(false)},
            {"ring_conformers",      V(false)},
            {"chirality",            V(false)},
            {"use_flexdee",          V(false)},
            {"permeability",         V(1.0)},
            {"rotamer_permeability", V(1.0)},
        })},
        {"thermodynamics", V(O{
            {"temperature", V(0)},
        })},
    });
}
