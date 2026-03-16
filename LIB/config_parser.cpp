// config_parser.cpp — JSON config loader & applier for FlexAIDdS
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

#include "config_parser.h"
#include "config_defaults.h"
#include "flexaid.h"
#include "gaboom.h"

#include <cstring>
#include <stdexcept>

// ─── Helper: safely get a value from section.key with fallback ───────────

static bool jbool(const json::Value& cfg, const char* section, const char* key, bool fb) {
    return cfg[section][key].is_null() ? fb : cfg[section][key].as_bool(fb);
}
static int jint(const json::Value& cfg, const char* section, const char* key, int fb) {
    return cfg[section][key].is_null() ? fb : cfg[section][key].as_int(fb);
}
static double jdbl(const json::Value& cfg, const char* section, const char* key, double fb) {
    return cfg[section][key].is_null() ? fb : cfg[section][key].as_double(fb);
}
static float jflt(const json::Value& cfg, const char* section, const char* key, float fb) {
    return cfg[section][key].is_null() ? fb : cfg[section][key].as_float(fb);
}
static std::string jstr(const json::Value& cfg, const char* section, const char* key, const char* fb) {
    return cfg[section][key].is_null() ? std::string(fb) : cfg[section][key].as_string(fb);
}

// ─── Public API ──────────────────────────────────────────────────────────

json::Value load_config(const std::string& config_path) {
    json::Value defaults = flexaid_default_config();

    if (config_path.empty())
        return defaults;

    json::Value user_config = json::parse_file(config_path);
    return json::merge(defaults, user_config);
}

void apply_config(const json::Value& config, FA_Global* FA, GB_Global* GB) {

    // ── Scoring ──
    {
        auto complf = jstr(config, "scoring", "function", "VCT");
        std::strncpy(FA->complf, complf.c_str(), sizeof(FA->complf) - 1);

        auto sc = jstr(config, "scoring", "self_consistency", "MAX");
        std::strncpy(FA->vcontacts_self_consistency, sc.c_str(), sizeof(FA->vcontacts_self_consistency) - 1);

        auto pd = jstr(config, "scoring", "plane_definition", "X");
        FA->vcontacts_planedef = pd.empty() ? 'X' : pd[0];

        FA->normalize_area = jbool(config, "scoring", "normalize_area", false) ? 1 : 0;
        FA->useacs          = jbool(config, "scoring", "accessible_surface", false) ? 1 : 0;
        FA->acsweight       = jflt(config, "scoring", "acs_weight", 1.0f);
        FA->solventterm     = jflt(config, "scoring", "solvent_penalty", 0.0f);
    }

    // ── Optimization ──
    {
        FA->delta_angstron = jdbl(config, "optimization", "translation_step", 0.25);
        FA->delta_angle    = jdbl(config, "optimization", "angle_step", 5.0);
        FA->delta_dihedral = jdbl(config, "optimization", "dihedral_step", 5.0);
        FA->delta_flexible = jdbl(config, "optimization", "flexible_step", 10.0);
        FA->spacer_length  = jflt(config, "optimization", "grid_spacing", 0.375f);
    }

    // ── Flexibility ──
    {
        FA->deelig_flex          = jbool(config, "flexibility", "ligand_torsions", true) ? 1 : 0;
        FA->intramolecular       = jbool(config, "flexibility", "intramolecular", true) ? 1 : 0;
        FA->intrafraction        = jflt(config, "flexibility", "intramolecular_fraction", 1.0f);
        FA->permeability         = jflt(config, "flexibility", "permeability", 1.0f);
        FA->rotamer_permeability = jflt(config, "flexibility", "rotamer_permeability", 0.8f);
        FA->pbloops              = jint(config, "flexibility", "binding_site_conformations", 1);
        FA->bloops               = jint(config, "flexibility", "bonded_loops", 2);
        FA->useflexdee           = jbool(config, "flexibility", "use_flexdee", false) ? 1 : 0;
        FA->dee_clash            = jflt(config, "flexibility", "dee_clash", 0.5f);
    }

    // ── Thermodynamics ──
    {
        FA->temperature = static_cast<unsigned int>(jint(config, "thermodynamics", "temperature", 300));
        if (FA->temperature > 0) {
            FA->beta = 1.0 / static_cast<double>(FA->temperature);
        } else {
            FA->beta = 0.0;
        }
        FA->cluster_rmsd = jflt(config, "thermodynamics", "cluster_rmsd", 2.0f);

        auto ca = jstr(config, "thermodynamics", "clustering_algorithm", "CF");
        std::strncpy(FA->clustering_algorithm, ca.c_str(), sizeof(FA->clustering_algorithm) - 1);
    }

    // ── GA ──
    {
        GB->num_chrom        = jint(config, "ga", "num_chromosomes", 1000);
        GB->max_generations  = jint(config, "ga", "num_generations", 500);
        GB->cross_rate       = jdbl(config, "ga", "crossover_rate", 0.8);
        GB->mut_rate         = jdbl(config, "ga", "mutation_rate", 0.03);

        auto fm = jstr(config, "ga", "fitness_model", "PSHARE");
        std::strncpy(GB->fitness_model, fm.c_str(), sizeof(GB->fitness_model) - 1);

        auto rm = jstr(config, "ga", "reproduction_model", "BOOM");
        std::strncpy(GB->rep_model, rm.c_str(), sizeof(GB->rep_model) - 1);

        GB->pbfrac = jdbl(config, "ga", "boom_fraction", 1.0);

        auto pi = jstr(config, "ga", "population_init", "RANDOM");
        std::strncpy(GB->pop_init_method, pi.c_str(), sizeof(GB->pop_init_method) - 1);

        GB->seed        = jint(config, "ga", "seed", 0);
        GB->adaptive_ga = jbool(config, "ga", "adaptive", false) ? 1 : 0;

        // adaptive_k array
        const auto& ak = config["ga"]["adaptive_k"];
        if (ak.is_array() && ak.size() >= 4) {
            GB->k1 = ak[static_cast<size_t>(0)].as_double(1.0);
            GB->k2 = ak[static_cast<size_t>(1)].as_double(0.5);
            GB->k3 = ak[static_cast<size_t>(2)].as_double(1.0);
            GB->k4 = ak[static_cast<size_t>(3)].as_double(0.5);
        }

        GB->alpha       = jdbl(config, "ga", "sharing_alpha", 1.0);
        GB->peaks       = jdbl(config, "ga", "sharing_peaks", 5.0);
        GB->scale       = jdbl(config, "ga", "sharing_scale", 10.0);
        GB->intragenes  = jbool(config, "ga", "intragenes", false) ? 1 : 0;
        GB->duplicates  = jbool(config, "ga", "duplicates", false) ? 1 : 0;
        GB->ini_mut_prob = jdbl(config, "ga", "initial_mutation_prob", 0.0);
        GB->end_mut_prob = jdbl(config, "ga", "end_mutation_prob", 0.0);
        GB->ssnum            = jint(config, "ga", "steady_state_num", 0);
        GB->entropy_weight   = jdbl(config, "ga", "entropy_weight", 0.5);
        GB->entropy_interval = jint(config, "ga", "entropy_interval", 0);
        GB->use_shannon      = jbool(config, "ga", "use_shannon", false) ? 1 : 0;
    }

    // ── Output ──
    {
        FA->max_results        = jint(config, "output", "max_results", 10);
        FA->output_scored_only = jbool(config, "output", "scored_only", false) ? 1 : 0;
        FA->score_ligand_only  = jbool(config, "output", "score_ligand_only", false) ? 1 : 0;
        FA->htpmode            = jbool(config, "output", "htp_mode", false);
        GB->num_print          = jint(config, "output", "print_chromosomes", 10);
        GB->print_int          = jint(config, "output", "print_interval", 1);
        GB->rrg_skip           = jint(config, "output", "rrg_skip", 0);
        GB->outgen             = jbool(config, "output", "output_generations", false) ? 1 : 0;
        FA->rotout             = jbool(config, "output", "rotamer_output", false) ? 1 : 0;
    }

    // ── Protein ──
    {
        FA->is_protein   = jbool(config, "protein", "is_protein", true) ? 1 : 0;
        FA->exclude_het  = jbool(config, "protein", "exclude_het", false) ? 1 : 0;
        FA->remove_water = jbool(config, "protein", "remove_water", true) ? 1 : 0;
        FA->omit_buried  = jbool(config, "protein", "omit_buried", false) ? 1 : 0;
    }

    // ── Advanced ──
    {
        FA->vindex             = jbool(config, "advanced", "vcontacts_index", false) ? 1 : 0;
        FA->supernode          = jbool(config, "advanced", "supernode", false) ? 1 : 0;
        FA->force_interaction  = jbool(config, "advanced", "force_interaction", false) ? 1 : 0;
        FA->interaction_factor = jflt(config, "advanced", "interaction_factor", 5.0f);
        FA->assume_folded      = jbool(config, "advanced", "assume_folded", false) ? 1 : 0;
    }

    // Always GA
    std::strcpy(FA->metopt, "GA");
}
