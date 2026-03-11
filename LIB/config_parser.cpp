#include "config_parser.h"
#include "config_defaults.h"
#include "flexaid.h"
#include "gaboom.h"

#include <fstream>
#include <stdexcept>

nlohmann::json merge_json(const nlohmann::json& a, const nlohmann::json& b) {
    nlohmann::json result = a;
    for (auto& [key, val] : b.items()) {
        if (result.contains(key) && result[key].is_object() && val.is_object()) {
            result[key] = merge_json(result[key], val);
        } else {
            result[key] = val;
        }
    }
    return result;
}

nlohmann::json load_config(const std::string& config_path) {
    nlohmann::json defaults = flexaid_default_config();

    if (config_path.empty()) {
        return defaults;
    }

    std::ifstream ifs(config_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open config file: " + config_path);
    }

    nlohmann::json user_config;
    try {
        ifs >> user_config;
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("JSON parse error in " + config_path + ": " + e.what());
    }

    return merge_json(defaults, user_config);
}

nlohmann::json rigid_overrides() {
    return {
        {"flexibility", {
            {"ligand_torsions", false},
            {"intramolecular", false},
            {"ring_conformers", false},
            {"chirality", false},
            {"use_flexdee", false},
            {"permeability", 1.0},
            {"rotamer_permeability", 1.0}
        }},
        {"thermodynamics", {
            {"temperature", 0}
        }}
    };
}

// Helper to safely get values with defaults
template<typename T>
static T jget(const nlohmann::json& j, const std::string& section, const std::string& key, T fallback) {
    if (j.contains(section) && j[section].contains(key)) {
        return j[section][key].get<T>();
    }
    return fallback;
}

void apply_config(const nlohmann::json& config, FA_Global* FA, GB_Global* GB) {

    // ── Scoring ──
    {
        auto complf = jget<std::string>(config, "scoring", "function", "VCT");
        strncpy(FA->complf, complf.c_str(), sizeof(FA->complf) - 1);

        auto sc = jget<std::string>(config, "scoring", "self_consistency", "MAX");
        strncpy(FA->vcontacts_self_consistency, sc.c_str(), sizeof(FA->vcontacts_self_consistency) - 1);

        auto pd = jget<std::string>(config, "scoring", "plane_definition", "X");
        FA->vcontacts_planedef = pd.empty() ? 'X' : pd[0];

        FA->normalize_area = jget<bool>(config, "scoring", "normalize_area", false) ? 1 : 0;
        FA->useacs = jget<bool>(config, "scoring", "accessible_surface", false) ? 1 : 0;
        FA->acsweight = jget<float>(config, "scoring", "acs_weight", 1.0f);
        FA->solventterm = jget<float>(config, "scoring", "solvent_penalty", 0.0f);
    }

    // ── Optimization ──
    {
        FA->delta_angstron = jget<double>(config, "optimization", "translation_step", 0.25);
        FA->delta_angle = jget<double>(config, "optimization", "angle_step", 5.0);
        FA->delta_dihedral = jget<double>(config, "optimization", "dihedral_step", 5.0);
        FA->delta_flexible = jget<double>(config, "optimization", "flexible_step", 10.0);
        FA->spacer_length = jget<float>(config, "optimization", "grid_spacing", 0.375f);
    }

    // ── Flexibility ──
    {
        FA->deelig_flex = jget<bool>(config, "flexibility", "ligand_torsions", true) ? 1 : 0;
        FA->intramolecular = jget<bool>(config, "flexibility", "intramolecular", true) ? 1 : 0;
        FA->intrafraction = jget<float>(config, "flexibility", "intramolecular_fraction", 1.0f);
        FA->permeability = jget<float>(config, "flexibility", "permeability", 1.0f);
        FA->rotamer_permeability = jget<float>(config, "flexibility", "rotamer_permeability", 0.8f);
        FA->pbloops = jget<int>(config, "flexibility", "binding_site_conformations", 1);
        FA->bloops = jget<int>(config, "flexibility", "bonded_loops", 2);
        FA->useflexdee = jget<bool>(config, "flexibility", "use_flexdee", false) ? 1 : 0;
        FA->dee_clash = jget<float>(config, "flexibility", "dee_clash", 0.5f);
    }

    // ── Thermodynamics ──
    {
        FA->temperature = jget<unsigned int>(config, "thermodynamics", "temperature", 300);
        if (FA->temperature > 0) {
            FA->beta = 1.0 / static_cast<double>(FA->temperature);
        } else {
            FA->beta = 0.0;
        }
        FA->cluster_rmsd = jget<float>(config, "thermodynamics", "cluster_rmsd", 2.0f);

        auto ca = jget<std::string>(config, "thermodynamics", "clustering_algorithm", "CF");
        strncpy(FA->clustering_algorithm, ca.c_str(), sizeof(FA->clustering_algorithm) - 1);
    }

    // ── GA ──
    {
        GB->num_chrom = jget<int>(config, "ga", "num_chromosomes", 1000);
        GB->max_generations = jget<int>(config, "ga", "num_generations", 500);
        GB->cross_rate = jget<double>(config, "ga", "crossover_rate", 0.8);
        GB->mut_rate = jget<double>(config, "ga", "mutation_rate", 0.03);

        auto fm = jget<std::string>(config, "ga", "fitness_model", "PSHARE");
        strncpy(GB->fitness_model, fm.c_str(), sizeof(GB->fitness_model) - 1);

        auto rm = jget<std::string>(config, "ga", "reproduction_model", "BOOM");
        strncpy(GB->rep_model, rm.c_str(), sizeof(GB->rep_model) - 1);

        GB->pbfrac = jget<double>(config, "ga", "boom_fraction", 1.0);

        auto pi = jget<std::string>(config, "ga", "population_init", "RANDOM");
        strncpy(GB->pop_init_method, pi.c_str(), sizeof(GB->pop_init_method) - 1);

        GB->seed = jget<int>(config, "ga", "seed", 0);
        GB->adaptive_ga = jget<bool>(config, "ga", "adaptive", false) ? 1 : 0;

        if (config.contains("ga") && config["ga"].contains("adaptive_k")) {
            auto& ak = config["ga"]["adaptive_k"];
            if (ak.is_array() && ak.size() >= 4) {
                GB->k1 = ak[0].get<double>();
                GB->k2 = ak[1].get<double>();
                GB->k3 = ak[2].get<double>();
                GB->k4 = ak[3].get<double>();
            }
        }

        GB->alpha = jget<double>(config, "ga", "sharing_alpha", 1.0);
        GB->peaks = jget<double>(config, "ga", "sharing_peaks", 5.0);
        GB->scale = jget<double>(config, "ga", "sharing_scale", 10.0);
        GB->intragenes = jget<bool>(config, "ga", "intragenes", false) ? 1 : 0;
        GB->duplicates = jget<bool>(config, "ga", "duplicates", false) ? 1 : 0;
        GB->ini_mut_prob = jget<double>(config, "ga", "initial_mutation_prob", 0.0);
        GB->end_mut_prob = jget<double>(config, "ga", "end_mutation_prob", 0.0);
        GB->ssnum = jget<int>(config, "ga", "steady_state_num", 0);
    }

    // ── Output ──
    {
        FA->max_results = jget<int>(config, "output", "max_results", 10);
        FA->output_scored_only = jget<bool>(config, "output", "scored_only", false) ? 1 : 0;
        FA->score_ligand_only = jget<bool>(config, "output", "score_ligand_only", false) ? 1 : 0;
        FA->htpmode = jget<bool>(config, "output", "htp_mode", false);
        GB->num_print = jget<int>(config, "output", "print_chromosomes", 10);
        GB->print_int = jget<int>(config, "output", "print_interval", 1);
        GB->rrg_skip = jget<int>(config, "output", "rrg_skip", 0);
        GB->outgen = jget<bool>(config, "output", "output_generations", false) ? 1 : 0;
        FA->rotout = jget<bool>(config, "output", "rotamer_output", false) ? 1 : 0;
    }

    // ── Protein ──
    {
        FA->is_protein = jget<bool>(config, "protein", "is_protein", true) ? 1 : 0;
        FA->exclude_het = jget<bool>(config, "protein", "exclude_het", false) ? 1 : 0;
        FA->remove_water = jget<bool>(config, "protein", "remove_water", true) ? 1 : 0;
        FA->omit_buried = jget<bool>(config, "protein", "omit_buried", false) ? 1 : 0;
    }

    // ── Advanced ──
    {
        FA->vindex = jget<bool>(config, "advanced", "vcontacts_index", false) ? 1 : 0;
        FA->supernode = jget<bool>(config, "advanced", "supernode", false) ? 1 : 0;
        FA->force_interaction = jget<bool>(config, "advanced", "force_interaction", false) ? 1 : 0;
        FA->interaction_factor = jget<float>(config, "advanced", "interaction_factor", 5.0f);
    }

    // Always set metopt to GA (the only supported method in FlexAIDdS)
    strcpy(FA->metopt, "GA");
}
