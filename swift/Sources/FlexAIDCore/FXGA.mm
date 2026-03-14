// FXGA.mm — Objective-C++ implementation of the GA + BindingPopulation C shim
//
// Wraps the full FlexAID genetic algorithm lifecycle.
// The FXGAContext owns all allocated memory (FA_Global, GB_Global, etc.).
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

#include "FXGA.h"
#include "FXStatMechEngine.h"

// C++ core headers
#include "gaboom.h"
#include "fileio.h"
#include "Vcontacts.h"
#include "BindingMode.h"

#include <cstring>
#include <cstdlib>
#include <memory>

// ─── GA context: owns all FlexAID state ─────────────────────────────────────

struct FXGAContextImpl {
    FA_Global* FA = nullptr;
    GB_Global* GB = nullptr;
    VC_Global* VC = nullptr;

    chromosome* chrom = nullptr;
    chromosome* chrom_snapshot = nullptr;
    genlim* gene_lim = nullptr;
    atom* atoms = nullptr;
    resid* residue = nullptr;
    rot* rotamer = nullptr;
    gridpoint* cleftgrid = nullptr;

    int memchrom = 0;
    int n_chrom_snapshot = 0;

    // Post-run binding population (owned, created after GA + clustering)
    BindingPopulation* population = nullptr;

    char config_path[MAX_PATH__];
    char ga_path[MAX_PATH__];

    bool initialized = false;
    bool ran = false;
};

// ─── BindingPopulation / BindingMode wrappers ───────────────────────────────

struct FXBindingPopulationImpl {
    BindingPopulation* pop;  // non-owning: GA context owns the population
};

struct FXBindingModeImpl {
    BindingMode* mode;  // non-owning: population owns the modes
};

// ─── Helper: initialize FA/GB/VC defaults (mirrors top.cpp) ─────────────────

static void init_defaults(FA_Global* FA, GB_Global* GB, VC_Global* VC) {
    std::memset(FA, 0, sizeof(FA_Global));
    std::memset(GB, 0, sizeof(GB_Global));
    std::memset(VC, 0, sizeof(VC_Global));

    FA->MIN_NUM_ATOM = 1000;
    FA->MIN_NUM_RESIDUE = 250;
    FA->MIN_ROTAMER_LIBRARY_SIZE = 155;
    FA->MIN_ROTAMER = 1;
    FA->MIN_FLEX_BONDS = 5;
    FA->MIN_CLEFTGRID_POINTS = 250;
    FA->MIN_PAR = 6;
    FA->MIN_FLEX_RESIDUE = 5;
    FA->MIN_NORMAL_GRID_POINTS = 250;
    FA->MIN_OPTRES = 1;
    FA->MIN_CONSTRAINTS = 1;

    FA->vindex = 0;
    FA->rotout = 0;
    FA->num_optres = 0;
    FA->nflexbonds = 0;
    FA->normal_grid = nullptr;
    FA->supernode = 0;
    FA->eigenvector = nullptr;
    FA->psFlexDEENode = nullptr;
    FA->FlexDEE_Nodes = 0;
    FA->dee_clash = 0.5;
    FA->intrafraction = 1.0;
    FA->cluster_rmsd = 2.0f;
    FA->rotamer_permeability = 0.8;
    FA->temperature = 0;
    FA->beta = 0.0;

    FA->force_interaction = 0;
    FA->interaction_factor = 5.0;
    FA->atm_cnt = 0;
    FA->atm_cnt_real = 0;
    FA->res_cnt = 0;
    FA->nors = 0;

    FA->htpmode = false;
    FA->nrg_suite = 0;
    FA->nrg_suite_timeout = 60;
    FA->translational = 0;
    FA->refstructure = 0;
    FA->omit_buried = 0;
    FA->is_protein = 1;

    FA->delta_angstron = 0.25;
    FA->delta_angle = 5.0;
    FA->delta_dihedral = 5.0;
    FA->delta_flexible = 10.0;
    FA->delta_index = 1.0;
    FA->max_results = 10;
    FA->deelig_flex = 0;
    FA->resligand = nullptr;
    FA->useacs = 0;
    FA->acsweight = 1.0;

    GB->outgen = 0;
    FA->num_grd = 0;
    FA->exclude_het = 0;
    FA->remove_water = 1;
    FA->normalize_area = 0;

    FA->recalci = 0;
    FA->skipped = 0;
    FA->clashed = 0;

    FA->spacer_length = 0.375;
    FA->opt_grid = 0;

    FA->pbloops = 1;
    FA->bloops = 2;

    FA->rotobs = 0;
    FA->contributions = nullptr;
    FA->output_scored_only = 0;
    FA->score_ligand_only = 0;
    FA->permeability = 1.0;
    FA->intramolecular = 1;
    FA->solventterm = 0.0f;

    FA->useflexdee = 0;
    FA->num_constraints = 0;
    FA->npar = 0;

    FA->mov[0] = nullptr;
    FA->mov[1] = nullptr;
    std::strcpy(FA->clustering_algorithm, "CF");
    std::strcpy(FA->vcontacts_self_consistency, "MAX");
    FA->vcontacts_planedef = 'X';
    std::strcpy(FA->base_path, ".");
}

// ─── GA lifecycle ───────────────────────────────────────────────────────────

extern "C" FXGAContextRef fx_ga_create(const char* config_path, const char* ga_path) {
    if (!config_path || !ga_path) return nullptr;

    auto ctx = new FXGAContextImpl();
    std::strncpy(ctx->config_path, config_path, MAX_PATH__ - 1);
    std::strncpy(ctx->ga_path, ga_path, MAX_PATH__ - 1);

    // Allocate global structs
    ctx->FA = (FA_Global*)std::malloc(sizeof(FA_Global));
    ctx->GB = (GB_Global*)std::malloc(sizeof(GB_Global));
    ctx->VC = (VC_Global*)std::malloc(sizeof(VC_Global));

    if (!ctx->FA || !ctx->GB || !ctx->VC) {
        fx_ga_destroy(ctx);
        return nullptr;
    }

    init_defaults(ctx->FA, ctx->GB, ctx->VC);

    // Allocate contacts array
    ctx->FA->contacts = (int*)std::malloc(100000 * sizeof(int));
    if (!ctx->FA->contacts) {
        fx_ga_destroy(ctx);
        return nullptr;
    }

    // Allocate Vcontacts workspace
    ctx->VC->ptorder = (ptindex*)std::malloc(MAX_PT * sizeof(ptindex));
    ctx->VC->centerpt = (vertex*)std::malloc(MAX_PT * sizeof(vertex));
    ctx->VC->poly = (vertex*)std::malloc(MAX_POLY * sizeof(vertex));
    ctx->VC->cont = (plane*)std::malloc(MAX_PT * sizeof(plane));
    ctx->VC->vedge = (edgevector*)std::malloc(MAX_POLY * sizeof(edgevector));

    if (!ctx->VC->ptorder || !ctx->VC->centerpt || !ctx->VC->poly ||
        !ctx->VC->cont || !ctx->VC->vedge) {
        fx_ga_destroy(ctx);
        return nullptr;
    }
    ctx->VC->recalc = 1;

    // Sphere initialization
    wif083(ctx->FA);

    // Parameter allocations
    ctx->FA->map_par = (optmap*)std::malloc(ctx->FA->MIN_PAR * sizeof(optmap));
    ctx->FA->opt_par = (double*)std::malloc(ctx->FA->MIN_PAR * sizeof(double));
    ctx->FA->del_opt_par = (double*)std::malloc(ctx->FA->MIN_PAR * sizeof(double));
    ctx->FA->min_opt_par = (double*)std::malloc(ctx->FA->MIN_PAR * sizeof(double));
    ctx->FA->max_opt_par = (double*)std::malloc(ctx->FA->MIN_PAR * sizeof(double));
    ctx->FA->map_opt_par = (int*)std::malloc(ctx->FA->MIN_PAR * sizeof(int));

    if (!ctx->FA->map_par || !ctx->FA->opt_par || !ctx->FA->del_opt_par ||
        !ctx->FA->min_opt_par || !ctx->FA->max_opt_par || !ctx->FA->map_opt_par) {
        fx_ga_destroy(ctx);
        return nullptr;
    }

    std::memset(ctx->FA->map_par, 0, ctx->FA->MIN_PAR * sizeof(optmap));
    std::memset(ctx->FA->opt_par, 0, ctx->FA->MIN_PAR * sizeof(double));
    std::memset(ctx->FA->del_opt_par, 0, ctx->FA->MIN_PAR * sizeof(double));
    std::memset(ctx->FA->min_opt_par, 0, ctx->FA->MIN_PAR * sizeof(double));
    std::memset(ctx->FA->max_opt_par, 0, ctx->FA->MIN_PAR * sizeof(double));

    ctx->FA->map_par_flexbond_first_index = -1;
    ctx->FA->map_par_flexbond_first = nullptr;
    ctx->FA->map_par_flexbond_last = nullptr;
    ctx->FA->map_par_sidechain_first_index = -1;
    ctx->FA->map_par_sidechain_first = nullptr;
    ctx->FA->map_par_sidechain_last = nullptr;

    // Read input files
    read_input(ctx->FA, &ctx->atoms, &ctx->residue, &ctx->rotamer, &ctx->cleftgrid, ctx->config_path);

    // Initialize Vcontacts if using VCT complement function
    if (std::strcmp(ctx->FA->complf, "VCT") == 0) {
        ctx->VC->planedef = ctx->FA->vcontacts_planedef;
        ctx->VC->Calc = (atomsas*)std::malloc(ctx->FA->atm_cnt_real * sizeof(atomsas));
        ctx->VC->Calclist = (int*)std::malloc(ctx->FA->atm_cnt_real * sizeof(int));
        ctx->VC->ca_index = (int*)std::malloc(ctx->FA->atm_cnt_real * sizeof(int));
        ctx->VC->seed = (int*)std::malloc(3 * ctx->FA->atm_cnt_real * sizeof(int));
        ctx->VC->contlist = (contactlist*)std::malloc(10000 * sizeof(contactlist));
        ctx->VC->ca_recsize = 5 * ctx->FA->atm_cnt_real;
        ctx->VC->ca_rec = (ca_struct*)std::malloc(ctx->VC->ca_recsize * sizeof(ca_struct));

        if (ctx->VC->Calc) {
            for (int i = 0; i < ctx->FA->atm_cnt_real; i++) {
                ctx->VC->Calc[i].atom = nullptr;
                ctx->VC->Calc[i].residue = nullptr;
                ctx->VC->Calc[i].exposed = true;
            }
        }
    }

    // DEE root node
    ctx->FA->deelig_root_node = new struct deelig_node_struct;
    if (ctx->FA->deelig_root_node) {
        ctx->FA->deelig_root_node->parent = nullptr;
    }

    // Contributions
    ctx->FA->contributions = (float*)std::malloc(ctx->FA->ntypes * ctx->FA->ntypes * sizeof(float));

    // Build rebuild list
    create_rebuild_list(ctx->FA, ctx->atoms, ctx->residue);

    ctx->initialized = true;
    return ctx;
}

extern "C" int fx_ga_run(FXGAContextRef context) {
    if (!context || !context->initialized) return -1;
    if (context->ran) return -2;  // already ran

    // Read GA parameters
    int num_chrom = 0, num_genes = 0;
    read_gainputs(context->FA, context->GB, &num_chrom, &num_genes, context->ga_path);

    // Run the GA
    context->n_chrom_snapshot = GA(
        context->FA, context->GB, context->VC,
        &context->chrom, &context->chrom_snapshot,
        &context->gene_lim, context->atoms, context->residue,
        &context->cleftgrid, context->ga_path,
        &context->memchrom, ic2cf);

    if (context->n_chrom_snapshot <= 0) return -3;

    context->ran = true;
    return 0;
}

extern "C" void fx_ga_destroy(FXGAContextRef context) {
    if (!context) return;

    // Clean up population
    delete context->population;

    // Clean up gene_lim
    if (context->gene_lim) std::free(context->gene_lim);

    // Clean up chromosomes
    if (context->chrom) {
        for (int i = 0; i < context->memchrom; ++i) {
            if (context->chrom[i].genes) std::free(context->chrom[i].genes);
        }
        std::free(context->chrom);
    }

    if (context->chrom_snapshot && context->GB) {
        int total = context->GB->num_chrom * context->GB->max_generations;
        for (int i = 0; i < total; ++i) {
            if (context->chrom_snapshot[i].genes) std::free(context->chrom_snapshot[i].genes);
        }
        std::free(context->chrom_snapshot);
    }

    // Vcontacts cleanup
    if (context->VC) {
        if (context->VC->Calc) { std::free(context->VC->Calc); std::free(context->VC->Calclist); }
        if (context->VC->ca_index) std::free(context->VC->ca_index);
        if (context->VC->seed) std::free(context->VC->seed);
        if (context->VC->contlist) std::free(context->VC->contlist);
        if (context->VC->ptorder) std::free(context->VC->ptorder);
        if (context->VC->centerpt) std::free(context->VC->centerpt);
        if (context->VC->poly) std::free(context->VC->poly);
        if (context->VC->cont) std::free(context->VC->cont);
        if (context->VC->vedge) std::free(context->VC->vedge);
        if (context->VC->ca_rec) std::free(context->VC->ca_rec);
        std::free(context->VC);
    }

    // Cleft grid
    if (context->cleftgrid) std::free(context->cleftgrid);

    // Atoms
    if (context->atoms && context->FA) {
        for (int i = 0; i < context->FA->MIN_NUM_ATOM; i++) {
            if (context->atoms[i].cons) std::free(context->atoms[i].cons);
            if (context->atoms[i].coor_ref) std::free(context->atoms[i].coor_ref);
            if (context->atoms[i].eigen) {
                for (int j = 0; j < context->FA->normal_modes; j++)
                    if (context->atoms[i].eigen[j]) std::free(context->atoms[i].eigen[j]);
                std::free(context->atoms[i].eigen);
            }
        }
        std::free(context->atoms);
    }

    // Residues
    if (context->residue && context->FA) {
        for (int i = 1; i <= context->FA->res_cnt; i++) {
            if (context->residue[i].bonded) {
                int natm = context->residue[i].latm[0] - context->residue[i].fatm[0] + 1;
                for (int j = 0; j < natm; j++) std::free(context->residue[i].bonded[j]);
                std::free(context->residue[i].bonded);
            }
            if (context->residue[i].gpa) std::free(context->residue[i].gpa);
            if (context->residue[i].fatm) std::free(context->residue[i].fatm);
            if (context->residue[i].latm) std::free(context->residue[i].latm);
            if (context->residue[i].bond) std::free(context->residue[i].bond);
        }
        std::free(context->residue);
    }

    // Rotamers
    if (context->rotamer) std::free(context->rotamer);

    // FA sub-allocations
    if (context->FA) {
        if (context->FA->num_atm) std::free(context->FA->num_atm);
        if (context->FA->energy_matrix) std::free(context->FA->energy_matrix);
        if (context->FA->constraints) std::free(context->FA->constraints);
        for (int i = 0; i < 2; i++) { if (context->FA->mov[i]) std::free(context->FA->mov[i]); }
        if (context->FA->optres) std::free(context->FA->optres);
        if (context->FA->flex_res) {
            for (int i = 0; i < context->FA->MIN_FLEX_RESIDUE; i++) {
                if (context->FA->flex_res[i].close) std::free(context->FA->flex_res[i].close);
            }
            std::free(context->FA->flex_res);
        }
        if (context->FA->eigenvector) {
            for (int i = 0; i < 3 * context->FA->MIN_NUM_ATOM; i++)
                if (context->FA->eigenvector[i]) std::free(context->FA->eigenvector[i]);
            std::free(context->FA->eigenvector);
        }
        if (context->FA->normal_grid) {
            for (int i = 0; i < context->FA->MIN_NORMAL_GRID_POINTS; i++)
                if (context->FA->normal_grid[i]) std::free(context->FA->normal_grid[i]);
            std::free(context->FA->normal_grid);
        }
        if (context->FA->map_par) std::free(context->FA->map_par);
        if (context->FA->opt_par) std::free(context->FA->opt_par);
        if (context->FA->del_opt_par) std::free(context->FA->del_opt_par);
        if (context->FA->min_opt_par) std::free(context->FA->min_opt_par);
        if (context->FA->max_opt_par) std::free(context->FA->max_opt_par);
        if (context->FA->map_opt_par) std::free(context->FA->map_opt_par);
        if (context->FA->contacts) std::free(context->FA->contacts);
        std::free(context->FA);
    }

    if (context->GB) std::free(context->GB);

    delete context;
}

// ─── GA configuration accessors ─────────────────────────────────────────────

extern "C" int fx_ga_num_chromosomes(FXGAContextRef context) {
    return (context && context->GB) ? context->GB->num_chrom : 0;
}

extern "C" int fx_ga_num_genes(FXGAContextRef context) {
    return (context && context->GB) ? context->GB->num_genes : 0;
}

extern "C" int fx_ga_max_generations(FXGAContextRef context) {
    return (context && context->GB) ? context->GB->max_generations : 0;
}

extern "C" double fx_ga_temperature(FXGAContextRef context) {
    return (context && context->FA) ? static_cast<double>(context->FA->temperature) : 0.0;
}

// ─── BindingPopulation access ───────────────────────────────────────────────

extern "C" FXBindingPopulationRef fx_ga_get_population(FXGAContextRef context) {
    if (!context || !context->ran) return nullptr;

    // Lazy-create the BindingPopulation wrapper
    if (!context->population) {
        context->population = new BindingPopulation(
            context->FA, context->GB, context->VC,
            context->chrom_snapshot, context->gene_lim,
            context->atoms, context->residue, context->cleftgrid,
            context->n_chrom_snapshot);
    }

    static FXBindingPopulationImpl wrapper;
    wrapper.pop = context->population;
    return &wrapper;
}

extern "C" int fx_population_size(FXBindingPopulationRef pop) {
    return pop && pop->pop ? pop->pop->get_Population_size() : 0;
}

extern "C" FXBindingModeRef fx_population_get_mode(FXBindingPopulationRef pop, int index) {
    if (!pop || !pop->pop) return nullptr;
    if (index < 0 || index >= pop->pop->get_Population_size()) return nullptr;

    auto* impl = new FXBindingModeImpl();
    impl->mode = &pop->pop->get_binding_mode(index);
    return impl;
}

extern "C" FXStatMechEngineRef fx_population_global_ensemble(FXBindingPopulationRef pop) {
    if (!pop || !pop->pop) return nullptr;

    // Get the global ensemble engine from the population
    auto engine_copy = pop->pop->get_global_ensemble();

    // Create a new FXStatMechEngineImpl that owns a copy
    auto* impl = new FXStatMechEngineImpl(engine_copy.temperature());

    // Copy all samples from the global ensemble
    // (The caller owns this and must call fx_statmech_destroy)
    auto thermo = engine_copy.compute();
    // Re-add energies not possible without access to raw ensemble.
    // Instead, return the engine as-is by moving
    // This requires a move constructor or copy — use placement new
    // For correctness, we construct from the returned engine
    impl->engine = std::move(engine_copy);

    return impl;
}

extern "C" double fx_population_delta_G(FXBindingPopulationRef pop, int mode1_index, int mode2_index) {
    if (!pop || !pop->pop) return 0.0;
    int size = pop->pop->get_Population_size();
    if (mode1_index < 0 || mode1_index >= size ||
        mode2_index < 0 || mode2_index >= size) return 0.0;

    const auto& mode1 = pop->pop->get_binding_mode(mode1_index);
    const auto& mode2 = pop->pop->get_binding_mode(mode2_index);
    return pop->pop->compute_delta_G(mode1, mode2);
}

// ─── BindingMode access ─────────────────────────────────────────────────────

extern "C" int fx_mode_size(FXBindingModeRef mode) {
    return mode && mode->mode ? mode->mode->get_BindingMode_size() : 0;
}

extern "C" FXBindingModeInfo fx_mode_info(FXBindingModeRef mode) {
    FXBindingModeInfo info = {};
    if (!mode || !mode->mode) return info;

    info.size = mode->mode->get_BindingMode_size();
    info.free_energy = mode->mode->compute_energy();
    info.entropy = mode->mode->compute_entropy();
    info.enthalpy = mode->mode->compute_enthalpy();
    info.heat_capacity = mode->mode->get_heat_capacity();
    return info;
}

extern "C" FXThermodynamics fx_mode_thermodynamics(FXBindingModeRef mode) {
    FXThermodynamics fx = {};
    if (!mode || !mode->mode) return fx;

    auto t = mode->mode->get_thermodynamics();
    fx.temperature    = t.temperature;
    fx.log_Z          = t.log_Z;
    fx.free_energy    = t.free_energy;
    fx.mean_energy    = t.mean_energy;
    fx.mean_energy_sq = t.mean_energy_sq;
    fx.heat_capacity  = t.heat_capacity;
    fx.entropy        = t.entropy;
    fx.std_energy     = t.std_energy;
    return fx;
}

extern "C" double* fx_mode_boltzmann_weights(FXBindingModeRef mode, int* out_count) {
    if (!mode || !mode->mode || !out_count) {
        if (out_count) *out_count = 0;
        return nullptr;
    }

    auto weights = mode->mode->get_boltzmann_weights();
    *out_count = static_cast<int>(weights.size());
    if (weights.empty()) return nullptr;

    double* result = new double[weights.size()];
    std::memcpy(result, weights.data(), weights.size() * sizeof(double));
    return result;
}

extern "C" FXPoseInfo fx_mode_get_pose(FXBindingModeRef mode, int index) {
    FXPoseInfo info = {};
    if (!mode || !mode->mode) return info;
    if (index < 0 || index >= mode->mode->get_BindingMode_size()) return info;

    const Pose& pose = mode->mode->get_pose(index);
    info.chrom_index = pose.chrom_index;
    info.order = pose.order;
    info.reach_dist = pose.reachDist;
    info.cf = pose.CF;
    info.boltzmann_weight = pose.boltzmann_weight;
    return info;
}

extern "C" FXWHAMBin* fx_mode_free_energy_profile(FXBindingModeRef mode,
                                                    const double* coordinates, int coord_count,
                                                    int n_bins, int* out_count) {
    if (!mode || !mode->mode || !coordinates || coord_count <= 0 || !out_count) {
        if (out_count) *out_count = 0;
        return nullptr;
    }

    std::vector<double> coords(coordinates, coordinates + coord_count);
    auto bins = mode->mode->free_energy_profile(coords, n_bins);

    *out_count = static_cast<int>(bins.size());
    if (bins.empty()) return nullptr;

    FXWHAMBin* result = new FXWHAMBin[bins.size()];
    for (size_t i = 0; i < bins.size(); ++i) {
        result[i].coord_center = bins[i].coord_center;
        result[i].count        = bins[i].count;
        result[i].free_energy  = bins[i].free_energy;
    }
    return result;
}
