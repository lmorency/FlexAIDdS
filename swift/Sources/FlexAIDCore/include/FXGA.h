// FXGA.h — C interface to the FlexAID Genetic Algorithm engine
//
// Wraps the full GA lifecycle: initialization, execution, and result access.
// The FXGAContext owns all memory (FA_Global, GB_Global, chromosomes, atoms, etc.).
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "FXTypes.h"
#include "FXStatMechEngine.h"

#ifdef __cplusplus
extern "C" {
#endif

// ─── Opaque handles ─────────────────────────────────────────────────────────

typedef struct FXGAContextImpl* FXGAContextRef;
typedef struct FXBindingPopulationImpl* FXBindingPopulationRef;
typedef struct FXBindingModeImpl* FXBindingModeRef;

// ─── GA lifecycle ───────────────────────────────────────────────────────────

// Create GA context from input files (reads config.inp + ga.inp)
// Returns NULL on failure.
FXGAContextRef fx_ga_create(const char* config_path, const char* ga_path);

// Run the genetic algorithm. Returns 0 on success, non-zero on error.
int fx_ga_run(FXGAContextRef context);

// Destroy GA context and all owned memory.
void fx_ga_destroy(FXGAContextRef context);

// ─── GA configuration accessors ─────────────────────────────────────────────

int    fx_ga_num_chromosomes(FXGAContextRef context);
int    fx_ga_num_genes(FXGAContextRef context);
int    fx_ga_max_generations(FXGAContextRef context);
double fx_ga_temperature(FXGAContextRef context);

// ─── BindingPopulation access (valid after fx_ga_run) ───────────────────────

// Get the binding population (non-owning pointer, valid while context lives)
FXBindingPopulationRef fx_ga_get_population(FXGAContextRef context);

// Population-level queries
int fx_population_size(FXBindingPopulationRef pop);

// Get binding mode by index (non-owning, valid while context lives)
FXBindingModeRef fx_population_get_mode(FXBindingPopulationRef pop, int index);

// Global ensemble across all binding modes
// Caller owns the returned engine and must call fx_statmech_destroy()
FXStatMechEngineRef fx_population_global_ensemble(FXBindingPopulationRef pop);

// Delta-G between two binding modes
double fx_population_delta_G(FXBindingPopulationRef pop, int mode1_index, int mode2_index);

// ─── BindingMode access ─────────────────────────────────────────────────────

int fx_mode_size(FXBindingModeRef mode);
FXBindingModeInfo fx_mode_info(FXBindingModeRef mode);
FXThermodynamics fx_mode_thermodynamics(FXBindingModeRef mode);

// Returns heap-allocated array; caller must free via fx_free_doubles()
double* fx_mode_boltzmann_weights(FXBindingModeRef mode, int* out_count);

// Get pose info by index within the binding mode
FXPoseInfo fx_mode_get_pose(FXBindingModeRef mode, int index);

// Free energy profile along a coordinate
// Returns heap-allocated array; caller must free via fx_free_wham_bins()
FXWHAMBin* fx_mode_free_energy_profile(FXBindingModeRef mode,
                                        const double* coordinates, int coord_count,
                                        int n_bins, int* out_count);

#ifdef __cplusplus
}
#endif
