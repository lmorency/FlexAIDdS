// direct_input.h — Direct PDB/MOL2 input pipeline (no .inp files)
//
// Implements direct loading of receptor PDB and ligand MOL2
// directly from file paths, auto-detect the binding cleft, and set up
// all FlexAID data structures required by the GA engine.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "flexaid.h"
#include "gaboom.h"

/*
 * setup_direct_input — replaces read_input() for the new CLI mode.
 *
 * Performs all the steps that read_input() does from legacy .inp files,
 * but using direct file paths:
 *   1. Read interaction matrix (from dependencies or base path)
 *   2. Read and clean receptor PDB (modify_pdb + read_pdb)
 *   3. Calculate protein center of geometry
 *   4. Assign residue connectivity and atom types
 *   5. Read ligand MOL2 via Mol2Reader
 *   6. Assign atomic radii
 *   7. Auto-detect binding cleft via CleftDetector
 *   8. Generate docking grid from detected spheres
 *   9. Set up optimization vectors for the ligand
 *  10. Compute IC bounds
 *  11. Update optres pointers
 *
 * Returns 0 on success, non-zero on failure.
 */
int setup_direct_input(FA_Global* FA, GB_Global* GB, VC_Global* VC,
                       atom** atoms, resid** residue,
                       rot** rotamer, gridpoint** cleftgrid,
                       const char* receptor_pdb, const char* ligand_mol2);
