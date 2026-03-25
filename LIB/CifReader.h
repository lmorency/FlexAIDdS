// CifReader.h — Reads PDBx/mmCIF files into FlexAID atom/resid arrays
//
// mmCIF is the mandatory PDB deposition format since July 2019.
// This reader parses the _atom_site loop to extract coordinates,
// atom names, residue info, and chain IDs — the same data that
// read_pdb() extracts from legacy PDB files.
//
// Supports both receptor (ATOM) and ligand (HETATM) records.
// Auto-detects column order from the _atom_site.* header.
//
// Copyright 2026 Le Bonhomme Pharma
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "flexaid.h"

// Read a receptor from mmCIF/CIF file.
// Populates atom/resid arrays the same way read_pdb() does.
// Returns 1 on success, 0 on failure.
int read_cif_receptor(FA_Global* FA, atom** atoms, resid** residue,
                      const char* cif_file);

// Read a ligand from mmCIF/CIF file (HETATM records only).
// Populates atom/resid arrays for the ligand residue.
// Returns 1 on success, 0 on failure.
int read_cif_ligand(FA_Global* FA, atom** atoms, resid** residue,
                    const char* cif_file);

// ═══ CCBM: Multi-model CIF/PDB reader ═══
// Extract all MODEL records from a PDB or mmCIF file into separate
// coordinate arrays. Each model shares topology but has its own coordinates.
// Populates FA->model_coords[model_idx][atom_idx*3+{0,1,2}] and
// sets FA->n_models. Returns the number of models found (≥1).
int read_multi_model_pdb(FA_Global* FA, atom** atoms, resid** residue,
                         const char* pdb_file);
int read_multi_model_cif(FA_Global* FA, atom** atoms, resid** residue,
                         const char* cif_file);
