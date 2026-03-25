// LibrarySplitter.h — Split multi-molecule ligand libraries into individual files
//
// Handles:
//   - Multi-molecule SDF (molecules separated by $$$$)
//   - SMILES file (.smi, .smiles, .txt) — one SMILES per line
//   - Directory of MOL2/SDF/PDB files
//   - Single file (passthrough — library of 1)
//
// Writes individual ligand files to a temp directory, returns the list.
// The caller iterates over the list and docks each ligand separately.
//
// Copyright 2026 Le Bonhomme Pharma
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <vector>
#include <filesystem>

namespace library {

struct LigandEntry {
    std::string path;       // path to individual ligand file (temp or original)
    std::string name;       // ligand name/identifier
    std::string format;     // "mol2", "sdf", "smiles"
    bool        is_temp;    // true if file was created by splitter (cleanup needed)
};

struct LibraryInfo {
    std::vector<LigandEntry> ligands;
    std::string temp_dir;   // directory for temp files (empty if none created)
    int total;
};

/// Detect if the input is a library (multi-molecule file, SMILES list, or directory).
/// Returns the number of ligands detected (1 = single molecule, >1 = library).
int detect_library_size(const std::string& path);

/// Split a ligand library into individual entries.
/// For single-molecule inputs, returns a library of 1 (no temp files).
/// For multi-molecule SDF, splits at $$$$ markers.
/// For SMILES files, splits at newlines.
/// For directories, lists all MOL2/SDF/PDB files.
LibraryInfo split_library(const std::string& path);

/// Clean up temp files created by split_library.
void cleanup_library(const LibraryInfo& lib);

} // namespace library
