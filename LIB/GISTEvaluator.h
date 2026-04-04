// GISTEvaluator.h — GIST blurry trilinear displacement scoring
//
// Evaluates the thermodynamic cost/reward of displacing water molecules
// from a binding site using pre-computed GIST .dx grid files.  Uses a
// Gaussian kernel (sigma = atom_radius / divisor) for continuous,
// position-dependent scoring via blurry trilinear interpolation.
//
// Reference: Grid Inhomogeneous Solvation Theory (GIST)
//
// Apache-2.0 © 2026 Le Bonhomme Pharma

#pragma once

#include <cmath>
#include <vector>
#include <string>
#include <cstdio>

// Forward declarations (avoid pulling in flexaid.h here)
struct FA_Global_struct;
struct atom_struct;
struct VC_Global_struct;

class GISTEvaluator {
public:
    // Grid dimensions
    int nx = 0, ny = 0, nz = 0;

    // Grid origin and spacing (Angstroms)
    double origin_x = 0.0, origin_y = 0.0, origin_z = 0.0;
    double spacing = 0.5;

    // GIST data arrays (flat, row-major: z fastest)
    std::vector<double> free_energy_grid;
    std::vector<double> density_grid;

    // Cutoff parameters for identifying unfavorable (displaceable) water
    double delta_G_cutoff = 1.0;   // free-energy cutoff (kcal/mol)
    double rho_cutoff     = 4.8;   // density cutoff (relative to bulk)

    // Gaussian kernel divisor: sigma = atom_radius / divisor
    double divisor = 2.0;

    // Weight applied to total GIST score before adding to CF
    double weight = 1.0;

    // Whether grid data has been loaded successfully
    bool loaded = false;

    GISTEvaluator() = default;

    // Load GIST grids from OpenDX .dx files.
    // Returns true on success, false on parse error or file not found.
    bool load_dx(const std::string& free_energy_dx,
                 const std::string& density_dx);

    // Evaluate blurry displacement score for a single atom.
    // Scans a 3x3x3 neighborhood around the nearest voxel and applies
    // Gaussian-weighted displacement scoring for unfavorable water.
    double score_atom(double x, double y, double z,
                      double atom_radius) const;

    // Evaluate total GIST displacement score for all ligand atoms.
    // Called once per chromosome evaluation.
    double score_ligand(const atom_struct* atoms,
                        const FA_Global_struct* FA) const;

private:
    // Convert 3D grid indices to 1D flat index (row-major, z fastest).
    inline int flat_index(int ix, int iy, int iz) const {
        return ix * (ny * nz) + iy * nz + iz;
    }

    // Check if grid indices are within bounds.
    inline bool in_bounds(int ix, int iy, int iz) const {
        return ix >= 0 && ix < nx &&
               iy >= 0 && iy < ny &&
               iz >= 0 && iz < nz;
    }

    // Parse a single OpenDX .dx file into a data vector.
    // Populates grid dimensions and origin on first call.
    bool parse_dx_file(const std::string& path, std::vector<double>& data);
};
