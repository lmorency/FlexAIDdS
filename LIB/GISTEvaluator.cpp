// GISTEvaluator.cpp — GIST blurry trilinear displacement scoring
//
// Apache-2.0 © 2026 Le Bonhomme Pharma

#include "GISTEvaluator.h"
#include "flexaid.h"

#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstring>

// ─── DX file parser ──────────────────────────────────────────────────────────
//
// Parses OpenDX scalar field format (.dx) as produced by cpptraj GIST:
//
//   object 1 class gridpositions counts NX NY NZ
//   origin OX OY OZ
//   delta DX 0 0
//   delta 0 DY 0
//   delta 0 0 DZ
//   object 2 class gridconnections counts NX NY NZ
//   object 3 class array type double rank 0 items N data follows
//   <data values...>
//   attribute "dep" string "positions"

bool GISTEvaluator::parse_dx_file(const std::string& path,
                                   std::vector<double>& data) {
    std::ifstream in(path);
    if (!in.is_open()) {
        std::fprintf(stderr, "GISTEvaluator: cannot open %s\n", path.c_str());
        return false;
    }

    int file_nx = 0, file_ny = 0, file_nz = 0;
    double file_ox = 0, file_oy = 0, file_oz = 0;
    double file_dx = 0, file_dy = 0, file_dz = 0;
    int n_items = 0;
    bool found_counts = false;

    std::string line;
    while (std::getline(in, line)) {
        // Skip comments
        if (line.empty() || line[0] == '#') continue;

        if (line.find("gridpositions counts") != std::string::npos) {
            std::sscanf(line.c_str(),
                        "object 1 class gridpositions counts %d %d %d",
                        &file_nx, &file_ny, &file_nz);
            found_counts = true;
        }
        else if (line.find("origin") != std::string::npos &&
                 line.find("object") == std::string::npos) {
            std::sscanf(line.c_str(), "origin %lf %lf %lf",
                        &file_ox, &file_oy, &file_oz);
        }
        else if (line.find("delta") != std::string::npos) {
            double d1 = 0, d2 = 0, d3 = 0;
            std::sscanf(line.c_str(), "delta %lf %lf %lf", &d1, &d2, &d3);
            if (d1 != 0.0) file_dx = d1;
            else if (d2 != 0.0) file_dy = d2;
            else if (d3 != 0.0) file_dz = d3;
        }
        else if (line.find("data follows") != std::string::npos) {
            std::sscanf(line.c_str(),
                        "object 3 class array type double rank 0 items %d",
                        &n_items);
            break;
        }
    }

    if (!found_counts || file_nx <= 0 || file_ny <= 0 || file_nz <= 0) {
        std::fprintf(stderr, "GISTEvaluator: invalid grid dimensions in %s\n",
                     path.c_str());
        return false;
    }

    // Store grid parameters (first file defines the grid; subsequent must match)
    if (nx == 0) {
        nx = file_nx;
        ny = file_ny;
        nz = file_nz;
        origin_x = file_ox;
        origin_y = file_oy;
        origin_z = file_oz;
        // Use average spacing (grids should be cubic)
        spacing = (file_dx + file_dy + file_dz) / 3.0;
        if (spacing <= 0.0) spacing = 0.5;
    }

    // Read data values
    int expected = file_nx * file_ny * file_nz;
    data.clear();
    data.reserve(static_cast<size_t>(expected));

    double val;
    while (in >> val) {
        data.push_back(val);
        if (static_cast<int>(data.size()) >= expected) break;
    }

    if (static_cast<int>(data.size()) != expected) {
        std::fprintf(stderr,
                     "GISTEvaluator: expected %d values, got %zu in %s\n",
                     expected, data.size(), path.c_str());
        return false;
    }

    return true;
}

bool GISTEvaluator::load_dx(const std::string& free_energy_dx,
                              const std::string& density_dx) {
    loaded = false;

    if (!parse_dx_file(free_energy_dx, free_energy_grid))
        return false;
    if (!parse_dx_file(density_dx, density_grid))
        return false;

    // Verify grid sizes match
    int expected = nx * ny * nz;
    if (static_cast<int>(free_energy_grid.size()) != expected ||
        static_cast<int>(density_grid.size()) != expected) {
        std::fprintf(stderr,
                     "GISTEvaluator: grid size mismatch between DX files\n");
        return false;
    }

    loaded = true;
    std::printf("GISTEvaluator: loaded %dx%dx%d grid (spacing=%.3f A)\n",
                nx, ny, nz, spacing);
    return true;
}

// ─── Blurry trilinear displacement score for one atom ────────────────────────
//
// Scans a 3×3×3 voxel neighborhood and applies Gaussian-weighted scoring
// for unfavorable water voxels.  The Gaussian sigma = atom_radius / divisor
// determines the sharpness of the displacement peak.

double GISTEvaluator::score_atom(double x, double y, double z,
                                  double atom_radius) const {
    if (!loaded) return 0.0;

    double sigma = atom_radius / divisor;
    double variance = sigma * sigma;
    if (variance < 1e-12) return 0.0;

    // Nearest grid voxel
    int ci = static_cast<int>((x - origin_x) / spacing);
    int cj = static_cast<int>((y - origin_y) / spacing);
    int ck = static_cast<int>((z - origin_z) / spacing);

    double score = 0.0;

    // Scan 3×3×3 neighborhood
    for (int di = -1; di <= 1; ++di) {
        for (int dj = -1; dj <= 1; ++dj) {
            for (int dk = -1; dk <= 1; ++dk) {
                int gi = ci + di;
                int gj = cj + dj;
                int gk = ck + dk;

                if (!in_bounds(gi, gj, gk)) continue;

                int idx = flat_index(gi, gj, gk);

                // Only displace unfavorable water (high density, high free energy)
                if (density_grid[idx] <= rho_cutoff ||
                    free_energy_grid[idx] <= delta_G_cutoff)
                    continue;

                // Voxel center coordinates
                double vx = origin_x + gi * spacing;
                double vy = origin_y + gj * spacing;
                double vz = origin_z + gk * spacing;

                double dx = x - vx;
                double dy = y - vy;
                double dz = z - vz;
                double dist_sq = dx * dx + dy * dy + dz * dz;

                // Gaussian weighting: blurry displacement kernel
                double w = std::exp(-dist_sq / (2.0 * variance));
                score += w * free_energy_grid[idx];
            }
        }
    }

    return score;
}

// ─── Score all ligand atoms ──────────────────────────────────────────────────

double GISTEvaluator::score_ligand(const atom_struct* atoms,
                                    const FA_Global_struct* FA) const {
    if (!loaded) return 0.0;

    double total = 0.0;

    // Iterate all atoms and score those belonging to the ligand
    for (int i = 0; i < FA->atm_cnt_real; ++i) {
        // Only score ligand atoms (optres type == 1)
        if (atoms[i].optres == nullptr) continue;
        if (atoms[i].optres->type != 1) continue;

        total += score_atom(
            static_cast<double>(atoms[i].coor[0]),
            static_cast<double>(atoms[i].coor[1]),
            static_cast<double>(atoms[i].coor[2]),
            static_cast<double>(atoms[i].radius));
    }

    return total * weight;
}
