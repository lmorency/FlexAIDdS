// GISTGrid.h — Grid Inhomogeneous Solvation Theory desolvation grid
//
// Reads OpenDX .dx format water thermodynamic grids (typically from GIST
// analysis of MD trajectories) and provides trilinear interpolation of
// desolvation free energy at arbitrary 3D coordinates.
//
// Integration: queried per-atom in vcfunction.cpp to add a solvent
// correction term (gist_desolv) to the scoring function.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>

namespace gist {

class GISTGrid {
public:
    GISTGrid() : nx_(0), ny_(0), nz_(0), loaded_(false) {
        origin_[0] = origin_[1] = origin_[2] = 0.0f;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                delta_[i][j] = (i == j) ? 1.0f : 0.0f;
    }

    // Load an OpenDX .dx grid file.
    // Returns true on success, false on parse error or file not found.
    bool load_dx(const std::string& filename) {
        std::ifstream ifs(filename);
        if (!ifs.is_open()) return false;

        std::string line;
        bool header_done = false;
        int expected_count = 0;

        while (std::getline(ifs, line)) {
            if (line.empty() || line[0] == '#') continue;

            // Parse grid dimensions: "object 1 class gridpositions counts NX NY NZ"
            if (line.find("gridpositions counts") != std::string::npos) {
                std::istringstream ss(line);
                std::string tok;
                // skip: "object" "1" "class" "gridpositions" "counts"
                ss >> tok >> tok >> tok >> tok >> tok;
                ss >> nx_ >> ny_ >> nz_;
                continue;
            }

            // Parse origin: "origin OX OY OZ"
            if (line.find("origin") == 0) {
                std::istringstream ss(line);
                std::string tok;
                ss >> tok >> origin_[0] >> origin_[1] >> origin_[2];
                continue;
            }

            // Parse delta vectors: "delta DX DY DZ" (3 lines)
            if (line.find("delta") == 0) {
                static int delta_row = 0;
                std::istringstream ss(line);
                std::string tok;
                ss >> tok >> delta_[delta_row][0] >> delta_[delta_row][1] >> delta_[delta_row][2];
                delta_row = (delta_row + 1) % 3;
                continue;
            }

            // Parse data count: "object 3 class array ... follows"
            if (line.find("class array") != std::string::npos) {
                // Extract count from: "... type float rank 0 items NNNN data follows"
                auto pos = line.find("items");
                if (pos != std::string::npos) {
                    std::istringstream ss(line.substr(pos + 5));
                    ss >> expected_count;
                }
                header_done = true;
                continue;
            }

            // Read data values
            if (header_done && !line.empty() && line[0] != 'o' && line[0] != 'a') {
                std::istringstream ss(line);
                float val;
                while (ss >> val) {
                    data_.push_back(val);
                }
            }
        }

        loaded_ = (nx_ > 0 && ny_ > 0 && nz_ > 0 &&
                   static_cast<int>(data_.size()) == nx_ * ny_ * nz_);
        return loaded_;
    }

    // Compute desolvation energy at a 3D point via trilinear interpolation.
    // Returns 0.0 if grid is not loaded or point is outside grid bounds.
    double desolvation_energy(float x, float y, float z) const {
        if (!loaded_) return 0.0;

        // Convert world coordinates to fractional grid coordinates
        // For orthogonal grids: frac = (world - origin) / delta_diagonal
        float fx = (x - origin_[0]) / delta_[0][0];
        float fy = (y - origin_[1]) / delta_[1][1];
        float fz = (z - origin_[2]) / delta_[2][2];

        // Check bounds (with 0.5 cell margin for interpolation)
        if (fx < 0.0f || fx >= (nx_ - 1) ||
            fy < 0.0f || fy >= (ny_ - 1) ||
            fz < 0.0f || fz >= (nz_ - 1)) {
            return 0.0;
        }

        // Integer grid indices and fractional remainders
        int ix = static_cast<int>(fx);
        int iy = static_cast<int>(fy);
        int iz = static_cast<int>(fz);
        float dx = fx - ix;
        float dy = fy - iy;
        float dz = fz - iz;

        // Trilinear interpolation of the 8 enclosing voxel corners
        auto idx = [&](int i, int j, int k) -> int {
            return i * ny_ * nz_ + j * nz_ + k;
        };

        float c000 = data_[idx(ix,     iy,     iz)];
        float c001 = data_[idx(ix,     iy,     iz + 1)];
        float c010 = data_[idx(ix,     iy + 1, iz)];
        float c011 = data_[idx(ix,     iy + 1, iz + 1)];
        float c100 = data_[idx(ix + 1, iy,     iz)];
        float c101 = data_[idx(ix + 1, iy,     iz + 1)];
        float c110 = data_[idx(ix + 1, iy + 1, iz)];
        float c111 = data_[idx(ix + 1, iy + 1, iz + 1)];

        float c00 = c000 * (1.0f - dz) + c001 * dz;
        float c01 = c010 * (1.0f - dz) + c011 * dz;
        float c10 = c100 * (1.0f - dz) + c101 * dz;
        float c11 = c110 * (1.0f - dz) + c111 * dz;

        float c0 = c00 * (1.0f - dy) + c01 * dy;
        float c1 = c10 * (1.0f - dy) + c11 * dy;

        return static_cast<double>(c0 * (1.0f - dx) + c1 * dx);
    }

    bool is_loaded() const noexcept { return loaded_; }
    int nx() const noexcept { return nx_; }
    int ny() const noexcept { return ny_; }
    int nz() const noexcept { return nz_; }

private:
    std::vector<float> data_;
    float origin_[3];
    float delta_[3][3];
    int nx_, ny_, nz_;
    bool loaded_;
};

} // namespace gist
