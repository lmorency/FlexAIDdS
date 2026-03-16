// SpatialGrid.h — Lightweight 3D cube grid spatial index for CavityDetect
// Algorithmically equivalent to Vcontacts' index_protein() + get_contlist4()
// and NRGRank's build_index_cubes(), but with clean C++ ownership.
// Apache-2.0 © 2026 Le Bonhomme Pharma

#pragma once
#include <vector>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include "flexaid.h"

namespace cavity_detect {

class SpatialGrid {
public:
    SpatialGrid() = default;

    // Build grid from atom vector. cell_size defaults to 6.5 Angstroms
    // (matches Vcontacts CELLSIZE and NRGRank's ~6.56 Å).
    // After construction the grid is immutable (safe for concurrent reads).
    void build(const std::vector<atom>& atoms, float cell_size = 6.5f) {
        m_atoms_ptr = &atoms;
        m_num_atoms = atoms.size();
        m_cell_size = cell_size;

        if (m_num_atoms == 0) {
            m_dim = m_dim2 = m_dim3 = 0;
            return;
        }

        // 1. Compute bounding box
        float bmin[3] = { 1e38f,  1e38f,  1e38f};
        float bmax[3] = {-1e38f, -1e38f, -1e38f};
        for (const auto& a : atoms) {
            for (int d = 0; d < 3; ++d) {
                if (a.coor[d] < bmin[d]) bmin[d] = a.coor[d];
                if (a.coor[d] > bmax[d]) bmax[d] = a.coor[d];
            }
        }
        // Pad by 1 Angstrom (same as Vcontacts' floor/ceil logic)
        for (int d = 0; d < 3; ++d)
            m_origin[d] = bmin[d] - 1.0f;

        // 2. Grid dimension from largest span
        float max_width = 0.f;
        for (int d = 0; d < 3; ++d) {
            float w = (bmax[d] + 1.0f) - m_origin[d];
            if (w > max_width) max_width = w;
        }
        m_dim  = static_cast<int>(max_width / m_cell_size) + 1;
        m_dim2 = m_dim * m_dim;
        m_dim3 = m_dim * m_dim * m_dim;

        // 3. Count atoms per cell (first pass)
        m_cell_count.assign(static_cast<std::size_t>(m_dim3), 0);
        for (std::size_t i = 0; i < m_num_atoms; ++i) {
            int ci = coord_to_cell(atoms[i].coor[0], atoms[i].coor[1], atoms[i].coor[2]);
            ++m_cell_count[static_cast<std::size_t>(ci)];
        }

        // 4. Prefix sum for start offsets
        m_cell_start.resize(static_cast<std::size_t>(m_dim3));
        int offset = 0;
        for (int c = 0; c < m_dim3; ++c) {
            m_cell_start[static_cast<std::size_t>(c)] = offset;
            offset += m_cell_count[static_cast<std::size_t>(c)];
        }

        // 5. Fill atom list (second pass)
        m_atom_list.resize(m_num_atoms);
        std::vector<int> running(static_cast<std::size_t>(m_dim3), 0);
        for (std::size_t i = 0; i < m_num_atoms; ++i) {
            int ci = coord_to_cell(atoms[i].coor[0], atoms[i].coor[1], atoms[i].coor[2]);
            int pos = m_cell_start[static_cast<std::size_t>(ci)]
                    + running[static_cast<std::size_t>(ci)]++;
            m_atom_list[static_cast<std::size_t>(pos)] = i;
        }
    }

    // Query: write atom indices within the 3x3x3 neighborhood into out_indices.
    // Returns count of indices written. Zero-allocation hot path.
    std::size_t query_neighbors(const float* coord,
                                std::size_t* out_indices,
                                std::size_t out_capacity) const {
        if (m_dim3 == 0) return 0;

        int cx = static_cast<int>((coord[0] - m_origin[0]) / m_cell_size);
        int cy = static_cast<int>((coord[1] - m_origin[1]) / m_cell_size);
        int cz = static_cast<int>((coord[2] - m_origin[2]) / m_cell_size);

        std::size_t count = 0;
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = cx + dx;
            if (nx < 0 || nx >= m_dim) continue;
            for (int dy = -1; dy <= 1; ++dy) {
                int ny = cy + dy;
                if (ny < 0 || ny >= m_dim) continue;
                for (int dz = -1; dz <= 1; ++dz) {
                    int nz = cz + dz;
                    if (nz < 0 || nz >= m_dim) continue;
                    int cell = nx * m_dim2 + ny * m_dim + nz;
                    int start = m_cell_start[static_cast<std::size_t>(cell)];
                    int cnt   = m_cell_count[static_cast<std::size_t>(cell)];
                    for (int a = 0; a < cnt && count < out_capacity; ++a) {
                        out_indices[count++] = m_atom_list[static_cast<std::size_t>(start + a)];
                    }
                }
            }
        }
        return count;
    }

    // Convenience overload (allocates; use in non-hot paths)
    std::vector<std::size_t> query_neighbors(const float* coord) const {
        // Upper bound: 27 cells × max atoms per cell. Use generous initial size.
        std::vector<std::size_t> result(512);
        std::size_t n = query_neighbors(coord, result.data(), result.size());
        result.resize(n);
        return result;
    }

    bool empty() const { return m_atoms_ptr == nullptr || m_num_atoms == 0; }
    std::size_t atom_count() const { return m_num_atoms; }

private:
    int coord_to_cell(float x, float y, float z) const {
        int ix = static_cast<int>((x - m_origin[0]) / m_cell_size);
        int iy = static_cast<int>((y - m_origin[1]) / m_cell_size);
        int iz = static_cast<int>((z - m_origin[2]) / m_cell_size);
        // Clamp to valid range
        ix = std::max(0, std::min(ix, m_dim - 1));
        iy = std::max(0, std::min(iy, m_dim - 1));
        iz = std::max(0, std::min(iz, m_dim - 1));
        return ix * m_dim2 + iy * m_dim + iz;
    }

    const std::vector<atom>* m_atoms_ptr = nullptr;
    std::size_t m_num_atoms = 0;

    float m_cell_size = 6.5f;
    float m_origin[3] = {0.f, 0.f, 0.f};
    int   m_dim  = 0;
    int   m_dim2 = 0;
    int   m_dim3 = 0;

    // CSR (Compressed Sparse Row) storage:
    //   m_cell_start[cell] = first index in m_atom_list
    //   m_cell_count[cell] = number of atoms in that cell
    //   m_atom_list[start..start+count] = atom indices
    std::vector<int>         m_cell_start;
    std::vector<int>         m_cell_count;
    std::vector<std::size_t> m_atom_list;
};

} // namespace cavity_detect
