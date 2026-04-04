// GAContext.h — Per-instance GA state for re-entrant parallel execution
//
// Replaces all static variables in gaboom.cpp's calculate_fitness() and
// reproduce() functions, enabling multiple concurrent GA instances
// (e.g., ParallelDock multi-region, ParallelCampaign multi-model).
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Forward declaration — avoid pulling in full TurboQuant.h
namespace turboquant { class QuantizedContactMatrix; }

struct GAContext {
    // Generation counter — was static in calculate_fitness (gaboom.cpp:1152)
    int gen_id = 0;

    // Rejected conformer counter — was static in reproduce (gaboom.cpp:855)
    int nrejected = 0;

    // One-time hardware dispatch logging flag
    bool dispatch_logged = false;

    // TurboQuant compressed energy matrix cache (type-erased to avoid Eigen dependency)
    // Rebuilt when ntypes changes (was static at gaboom.cpp:1162-1163)
    // Managed via set_tqcm() / get_tqcm() / release_tqcm() in gaboom.cpp
    turboquant::QuantizedContactMatrix* tqcm = nullptr;
    int tqcm_ntypes = 0;

    GAContext() = default;
    ~GAContext();  // out-of-line: deletes tqcm if non-null

    // Movable, not copyable
    GAContext(GAContext&& o) noexcept
        : gen_id(o.gen_id), nrejected(o.nrejected),
          dispatch_logged(o.dispatch_logged),
          tqcm(o.tqcm), tqcm_ntypes(o.tqcm_ntypes) {
        o.tqcm = nullptr;
    }
    GAContext& operator=(GAContext&& o) noexcept;
    GAContext(const GAContext&) = delete;
    GAContext& operator=(const GAContext&) = delete;
};
