// FXBoltzmannLUT.h — C interface to statmech::BoltzmannLUT
//
// Pre-tabulated exp(-beta*E) lookup for O(1) inner-loop evaluation.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// ─── Opaque handle ──────────────────────────────────────────────────────────

typedef struct FXBoltzmannLUTImpl* FXBoltzmannLUTRef;

// ─── Lifecycle ──────────────────────────────────────────────────────────────

FXBoltzmannLUTRef fx_lut_create(double beta, double e_min, double e_max, int n_bins);
void fx_lut_destroy(FXBoltzmannLUTRef lut);

// ─── Lookup ─────────────────────────────────────────────────────────────────

// Returns exp(-beta * energy) via O(1) table lookup
double fx_lut_lookup(FXBoltzmannLUTRef lut, double energy);

#ifdef __cplusplus
}
#endif
