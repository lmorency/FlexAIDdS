// FXBoltzmannLUT.mm — Objective-C++ implementation of the BoltzmannLUT C shim
//
// Bridges statmech::BoltzmannLUT (C++20) to plain C for Swift.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

#include "FXBoltzmannLUT.h"
#include "statmech.h"

struct FXBoltzmannLUTImpl {
    statmech::BoltzmannLUT lut;
    FXBoltzmannLUTImpl(double beta, double e_min, double e_max, int n_bins)
        : lut(beta, e_min, e_max, n_bins) {}
};

extern "C" FXBoltzmannLUTRef fx_lut_create(double beta, double e_min, double e_max, int n_bins) {
    return new FXBoltzmannLUTImpl(beta, e_min, e_max, n_bins);
}

extern "C" void fx_lut_destroy(FXBoltzmannLUTRef lut) {
    delete lut;
}

extern "C" double fx_lut_lookup(FXBoltzmannLUTRef lut, double energy) {
    if (!lut) return 0.0;
    return lut->lut(energy);
}
