// bindings.cpp — Emscripten/Embind wrappers for FlexAIDdS WASM build
//
// Exposes StatMechEngine, BoltzmannLUT, ENCoMEngine to JavaScript/TypeScript.
// Full GA/BindingPopulation excluded — too heavy for browser; results consumed as JSON.
//
// Build: emcmake cmake .. && cmake --build .
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

#include <emscripten/bind.h>
#include "statmech.h"
#include "encom.h"

using namespace emscripten;

// ─── Thermodynamics struct ──────────────────────────────────────────────────

EMSCRIPTEN_BINDINGS(flexaidds) {

    // Physical constants
    constant("kB_kcal", statmech::kB_kcal);
    constant("kB_SI", statmech::kB_SI);

    // Thermodynamics
    value_object<statmech::Thermodynamics>("Thermodynamics")
        .field("temperature", &statmech::Thermodynamics::temperature)
        .field("logZ", &statmech::Thermodynamics::log_Z)
        .field("freeEnergy", &statmech::Thermodynamics::free_energy)
        .field("meanEnergy", &statmech::Thermodynamics::mean_energy)
        .field("meanEnergySq", &statmech::Thermodynamics::mean_energy_sq)
        .field("heatCapacity", &statmech::Thermodynamics::heat_capacity)
        .field("entropy", &statmech::Thermodynamics::entropy)
        .field("stdEnergy", &statmech::Thermodynamics::std_energy);

    // WHAMBin
    value_object<statmech::WHAMBin>("WHAMBin")
        .field("coordCenter", &statmech::WHAMBin::coord_center)
        .field("count", &statmech::WHAMBin::count)
        .field("freeEnergy", &statmech::WHAMBin::free_energy);

    // TIPoint
    value_object<statmech::TIPoint>("TIPoint")
        .field("lambda", &statmech::TIPoint::lambda)
        .field("dVdLambda", &statmech::TIPoint::dV_dlambda);

    // State
    value_object<statmech::State>("State")
        .field("energy", &statmech::State::energy)
        .field("count", &statmech::State::count);

    // StatMechEngine
    class_<statmech::StatMechEngine>("StatMechEngine")
        .constructor<double>()
        .function("addSample", &statmech::StatMechEngine::add_sample)
        .function("compute", &statmech::StatMechEngine::compute)
        .function("boltzmannWeights", &statmech::StatMechEngine::boltzmann_weights)
        .function("temperature", &statmech::StatMechEngine::temperature)
        .function("beta", &statmech::StatMechEngine::beta)
        .function("size", &statmech::StatMechEngine::size)
        .function("clear", &statmech::StatMechEngine::clear)
        .class_function("helmholtz", optional_override([](val energies, double T) -> double {
            auto vec = vecFromJSArray<double>(energies);
            return statmech::StatMechEngine::helmholtz(vec, T);
        }))
        .class_function("thermodynamicIntegration", optional_override([](val points) -> double {
            auto len = points["length"].as<int>();
            std::vector<statmech::TIPoint> ti_points(len);
            for (int i = 0; i < len; i++) {
                auto pt = points[i];
                ti_points[i].lambda = pt["lambda"].as<double>();
                ti_points[i].dV_dlambda = pt["dVdLambda"].as<double>();
            }
            return statmech::StatMechEngine::thermodynamic_integration(ti_points);
        }));

    // BoltzmannLUT
    class_<statmech::BoltzmannLUT>("BoltzmannLUT")
        .constructor<double, double, double, int>()
        .function("lookup", &statmech::BoltzmannLUT::operator());

    // ENCoM VibrationalEntropy
    value_object<encom::VibrationalEntropy>("VibrationalEntropy")
        .field("entropy", &encom::VibrationalEntropy::S_vib_kcal_mol_K)
        .field("entropySI", &encom::VibrationalEntropy::S_vib_J_mol_K)
        .field("omegaEff", &encom::VibrationalEntropy::omega_eff)
        .field("nModes", &encom::VibrationalEntropy::n_modes)
        .field("temperature", &encom::VibrationalEntropy::temperature);

    // ENCoMEngine static methods
    class_<encom::ENCoMEngine>("ENCoMEngine")
        .class_function("totalEntropy", &encom::ENCoMEngine::total_entropy)
        .class_function("freeEnergyWithVibrations", &encom::ENCoMEngine::free_energy_with_vibrations);

    // Register vector types
    register_vector<double>("VectorDouble");
    register_vector<statmech::WHAMBin>("VectorWHAMBin");
}
