// DockingRunner.swift — Swift actor wrapping the full FlexAID GA lifecycle
//
// Owns the entire GA context (FA_Global, GB_Global, chromosomes, etc.).
// All child references (BindingPopulation, BindingModes) are invalidated on deinit.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import FlexAIDCore
import Foundation

/// Actor managing the full FlexAID genetic algorithm docking lifecycle.
///
/// Usage:
/// ```swift
/// let runner = DockingRunner(configPath: "config.inp", gaPath: "ga.inp")
/// let result = try await runner.run()
/// print("Found \(result.bindingModes.count) binding modes")
/// print("Best free energy: \(result.bindingModes.first?.freeEnergy ?? 0)")
/// ```
public actor DockingRunner {
    private var contextRef: FXGAContextRef?
    private var populationRef: FXBindingPopulationRef?
    private let configPath: String
    private let gaPath: String

    /// Create a docking runner from FlexAID input files.
    /// - Parameters:
    ///   - configPath: Path to the docking configuration file (config.inp)
    ///   - gaPath: Path to the GA parameters file (ga.inp)
    public init(configPath: String, gaPath: String) {
        self.configPath = configPath
        self.gaPath = gaPath
    }

    deinit {
        if let ctx = contextRef {
            fx_ga_destroy(ctx)
        }
    }

    // MARK: - GA Lifecycle

    /// Run the genetic algorithm and return docking results.
    ///
    /// This runs the full pipeline: GA optimization, scoring, clustering,
    /// and thermodynamic analysis of the binding population.
    ///
    /// - Throws: `DockingError` if initialization or execution fails
    /// - Returns: Complete docking result with binding modes and thermodynamics
    public func run() async throws -> DockingResult {
        // Create GA context
        guard let ctx = fx_ga_create(configPath, gaPath) else {
            throw DockingError.initializationFailed
        }
        contextRef = ctx

        // Run the genetic algorithm (blocking — runs on actor's executor)
        let status = fx_ga_run(ctx)
        guard status == 0 else {
            throw DockingError.executionFailed(code: Int(status))
        }

        // Extract results
        return extractResults()
    }

    // MARK: - Result Access

    /// Get the binding population (available after run() completes).
    public var population: FXBindingPopulationRef? {
        guard let ctx = contextRef else { return nil }
        if populationRef == nil {
            populationRef = fx_ga_get_population(ctx)
        }
        return populationRef
    }

    /// Number of binding modes in the population.
    public var bindingModeCount: Int {
        guard let pop = population else { return 0 }
        return Int(fx_population_size(pop))
    }

    /// Temperature used for the simulation.
    public var temperature: Double {
        guard let ctx = contextRef else { return 0 }
        return fx_ga_temperature(ctx)
    }

    /// Global ensemble thermodynamics across all binding modes.
    public func globalThermodynamics() -> ThermodynamicResult? {
        guard let pop = population else { return nil }
        guard let engineRef = fx_population_global_ensemble(pop) else { return nil }
        defer { fx_statmech_destroy(engineRef) }
        return ThermodynamicResult(from: fx_statmech_compute(engineRef))
    }

    // MARK: - Private Helpers

    private func extractResults() -> DockingResult {
        guard let pop = population else {
            return DockingResult(
                bindingModes: [],
                globalThermodynamics: ThermodynamicResult(
                    temperature: 0, logZ: 0, freeEnergy: 0,
                    meanEnergy: 0, meanEnergySq: 0,
                    heatCapacity: 0, entropy: 0, stdEnergy: 0),
                temperature: 0,
                populationSize: 0
            )
        }

        let modeCount = Int(fx_population_size(pop))
        var modes: [BindingModeResult] = []

        for i in 0..<modeCount {
            if let modeRef = fx_population_get_mode(pop, Int32(i)) {
                var result = BindingModeResult(from: fx_mode_info(modeRef))
                result.thermodynamics = ThermodynamicResult(from: fx_mode_thermodynamics(modeRef))
                modes.append(result)
            }
        }

        let globalThermo = globalThermodynamics() ?? ThermodynamicResult(
            temperature: temperature, logZ: 0, freeEnergy: 0,
            meanEnergy: 0, meanEnergySq: 0,
            heatCapacity: 0, entropy: 0, stdEnergy: 0)

        return DockingResult(
            bindingModes: modes,
            globalThermodynamics: globalThermo,
            temperature: temperature,
            populationSize: Int(fx_ga_num_chromosomes(contextRef!))
        )
    }
}

// MARK: - Errors

/// Errors that can occur during docking.
public enum DockingError: Error, LocalizedError {
    case initializationFailed
    case executionFailed(code: Int)
    case notRun

    public var errorDescription: String? {
        switch self {
        case .initializationFailed:
            return "Failed to initialize GA context from input files"
        case .executionFailed(let code):
            return "GA execution failed with status code \(code)"
        case .notRun:
            return "Docking has not been run yet — call run() first"
        }
    }
}
