// ThermoRefereeTools.swift — FoundationModels Tool conformances for thermodynamic callbacks
//
// Allows the on-device Apple Intelligence model to autonomously invoke
// thermodynamic recomputation functions during referee analysis.
// Tools are registered with the LanguageModelSession and called by the model
// when it needs additional data to complete its verdict.
//
// Example: The model sees a borderline entropy value and decides to recompute
// at a different temperature to check sensitivity. It calls the
// RecomputeAtTemperature tool, receives the result, and factors it into
// its verdict — all on-device, zero latency.
//
// Requires macOS 26+ / iOS 26+ for FoundationModels Tool protocol.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import Foundation
import FlexAIDdS

#if canImport(FoundationModels)
import FoundationModels

// MARK: - Recompute Shannon Entropy at Different Temperature

/// Tool that lets the on-device model recompute the ShannonThermoStack
/// at a different temperature for sensitivity analysis.
///
/// The model might invoke this when it detects borderline convergence
/// or wants to verify enthalpy-entropy compensation at physiological temperature.
@available(macOS 26.0, iOS 26.0, *)
struct RecomputeAtTemperatureTool: Tool {
    let name = "recompute_at_temperature"
    let description = """
        Recompute Shannon configurational entropy and free energy at a different \
        temperature (Kelvin). Use this to check temperature sensitivity of the \
        thermodynamic results. Returns S_conf (nats), S_vib (kcal/mol/K), \
        delta_G (kcal/mol), and convergence status at the new temperature.
        """

    @Generable
    struct Input: Sendable {
        /// Target temperature in Kelvin (typically 280-320 K)
        var temperature: Double
    }

    @Generable
    struct Output: Sendable {
        var shannonEntropy: Double      // nats
        var vibrationalEntropy: Double  // kcal/mol/K
        var deltaG: Double              // kcal/mol
        var isConverged: Bool
        var report: String
    }

    /// Reference to the GA context for recomputation.
    let gaContext: FXGAContextRef

    func call(_ input: Input) async throws -> Output {
        var result = FXShannonThermoResult()
        let success = fx_ga_recompute_shannon_at_temperature(
            gaContext, input.temperature, &result)

        guard success != 0 else {
            return Output(
                shannonEntropy: 0, vibrationalEntropy: 0, deltaG: 0,
                isConverged: false,
                report: "Recomputation failed — ShannonThermoStack unavailable"
            )
        }

        return Output(
            shannonEntropy: result.shannon_entropy,
            vibrationalEntropy: result.torsional_vib_entropy,
            deltaG: result.delta_G,
            isConverged: result.is_converged != 0,
            report: "Recomputed at \(input.temperature) K: ΔG = \(String(format: "%.3f", result.delta_G)) kcal/mol, S_conf = \(String(format: "%.4f", result.shannon_entropy)) nats"
        )
    }
}

// MARK: - Query Per-Mode Entropy Breakdown

/// Tool that lets the model request per-binding-mode entropy decomposition.
/// Useful when the model suspects entropy imbalance across modes.
@available(macOS 26.0, iOS 26.0, *)
struct PerModeEntropyTool: Tool {
    let name = "per_mode_entropy"
    let description = """
        Get the Shannon configurational entropy for each binding mode separately. \
        Returns an array of (mode_index, entropy_nats, pose_count) tuples. \
        Use this to detect if one binding mode absorbs most conformational diversity.
        """

    @Generable
    struct Input: Sendable {
        /// Maximum number of modes to return (0 = all)
        var maxModes: Int
    }

    @Generable
    struct Output: Sendable {
        var modeCount: Int
        var report: String
    }

    let gaContext: FXGAContextRef

    func call(_ input: Input) async throws -> Output {
        let maxModes = input.maxModes > 0 ? input.maxModes : 100
        var entropies = [Double](repeating: 0, count: maxModes)
        let count = fx_ga_per_mode_shannon(gaContext, &entropies, Int32(maxModes))

        guard count > 0 else {
            return Output(modeCount: 0, report: "No binding modes available")
        }

        let modes = entropies.prefix(Int(count))
        var report = "Per-mode Shannon entropy (\(count) modes):\n"
        for (i, s) in modes.enumerated() {
            report += "  Mode \(i): S = \(String(format: "%.6f", s)) nats\n"
        }

        if let maxS = modes.max(), let minPositive = modes.filter({ $0 > 0 }).min() {
            let ratio = maxS / minPositive
            if ratio > 5.0 {
                report += "WARNING: Entropy imbalance ratio = \(String(format: "%.1f", ratio))x"
            }
        }

        return Output(modeCount: Int(count), report: report)
    }
}

// MARK: - Helmholtz Free Energy Calculator

/// Tool that computes Helmholtz free energy from a subset of energies.
/// Useful for the model to verify or cross-check thermodynamic values.
@available(macOS 26.0, iOS 26.0, *)
struct HelmholtzCalculatorTool: Tool {
    let name = "helmholtz_free_energy"
    let description = """
        Compute Helmholtz free energy F = -kT ln Z from energy values at a given \
        temperature. Use this to verify free energy calculations or compute F \
        for a subset of poses. Returns F in kcal/mol.
        """

    @Generable
    struct Input: Sendable {
        /// Temperature in Kelvin
        var temperature: Double
        /// Up to 10 representative energy values (kcal/mol) to compute F from
        var energies: [Double]
    }

    @Generable
    struct Output: Sendable {
        var freeEnergy: Double   // kcal/mol
        var report: String
    }

    func call(_ input: Input) async throws -> Output {
        guard !input.energies.isEmpty else {
            return Output(freeEnergy: 0, report: "No energies provided")
        }

        let F = FlexAIDRunner.helmholtz(energies: input.energies, temperature: input.temperature)
        return Output(
            freeEnergy: F,
            report: "F = \(String(format: "%.3f", F)) kcal/mol from \(input.energies.count) samples at \(input.temperature) K"
        )
    }
}

// MARK: - Vibrational Entropy Calculator

/// Tool that computes ENCoM vibrational entropy from eigenvalues.
/// Lets the model assess vibrational contributions independently.
@available(macOS 26.0, iOS 26.0, *)
struct VibrationalEntropyTool: Tool {
    let name = "vibrational_entropy"
    let description = """
        Compute ENCoM vibrational entropy from normal mode eigenvalues using \
        the Schlitter formula. Returns S_vib in kcal/mol/K and the number of \
        non-trivial modes. Use this to assess the vibrational entropy contribution.
        """

    @Generable
    struct Input: Sendable {
        /// Temperature in Kelvin
        var temperature: Double
    }

    @Generable
    struct Output: Sendable {
        var vibrationalEntropy: Double  // kcal/mol/K
        var modeCount: Int
        var effectiveFrequency: Double  // rad/s
        var report: String
    }

    /// Cached eigenvalues from the docking run.
    let eigenvalues: [Double]

    func call(_ input: Input) async throws -> Output {
        guard !eigenvalues.isEmpty else {
            return Output(
                vibrationalEntropy: 0, modeCount: 0, effectiveFrequency: 0,
                report: "No eigenvalues available — ENCoM not computed for this run"
            )
        }

        let result = ENCoMRunner.computeVibrationalEntropy(
            eigenvalues: eigenvalues, temperature: input.temperature)

        return Output(
            vibrationalEntropy: result.entropy,
            modeCount: result.modeCount,
            effectiveFrequency: result.effectiveFrequency,
            report: "S_vib = \(String(format: "%.6f", result.entropy)) kcal/mol/K from \(result.modeCount) modes at \(input.temperature) K"
        )
    }
}

#endif
