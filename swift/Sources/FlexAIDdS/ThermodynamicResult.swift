// ThermodynamicResult.swift — Swift model for statistical mechanics thermodynamics
//
// Maps from the C FXThermodynamics struct to a Swift-native Sendable, Codable type.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import FlexAIDCore

/// Full thermodynamic analysis of a conformational ensemble.
///
/// Computed from a partition function Z(T) via log-sum-exp for numerical stability:
/// - Free energy: F = -kT ln Z
/// - Entropy: S = (<E> - F) / T
/// - Heat capacity: C_v = (<E^2> - <E>^2) / (kT^2)
public struct ThermodynamicResult: Sendable, Codable, Hashable {
    /// Temperature in Kelvin
    public let temperature: Double

    /// Natural log of the partition function ln(Z)
    public let logZ: Double

    /// Helmholtz free energy F = -kT ln Z (kcal/mol)
    public let freeEnergy: Double

    /// Boltzmann-weighted mean energy <E> (kcal/mol)
    public let meanEnergy: Double

    /// Mean squared energy <E^2>
    public let meanEnergySq: Double

    /// Heat capacity C_v = (<E^2> - <E>^2) / (kT^2)
    public let heatCapacity: Double

    /// Conformational entropy S = (<E> - F) / T (kcal mol^-1 K^-1)
    public let entropy: Double

    /// Standard deviation of energy sigma_E (kcal/mol)
    public let stdEnergy: Double

    /// Initialize from a C FXThermodynamics struct
    init(from c: FXThermodynamics) {
        self.temperature = c.temperature
        self.logZ = c.log_Z
        self.freeEnergy = c.free_energy
        self.meanEnergy = c.mean_energy
        self.meanEnergySq = c.mean_energy_sq
        self.heatCapacity = c.heat_capacity
        self.entropy = c.entropy
        self.stdEnergy = c.std_energy
    }

    /// Initialize with explicit values
    public init(
        temperature: Double, logZ: Double, freeEnergy: Double,
        meanEnergy: Double, meanEnergySq: Double,
        heatCapacity: Double, entropy: Double, stdEnergy: Double
    ) {
        self.temperature = temperature
        self.logZ = logZ
        self.freeEnergy = freeEnergy
        self.meanEnergy = meanEnergy
        self.meanEnergySq = meanEnergySq
        self.heatCapacity = heatCapacity
        self.entropy = entropy
        self.stdEnergy = stdEnergy
    }
}
