// BindingEntropyScore.swift — Entropy-health correlation data
//
// Links docking entropy metrics with HealthKit biometrics.
// Provides decomposed Shannon entropy (configurational + vibrational)
// for the IntelligenceOracle referee pipeline.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import Foundation
import FlexAIDdS

// MARK: - Shannon Entropy Decomposition

/// Decomposed entropy report from ShannonThermoStack.
///
/// Breaks the scalar Shannon entropy into configurational and vibrational
/// components, with convergence diagnostics and histogram summary for
/// the IntelligenceOracle referee.
public struct ShannonEntropyDecomposition: Sendable, Codable, Hashable {
    /// Configurational entropy from GA ensemble histogram (nats)
    public let configurational: Double

    /// Torsional vibrational entropy from ENCoM modes (kcal/mol/K)
    public let vibrational: Double

    /// Combined -T*S entropy contribution to free energy (kcal/mol)
    public let entropyContribution: Double

    /// Whether the Shannon entropy has reached a convergence plateau
    public let isConverged: Bool

    /// Relative change in entropy over the last convergence window
    /// (< 0.01 indicates plateau reached)
    public let convergenceRate: Double

    /// Hardware backend used for computation (e.g., "Metal", "AVX-512", "OpenMP", "scalar")
    public let hardwareBackend: String

    /// Number of non-zero histogram bins out of total (e.g., 18/20)
    public let occupiedBins: Int

    /// Total histogram bins used
    public let totalBins: Int

    /// Per-binding-mode Shannon entropy breakdown (nats, indexed by mode)
    public let perModeEntropy: [Double]

    /// Histogram bin edges and normalized probabilities for the oracle
    /// (compact summary: top 5 most populated bins as (binCenter, probability) pairs)
    public let dominantBins: [(center: Double, probability: Double)]

    // Codable conformance for tuple array
    private enum CodingKeys: String, CodingKey {
        case configurational, vibrational, entropyContribution, isConverged
        case convergenceRate, hardwareBackend, occupiedBins, totalBins
        case perModeEntropy, dominantBinCenters, dominantBinProbabilities
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        configurational = try c.decode(Double.self, forKey: .configurational)
        vibrational = try c.decode(Double.self, forKey: .vibrational)
        entropyContribution = try c.decode(Double.self, forKey: .entropyContribution)
        isConverged = try c.decode(Bool.self, forKey: .isConverged)
        convergenceRate = try c.decode(Double.self, forKey: .convergenceRate)
        hardwareBackend = try c.decode(String.self, forKey: .hardwareBackend)
        occupiedBins = try c.decode(Int.self, forKey: .occupiedBins)
        totalBins = try c.decode(Int.self, forKey: .totalBins)
        perModeEntropy = try c.decode([Double].self, forKey: .perModeEntropy)
        let centers = try c.decode([Double].self, forKey: .dominantBinCenters)
        let probs = try c.decode([Double].self, forKey: .dominantBinProbabilities)
        dominantBins = zip(centers, probs).map { ($0, $1) }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(configurational, forKey: .configurational)
        try c.encode(vibrational, forKey: .vibrational)
        try c.encode(entropyContribution, forKey: .entropyContribution)
        try c.encode(isConverged, forKey: .isConverged)
        try c.encode(convergenceRate, forKey: .convergenceRate)
        try c.encode(hardwareBackend, forKey: .hardwareBackend)
        try c.encode(occupiedBins, forKey: .occupiedBins)
        try c.encode(totalBins, forKey: .totalBins)
        try c.encode(perModeEntropy, forKey: .perModeEntropy)
        try c.encode(dominantBins.map(\.center), forKey: .dominantBinCenters)
        try c.encode(dominantBins.map(\.probability), forKey: .dominantBinProbabilities)
    }

    public static func == (lhs: ShannonEntropyDecomposition, rhs: ShannonEntropyDecomposition) -> Bool {
        lhs.configurational == rhs.configurational
        && lhs.vibrational == rhs.vibrational
        && lhs.entropyContribution == rhs.entropyContribution
        && lhs.isConverged == rhs.isConverged
        && lhs.hardwareBackend == rhs.hardwareBackend
        && lhs.occupiedBins == rhs.occupiedBins
        && lhs.perModeEntropy == rhs.perModeEntropy
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(configurational)
        hasher.combine(vibrational)
        hasher.combine(entropyContribution)
        hasher.combine(isConverged)
        hasher.combine(hardwareBackend)
    }

    public init(
        configurational: Double, vibrational: Double, entropyContribution: Double,
        isConverged: Bool, convergenceRate: Double, hardwareBackend: String,
        occupiedBins: Int, totalBins: Int,
        perModeEntropy: [Double] = [],
        dominantBins: [(center: Double, probability: Double)] = []
    ) {
        self.configurational = configurational
        self.vibrational = vibrational
        self.entropyContribution = entropyContribution
        self.isConverged = isConverged
        self.convergenceRate = convergenceRate
        self.hardwareBackend = hardwareBackend
        self.occupiedBins = occupiedBins
        self.totalBins = totalBins
        self.perModeEntropy = perModeEntropy
        self.dominantBins = dominantBins
    }
}

// MARK: - Binding Entropy Score

/// Correlates binding population entropy with health biometrics.
///
/// This is the core data object that connects molecular docking results
/// with HealthKit measurements (HRV, sleep, etc.) and the decomposed
/// Shannon entropy for the IntelligenceOracle referee pipeline.
public struct BindingEntropyScore: Sendable, Codable, Hashable, Identifiable {
    public let id: UUID

    // MARK: - Docking Metrics

    /// Shannon configurational entropy of the binding population
    public let shannonS: Double

    /// Temperature of the simulation (K)
    public let temperature: Double

    /// Number of binding modes in the population
    public let bindingModeCount: Int

    /// Helmholtz free energy of the lowest-energy binding mode (kcal/mol)
    public let bestFreeEnergy: Double

    /// Heat capacity of the global ensemble
    public let heatCapacity: Double

    // MARK: - Shannon Decomposition (from ShannonThermoStack)

    /// Decomposed entropy with configurational/vibrational split and convergence data.
    /// Available when `DockingRunner` has access to the ShannonThermoStack output.
    public var shannonDecomposition: ShannonEntropyDecomposition?

    // MARK: - Health Metrics (optional, filled when HealthKit data available)

    /// Heart rate variability SDNN (ms), if available
    public var hrvSDNN: Double?

    /// Resting heart rate (bpm), if available
    public var restingHeartRate: Double?

    /// Sleep duration (hours), if available
    public var sleepHours: Double?

    /// Timestamp of measurement
    public let timestamp: Date

    /// Whether the entropy has "collapsed" (shannonS below threshold)
    public var isCollapsed: Bool {
        shannonS < 0.1  // Threshold for entropy collapse
    }

    /// Whether the Shannon entropy has converged (plateau reached)
    public var isConverged: Bool {
        shannonDecomposition?.isConverged ?? false
    }

    /// Vibrational entropy contribution, if decomposition available (kcal/mol/K)
    public var vibrationalEntropy: Double? {
        shannonDecomposition?.vibrational
    }

    /// Configurational entropy contribution, if decomposition available (nats)
    public var configurationalEntropy: Double? {
        shannonDecomposition?.configurational
    }

    public init(
        shannonS: Double, temperature: Double,
        bindingModeCount: Int, bestFreeEnergy: Double,
        heatCapacity: Double, timestamp: Date = Date()
    ) {
        self.id = UUID()
        self.shannonS = shannonS
        self.temperature = temperature
        self.bindingModeCount = bindingModeCount
        self.bestFreeEnergy = bestFreeEnergy
        self.heatCapacity = heatCapacity
        self.timestamp = timestamp
    }

    /// Create from a DockingResult with optional Shannon decomposition.
    public init(from result: DockingResult, decomposition: ShannonEntropyDecomposition? = nil) {
        self.id = UUID()
        self.shannonS = result.globalThermodynamics.entropy
        self.temperature = result.temperature
        self.bindingModeCount = result.bindingModes.count
        self.bestFreeEnergy = result.bindingModes.first?.freeEnergy ?? 0
        self.heatCapacity = result.globalThermodynamics.heatCapacity
        self.timestamp = result.timestamp
        self.shannonDecomposition = decomposition
    }
}
