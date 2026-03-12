// IntelligenceOracle.swift — Apple Intelligence integration for FlexAIDdS
//
// Uses the on-device FoundationModels framework (macOS 26+ / iOS 26+)
// to analyze BindingPopulation thermodynamics + health correlation.
// Returns a 3-bullet analysis.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import Foundation
import FlexAIDdS

// MARK: - Oracle Result

/// Three-bullet analysis from the Intelligence oracle.
public struct OracleAnalysis: Sendable, Codable {
    /// Analysis bullets (typically 3)
    public let bullets: [String]

    /// Timestamp of analysis
    public let timestamp: Date

    /// Input summary used for the analysis
    public let inputSummary: String

    public init(bullets: [String], inputSummary: String) {
        self.bullets = bullets
        self.timestamp = Date()
        self.inputSummary = inputSummary
    }
}

// MARK: - FoundationModels Integration

#if canImport(FoundationModels)
import FoundationModels

/// Actor providing on-device intelligence analysis of docking results.
///
/// Uses Apple's on-device language model (FoundationModels framework)
/// to analyze binding population thermodynamics and health correlations.
///
/// Usage:
/// ```swift
/// let oracle = try await IntelligenceOracle()
/// let analysis = try await oracle.analyze(
///     thermodynamics: result.globalThermodynamics,
///     entropyScore: score
/// )
/// for bullet in analysis.bullets {
///     print("- \(bullet)")
/// }
/// ```
@available(macOS 26.0, iOS 26.0, *)
public actor IntelligenceOracle {
    private let session: LanguageModelSession

    public init() async throws {
        self.session = LanguageModelSession()
    }

    /// Analyze binding population state with optional health data.
    /// - Parameters:
    ///   - thermodynamics: Global ensemble thermodynamic result
    ///   - entropyScore: Binding entropy score with optional health correlation
    /// - Returns: Three-bullet analysis
    public func analyze(
        thermodynamics: ThermodynamicResult,
        entropyScore: BindingEntropyScore? = nil
    ) async throws -> OracleAnalysis {
        let prompt = buildPrompt(thermodynamics: thermodynamics, entropyScore: entropyScore)
        let response = try await session.respond(to: prompt)
        let bullets = parseBullets(from: response.content)

        return OracleAnalysis(
            bullets: bullets,
            inputSummary: "T=\(thermodynamics.temperature)K, F=\(String(format: "%.2f", thermodynamics.freeEnergy)) kcal/mol, S=\(String(format: "%.4f", thermodynamics.entropy)) kcal/mol/K"
        )
    }

    private func buildPrompt(
        thermodynamics: ThermodynamicResult,
        entropyScore: BindingEntropyScore?
    ) -> String {
        var prompt = """
        Analyze this molecular docking binding population. Respond with exactly 3 concise bullet points.

        Thermodynamics:
        - Temperature: \(thermodynamics.temperature) K
        - Free energy F: \(String(format: "%.3f", thermodynamics.freeEnergy)) kcal/mol
        - Mean energy <E>: \(String(format: "%.3f", thermodynamics.meanEnergy)) kcal/mol
        - Entropy S: \(String(format: "%.6f", thermodynamics.entropy)) kcal/mol/K
        - Heat capacity Cv: \(String(format: "%.6f", thermodynamics.heatCapacity))
        - Energy std dev: \(String(format: "%.3f", thermodynamics.stdEnergy)) kcal/mol
        """

        if let score = entropyScore {
            prompt += "\n\nBinding Population:"
            prompt += "\n- Binding modes: \(score.bindingModeCount)"
            prompt += "\n- Shannon S: \(String(format: "%.6f", score.shannonS))"
            prompt += "\n- Best F: \(String(format: "%.3f", score.bestFreeEnergy)) kcal/mol"

            if score.isCollapsed {
                prompt += "\n- STATUS: Entropy COLLAPSED (single dominant binding mode)"
            }

            if let hrv = score.hrvSDNN {
                prompt += "\n\nHealth Correlation:"
                prompt += "\n- HRV SDNN: \(String(format: "%.0f", hrv)) ms"
            }
            if let sleep = score.sleepHours {
                prompt += "\n- Sleep: \(String(format: "%.1f", sleep)) hours"
            }

            prompt += "\n\nConsider the effect of target PTM/glycan modifications on the population."
        }

        prompt += "\n\nRespond with exactly 3 bullet points, each starting with '- '."
        return prompt
    }

    private func parseBullets(from text: String) -> [String] {
        let lines = text.split(separator: "\n")
        let bullets = lines.compactMap { line -> String? in
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("- ") {
                return String(trimmed.dropFirst(2))
            }
            if trimmed.hasPrefix("* ") {
                return String(trimmed.dropFirst(2))
            }
            return nil
        }
        return Array(bullets.prefix(3))
    }
}
#endif

// MARK: - Fallback (non-FoundationModels platforms)

/// Rule-based oracle for platforms without FoundationModels.
///
/// Provides deterministic 3-bullet analysis based on thermodynamic thresholds.
public struct RuleBasedOracle: Sendable {

    public init() {}

    /// Generate rule-based analysis of binding population state.
    public func analyze(
        thermodynamics: ThermodynamicResult,
        entropyScore: BindingEntropyScore? = nil
    ) -> OracleAnalysis {
        var bullets: [String] = []

        // Bullet 1: Free energy assessment
        if thermodynamics.freeEnergy < -10 {
            bullets.append("Strong binding affinity (F = \(String(format: "%.1f", thermodynamics.freeEnergy)) kcal/mol) — high confidence in drug-target interaction.")
        } else if thermodynamics.freeEnergy < -5 {
            bullets.append("Moderate binding affinity (F = \(String(format: "%.1f", thermodynamics.freeEnergy)) kcal/mol) — reasonable drug candidate.")
        } else {
            bullets.append("Weak binding affinity (F = \(String(format: "%.1f", thermodynamics.freeEnergy)) kcal/mol) — consider structural optimization.")
        }

        // Bullet 2: Entropy state
        if let score = entropyScore {
            if score.isCollapsed {
                bullets.append("Entropy collapsed to \(score.bindingModeCount) mode(s) — high specificity but check for enthalpy-entropy compensation.")
            } else if score.shannonS > 0.5 {
                bullets.append("High conformational entropy (S = \(String(format: "%.4f", score.shannonS))) — population still exploring. More sampling may refine the landscape.")
            } else {
                bullets.append("Moderate entropy with \(score.bindingModeCount) binding modes — population is converging on preferred conformations.")
            }
        } else {
            bullets.append("Entropy S = \(String(format: "%.4f", thermodynamics.entropy)) kcal/mol/K with Cv = \(String(format: "%.4f", thermodynamics.heatCapacity)).")
        }

        // Bullet 3: Health correlation or recommendation
        if let score = entropyScore, let hrv = score.hrvSDNN {
            if score.isCollapsed && hrv > 60 {
                bullets.append("Entropy collapse correlates with good HRV (\(String(format: "%.0f", hrv)) ms) — the system is recovering. Gentle activity recommended.")
            } else if hrv < 40 {
                bullets.append("Low HRV (\(String(format: "%.0f", hrv)) ms) detected — prioritize rest and recovery before interpreting docking thermodynamics.")
            } else {
                bullets.append("HRV at \(String(format: "%.0f", hrv)) ms — physiological state is stable for computational analysis.")
            }
        } else {
            bullets.append("Connect HealthKit for entropy-health correlation insights. Run with fleet mode for distributed compute.")
        }

        let inputSummary = "T=\(thermodynamics.temperature)K, F=\(String(format: "%.2f", thermodynamics.freeEnergy)) kcal/mol"
        return OracleAnalysis(bullets: bullets, inputSummary: inputSummary)
    }
}
