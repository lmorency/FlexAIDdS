// IntelligenceOracle.swift — Apple Intelligence integration for FlexAIDdS
//
// Uses the on-device FoundationModels framework (macOS 26+ / iOS 26+)
// to analyze BindingPopulation thermodynamics + health correlation.
// Returns structured analysis with confidence scoring, trend tracking,
// and multi-turn conversational context.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import Foundation
import FlexAIDdS

// MARK: - Oracle Result

/// Confidence level for oracle analysis bullets.
public enum AnalysisConfidence: String, Sendable, Codable, CaseIterable {
    case high     // Strong signal, well-supported by data
    case moderate // Reasonable inference, some uncertainty
    case low      // Speculative, needs more data or sampling
}

/// A single analysis bullet with metadata.
public struct AnalysisBullet: Sendable, Codable {
    public let text: String
    public let confidence: AnalysisConfidence
    public let category: BulletCategory

    public enum BulletCategory: String, Sendable, Codable {
        case binding       // Free energy / affinity assessment
        case entropy       // Conformational or vibrational entropy
        case health        // HealthKit correlation
        case modification  // PTM / glycan effects
        case fleet         // Fleet compute recommendation
        case trend         // Historical comparison
    }

    public init(text: String, confidence: AnalysisConfidence, category: BulletCategory) {
        self.text = text
        self.confidence = confidence
        self.category = category
    }
}

/// Structured analysis from the Intelligence oracle.
public struct OracleAnalysis: Sendable, Codable {
    /// Analysis bullets (typically 3-5)
    public let bullets: [String]

    /// Structured bullets with confidence and category metadata
    public let structuredBullets: [AnalysisBullet]

    /// Timestamp of analysis
    public let timestamp: Date

    /// Input summary used for the analysis
    public let inputSummary: String

    /// Overall confidence of the analysis
    public let overallConfidence: AnalysisConfidence

    public init(bullets: [String], inputSummary: String) {
        self.bullets = bullets
        self.structuredBullets = bullets.map {
            AnalysisBullet(text: $0, confidence: .moderate, category: .binding)
        }
        self.timestamp = Date()
        self.inputSummary = inputSummary
        self.overallConfidence = .moderate
    }

    public init(structuredBullets: [AnalysisBullet], inputSummary: String, overallConfidence: AnalysisConfidence) {
        self.structuredBullets = structuredBullets
        self.bullets = structuredBullets.map(\.text)
        self.timestamp = Date()
        self.inputSummary = inputSummary
        self.overallConfidence = overallConfidence
    }
}

// MARK: - Analysis History (trend tracking)

/// Stores past analyses for trend detection across docking campaigns.
public actor AnalysisHistory {
    public static let shared = AnalysisHistory()

    private var entries: [(key: String, analysis: OracleAnalysis)] = []
    private let maxEntries = 50

    private init() {}

    /// Record an analysis keyed by receptor-ligand pair.
    public func record(key: String, analysis: OracleAnalysis) {
        entries.append((key: key, analysis: analysis))
        if entries.count > maxEntries {
            entries.removeFirst()
        }
    }

    /// Retrieve past analyses for a receptor-ligand pair.
    public func history(for key: String) -> [OracleAnalysis] {
        entries.filter { $0.key == key }.map(\.analysis)
    }

    /// Get the most recent analysis for comparison.
    public func lastAnalysis(for key: String) -> OracleAnalysis? {
        entries.last(where: { $0.key == key })?.analysis
    }
}

// MARK: - FoundationModels Integration

#if canImport(FoundationModels)
import FoundationModels

/// Actor providing on-device intelligence analysis of docking results.
///
/// Uses Apple's on-device language model (FoundationModels framework)
/// to analyze binding population thermodynamics and health correlations.
/// Maintains multi-turn session context for follow-up questions.
///
/// Usage:
/// ```swift
/// let oracle = try await IntelligenceOracle()
/// let analysis = try await oracle.analyze(
///     thermodynamics: result.globalThermodynamics,
///     entropyScore: score
/// )
/// for bullet in analysis.structuredBullets {
///     print("[\(bullet.confidence.rawValue)] - \(bullet.text)")
/// }
///
/// // Follow-up question using session context
/// let followUp = try await oracle.askFollowUp(
///     "Should I increase sampling for binding mode 2?"
/// )
/// ```
@available(macOS 26.0, iOS 26.0, *)
public actor IntelligenceOracle {
    private let session: LanguageModelSession
    private var analysisCount: Int = 0

    /// System instructions that ground the model in molecular docking context.
    private static let systemInstructions = """
    You are a molecular docking analysis assistant for FlexAIDdS, an entropy-driven \
    docking engine. You analyze binding populations using statistical mechanics \
    thermodynamics (free energy F, entropy S, heat capacity Cv) and correlate with \
    HealthKit biometrics when available. Be concise, precise, and quantitative. \
    Use kcal/mol for energies, kcal/mol/K for entropy. Flag enthalpy-entropy \
    compensation when both F and S are large. Recommend increased sampling when \
    entropy is high (S > 0.5) or population has not converged.
    """

    public init() async throws {
        self.session = LanguageModelSession(
            instructions: Self.systemInstructions
        )
    }

    /// Analyze binding population state with optional health data.
    /// - Parameters:
    ///   - thermodynamics: Global ensemble thermodynamic result
    ///   - entropyScore: Binding entropy score with optional health correlation
    ///   - campaignKey: Optional key for trend tracking (e.g., "5HT2A-psilocin")
    /// - Returns: Structured analysis with confidence metadata
    public func analyze(
        thermodynamics: ThermodynamicResult,
        entropyScore: BindingEntropyScore? = nil,
        campaignKey: String? = nil
    ) async throws -> OracleAnalysis {
        let prompt = buildPrompt(thermodynamics: thermodynamics, entropyScore: entropyScore)
        let response = try await session.respond(to: prompt)
        let bullets = parseBullets(from: response.content)

        // Build structured bullets with confidence assessment
        let structured = assessConfidence(
            bullets: bullets,
            thermodynamics: thermodynamics,
            entropyScore: entropyScore
        )

        let overall = computeOverallConfidence(structured)

        let inputSummary = "T=\(thermodynamics.temperature)K, F=\(String(format: "%.2f", thermodynamics.freeEnergy)) kcal/mol, S=\(String(format: "%.4f", thermodynamics.entropy)) kcal/mol/K"

        let analysis = OracleAnalysis(
            structuredBullets: structured,
            inputSummary: inputSummary,
            overallConfidence: overall
        )

        // Track for trend analysis
        if let key = campaignKey {
            await AnalysisHistory.shared.record(key: key, analysis: analysis)
        }

        analysisCount += 1
        return analysis
    }

    /// Ask a follow-up question using the existing session context.
    /// The FoundationModels session retains prior conversation turns.
    public func askFollowUp(_ question: String) async throws -> String {
        let response = try await session.respond(to: question)
        return response.content
    }

    /// Compare current results with a previous campaign run.
    public func compareWithPrevious(
        current: ThermodynamicResult,
        previous: ThermodynamicResult,
        entropyScore: BindingEntropyScore? = nil
    ) async throws -> OracleAnalysis {
        let deltaF = current.freeEnergy - previous.freeEnergy
        let deltaS = current.entropy - previous.entropy
        let deltaCv = current.heatCapacity - previous.heatCapacity

        let prompt = """
        Compare these two docking campaigns. Respond with exactly 3 concise bullet points.

        Current run:
        - Free energy F: \(String(format: "%.3f", current.freeEnergy)) kcal/mol
        - Entropy S: \(String(format: "%.6f", current.entropy)) kcal/mol/K
        - Heat capacity Cv: \(String(format: "%.6f", current.heatCapacity))

        Previous run:
        - Free energy F: \(String(format: "%.3f", previous.freeEnergy)) kcal/mol
        - Entropy S: \(String(format: "%.6f", previous.entropy)) kcal/mol/K
        - Heat capacity Cv: \(String(format: "%.6f", previous.heatCapacity))

        Changes: ΔF = \(String(format: "%.3f", deltaF)) kcal/mol, ΔS = \(String(format: "%.6f", deltaS)) kcal/mol/K, ΔCv = \(String(format: "%.6f", deltaCv))

        Assess whether sampling has improved, binding affinity changed, or entropy compensation occurred.
        Respond with exactly 3 bullet points, each starting with '- '.
        """

        let response = try await session.respond(to: prompt)
        let bullets = parseBullets(from: response.content)

        let structured = bullets.map {
            AnalysisBullet(text: $0, confidence: .moderate, category: .trend)
        }

        return OracleAnalysis(
            structuredBullets: structured,
            inputSummary: "ΔF=\(String(format: "%.2f", deltaF)) kcal/mol, ΔS=\(String(format: "%.4f", deltaS))",
            overallConfidence: .moderate
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

        // Enthalpy-entropy compensation detection
        if thermodynamics.freeEnergy < -5 && thermodynamics.entropy > 0.01 {
            prompt += "\n- NOTE: Possible enthalpy-entropy compensation (large |F| with significant S)"
        }

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
                if hrv < 30 {
                    prompt += " (critically low — acute stress indicator)"
                } else if hrv < 50 {
                    prompt += " (below average — consider recovery)"
                }
            }
            if let sleep = score.sleepHours {
                prompt += "\n- Sleep: \(String(format: "%.1f", sleep)) hours"
                if sleep < 6.0 {
                    prompt += " (insufficient — cognitive load may affect interpretation)"
                }
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
            // Handle numbered bullets (1. 2. 3.)
            if let range = trimmed.range(of: #"^\d+\.\s+"#, options: .regularExpression) {
                return String(trimmed[range.upperBound...])
            }
            return nil
        }
        return Array(bullets.prefix(5))
    }

    /// Assess confidence of each bullet based on data quality.
    private func assessConfidence(
        bullets: [String],
        thermodynamics: ThermodynamicResult,
        entropyScore: BindingEntropyScore?
    ) -> [AnalysisBullet] {
        bullets.enumerated().map { index, text in
            let category: AnalysisBullet.BulletCategory
            let confidence: AnalysisConfidence

            switch index {
            case 0:
                category = .binding
                // High confidence if energy std dev is small relative to F
                confidence = thermodynamics.stdEnergy < abs(thermodynamics.freeEnergy) * 0.5 ? .high : .moderate
            case 1:
                category = .entropy
                // High confidence if sufficient binding modes
                if let score = entropyScore {
                    confidence = score.bindingModeCount >= 3 ? .high : .low
                } else {
                    confidence = .low
                }
            default:
                if text.lowercased().contains("hrv") || text.lowercased().contains("sleep") {
                    category = .health
                    confidence = entropyScore?.hrvSDNN != nil ? .moderate : .low
                } else if text.lowercased().contains("ptm") || text.lowercased().contains("glycan") {
                    category = .modification
                    confidence = .moderate
                } else {
                    category = .binding
                    confidence = .moderate
                }
            }

            return AnalysisBullet(text: text, confidence: confidence, category: category)
        }
    }

    private func computeOverallConfidence(_ bullets: [AnalysisBullet]) -> AnalysisConfidence {
        let highCount = bullets.filter { $0.confidence == .high }.count
        let lowCount = bullets.filter { $0.confidence == .low }.count
        if highCount >= 2 { return .high }
        if lowCount >= 2 { return .low }
        return .moderate
    }
}
#endif

// MARK: - Fallback (non-FoundationModels platforms)

/// Rule-based oracle for platforms without FoundationModels.
///
/// Provides deterministic analysis based on thermodynamic thresholds
/// with structured confidence metadata and trend detection.
public struct RuleBasedOracle: Sendable {

    public init() {}

    /// Generate rule-based analysis of binding population state.
    public func analyze(
        thermodynamics: ThermodynamicResult,
        entropyScore: BindingEntropyScore? = nil,
        campaignKey: String? = nil
    ) -> OracleAnalysis {
        var structured: [AnalysisBullet] = []

        // Bullet 1: Free energy assessment with confidence
        let fConfidence: AnalysisConfidence = thermodynamics.stdEnergy < abs(thermodynamics.freeEnergy) * 0.5 ? .high : .moderate
        if thermodynamics.freeEnergy < -10 {
            structured.append(AnalysisBullet(
                text: "Strong binding affinity (F = \(String(format: "%.1f", thermodynamics.freeEnergy)) kcal/mol) — high confidence in drug-target interaction.",
                confidence: fConfidence, category: .binding
            ))
        } else if thermodynamics.freeEnergy < -5 {
            structured.append(AnalysisBullet(
                text: "Moderate binding affinity (F = \(String(format: "%.1f", thermodynamics.freeEnergy)) kcal/mol) — reasonable drug candidate.",
                confidence: fConfidence, category: .binding
            ))
        } else {
            structured.append(AnalysisBullet(
                text: "Weak binding affinity (F = \(String(format: "%.1f", thermodynamics.freeEnergy)) kcal/mol) — consider structural optimization.",
                confidence: fConfidence, category: .binding
            ))
        }

        // Bullet 2: Entropy state with mode-count confidence
        if let score = entropyScore {
            let sConfidence: AnalysisConfidence = score.bindingModeCount >= 3 ? .high : .low
            if score.isCollapsed {
                structured.append(AnalysisBullet(
                    text: "Entropy collapsed to \(score.bindingModeCount) mode(s) — high specificity but check for enthalpy-entropy compensation.",
                    confidence: sConfidence, category: .entropy
                ))
            } else if score.shannonS > 0.5 {
                structured.append(AnalysisBullet(
                    text: "High conformational entropy (S = \(String(format: "%.4f", score.shannonS))) — population still exploring. More sampling may refine the landscape.",
                    confidence: .moderate, category: .entropy
                ))
            } else {
                structured.append(AnalysisBullet(
                    text: "Moderate entropy with \(score.bindingModeCount) binding modes — population is converging on preferred conformations.",
                    confidence: sConfidence, category: .entropy
                ))
            }
        } else {
            structured.append(AnalysisBullet(
                text: "Entropy S = \(String(format: "%.4f", thermodynamics.entropy)) kcal/mol/K with Cv = \(String(format: "%.4f", thermodynamics.heatCapacity)).",
                confidence: .low, category: .entropy
            ))
        }

        // Bullet 3: Enthalpy-entropy compensation detection
        if thermodynamics.freeEnergy < -5 && thermodynamics.entropy > 0.01 {
            structured.append(AnalysisBullet(
                text: "Enthalpy-entropy compensation detected: strong binding (F = \(String(format: "%.1f", thermodynamics.freeEnergy))) offset by conformational flexibility (S = \(String(format: "%.4f", thermodynamics.entropy))). Net ΔG may be less favorable than F alone suggests.",
                confidence: .moderate, category: .binding
            ))
        }

        // Bullet 4: Health correlation or recommendation
        if let score = entropyScore, let hrv = score.hrvSDNN {
            if score.isCollapsed && hrv > 60 {
                structured.append(AnalysisBullet(
                    text: "Entropy collapse correlates with good HRV (\(String(format: "%.0f", hrv)) ms) — the system is recovering. Gentle activity recommended.",
                    confidence: .moderate, category: .health
                ))
            } else if hrv < 40 {
                structured.append(AnalysisBullet(
                    text: "Low HRV (\(String(format: "%.0f", hrv)) ms) detected — prioritize rest and recovery before interpreting docking thermodynamics.",
                    confidence: .high, category: .health
                ))
            } else {
                structured.append(AnalysisBullet(
                    text: "HRV at \(String(format: "%.0f", hrv)) ms — physiological state is stable for computational analysis.",
                    confidence: .moderate, category: .health
                ))
            }
        } else {
            structured.append(AnalysisBullet(
                text: "Connect HealthKit for entropy-health correlation insights. Run with fleet mode for distributed compute.",
                confidence: .low, category: .fleet
            ))
        }

        let overall = computeOverallConfidence(structured)
        let inputSummary = "T=\(thermodynamics.temperature)K, F=\(String(format: "%.2f", thermodynamics.freeEnergy)) kcal/mol"
        return OracleAnalysis(structuredBullets: structured, inputSummary: inputSummary, overallConfidence: overall)
    }

    private func computeOverallConfidence(_ bullets: [AnalysisBullet]) -> AnalysisConfidence {
        let highCount = bullets.filter { $0.confidence == .high }.count
        let lowCount = bullets.filter { $0.confidence == .low }.count
        if highCount >= 2 { return .high }
        if lowCount >= 2 { return .low }
        return .moderate
    }
}
