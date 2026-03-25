// ThermoRefereeTypes.swift — @Generable structured output for FoundationModels referee
//
// Defines typed verdict structures that Apple's on-device model produces
// via guided generation (constrained decoding). Eliminates text parsing —
// the model outputs native Swift structs directly.
//
// Requires macOS 26+ / iOS 26+ for @Generable macro.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import Foundation
import FlexAIDdS

#if canImport(FoundationModels)
import FoundationModels

// MARK: - Referee Verdict (Structured Output)

/// Severity level for a referee finding.
@available(macOS 26.0, iOS 26.0, *)
@Generable
public enum RefereeSeverity: String, Sendable, Codable {
    case pass        // No issue
    case advisory    // Informational, no action needed
    case warning     // Potential issue, consider action
    case critical    // Must address before trusting results
}

/// Category of thermodynamic referee finding.
@available(macOS 26.0, iOS 26.0, *)
@Generable
public enum RefereeCategory: String, Sendable, Codable {
    case convergence       // Entropy plateau / sampling adequacy
    case entropyBalance    // S_conf vs S_vib decomposition
    case compensation      // Enthalpy-entropy compensation
    case histogram         // Energy landscape sampling quality
    case modeBalance       // Per-mode entropy distribution
    case affinity          // Binding free energy assessment
    case recommendation    // Actionable next steps
}

/// A single referee finding with typed severity and category.
@available(macOS 26.0, iOS 26.0, *)
@Generable
public struct RefereeFinding: Sendable, Codable {
    /// Short title (3-8 words)
    public var title: String

    /// Detailed explanation with quantitative values
    public var detail: String

    /// Severity of this finding
    public var severity: RefereeSeverity

    /// Category of thermodynamic concern
    public var category: RefereeCategory
}

/// Complete referee verdict produced by FoundationModels guided generation.
@available(macOS 26.0, iOS 26.0, *)
@Generable
public struct RefereeVerdict: Sendable, Codable {
    /// List of findings (typically 3-5)
    public var findings: [RefereeFinding]

    /// Overall assessment: can we trust these thermodynamic values?
    public var overallTrustworthy: Bool

    /// Recommended next action (concise, actionable)
    public var recommendedAction: String

    /// Confidence in the verdict itself (0.0-1.0)
    public var confidence: Double
}

// MARK: - Temperature Sensitivity Analysis (Structured Output)

/// Result of a what-if temperature recomputation.
@available(macOS 26.0, iOS 26.0, *)
@Generable
public struct TemperatureSensitivity: Sendable, Codable {
    /// Does free energy change significantly with temperature?
    public var isSensitive: Bool

    /// Assessment of the temperature sensitivity
    public var assessment: String

    /// Recommended temperature range for reliable results
    public var recommendedTempRange: String
}

// MARK: - Comparative Verdict (Structured Output)

/// Verdict comparing two docking campaigns.
@available(macOS 26.0, iOS 26.0, *)
@Generable
public struct ComparativeVerdict: Sendable, Codable {
    /// Did binding improve, worsen, or stay similar?
    public var bindingTrend: String

    /// Did entropy converge better or worse?
    public var convergenceTrend: String

    /// Overall comparison summary
    public var summary: String

    /// Is the new run an improvement?
    public var improved: Bool
}

#endif

// MARK: - Cross-Platform Verdict (non-FoundationModels)

/// Platform-independent referee verdict that mirrors the @Generable version.
/// Used by RuleBasedOracle and TypeScript IntelligenceEngine.
public struct CrossPlatformRefereeFinding: Sendable, Codable, Hashable {
    public let title: String
    public let detail: String
    public let severity: String   // "pass", "advisory", "warning", "critical"
    public let category: String   // matches RefereeCategory raw values

    public init(title: String, detail: String, severity: String, category: String) {
        self.title = title
        self.detail = detail
        self.severity = severity
        self.category = category
    }
}

/// Platform-independent verdict.
public struct CrossPlatformRefereeVerdict: Sendable, Codable, Hashable {
    public let findings: [CrossPlatformRefereeFinding]
    public let overallTrustworthy: Bool
    public let recommendedAction: String
    public let confidence: Double

    public init(findings: [CrossPlatformRefereeFinding], overallTrustworthy: Bool,
                recommendedAction: String, confidence: Double) {
        self.findings = findings
        self.overallTrustworthy = overallTrustworthy
        self.recommendedAction = recommendedAction
        self.confidence = confidence
    }
}
