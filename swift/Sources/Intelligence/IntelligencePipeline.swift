// IntelligencePipeline.swift — Unified Intelligence feature orchestrator
//
// Runs all applicable rule-based Intelligence features on a docking result
// and collects outputs into a single IntelligenceResult. Each feature runs
// independently — one failure does not block others.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import Foundation
import FlexAIDdS

/// Orchestrates all rule-based Intelligence features on a docking result.
///
/// Usage:
/// ```swift
/// let pipeline = IntelligencePipeline()
/// let result = pipeline.analyze(dockingResult: result)
/// print(result.modeNarrative?.selectivityInsight ?? "N/A")
/// ```
public struct IntelligencePipeline: Sendable {

    public init() {}

    /// Run all applicable Intelligence features on a docking result.
    ///
    /// Each feature runs independently. If one fails, an error message is
    /// collected and the remaining features continue.
    ///
    /// - Parameters:
    ///   - dockingResult: The docking result to analyze.
    ///   - gaSnapshot: Optional GA progress snapshot for convergence coaching.
    ///   - cleftFeatures: Optional cleft geometry for druggability assessment.
    /// - Returns: Aggregate IntelligenceResult with all available analyses.
    public func analyze(
        dockingResult: DockingResult,
        gaSnapshot: GAProgressSnapshot? = nil,
        cleftFeatures: CleftFeatures? = nil
    ) -> IntelligenceResult {
        var errors: [String] = []

        // 1. Binding mode narrative
        let modeNarrative: CrossPlatformModeNarrative? = {
            guard !dockingResult.bindingModes.isEmpty else {
                errors.append("BindingModeNarrator: no binding modes available")
                return nil
            }
            let narrator = RuleBasedModeNarrator()
            let context = narrator.buildContext(from: dockingResult)
            return narrator.narrate(context: context)
        }()

        // 2. Per-mode pose quality
        let poseQualityReports: [CrossPlatformPoseQualityReport] = {
            let critic = RuleBasedLigandFitCritic()
            return dockingResult.bindingModes.prefix(3).enumerated().compactMap { i, mode in
                let context = critic.buildContext(
                    modeIndex: i,
                    mode: mode,
                    poses: mode.poses,
                    temperature: dockingResult.temperature
                )
                return critic.evaluate(context: context)
            }
        }()

        // 3. Convergence coaching
        let convergenceCoaching: CrossPlatformConvergenceCoaching? = {
            guard let snapshot = gaSnapshot else { return nil }
            let coach = RuleBasedConvergenceCoach()
            return coach.coach(snapshot: snapshot)
        }()

        // 4. Cleft assessment
        let cleftAssessment: CrossPlatformCleftAssessment? = {
            guard let features = cleftFeatures else { return nil }
            let assessor = RuleBasedCleftAssessor()
            return assessor.assess(cleft: features)
        }()

        return IntelligenceResult(
            refereeVerdict: nil,  // Referee requires ShannonDecomposition — call separately
            modeNarrative: modeNarrative,
            poseQualityReports: poseQualityReports,
            convergenceCoaching: convergenceCoaching,
            cleftAssessment: cleftAssessment,
            errors: errors
        )
    }
}

// MARK: - DockingRunner Convenience Extension

/// Extends DockingRunner with a unified intelligence pipeline method.
/// Lives in the Intelligence module to avoid circular dependencies
/// (FlexAIDdS -> Intelligence is not allowed, but Intelligence -> FlexAIDdS is).
extension DockingRunner {

    /// Run the GA and then analyze the result with all applicable Intelligence features.
    ///
    /// Chains: `run()` -> `extractShannonDecomposition()` -> `IntelligencePipeline.analyze()`.
    /// Each Intelligence feature runs independently — one failure does not block others.
    ///
    /// - Parameters:
    ///   - gaSnapshot: Optional GA progress snapshot for convergence coaching.
    ///   - cleftFeatures: Optional cleft geometry for druggability assessment.
    /// - Returns: Tuple of docking result and aggregate intelligence analysis.
    public func runWithIntelligence(
        gaSnapshot: GAProgressSnapshot? = nil,
        cleftFeatures: CleftFeatures? = nil
    ) async throws -> (DockingResult, IntelligenceResult) {
        let result = try await run()
        let pipeline = IntelligencePipeline()
        let intelligence = pipeline.analyze(
            dockingResult: result,
            gaSnapshot: gaSnapshot,
            cleftFeatures: cleftFeatures
        )
        return (result, intelligence)
    }
}
