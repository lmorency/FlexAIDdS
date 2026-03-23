// IntelligenceResult.swift — Aggregate intelligence analysis result
//
// Collects outputs from all Intelligence features into a single result.
// Used by IntelligencePipeline to return all available analyses.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import Foundation

/// Aggregate result from running all Intelligence features on a docking result.
public struct IntelligenceResult: Sendable, Codable {
    /// Thermodynamic referee verdict
    public let refereeVerdict: CrossPlatformRefereeVerdict?
    /// Binding mode narrative
    public let modeNarrative: CrossPlatformModeNarrative?
    /// Per-mode pose quality reports
    public let poseQualityReports: [CrossPlatformPoseQualityReport]
    /// GA convergence coaching
    public let convergenceCoaching: CrossPlatformConvergenceCoaching?
    /// Selectivity analysis (only when multi-target context is provided)
    public let selectivityAnalysis: CrossPlatformSelectivityAnalysis?
    /// Fleet explanation (only when fleet context is provided)
    public let fleetExplanation: CrossPlatformFleetExplanation?
    /// Health entropy insight (only when health context is provided)
    public let healthInsight: CrossPlatformHealthEntropyInsight?
    /// Vibrational entropy insight
    public let vibrationalInsight: CrossPlatformVibrationalInsight?
    /// Cleft druggability assessment (only when cleft features are provided)
    public let cleftAssessment: CrossPlatformCleftAssessment?
    /// Campaign summary (only when campaign context is provided)
    public let campaignSummary: CrossPlatformCampaignSummary?
    /// Feature names that failed during analysis, with error descriptions
    public let errors: [String]

    public init(
        refereeVerdict: CrossPlatformRefereeVerdict? = nil,
        modeNarrative: CrossPlatformModeNarrative? = nil,
        poseQualityReports: [CrossPlatformPoseQualityReport] = [],
        convergenceCoaching: CrossPlatformConvergenceCoaching? = nil,
        selectivityAnalysis: CrossPlatformSelectivityAnalysis? = nil,
        fleetExplanation: CrossPlatformFleetExplanation? = nil,
        healthInsight: CrossPlatformHealthEntropyInsight? = nil,
        vibrationalInsight: CrossPlatformVibrationalInsight? = nil,
        cleftAssessment: CrossPlatformCleftAssessment? = nil,
        campaignSummary: CrossPlatformCampaignSummary? = nil,
        errors: [String] = []
    ) {
        self.refereeVerdict = refereeVerdict
        self.modeNarrative = modeNarrative
        self.poseQualityReports = poseQualityReports
        self.convergenceCoaching = convergenceCoaching
        self.selectivityAnalysis = selectivityAnalysis
        self.fleetExplanation = fleetExplanation
        self.healthInsight = healthInsight
        self.vibrationalInsight = vibrationalInsight
        self.cleftAssessment = cleftAssessment
        self.campaignSummary = campaignSummary
        self.errors = errors
    }
}
