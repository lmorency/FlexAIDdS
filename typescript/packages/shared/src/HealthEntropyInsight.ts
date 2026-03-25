// HealthEntropyInsight.ts — Cross-platform health-entropy insight types
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

/** Insight linking binding entropy to health metrics. */
export interface HealthEntropyInsight {
  /** Summary of the entropy-health correlation */
  correlationSummary: string;
  /** Wellness recommendation based on the correlation */
  wellnessRecommendation: string;
  /** Note on the quality of the underlying data */
  dataQualityNote: string;
  /** Confidence in the insight (0.0-1.0) */
  confidence: number;
}
