// PoseQualityReport.ts — Cross-platform pose quality report types
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

/** Quality report for the top pose and pose ensemble. */
export interface PoseQualityReport {
  /** Summary of the top-ranked pose */
  topPoseSummary: string;
  /** Consensus assessment across the pose ensemble */
  poseConsensus: string;
  /** Alignment between score and Boltzmann weight */
  scoreWeightAlignment: string;
  /** Confidence in the top pose (0.0-1.0) */
  confidenceInTopPose: number;
  /** Medicinal chemistry note for the top pose */
  medicinalChemistryNote: string;
}
