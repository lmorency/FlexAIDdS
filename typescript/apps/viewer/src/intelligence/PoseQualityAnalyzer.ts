// PoseQualityAnalyzer.ts — Rule-based pose quality critic
//
// Ports the Swift RuleBasedLigandFitCritic to TypeScript for web parity.
// Evaluates pose ensemble quality using Boltzmann weight distribution,
// CF score alignment, and RMSD spread.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import type { PoseQualityContext } from '@bonhomme/shared';
import type { PoseQualityReport } from '@bonhomme/shared';

// Named thresholds (matching Swift RuleBasedLigandFitCritic)
const STRONG_CONSENSUS_RMSD = 2.0;
const MODERATE_CONSENSUS_RMSD = 3.0;
const HIGH_CORRELATION = 0.8;
const MODERATE_CORRELATION = 0.4;
const LARGE_RMSD_SPREAD = 3.0;
const MAX_CONFIDENCE = 0.95;

export class PoseQualityAnalyzer {
  /**
   * Evaluate pose quality using threshold logic.
   */
  static evaluate(context: PoseQualityContext): PoseQualityReport {
    if (context.topPoses.length === 0) {
      return {
        topPoseSummary: 'No poses available for this binding mode.',
        poseConsensus: 'weak',
        scoreWeightAlignment: 'No data available.',
        confidenceInTopPose: 0.0,
        medicinalChemistryNote: 'No poses to evaluate. Check docking parameters and re-run.',
      };
    }

    const top = context.topPoses[0];

    // Top pose summary
    const topPoseSummary = `Pose ${top.rank + 1} (CF = ${top.cfScore.toFixed(1)} kcal/mol) carries ${(top.boltzmannWeight * 100).toFixed(0)}% Boltzmann weight at RMSD ${top.rmsdToCentroid.toFixed(1)} A from mode centroid.`;

    // Consensus
    let poseConsensus: string;
    if (context.hasDominantPose && context.rmsdSpread < STRONG_CONSENSUS_RMSD) {
      poseConsensus = 'strong';
    } else if (context.hasDominantPose || context.rmsdSpread < MODERATE_CONSENSUS_RMSD) {
      poseConsensus = 'moderate';
    } else if (context.topPoses.length >= 3 && !context.hasDominantPose) {
      poseConsensus = 'ambiguous';
    } else {
      poseConsensus = 'weak';
    }

    // Score-weight alignment
    let scoreWeightAlignment: string;
    if (context.scoreWeightCorrelation > HIGH_CORRELATION) {
      scoreWeightAlignment = 'Well aligned — best CF scores correspond to highest Boltzmann weights. Entropy and enthalpy agree.';
    } else if (context.scoreWeightCorrelation > MODERATE_CORRELATION) {
      scoreWeightAlignment = 'Partially aligned — some entropy-enthalpy tension in pose ranking.';
    } else {
      scoreWeightAlignment = 'Misaligned — entropy fights enthalpy. The energetically best pose is not the most thermodynamically populated. Consider ensemble-averaged analysis.';
    }

    // Confidence
    let confidence = 0.5;
    if (context.hasDominantPose) confidence += 0.2;
    if (context.scoreWeightAligned) confidence += 0.15;
    if (context.rmsdSpread < STRONG_CONSENSUS_RMSD) confidence += 0.1;
    if (context.totalPoses >= 10) confidence += 0.05;
    confidence = Math.min(Math.max(confidence, 0.0), MAX_CONFIDENCE);

    // Medicinal chemistry note
    let medicinalChemistryNote: string;
    if (poseConsensus === 'strong' && context.scoreWeightAligned) {
      medicinalChemistryNote = 'High-confidence binding pose. The predicted geometry is suitable for structure-based design and interaction analysis.';
    } else if (poseConsensus === 'ambiguous') {
      medicinalChemistryNote = 'Multiple competing poses with similar weights. Consider whether the ligand has internal symmetry or if the binding site accommodates multiple orientations. Validate with experimental data before committing to a single geometry.';
    } else if (!context.scoreWeightAligned) {
      medicinalChemistryNote = 'The energetically best pose differs from the thermodynamically most populated one. This suggests conformational entropy plays a significant role. Consider rigidifying the ligand scaffold to resolve the ambiguity.';
    } else if (context.rmsdSpread > LARGE_RMSD_SPREAD) {
      medicinalChemistryNote = `Large positional spread (${context.rmsdSpread.toFixed(1)} A) indicates the ligand samples multiple orientations within this mode. The binding orientation is not fully resolved — increase sampling or constrain the docking.`;
    } else {
      medicinalChemistryNote = 'Moderate pose quality. Results are usable for preliminary SAR but verify key interactions with molecular dynamics or experimental data.';
    }

    return { topPoseSummary, poseConsensus, scoreWeightAlignment, confidenceInTopPose: confidence, medicinalChemistryNote };
  }
}
