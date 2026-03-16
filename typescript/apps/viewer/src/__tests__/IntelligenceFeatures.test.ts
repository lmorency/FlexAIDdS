// IntelligenceFeatures.test.ts — TypeScript parity tests for Intelligence features
//
// Tests that TypeScript types mirror Swift CrossPlatform* types correctly.
// JSON round-trip and structural validation for all 9 features.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect } from 'vitest';
import type {
  ModeDescription,
  BindingModeNarrative,
  DruggabilityTier,
  CleftAssessment,
  GAAdvice,
  ConvergenceCoaching,
  FleetExplanation,
  HealthEntropyInsight,
  VibrationalInsight,
  SelectivityDriver,
  DeltaDeltaG,
  SelectivityAnalysis,
  CampaignSummary,
  PoseQualityReport,
} from '@bonhomme/shared';

// MARK: - BindingModeNarrative

describe('BindingModeNarrative', () => {
  it('creates well-typed narrative', () => {
    const desc: ModeDescription = {
      characterization: 'tight enthalpy-driven lock',
      optimizationHint: 'Add flexible substituents',
    };
    const narrative: BindingModeNarrative = {
      modeDescriptions: [desc],
      selectivityInsight: 'Mode 1 dominates (60% weight).',
      confidence: 0.8,
    };
    expect(narrative.modeDescriptions).toHaveLength(1);
    expect(narrative.confidence).toBe(0.8);
  });

  it('round-trips through JSON', () => {
    const original: BindingModeNarrative = {
      modeDescriptions: [
        { characterization: 'entropy-driven', optimizationHint: 'Rigidify' },
        { characterization: 'enthalpy-driven', optimizationHint: 'Add flexibility' },
      ],
      selectivityInsight: 'Focus on mode 1.',
      confidence: 0.75,
    };
    const decoded = JSON.parse(JSON.stringify(original)) as BindingModeNarrative;
    expect(decoded.modeDescriptions).toHaveLength(2);
    expect(decoded.modeDescriptions[0].characterization).toBe('entropy-driven');
    expect(decoded.confidence).toBeCloseTo(0.75);
  });
});

// MARK: - CleftAssessment

describe('CleftAssessment', () => {
  it('creates high druggability assessment', () => {
    const assessment: CleftAssessment = {
      druggability: 'high',
      summary: 'Medium oval pocket (600 Å³, 60% hydrophobic).',
      suggestedLigandProperties: 'Balanced compounds, MW 300-500 Da',
      warnings: [],
    };
    expect(assessment.druggability).toBe('high');
    expect(assessment.warnings).toHaveLength(0);
  });

  it('validates all druggability tiers', () => {
    const tiers: DruggabilityTier[] = ['high', 'moderate', 'low', 'undruggable'];
    for (const tier of tiers) {
      const a: CleftAssessment = {
        druggability: tier,
        summary: 'test',
        suggestedLigandProperties: 'test',
        warnings: [],
      };
      expect(a.druggability).toBe(tier);
    }
  });

  it('round-trips with warnings', () => {
    const original: CleftAssessment = {
      druggability: 'low',
      summary: 'Small pocket',
      suggestedLigandProperties: 'Fragment-like',
      warnings: ['Too small', 'High exposure'],
    };
    const decoded = JSON.parse(JSON.stringify(original)) as CleftAssessment;
    expect(decoded.warnings).toHaveLength(2);
    expect(decoded.warnings[0]).toBe('Too small');
  });
});

// MARK: - ConvergenceCoaching

describe('ConvergenceCoaching', () => {
  it('creates stop-early coaching', () => {
    const coaching: ConvergenceCoaching = {
      advice: 'stopEarly',
      reasoning: 'Stagnated for 300 generations.',
      estimatedGenerationsRemaining: 0,
      confidence: 0.85,
    };
    expect(coaching.advice).toBe('stopEarly');
    expect(coaching.estimatedGenerationsRemaining).toBe(0);
  });

  it('handles null estimatedGenerationsRemaining', () => {
    const coaching: ConvergenceCoaching = {
      advice: 'increasePopulation',
      reasoning: 'Diversity collapsed.',
      estimatedGenerationsRemaining: null,
      confidence: 0.8,
    };
    expect(coaching.estimatedGenerationsRemaining).toBeNull();
  });

  it('validates all advice types', () => {
    const advices: GAAdvice[] = [
      'continueRun', 'stopEarly', 'increasePopulation',
      'increaseMutationRate', 'restart',
    ];
    for (const advice of advices) {
      const c: ConvergenceCoaching = {
        advice,
        reasoning: 'test',
        estimatedGenerationsRemaining: null,
        confidence: 0.5,
      };
      expect(c.advice).toBe(advice);
    }
  });
});

// MARK: - FleetExplanation

describe('FleetExplanation', () => {
  it('creates fleet explanation', () => {
    const explanation: FleetExplanation = {
      allocationRationale: 'Mac Pro (40%), MacBook Air (30%), iPad (30%).',
      bottleneckAnalysis: 'MacBook Air thermally throttled.',
      actionItems: ['Allow MacBook Air to cool', 'Plug in iPad'],
      estimatedCompletion: '~30 minutes remaining.',
    };
    expect(explanation.actionItems).toHaveLength(2);
  });

  it('round-trips through JSON', () => {
    const original: FleetExplanation = {
      allocationRationale: 'By TFLOPS',
      bottleneckAnalysis: 'None',
      actionItems: ['No action needed'],
      estimatedCompletion: '~10 min',
    };
    const decoded = JSON.parse(JSON.stringify(original)) as FleetExplanation;
    expect(decoded.actionItems[0]).toBe('No action needed');
  });
});

// MARK: - HealthEntropyInsight

describe('HealthEntropyInsight', () => {
  it('creates insight with high confidence', () => {
    const insight: HealthEntropyInsight = {
      correlationSummary: 'Good HRV (70ms) correlates with converged results.',
      wellnessRecommendation: 'Health metrics are within normal range.',
      dataQualityNote: 'Results are reliable.',
      confidence: 0.8,
    };
    expect(insight.confidence).toBe(0.8);
  });

  it('round-trips through JSON', () => {
    const original: HealthEntropyInsight = {
      correlationSummary: 'Low HRV',
      wellnessRecommendation: 'Rest',
      dataQualityNote: 'Review with fresh eyes',
      confidence: 0.5,
    };
    const decoded = JSON.parse(JSON.stringify(original)) as HealthEntropyInsight;
    expect(decoded.confidence).toBeCloseTo(0.5);
  });
});

// MARK: - VibrationalInsight

describe('VibrationalInsight', () => {
  it('creates entropy-driven insight', () => {
    const insight: VibrationalInsight = {
      dominantMotionDescription: 'Very low-frequency collective breathing (mode 0, 20 cm⁻¹)',
      bindingImpact: 'Ligand binding restricts protein motion.',
      designImplication: 'Use smaller fragment to preserve loop flexibility.',
      isEntropicallyDriven: true,
    };
    expect(insight.isEntropicallyDriven).toBe(true);
  });

  it('round-trips boolean correctly', () => {
    const original: VibrationalInsight = {
      dominantMotionDescription: 'High-frequency side-chain vibration',
      bindingImpact: 'No significant restriction.',
      designImplication: 'Standard optimization.',
      isEntropicallyDriven: false,
    };
    const decoded = JSON.parse(JSON.stringify(original)) as VibrationalInsight;
    expect(decoded.isEntropicallyDriven).toBe(false);
  });
});

// MARK: - SelectivityAnalysis

describe('SelectivityAnalysis', () => {
  it('creates enthalpic selectivity result', () => {
    const analysis: SelectivityAnalysis = {
      preferredTarget: '5HT2A',
      deltaG: -3.5,
      driver: 'enthalpic',
      explanation: 'Stronger direct interactions at 5HT2A.',
      designSuggestion: 'Optimize interaction geometry at D2R.',
    };
    expect(analysis.driver).toBe('enthalpic');
    expect(analysis.deltaG).toBe(-3.5);
  });

  it('validates all selectivity drivers', () => {
    const drivers: SelectivityDriver[] = ['enthalpic', 'entropic', 'mixed', 'inconclusive'];
    for (const driver of drivers) {
      const a: SelectivityAnalysis = {
        preferredTarget: 'test',
        deltaG: -1.0,
        driver,
        explanation: 'test',
        designSuggestion: 'test',
      };
      expect(a.driver).toBe(driver);
    }
  });

  it('validates DeltaDeltaG type', () => {
    const ddg: DeltaDeltaG = {
      targetA: '5HT2A',
      targetB: 'D2R',
      ddg: -2.5,
    };
    expect(ddg.targetA).toBe('5HT2A');
    expect(ddg.ddg).toBe(-2.5);
  });

  it('round-trips through JSON', () => {
    const original: SelectivityAnalysis = {
      preferredTarget: '5HT2A',
      deltaG: -2.5,
      driver: 'entropic',
      explanation: 'Entropy-driven selectivity.',
      designSuggestion: 'Rigidify ligand.',
    };
    const decoded = JSON.parse(JSON.stringify(original)) as SelectivityAnalysis;
    expect(decoded.driver).toBe('entropic');
    expect(decoded.deltaG).toBeCloseTo(-2.5);
  });
});

// MARK: - CampaignSummary

describe('CampaignSummary', () => {
  it('creates publication-ready summary', () => {
    const summary: CampaignSummary = {
      campaignKey: '5HT2A-psilocin',
      runCount: 5,
      progressNarrative: 'Binding affinity improved across runs.',
      bestResult: 'Run 3: F = -12.50 kcal/mol (converged)',
      trend: 'improving',
      nextStepRecommendation: 'Continue optimization.',
      readyForPublication: true,
    };
    expect(summary.readyForPublication).toBe(true);
    expect(summary.runCount).toBe(5);
  });

  it('round-trips through JSON', () => {
    const original: CampaignSummary = {
      campaignKey: 'test-campaign',
      runCount: 3,
      progressNarrative: 'Stagnating.',
      bestResult: 'Run 1',
      trend: 'stagnating',
      nextStepRecommendation: 'Modify scaffold.',
      readyForPublication: false,
    };
    const decoded = JSON.parse(JSON.stringify(original)) as CampaignSummary;
    expect(decoded.campaignKey).toBe('test-campaign');
    expect(decoded.readyForPublication).toBe(false);
  });
});

// MARK: - PoseQualityReport

describe('PoseQualityReport', () => {
  it('creates strong-consensus report', () => {
    const report: PoseQualityReport = {
      topPoseSummary: 'Pose 1 (CF = -10.0 kcal/mol) carries 70% Boltzmann weight.',
      poseConsensus: 'strong',
      scoreWeightAlignment: 'Well aligned — entropy and enthalpy agree.',
      confidenceInTopPose: 0.9,
      medicinalChemistryNote: 'High-confidence binding pose.',
    };
    expect(report.poseConsensus).toBe('strong');
    expect(report.confidenceInTopPose).toBe(0.9);
  });

  it('round-trips through JSON', () => {
    const original: PoseQualityReport = {
      topPoseSummary: 'Pose 1',
      poseConsensus: 'ambiguous',
      scoreWeightAlignment: 'Misaligned',
      confidenceInTopPose: 0.45,
      medicinalChemistryNote: 'Validate experimentally.',
    };
    const decoded = JSON.parse(JSON.stringify(original)) as PoseQualityReport;
    expect(decoded.confidenceInTopPose).toBeCloseTo(0.45);
    expect(decoded.poseConsensus).toBe('ambiguous');
  });
});
