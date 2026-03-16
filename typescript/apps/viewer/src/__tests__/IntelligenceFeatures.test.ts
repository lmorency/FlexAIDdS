// IntelligenceFeatures.test.ts — TypeScript parity tests for Intelligence features
// Tests that TypeScript types mirror Swift CrossPlatform* types correctly.
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

// ─── Factory Functions ──────────────────────────────────────────────────────

function makeModeDescription(
  overrides: Partial<ModeDescription> = {},
): ModeDescription {
  return {
    characterization: 'Deep burial in orthosteric pocket with H-bond network',
    optimizationHint: 'Extend into sub-pocket with hydrophobic substituent',
    ...overrides,
  };
}

function makeBindingModeNarrative(
  overrides: Partial<BindingModeNarrative> = {},
): BindingModeNarrative {
  return {
    modeDescriptions: [
      makeModeDescription(),
      makeModeDescription({
        characterization: 'Shallow surface binding near allosteric site',
        optimizationHint: 'Add polar group to improve solvent-exposed contacts',
      }),
    ],
    selectivityInsight: 'Mode 1 dominates at 298K; mode 2 becomes relevant above 310K',
    confidence: 0.85,
    ...overrides,
  };
}

function makeCleftAssessment(
  overrides: Partial<CleftAssessment> = {},
): CleftAssessment {
  return {
    druggability: 'high',
    summary: 'Well-defined pocket with mixed hydrophobic/polar character',
    suggestedLigandProperties: 'MW 350-500, cLogP 2-4, 2-3 HBD',
    warnings: [],
    ...overrides,
  };
}

function makeConvergenceCoaching(
  overrides: Partial<ConvergenceCoaching> = {},
): ConvergenceCoaching {
  return {
    advice: 'continueRun',
    reasoning: 'Entropy still decreasing; population diversity adequate',
    estimatedGenerationsRemaining: 150,
    confidence: 0.72,
    ...overrides,
  };
}

function makeFleetExplanation(
  overrides: Partial<FleetExplanation> = {},
): FleetExplanation {
  return {
    allocationRationale: '70% compute on target A due to higher druggability score',
    bottleneckAnalysis: 'Target B cleft search consuming disproportionate cycles',
    actionItems: [
      'Increase population size for target A',
      'Reduce target B grid resolution',
    ],
    estimatedCompletion: '~45 minutes at current throughput',
    ...overrides,
  };
}

function makeHealthEntropyInsight(
  overrides: Partial<HealthEntropyInsight> = {},
): HealthEntropyInsight {
  return {
    correlationSummary: 'Binding entropy tracks inversely with receptor flexibility',
    wellnessRecommendation: 'Monitor S_vib contribution at physiological temperature',
    dataQualityNote: 'Based on 500+ converged poses; histogram well-populated',
    confidence: 0.91,
    ...overrides,
  };
}

function makeVibrationalInsight(
  overrides: Partial<VibrationalInsight> = {},
): VibrationalInsight {
  return {
    dominantMotionDescription: 'Hinge-bending mode between lobes at 12 cm^-1',
    bindingImpact: 'Ligand binding stiffens hinge, reducing S_vib by 1.2 kcal/mol',
    designImplication: 'Rigid analogs may improve affinity by pre-paying entropic cost',
    isEntropicallyDriven: true,
    ...overrides,
  };
}

function makeDeltaDeltaG(
  overrides: Partial<DeltaDeltaG> = {},
): DeltaDeltaG {
  return {
    targetA: 'D2R',
    targetB: '5-HT2A',
    ddg: -2.3,
    ...overrides,
  };
}

function makeSelectivityAnalysis(
  overrides: Partial<SelectivityAnalysis> = {},
): SelectivityAnalysis {
  return {
    preferredTarget: 'D2R',
    deltaG: -9.8,
    driver: 'enthalpic',
    explanation: 'Stronger van der Waals contacts in D2R orthosteric pocket',
    designSuggestion: 'Introduce fluorine at C-3 to exploit halogen bond with Ser193',
    ...overrides,
  };
}

function makeCampaignSummary(
  overrides: Partial<CampaignSummary> = {},
): CampaignSummary {
  return {
    campaignKey: 'D2R-aripiprazole-screen-2026Q1',
    runCount: 12,
    progressNarrative: '10 of 12 runs converged; 2 still refining entropy',
    bestResult: 'Run 7: dG = -11.2 kcal/mol, 3 binding modes, high confidence',
    trend: 'Affinity improving with successive scaffold modifications',
    nextStepRecommendation: 'Proceed to FEP validation for top 3 scaffolds',
    readyForPublication: false,
    ...overrides,
  };
}

function makePoseQualityReport(
  overrides: Partial<PoseQualityReport> = {},
): PoseQualityReport {
  return {
    topPoseSummary: 'Pose 1: dG = -10.5 kcal/mol, Boltzmann weight 0.62',
    poseConsensus: 'Top 5 poses cluster within 1.2 A RMSD',
    scoreWeightAlignment: 'Score and Boltzmann weight agree on rank order',
    confidenceInTopPose: 0.88,
    medicinalChemistryNote: 'No PAINS alerts; Lipinski-compliant',
    ...overrides,
  };
}

// ─── Tests ──────────────────────────────────────────────────────────────────

describe('BindingModeNarrative', () => {
  it('creates a well-typed narrative with mode descriptions', () => {
    const narrative = makeBindingModeNarrative();

    expect(narrative.modeDescriptions).toHaveLength(2);
    expect(narrative.modeDescriptions[0].characterization).toEqual(
      expect.any(String),
    );
    expect(narrative.modeDescriptions[0].optimizationHint).toEqual(
      expect.any(String),
    );
    expect(narrative.selectivityInsight).toEqual(expect.any(String));
    expect(narrative.confidence).toBeGreaterThanOrEqual(0.0);
    expect(narrative.confidence).toBeLessThanOrEqual(1.0);
  });

  it('supports empty mode descriptions array', () => {
    const narrative = makeBindingModeNarrative({ modeDescriptions: [] });

    expect(narrative.modeDescriptions).toHaveLength(0);
    expect(narrative.confidence).toEqual(expect.any(Number));
  });

  it('round-trips through JSON serialization', () => {
    const narrative = makeBindingModeNarrative();
    const json = JSON.stringify(narrative);
    const parsed: BindingModeNarrative = JSON.parse(json);

    expect(parsed.modeDescriptions).toHaveLength(
      narrative.modeDescriptions.length,
    );
    expect(parsed.modeDescriptions[0].characterization).toBe(
      narrative.modeDescriptions[0].characterization,
    );
    expect(parsed.modeDescriptions[0].optimizationHint).toBe(
      narrative.modeDescriptions[0].optimizationHint,
    );
    expect(parsed.selectivityInsight).toBe(narrative.selectivityInsight);
    expect(parsed.confidence).toBe(narrative.confidence);
  });
});

describe('CleftAssessment', () => {
  it('creates a well-typed assessment with druggability tier', () => {
    const assessment = makeCleftAssessment();

    expect(assessment.druggability).toBe('high');
    expect(assessment.summary).toEqual(expect.any(String));
    expect(assessment.suggestedLigandProperties).toEqual(expect.any(String));
    expect(assessment.warnings).toEqual(expect.any(Array));
  });

  it('accepts all DruggabilityTier values', () => {
    const tiers: DruggabilityTier[] = ['high', 'moderate', 'low', 'undruggable'];

    for (const tier of tiers) {
      const assessment = makeCleftAssessment({ druggability: tier });
      expect(assessment.druggability).toBe(tier);
    }
  });

  it('supports warnings array', () => {
    const assessment = makeCleftAssessment({
      warnings: ['Shallow pocket', 'High solvent exposure'],
    });

    expect(assessment.warnings).toHaveLength(2);
    expect(assessment.warnings[0]).toBe('Shallow pocket');
  });

  it('round-trips through JSON serialization', () => {
    const assessment = makeCleftAssessment({
      warnings: ['Flexible loop region near binding site'],
    });
    const json = JSON.stringify(assessment);
    const parsed: CleftAssessment = JSON.parse(json);

    expect(parsed.druggability).toBe(assessment.druggability);
    expect(parsed.summary).toBe(assessment.summary);
    expect(parsed.suggestedLigandProperties).toBe(
      assessment.suggestedLigandProperties,
    );
    expect(parsed.warnings).toEqual(assessment.warnings);
  });
});

describe('ConvergenceCoaching', () => {
  it('creates a well-typed coaching with GA advice', () => {
    const coaching = makeConvergenceCoaching();

    expect(coaching.advice).toBe('continueRun');
    expect(coaching.reasoning).toEqual(expect.any(String));
    expect(coaching.estimatedGenerationsRemaining).toBe(150);
    expect(coaching.confidence).toBeGreaterThanOrEqual(0.0);
    expect(coaching.confidence).toBeLessThanOrEqual(1.0);
  });

  it('accepts all GAAdvice values', () => {
    const advices: GAAdvice[] = [
      'continueRun',
      'stopEarly',
      'increasePopulation',
      'increaseMutationRate',
      'restart',
    ];

    for (const advice of advices) {
      const coaching = makeConvergenceCoaching({ advice });
      expect(coaching.advice).toBe(advice);
    }
  });

  it('supports null estimatedGenerationsRemaining', () => {
    const coaching = makeConvergenceCoaching({
      estimatedGenerationsRemaining: null,
    });

    expect(coaching.estimatedGenerationsRemaining).toBeNull();
  });

  it('round-trips through JSON serialization', () => {
    const coaching = makeConvergenceCoaching();
    const json = JSON.stringify(coaching);
    const parsed: ConvergenceCoaching = JSON.parse(json);

    expect(parsed.advice).toBe(coaching.advice);
    expect(parsed.reasoning).toBe(coaching.reasoning);
    expect(parsed.estimatedGenerationsRemaining).toBe(
      coaching.estimatedGenerationsRemaining,
    );
    expect(parsed.confidence).toBe(coaching.confidence);
  });

  it('round-trips null estimatedGenerationsRemaining through JSON', () => {
    const coaching = makeConvergenceCoaching({
      estimatedGenerationsRemaining: null,
    });
    const json = JSON.stringify(coaching);
    const parsed: ConvergenceCoaching = JSON.parse(json);

    expect(parsed.estimatedGenerationsRemaining).toBeNull();
  });
});

describe('FleetExplanation', () => {
  it('creates a well-typed explanation with action items', () => {
    const explanation = makeFleetExplanation();

    expect(explanation.allocationRationale).toEqual(expect.any(String));
    expect(explanation.bottleneckAnalysis).toEqual(expect.any(String));
    expect(explanation.actionItems).toHaveLength(2);
    expect(explanation.actionItems[0]).toEqual(expect.any(String));
    expect(explanation.estimatedCompletion).toEqual(expect.any(String));
  });

  it('supports empty action items', () => {
    const explanation = makeFleetExplanation({ actionItems: [] });

    expect(explanation.actionItems).toHaveLength(0);
  });

  it('round-trips through JSON serialization', () => {
    const explanation = makeFleetExplanation();
    const json = JSON.stringify(explanation);
    const parsed: FleetExplanation = JSON.parse(json);

    expect(parsed.allocationRationale).toBe(explanation.allocationRationale);
    expect(parsed.bottleneckAnalysis).toBe(explanation.bottleneckAnalysis);
    expect(parsed.actionItems).toEqual(explanation.actionItems);
    expect(parsed.estimatedCompletion).toBe(explanation.estimatedCompletion);
  });
});

describe('HealthEntropyInsight', () => {
  it('creates a well-typed insight with confidence', () => {
    const insight = makeHealthEntropyInsight();

    expect(insight.correlationSummary).toEqual(expect.any(String));
    expect(insight.wellnessRecommendation).toEqual(expect.any(String));
    expect(insight.dataQualityNote).toEqual(expect.any(String));
    expect(insight.confidence).toBeGreaterThanOrEqual(0.0);
    expect(insight.confidence).toBeLessThanOrEqual(1.0);
  });

  it('accepts boundary confidence values', () => {
    const low = makeHealthEntropyInsight({ confidence: 0.0 });
    const high = makeHealthEntropyInsight({ confidence: 1.0 });

    expect(low.confidence).toBe(0.0);
    expect(high.confidence).toBe(1.0);
  });

  it('round-trips through JSON serialization', () => {
    const insight = makeHealthEntropyInsight();
    const json = JSON.stringify(insight);
    const parsed: HealthEntropyInsight = JSON.parse(json);

    expect(parsed.correlationSummary).toBe(insight.correlationSummary);
    expect(parsed.wellnessRecommendation).toBe(insight.wellnessRecommendation);
    expect(parsed.dataQualityNote).toBe(insight.dataQualityNote);
    expect(parsed.confidence).toBe(insight.confidence);
  });
});

describe('VibrationalInsight', () => {
  it('creates a well-typed insight with isEntropicallyDriven boolean', () => {
    const insight = makeVibrationalInsight();

    expect(insight.dominantMotionDescription).toEqual(expect.any(String));
    expect(insight.bindingImpact).toEqual(expect.any(String));
    expect(insight.designImplication).toEqual(expect.any(String));
    expect(insight.isEntropicallyDriven).toBe(true);
  });

  it('accepts false for isEntropicallyDriven', () => {
    const insight = makeVibrationalInsight({ isEntropicallyDriven: false });

    expect(insight.isEntropicallyDriven).toBe(false);
  });

  it('round-trips through JSON serialization', () => {
    const insight = makeVibrationalInsight();
    const json = JSON.stringify(insight);
    const parsed: VibrationalInsight = JSON.parse(json);

    expect(parsed.dominantMotionDescription).toBe(
      insight.dominantMotionDescription,
    );
    expect(parsed.bindingImpact).toBe(insight.bindingImpact);
    expect(parsed.designImplication).toBe(insight.designImplication);
    expect(parsed.isEntropicallyDriven).toBe(insight.isEntropicallyDriven);
  });

  it('preserves boolean type through JSON (not coerced to number)', () => {
    const insight = makeVibrationalInsight({ isEntropicallyDriven: false });
    const json = JSON.stringify(insight);
    const parsed: VibrationalInsight = JSON.parse(json);

    expect(typeof parsed.isEntropicallyDriven).toBe('boolean');
  });
});

describe('SelectivityAnalysis', () => {
  it('creates a well-typed analysis with driver and delta-G', () => {
    const analysis = makeSelectivityAnalysis();

    expect(analysis.preferredTarget).toBe('D2R');
    expect(analysis.deltaG).toBe(-9.8);
    expect(analysis.driver).toBe('enthalpic');
    expect(analysis.explanation).toEqual(expect.any(String));
    expect(analysis.designSuggestion).toEqual(expect.any(String));
  });

  it('accepts all SelectivityDriver values', () => {
    const drivers: SelectivityDriver[] = [
      'enthalpic',
      'entropic',
      'mixed',
      'inconclusive',
    ];

    for (const driver of drivers) {
      const analysis = makeSelectivityAnalysis({ driver });
      expect(analysis.driver).toBe(driver);
    }
  });

  it('creates a well-typed DeltaDeltaG', () => {
    const ddg = makeDeltaDeltaG();

    expect(ddg.targetA).toBe('D2R');
    expect(ddg.targetB).toBe('5-HT2A');
    expect(ddg.ddg).toBe(-2.3);
    expect(typeof ddg.ddg).toBe('number');
  });

  it('round-trips SelectivityAnalysis through JSON serialization', () => {
    const analysis = makeSelectivityAnalysis();
    const json = JSON.stringify(analysis);
    const parsed: SelectivityAnalysis = JSON.parse(json);

    expect(parsed.preferredTarget).toBe(analysis.preferredTarget);
    expect(parsed.deltaG).toBe(analysis.deltaG);
    expect(parsed.driver).toBe(analysis.driver);
    expect(parsed.explanation).toBe(analysis.explanation);
    expect(parsed.designSuggestion).toBe(analysis.designSuggestion);
  });

  it('round-trips DeltaDeltaG through JSON serialization', () => {
    const ddg = makeDeltaDeltaG();
    const json = JSON.stringify(ddg);
    const parsed: DeltaDeltaG = JSON.parse(json);

    expect(parsed.targetA).toBe(ddg.targetA);
    expect(parsed.targetB).toBe(ddg.targetB);
    expect(parsed.ddg).toBe(ddg.ddg);
  });
});

describe('CampaignSummary', () => {
  it('creates a well-typed summary with readyForPublication boolean', () => {
    const summary = makeCampaignSummary();

    expect(summary.campaignKey).toEqual(expect.any(String));
    expect(summary.runCount).toBe(12);
    expect(summary.progressNarrative).toEqual(expect.any(String));
    expect(summary.bestResult).toEqual(expect.any(String));
    expect(summary.trend).toEqual(expect.any(String));
    expect(summary.nextStepRecommendation).toEqual(expect.any(String));
    expect(summary.readyForPublication).toBe(false);
  });

  it('accepts true for readyForPublication', () => {
    const summary = makeCampaignSummary({ readyForPublication: true });

    expect(summary.readyForPublication).toBe(true);
  });

  it('round-trips through JSON serialization', () => {
    const summary = makeCampaignSummary();
    const json = JSON.stringify(summary);
    const parsed: CampaignSummary = JSON.parse(json);

    expect(parsed.campaignKey).toBe(summary.campaignKey);
    expect(parsed.runCount).toBe(summary.runCount);
    expect(parsed.progressNarrative).toBe(summary.progressNarrative);
    expect(parsed.bestResult).toBe(summary.bestResult);
    expect(parsed.trend).toBe(summary.trend);
    expect(parsed.nextStepRecommendation).toBe(summary.nextStepRecommendation);
    expect(parsed.readyForPublication).toBe(summary.readyForPublication);
  });
});

describe('PoseQualityReport', () => {
  it('creates a well-typed report with confidenceInTopPose', () => {
    const report = makePoseQualityReport();

    expect(report.topPoseSummary).toEqual(expect.any(String));
    expect(report.poseConsensus).toEqual(expect.any(String));
    expect(report.scoreWeightAlignment).toEqual(expect.any(String));
    expect(report.confidenceInTopPose).toBeGreaterThanOrEqual(0.0);
    expect(report.confidenceInTopPose).toBeLessThanOrEqual(1.0);
    expect(report.medicinalChemistryNote).toEqual(expect.any(String));
  });

  it('accepts boundary confidence values', () => {
    const low = makePoseQualityReport({ confidenceInTopPose: 0.0 });
    const high = makePoseQualityReport({ confidenceInTopPose: 1.0 });

    expect(low.confidenceInTopPose).toBe(0.0);
    expect(high.confidenceInTopPose).toBe(1.0);
  });

  it('round-trips through JSON serialization', () => {
    const report = makePoseQualityReport();
    const json = JSON.stringify(report);
    const parsed: PoseQualityReport = JSON.parse(json);

    expect(parsed.topPoseSummary).toBe(report.topPoseSummary);
    expect(parsed.poseConsensus).toBe(report.poseConsensus);
    expect(parsed.scoreWeightAlignment).toBe(report.scoreWeightAlignment);
    expect(parsed.confidenceInTopPose).toBe(report.confidenceInTopPose);
    expect(parsed.medicinalChemistryNote).toBe(report.medicinalChemistryNote);
  });
});
