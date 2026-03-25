// IntelligenceAnalyzers.test.ts — Tests for TypeScript rule-based intelligence analyzers
//
// Validates parity with Swift RuleBased* implementations using identical
// thresholds and expected outputs.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect } from 'vitest';
import { BindingModeAnalyzer } from '../intelligence/BindingModeAnalyzer.js';
import { CleftAnalyzer } from '../intelligence/CleftAnalyzer.js';
import { ConvergenceAnalyzer } from '../intelligence/ConvergenceAnalyzer.js';
import { SelectivityAnalyzer } from '../intelligence/SelectivityAnalyzer.js';
import { PoseQualityAnalyzer } from '../intelligence/PoseQualityAnalyzer.js';
import type { BindingPopulation, BindingMode } from '@bonhomme/shared';
import type { CleftFeatures } from '@bonhomme/shared';
import type { GAProgressSnapshot } from '@bonhomme/shared';
import type { SelectivityContext, TargetDockingSummary } from '@bonhomme/shared';
import type { PoseQualityContext, PoseProfile } from '@bonhomme/shared';

// ─── Factories ──────────────────────────────────────────────────────────────

function makeMode(overrides: Partial<BindingMode> = {}): BindingMode {
  return {
    index: 0,
    size: 15,
    freeEnergy: -8.5,
    entropy: 0.005,
    enthalpy: -10.0,
    heatCapacity: 0.02,
    probability: 0.6,
    ...overrides,
  };
}

function makePopulation(overrides: Partial<BindingPopulation> = {}): BindingPopulation {
  return {
    modes: [makeMode({ index: 0 }), makeMode({ index: 1, freeEnergy: -6.2, probability: 0.3 })],
    globalThermodynamics: {
      temperature: 298.15,
      logZ: 10.0,
      freeEnergy: -8.0,
      meanEnergy: -9.5,
      meanEnergySq: 92.0,
      heatCapacity: 0.03,
      entropy: 0.006,
      stdEnergy: 1.2,
    },
    temperature: 298.15,
    totalPoses: 50,
    shannonS: 0.4,
    isCollapsed: false,
    ...overrides,
  };
}

function makeCleftFeatures(overrides: Partial<CleftFeatures> = {}): CleftFeatures {
  return {
    volume: 500,
    depth: 7.0,
    sphereCount: 25,
    maxSphereRadius: 3.5,
    hydrophobicFraction: 0.55,
    anchorResidueCount: 6,
    elongation: 0.4,
    solventExposure: 0.25,
    ...overrides,
  };
}

function makeGASnapshot(overrides: Partial<GAProgressSnapshot> = {}): GAProgressSnapshot {
  return {
    currentGeneration: 50,
    maxGenerations: 200,
    bestFitness: -12.5,
    meanFitness: -8.3,
    populationDiversity: 0.65,
    generationsSinceImprovement: 5,
    fitnessTrajectory: [-10.0, -11.0, -11.5, -12.0, -12.5],
    diversityTrajectory: [0.8, 0.75, 0.7, 0.68, 0.65],
    isImproving: true,
    isDiversityCollapsed: false,
    populationSize: 300,
    ...overrides,
  };
}

function makeTarget(overrides: Partial<TargetDockingSummary> = {}): TargetDockingSummary {
  return {
    targetName: '5HT2A',
    bestFreeEnergy: -9.5,
    modeCount: 3,
    sConf: 1.8,
    sVib: 0.002,
    isConverged: true,
    cavityVolume: 450,
    populationSize: 500,
    ...overrides,
  };
}

function makeSelectivityContext(overrides: Partial<SelectivityContext> = {}): SelectivityContext {
  const targets = overrides.targets ?? [
    makeTarget({ targetName: '5HT2A', bestFreeEnergy: -9.5, sConf: 1.8 }),
    makeTarget({ targetName: 'D2R', bestFreeEnergy: -6.2, sConf: 1.2 }),
  ];
  return {
    ligandName: 'LSD',
    targets,
    deltaDeltas: [{ targetA: targets[0].targetName, targetB: targets[1].targetName, ddg: targets[0].bestFreeEnergy - targets[1].bestFreeEnergy }],
    ...overrides,
  };
}

function makePoseProfile(overrides: Partial<PoseProfile> = {}): PoseProfile {
  return {
    rank: 0,
    cfScore: -12.5,
    boltzmannWeight: 0.65,
    rmsdToCentroid: 0.8,
    ...overrides,
  };
}

function makePoseQualityContext(overrides: Partial<PoseQualityContext> = {}): PoseQualityContext {
  return {
    modeIndex: 0,
    topPoses: [
      makePoseProfile({ rank: 0, cfScore: -12.5, boltzmannWeight: 0.65, rmsdToCentroid: 0.8 }),
      makePoseProfile({ rank: 1, cfScore: -11.0, boltzmannWeight: 0.20, rmsdToCentroid: 1.2 }),
      makePoseProfile({ rank: 2, cfScore: -10.5, boltzmannWeight: 0.10, rmsdToCentroid: 1.5 }),
    ],
    totalPoses: 30,
    dominantPoseWeight: 0.65,
    hasDominantPose: true,
    rmsdSpread: 0.7,
    scoreWeightAligned: true,
    scoreWeightCorrelation: 0.95,
    modeFreeEnergy: -8.5,
    ...overrides,
  };
}

// ─── BindingModeAnalyzer ────────────────────────────────────────────────────

describe('BindingModeAnalyzer', () => {
  it('should generate narrative for two modes', () => {
    const result = BindingModeAnalyzer.analyze(makePopulation());
    expect(result.modeDescriptions).toHaveLength(2);
    expect(result.confidence).toBe(0.8);
    expect(result.selectivityInsight).toContain('Mode');
  });

  it('should classify enthalpy-driven mode', () => {
    const pop = makePopulation({
      modes: [makeMode({ enthalpy: -15.0, entropy: 0.001 })],
    });
    const result = BindingModeAnalyzer.analyze(pop);
    expect(result.modeDescriptions[0].characterization).toContain('enthalpy-driven');
    expect(result.confidence).toBe(0.5); // single mode
  });

  it('should classify entropy-driven mode', () => {
    const pop = makePopulation({
      modes: [makeMode({ enthalpy: -2.0, entropy: 0.05 })],
    });
    const result = BindingModeAnalyzer.analyze(pop);
    expect(result.modeDescriptions[0].characterization).toContain('entropy-driven');
  });

  it('should classify tight cluster', () => {
    const pop = makePopulation({
      modes: [makeMode({ size: 3 })],
    });
    const result = BindingModeAnalyzer.analyze(pop);
    expect(result.modeDescriptions[0].characterization).toContain('tight cluster');
  });
});

// ─── CleftAnalyzer ──────────────────────────────────────────────────────────

describe('CleftAnalyzer', () => {
  it('should assess high druggability for ideal pocket', () => {
    const result = CleftAnalyzer.assess(makeCleftFeatures());
    expect(result.druggability).toBe('high');
    expect(result.warnings).toHaveLength(0);
  });

  it('should assess undruggable for tiny shallow pocket', () => {
    const result = CleftAnalyzer.assess(makeCleftFeatures({
      volume: 80, depth: 2.0, hydrophobicFraction: 0.2,
      solventExposure: 0.7, anchorResidueCount: 1,
    }));
    expect(result.druggability).toBe('undruggable');
    expect(result.warnings.length).toBeGreaterThan(0);
  });

  it('should warn about high solvent exposure', () => {
    const result = CleftAnalyzer.assess(makeCleftFeatures({ solventExposure: 0.75 }));
    expect(result.warnings.some(w => w.includes('solvent'))).toBe(true);
  });

  it('should suggest lipophilic compounds for hydrophobic pocket', () => {
    const result = CleftAnalyzer.assess(makeCleftFeatures({ hydrophobicFraction: 0.85 }));
    expect(result.suggestedLigandProperties).toContain('Lipophilic');
  });

  it('should suggest polar compounds for hydrophilic pocket', () => {
    const result = CleftAnalyzer.assess(makeCleftFeatures({ hydrophobicFraction: 0.2 }));
    expect(result.suggestedLigandProperties).toContain('Polar');
  });
});

// ─── ConvergenceAnalyzer ────────────────────────────────────────────────────

describe('ConvergenceAnalyzer', () => {
  it('should advise continue for early run', () => {
    const result = ConvergenceAnalyzer.coach(makeGASnapshot({ currentGeneration: 10 }));
    expect(result.advice).toBe('continueRun');
    expect(result.confidence).toBe(0.7);
  });

  it('should advise stop early for long stagnation', () => {
    const result = ConvergenceAnalyzer.coach(makeGASnapshot({
      currentGeneration: 150,
      generationsSinceImprovement: 60,
      isImproving: false,
    }));
    expect(result.advice).toBe('stopEarly');
    expect(result.confidence).toBe(0.85);
  });

  it('should advise increase population on diversity collapse with small pop', () => {
    const result = ConvergenceAnalyzer.coach(makeGASnapshot({
      currentGeneration: 100,
      isDiversityCollapsed: true,
      generationsSinceImprovement: 40,
      populationSize: 100,
    }));
    expect(result.advice).toBe('increasePopulation');
  });

  it('should advise increase mutation rate on diversity collapse with large pop', () => {
    const result = ConvergenceAnalyzer.coach(makeGASnapshot({
      currentGeneration: 100,
      isDiversityCollapsed: true,
      generationsSinceImprovement: 40,
      populationSize: 500,
    }));
    expect(result.advice).toBe('increaseMutationRate');
  });

  it('should advise continue when still improving', () => {
    const result = ConvergenceAnalyzer.coach(makeGASnapshot({
      currentGeneration: 100,
      isImproving: true,
    }));
    expect(result.advice).toBe('continueRun');
    expect(result.estimatedGenerationsRemaining).toBe(100);
  });
});

// ─── SelectivityAnalyzer ────────────────────────────────────────────────────

describe('SelectivityAnalyzer', () => {
  it('should detect enthalpic driver for large DDG', () => {
    const result = SelectivityAnalyzer.analyze(makeSelectivityContext());
    expect(result.preferredTarget).toBe('5HT2A');
    expect(result.driver).toBe('enthalpic');
    expect(result.deltaG).toBeCloseTo(-3.3, 1);
  });

  it('should detect inconclusive for single target', () => {
    const result = SelectivityAnalyzer.analyze(makeSelectivityContext({
      targets: [makeTarget()],
      deltaDeltas: [],
    }));
    expect(result.driver).toBe('inconclusive');
    expect(result.explanation).toContain('Only one target');
  });

  it('should gate on convergence', () => {
    const result = SelectivityAnalyzer.analyze(makeSelectivityContext({
      targets: [
        makeTarget({ targetName: '5HT2A', isConverged: false }),
        makeTarget({ targetName: 'D2R', bestFreeEnergy: -6.0 }),
      ],
    }));
    expect(result.driver).toBe('inconclusive');
    expect(result.explanation).toContain('not converged');
  });

  it('should detect inconclusive for marginal differences', () => {
    const result = SelectivityAnalyzer.analyze(makeSelectivityContext({
      targets: [
        makeTarget({ targetName: '5HT2A', bestFreeEnergy: -8.0, sConf: 1.5 }),
        makeTarget({ targetName: 'D2R', bestFreeEnergy: -7.8, sConf: 1.5 }),
      ],
      deltaDeltas: [{ targetA: '5HT2A', targetB: 'D2R', ddg: -0.2 }],
    }));
    expect(result.driver).toBe('inconclusive');
  });
});

// ─── PoseQualityAnalyzer ────────────────────────────────────────────────────

describe('PoseQualityAnalyzer', () => {
  it('should report strong consensus for dominant aligned pose', () => {
    const result = PoseQualityAnalyzer.evaluate(makePoseQualityContext());
    expect(result.poseConsensus).toBe('strong');
    expect(result.confidenceInTopPose).toBeGreaterThan(0.8);
    expect(result.medicinalChemistryNote).toContain('High-confidence');
  });

  it('should report ambiguous for many non-dominant poses', () => {
    const result = PoseQualityAnalyzer.evaluate(makePoseQualityContext({
      hasDominantPose: false,
      dominantPoseWeight: 0.3,
      rmsdSpread: 4.0,
      topPoses: [
        makePoseProfile({ rank: 0, boltzmannWeight: 0.3 }),
        makePoseProfile({ rank: 1, boltzmannWeight: 0.25 }),
        makePoseProfile({ rank: 2, boltzmannWeight: 0.2 }),
      ],
    }));
    expect(result.poseConsensus).toBe('ambiguous');
  });

  it('should handle empty poses', () => {
    const result = PoseQualityAnalyzer.evaluate(makePoseQualityContext({
      topPoses: [],
      totalPoses: 0,
      hasDominantPose: false,
      dominantPoseWeight: 0,
    }));
    expect(result.poseConsensus).toBe('weak');
    expect(result.confidenceInTopPose).toBe(0.0);
  });

  it('should detect score-weight misalignment', () => {
    const result = PoseQualityAnalyzer.evaluate(makePoseQualityContext({
      scoreWeightAligned: false,
      scoreWeightCorrelation: 0.2,
    }));
    expect(result.scoreWeightAlignment).toContain('Misaligned');
    expect(result.medicinalChemistryNote).toContain('entropy');
  });

  it('should clamp confidence to max 0.95', () => {
    const result = PoseQualityAnalyzer.evaluate(makePoseQualityContext({
      hasDominantPose: true,
      scoreWeightAligned: true,
      rmsdSpread: 0.5,
      totalPoses: 100,
    }));
    expect(result.confidenceInTopPose).toBeLessThanOrEqual(0.95);
  });
});
