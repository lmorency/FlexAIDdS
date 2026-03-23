// IntelligenceEngineReferee.test.ts — Parity tests for RuleBasedReferee
//
// Mirrors Swift ThermoRefereeTests with identical inputs and expected outputs.
// Verifies cross-platform deterministic referee produces the same findings.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect } from 'vitest';
import { RuleBasedReferee } from '../IntelligenceEngine.js';
import type {
  BindingPopulation,
  Thermodynamics,
  HealthCorrelation,
  ShannonEntropyDecomposition,
} from '@bonhomme/shared';

// ─── Helpers ────────────────────────────────────────────────────────────────

function makeThermo(overrides: Partial<Thermodynamics> = {}): Thermodynamics {
  return {
    temperature: 298.15,
    logZ: 10.0,
    freeEnergy: -8.0,
    meanEnergy: -7.0,
    meanEnergySq: 50.0,
    heatCapacity: 0.1,
    entropy: 0.005,
    stdEnergy: 1.0,
    ...overrides,
  };
}

function makeDecomp(overrides: Partial<ShannonEntropyDecomposition> = {}): ShannonEntropyDecomposition {
  return {
    configurational: 2.5,
    vibrational: 0.0005,
    entropyContribution: -1.5,
    isConverged: true,
    convergenceRate: 0.001,
    hardwareBackend: 'scalar',
    occupiedBins: 18,
    totalBins: 20,
    perModeEntropy: [0.5, 0.4, 0.3, 0.2, 0.1],
    dominantBins: [],
    ...overrides,
  };
}

function makePopulation(
  thermo: Thermodynamics,
  modeCount: number = 5,
): BindingPopulation {
  return {
    modes: Array.from({ length: modeCount }, (_, i) => ({
      index: i,
      size: 10,
      freeEnergy: thermo.freeEnergy + i * 0.5,
      entropy: 0.005,
      enthalpy: -7.0 + i * 0.3,
      heatCapacity: 0.1,
      probability: 1.0 / modeCount,
    })),
    globalThermodynamics: thermo,
    temperature: thermo.temperature,
    totalPoses: modeCount * 10,
    shannonS: 1.0,
    isCollapsed: false,
  };
}

function makeHealth(
  decomp?: ShannonEntropyDecomposition,
  overrides: Partial<HealthCorrelation> = {},
): HealthCorrelation {
  return {
    shannonS: 1.0,
    timestamp: new Date().toISOString(),
    shannonDecomposition: decomp,
    ...overrides,
  };
}

// ─── Tests ──────────────────────────────────────────────────────────────────

describe('RuleBasedReferee', () => {
  describe('Convergence', () => {
    it('flags critical when entropy not converged', () => {
      const decomp = makeDecomp({ isConverged: false, convergenceRate: 0.05 });
      const thermo = makeThermo();
      const pop = makePopulation(thermo);
      const health = makeHealth(decomp);

      const verdict = RuleBasedReferee.referee(pop, health);

      const convergence = verdict.findings.find((f) => f.category === 'convergence');
      expect(convergence).toBeDefined();
      expect(convergence!.severity).toBe('critical');
      expect(verdict.overallTrustworthy).toBe(false);
      expect(verdict.confidence).toBeLessThanOrEqual(0.5);
    });

    it('passes trust when converged with good histogram', () => {
      const decomp = makeDecomp({ isConverged: true, occupiedBins: 18, totalBins: 20 });
      const thermo = makeThermo({ freeEnergy: -12.0 });
      const pop = makePopulation(thermo);
      const health = makeHealth(decomp);

      const verdict = RuleBasedReferee.referee(pop, health);

      expect(verdict.overallTrustworthy).toBe(true);
      const convergence = verdict.findings.find((f) => f.category === 'convergence');
      expect(convergence!.severity).toBe('pass');
      expect(verdict.confidence).toBeGreaterThanOrEqual(0.8);
    });
  });

  describe('Histogram quality', () => {
    it('flags warning for sparse histogram (<50%)', () => {
      const decomp = makeDecomp({ occupiedBins: 8, totalBins: 20 });
      const thermo = makeThermo();
      const pop = makePopulation(thermo);
      const health = makeHealth(decomp);

      const verdict = RuleBasedReferee.referee(pop, health);

      const hist = verdict.findings.find((f) => f.category === 'histogram');
      expect(hist).toBeDefined();
      expect(hist!.severity).toBe('warning');
    });

    it('flags critical for critically sparse histogram (<30%)', () => {
      const decomp = makeDecomp({ occupiedBins: 4, totalBins: 20 });
      const thermo = makeThermo();
      const pop = makePopulation(thermo);
      const health = makeHealth(decomp);

      const verdict = RuleBasedReferee.referee(pop, health);

      const hist = verdict.findings.find((f) => f.category === 'histogram');
      expect(hist).toBeDefined();
      expect(hist!.severity).toBe('critical');
      expect(verdict.overallTrustworthy).toBe(false);
    });
  });

  describe('Entropy balance', () => {
    it('flags vibrational dominance when S_vib >> S_conf', () => {
      // S_vib = 0.01, S_conf_physical ≈ 0.5 * 0.001987 ≈ 0.001
      const decomp = makeDecomp({ configurational: 0.5, vibrational: 0.01 });
      const thermo = makeThermo();
      const pop = makePopulation(thermo);
      const health = makeHealth(decomp);

      const verdict = RuleBasedReferee.referee(pop, health);

      const balance = verdict.findings.find((f) => f.category === 'entropyBalance');
      expect(balance).toBeDefined();
      expect(balance!.severity).toBe('warning');
    });
  });

  describe('Mode imbalance', () => {
    it('detects 10x+ mode entropy ratio', () => {
      const decomp = makeDecomp({ perModeEntropy: [5.0, 0.3, 0.2] });
      const thermo = makeThermo();
      const pop = makePopulation(thermo);
      const health = makeHealth(decomp);

      const verdict = RuleBasedReferee.referee(pop, health);

      const mode = verdict.findings.find((f) => f.category === 'modeBalance');
      expect(mode).toBeDefined();
      expect(mode!.severity).toBe('warning');
      expect(mode!.detail).toContain('25.0x');
    });
  });

  describe('Enthalpy-entropy compensation', () => {
    it('flags advisory when F < -5 and S > 0.01', () => {
      const decomp = makeDecomp();
      const thermo = makeThermo({ freeEnergy: -12.0, entropy: 0.02 });
      const pop = makePopulation(thermo);
      const health = makeHealth(decomp);

      const verdict = RuleBasedReferee.referee(pop, health);

      const comp = verdict.findings.find((f) => f.category === 'compensation');
      expect(comp).toBeDefined();
      expect(comp!.severity).toBe('advisory');
    });
  });

  describe('No decomposition', () => {
    it('warns when ShannonThermoStack data unavailable', () => {
      const thermo = makeThermo();
      const pop = makePopulation(thermo);

      const verdict = RuleBasedReferee.referee(pop);

      const convergence = verdict.findings.find((f) => f.category === 'convergence');
      expect(convergence).toBeDefined();
      expect(convergence!.severity).toBe('warning');
      expect(convergence!.detail).toContain('ShannonThermoStack');
    });
  });

  describe('Binding affinity', () => {
    it('reports strong affinity as pass when converged', () => {
      const decomp = makeDecomp({ isConverged: true });
      const thermo = makeThermo({ freeEnergy: -15.0 });
      const pop = makePopulation(thermo);
      const health = makeHealth(decomp);

      const verdict = RuleBasedReferee.referee(pop, health);

      const affinity = verdict.findings.find((f) => f.category === 'affinity');
      expect(affinity).toBeDefined();
      expect(affinity!.severity).toBe('pass');
      expect(affinity!.title).toContain('Strong');
    });

    it('reports weak affinity as warning', () => {
      const decomp = makeDecomp({ isConverged: true });
      const thermo = makeThermo({ freeEnergy: -2.0 });
      const pop = makePopulation(thermo);
      const health = makeHealth(decomp);

      const verdict = RuleBasedReferee.referee(pop, health);

      const affinity = verdict.findings.find((f) => f.category === 'affinity');
      expect(affinity).toBeDefined();
      expect(affinity!.severity).toBe('warning');
      expect(affinity!.title).toContain('Weak');
    });
  });

  describe('Recommendations', () => {
    it('recommends against trusting when not trustworthy', () => {
      const decomp = makeDecomp({ isConverged: false });
      const thermo = makeThermo();
      const pop = makePopulation(thermo);
      const health = makeHealth(decomp);

      const verdict = RuleBasedReferee.referee(pop, health);

      expect(verdict.recommendedAction).toContain('Do not trust');
    });

    it('recommends proceeding when reliable', () => {
      const decomp = makeDecomp({ isConverged: true });
      const thermo = makeThermo({ freeEnergy: -8.0 });
      const pop = makePopulation(thermo);
      const health = makeHealth(decomp);

      const verdict = RuleBasedReferee.referee(pop, health);

      expect(verdict.recommendedAction).toMatch(/reliable|Proceed/);
    });
  });

  describe('Cross-platform verdict JSON', () => {
    it('round-trips through JSON serialization', () => {
      const decomp = makeDecomp();
      const thermo = makeThermo({ freeEnergy: -12.0 });
      const pop = makePopulation(thermo);
      const health = makeHealth(decomp);

      const verdict = RuleBasedReferee.referee(pop, health);
      const json = JSON.stringify(verdict);
      const parsed = JSON.parse(json);

      expect(parsed.findings).toHaveLength(verdict.findings.length);
      expect(parsed.overallTrustworthy).toBe(verdict.overallTrustworthy);
      expect(parsed.confidence).toBe(verdict.confidence);
      expect(parsed.recommendedAction).toBe(verdict.recommendedAction);
      expect(parsed.findings[0].title).toBe(verdict.findings[0].title);
      expect(parsed.findings[0].severity).toBe(verdict.findings[0].severity);
    });
  });
});
