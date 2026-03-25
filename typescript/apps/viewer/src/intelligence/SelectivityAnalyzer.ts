// SelectivityAnalyzer.ts — Rule-based multi-target selectivity analysis
//
// Ports the Swift RuleBasedSelectivityAnalyst to TypeScript for web parity.
// Compares docking results across protein targets to determine
// selectivity drivers (enthalpic vs entropic).
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import type { SelectivityContext } from '@bonhomme/shared';
import type { SelectivityDriver, SelectivityAnalysis } from '@bonhomme/shared';

// Named thresholds (matching Swift RuleBasedSelectivityAnalyst)
const INCONCLUSIVE_ENTHALPIC_THRESHOLD = 0.5; // kcal/mol
const INCONCLUSIVE_ENTROPIC_THRESHOLD = 0.0001; // kcal/mol/K
const ENTROPIC_DOMINANCE_RATIO = 1.3;
const ENTHALPIC_SIGNIFICANCE_THRESHOLD = 2.0; // kcal/mol
const kB = 0.001987206;

export class SelectivityAnalyzer {
  /**
   * Analyze selectivity across targets using threshold logic.
   */
  static analyze(context: SelectivityContext): SelectivityAnalysis {
    if (context.targets.length < 2) {
      return {
        preferredTarget: context.targets[0]?.targetName ?? 'unknown',
        deltaG: 0,
        driver: 'inconclusive',
        explanation: 'Only one target available — selectivity analysis requires at least two targets.',
        designSuggestion: 'Dock against additional targets for selectivity analysis.',
      };
    }

    const sorted = [...context.targets].sort((a, b) => a.bestFreeEnergy - b.bestFreeEnergy);
    const preferred = sorted[0];
    const second = sorted[1];
    const ddg = preferred.bestFreeEnergy - second.bestFreeEnergy;

    // Determine driver
    const sConfPhysA = preferred.sConf * kB;
    const sConfPhysB = second.sConf * kB;
    const entropicDiff = Math.abs(sConfPhysA - sConfPhysB);
    const enthalpicDiff = Math.abs(ddg);

    let driver: SelectivityDriver;
    if (enthalpicDiff < INCONCLUSIVE_ENTHALPIC_THRESHOLD && entropicDiff < INCONCLUSIVE_ENTROPIC_THRESHOLD) {
      driver = 'inconclusive';
    } else if (entropicDiff > enthalpicDiff * 0.001 && preferred.sConf > second.sConf * ENTROPIC_DOMINANCE_RATIO) {
      driver = 'entropic';
    } else if (enthalpicDiff > ENTHALPIC_SIGNIFICANCE_THRESHOLD) {
      driver = 'enthalpic';
    } else {
      driver = 'mixed';
    }

    // Explanation
    let explanation = `${context.ligandName} prefers ${preferred.targetName} (F = ${preferred.bestFreeEnergy.toFixed(1)} kcal/mol) over ${second.targetName} (F = ${second.bestFreeEnergy.toFixed(1)} kcal/mol). `;
    explanation += `DDG = ${ddg.toFixed(2)} kcal/mol. `;

    switch (driver) {
      case 'entropic':
        explanation += `Selectivity is entropy-driven: ${preferred.targetName} has broader conformational ensemble (S_conf = ${preferred.sConf.toFixed(4)} vs ${second.sConf.toFixed(4)} nats).`;
        break;
      case 'enthalpic':
        explanation += `Selectivity is enthalpy-driven: stronger direct interactions at ${preferred.targetName}.`;
        break;
      case 'mixed':
        explanation += 'Both enthalpy and entropy contribute to selectivity.';
        break;
      case 'inconclusive':
        explanation += 'Selectivity is marginal — results may not be significant.';
        break;
    }

    // Design suggestion
    let designSuggestion: string;
    switch (driver) {
      case 'entropic':
        designSuggestion = `Rigidify ligand to reduce entropy-driven selectivity for ${preferred.targetName}, or add flexible groups to enhance selectivity.`;
        break;
      case 'enthalpic':
        designSuggestion = `Optimize interaction geometry at ${second.targetName} binding site to improve affinity and shift selectivity.`;
        break;
      case 'mixed':
        designSuggestion = 'Balanced selectivity — consider fragment-based approach targeting unique pocket features of each receptor.';
        break;
      case 'inconclusive':
        designSuggestion = 'Increase sampling (more GA generations) and verify convergence before drawing selectivity conclusions.';
        break;
    }

    // Convergence gating
    const convergedBoth = preferred.isConverged && second.isConverged;
    if (!convergedBoth) {
      const notConverged = [preferred, second].filter((t) => !t.isConverged).map((t) => t.targetName);
      return {
        preferredTarget: preferred.targetName,
        deltaG: ddg,
        driver: 'inconclusive',
        explanation: explanation + ` Note: ${notConverged.join(', ')} not converged — selectivity assessment unreliable.`,
        designSuggestion: 'Achieve convergence at all targets before selectivity-driven optimization.',
      };
    }

    return { preferredTarget: preferred.targetName, deltaG: ddg, driver, explanation, designSuggestion };
  }
}
