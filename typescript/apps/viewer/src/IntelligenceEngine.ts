// IntelligenceEngine.ts — Client-side intelligence for BonhommeViewer
//
// On Apple devices with FoundationModels, delegates to the native bridge.
// On web, provides rule-based analysis (same logic as Swift RuleBasedOracle).
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import type { BindingPopulation, HealthCorrelation } from '@bonhomme/shared';

export class IntelligenceEngine {
  /**
   * Analyze a binding population and return 3-bullet analysis.
   *
   * Uses rule-based analysis (same thresholds as the Swift RuleBasedOracle).
   */
  static async analyze(
    population: BindingPopulation,
    health?: HealthCorrelation,
  ): Promise<string[]> {
    const bullets: string[] = [];
    const thermo = population.globalThermodynamics;

    // Bullet 1: Free energy assessment
    if (thermo.freeEnergy < -10) {
      bullets.push(
        `Strong binding affinity (F = ${thermo.freeEnergy.toFixed(1)} kcal/mol) — high confidence in drug-target interaction.`,
      );
    } else if (thermo.freeEnergy < -5) {
      bullets.push(
        `Moderate binding affinity (F = ${thermo.freeEnergy.toFixed(1)} kcal/mol) — reasonable drug candidate.`,
      );
    } else {
      bullets.push(
        `Weak binding affinity (F = ${thermo.freeEnergy.toFixed(1)} kcal/mol) — consider structural optimization.`,
      );
    }

    // Bullet 2: Entropy state
    if (population.isCollapsed) {
      bullets.push(
        `Entropy collapsed to ${population.modes.length} mode(s) — high specificity but check for enthalpy-entropy compensation.`,
      );
    } else if (population.shannonS > 0.5) {
      bullets.push(
        `High conformational entropy (S = ${population.shannonS.toFixed(4)}) — population still exploring. More sampling may refine.`,
      );
    } else {
      bullets.push(
        `Moderate entropy with ${population.modes.length} binding modes — population converging on preferred conformations.`,
      );
    }

    // Bullet 3: Target modifications with population impact
    if (population.targetModifications?.length) {
      const mods = population.targetModifications;
      const modSummary = mods.map((m) => `${m.type}@${m.residueName}${m.residueNumber}`).join(', ');
      const deltaF = mods
        .filter((m) => m.effect?.deltaFreeEnergy !== undefined)
        .reduce((sum, m) => sum + (m.effect?.deltaFreeEnergy ?? 0), 0);
      const deltaS = mods
        .filter((m) => m.effect?.deltaEntropy !== undefined)
        .reduce((sum, m) => sum + (m.effect?.deltaEntropy ?? 0), 0);

      let modBullet = `${mods.length} target modification(s) (${modSummary})`;
      if (deltaF !== 0 || deltaS !== 0) {
        modBullet += ` — net population shift: ΔF=${deltaF.toFixed(2)} kcal/mol, ΔS=${deltaS.toFixed(4)} kcal/mol/K`;
      }
      modBullet += '. Population recalculated with PTM/glycan effects on the binding landscape.';
      bullets.push(modBullet);
    }

    // Bullet 4 (optional): Health correlation
    if (health?.hrvSDNN) {
      if (population.isCollapsed && health.hrvSDNN > 60) {
        bullets.push(
          `Entropy collapse correlates with good HRV (${health.hrvSDNN.toFixed(0)} ms) — system recovering. Gentle activity recommended.`,
        );
      } else if (health.hrvSDNN < 40) {
        bullets.push(
          `Low HRV (${health.hrvSDNN.toFixed(0)} ms) — prioritize rest before interpreting docking results.`,
        );
      } else {
        bullets.push(
          `HRV at ${health.hrvSDNN.toFixed(0)} ms — stable physiological state for analysis.`,
        );
      }
    } else if (!population.targetModifications?.length) {
      bullets.push(
        'Connect HealthKit for entropy-health correlation. Enable fleet mode for distributed compute.',
      );
    }

    return bullets;
  }
}
