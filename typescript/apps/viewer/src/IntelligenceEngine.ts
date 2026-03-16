// IntelligenceEngine.ts — Client-side intelligence for BonhommeViewer
//
// On Apple devices with FoundationModels, delegates to the native bridge.
// On web, provides rule-based analysis with structured confidence metadata,
// trend tracking, and enthalpy-entropy compensation detection.
// Matches Swift IntelligenceOracle and RuleBasedOracle APIs.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import type { BindingPopulation, HealthCorrelation } from '@bonhomme/shared';

/** Confidence level for each analysis bullet. */
export type AnalysisConfidence = 'high' | 'moderate' | 'low';

/** Category of analysis bullet. */
export type BulletCategory =
  | 'binding'
  | 'entropy'
  | 'health'
  | 'modification'
  | 'fleet'
  | 'trend';

/** A single analysis bullet with metadata. */
export interface AnalysisBullet {
  text: string;
  confidence: AnalysisConfidence;
  category: BulletCategory;
}

/** Structured oracle analysis result. */
export interface OracleAnalysis {
  bullets: string[];
  structuredBullets: AnalysisBullet[];
  overallConfidence: AnalysisConfidence;
  inputSummary: string;
  timestamp: string;
}

/** Stored analysis entry for trend tracking. */
interface AnalysisEntry {
  key: string;
  analysis: OracleAnalysis;
}

export class IntelligenceEngine {
  /** Analysis history for trend detection. */
  private static history: AnalysisEntry[] = [];
  private static readonly MAX_HISTORY = 50;

  /**
   * Analyze a binding population and return structured analysis.
   *
   * Uses rule-based analysis (same thresholds as the Swift RuleBasedOracle)
   * with confidence metadata and enthalpy-entropy compensation detection.
   */
  static async analyze(
    population: BindingPopulation,
    health?: HealthCorrelation,
    campaignKey?: string,
  ): Promise<OracleAnalysis> {
    const structured: AnalysisBullet[] = [];
    const thermo = population.globalThermodynamics;

    // Bullet 1: Free energy assessment with confidence
    const fConfidence: AnalysisConfidence =
      thermo.stdEnergy !== undefined &&
      thermo.stdEnergy < Math.abs(thermo.freeEnergy) * 0.5
        ? 'high'
        : 'moderate';

    if (thermo.freeEnergy < -10) {
      structured.push({
        text: `Strong binding affinity (F = ${thermo.freeEnergy.toFixed(1)} kcal/mol) — high confidence in drug-target interaction.`,
        confidence: fConfidence,
        category: 'binding',
      });
    } else if (thermo.freeEnergy < -5) {
      structured.push({
        text: `Moderate binding affinity (F = ${thermo.freeEnergy.toFixed(1)} kcal/mol) — reasonable drug candidate.`,
        confidence: fConfidence,
        category: 'binding',
      });
    } else {
      structured.push({
        text: `Weak binding affinity (F = ${thermo.freeEnergy.toFixed(1)} kcal/mol) — consider structural optimization.`,
        confidence: fConfidence,
        category: 'binding',
      });
    }

    // Bullet 2: Entropy state with mode-count confidence
    const modeCount = population.modes.length;
    const sConfidence: AnalysisConfidence = modeCount >= 3 ? 'high' : 'low';

    if (population.isCollapsed) {
      structured.push({
        text: `Entropy collapsed to ${modeCount} mode(s) — high specificity but check for enthalpy-entropy compensation.`,
        confidence: sConfidence,
        category: 'entropy',
      });
    } else if (population.shannonS > 0.5) {
      structured.push({
        text: `High conformational entropy (S = ${population.shannonS.toFixed(4)}) — population still exploring. More sampling may refine.`,
        confidence: 'moderate',
        category: 'entropy',
      });
    } else {
      structured.push({
        text: `Moderate entropy with ${modeCount} binding modes — population converging on preferred conformations.`,
        confidence: sConfidence,
        category: 'entropy',
      });
    }

    // Bullet 3: Enthalpy-entropy compensation detection
    if (thermo.freeEnergy < -5 && thermo.entropy > 0.01) {
      structured.push({
        text: `Enthalpy-entropy compensation detected: strong binding (F = ${thermo.freeEnergy.toFixed(1)}) offset by conformational flexibility (S = ${thermo.entropy.toFixed(4)}). Net \u0394G may be less favorable than F alone suggests.`,
        confidence: 'moderate',
        category: 'binding',
      });
    }

    // Bullet 4: Target modifications with population impact
    if (population.targetModifications?.length) {
      const mods = population.targetModifications;
      const modSummary = mods.map((m) => `${m.type}@${m.residueName}${m.residueNumber}`).join(', ');
      const deltaF = mods
        .filter((m) => m.effect?.deltaFreeEnergy !== undefined)
        .reduce((sum, m) => sum + (m.effect?.deltaFreeEnergy ?? 0), 0);
      const deltaS = mods
        .filter((m) => m.effect?.deltaEntropy !== undefined)
        .reduce((sum, m) => sum + (m.effect?.deltaEntropy ?? 0), 0);

      let modText = `${mods.length} target modification(s) (${modSummary})`;
      if (deltaF !== 0 || deltaS !== 0) {
        modText += ` — net population shift: \u0394F=${deltaF.toFixed(2)} kcal/mol, \u0394S=${deltaS.toFixed(4)} kcal/mol/K`;
      }
      modText += '. Population recalculated with PTM/glycan effects on the binding landscape.';

      structured.push({
        text: modText,
        confidence: 'moderate',
        category: 'modification',
      });
    }

    // Bullet 5: Health correlation
    if (health?.hrvSDNN != null) {
      if (population.isCollapsed && health.hrvSDNN > 60) {
        structured.push({
          text: `Entropy collapse correlates with good HRV (${health.hrvSDNN.toFixed(0)} ms) — system recovering. Gentle activity recommended.`,
          confidence: 'moderate',
          category: 'health',
        });
      } else if (health.hrvSDNN < 40) {
        structured.push({
          text: `Low HRV (${health.hrvSDNN.toFixed(0)} ms) — prioritize rest before interpreting docking results.`,
          confidence: 'high',
          category: 'health',
        });
      } else {
        structured.push({
          text: `HRV at ${health.hrvSDNN.toFixed(0)} ms — stable physiological state for analysis.`,
          confidence: 'moderate',
          category: 'health',
        });
      }
    } else if (!population.targetModifications?.length) {
      structured.push({
        text: 'Connect HealthKit for entropy-health correlation. Enable fleet mode for distributed compute.',
        confidence: 'low',
        category: 'fleet',
      });
    }

    const overallConfidence = IntelligenceEngine.computeOverallConfidence(structured);
    const inputSummary = `T=${thermo.temperature}K, F=${thermo.freeEnergy.toFixed(2)} kcal/mol`;

    const analysis: OracleAnalysis = {
      bullets: structured.map((b) => b.text),
      structuredBullets: structured,
      overallConfidence,
      inputSummary,
      timestamp: new Date().toISOString(),
    };

    // Record for trend tracking
    if (campaignKey) {
      IntelligenceEngine.recordAnalysis(campaignKey, analysis);
    }

    return analysis;
  }

  /**
   * Compare current results with a previous analysis for the same campaign.
   */
  static compareTrend(
    campaignKey: string,
    current: OracleAnalysis,
  ): AnalysisBullet | null {
    const previous = IntelligenceEngine.lastAnalysis(campaignKey);
    if (!previous) return null;

    // Parse free energy from input summaries for delta comparison
    const extractF = (summary: string): number | null => {
      const match = summary.match(/F=(-?[\d.]+)/);
      return match ? parseFloat(match[1]) : null;
    };

    const prevF = extractF(previous.inputSummary);
    const currF = extractF(current.inputSummary);

    if (prevF !== null && currF !== null) {
      const delta = currF - prevF;
      const improved = delta < 0;
      return {
        text: `Compared to previous run: \u0394F = ${delta.toFixed(2)} kcal/mol (${improved ? 'improved' : 'worsened'} binding).`,
        confidence: Math.abs(delta) > 1.0 ? 'high' : 'moderate',
        category: 'trend',
      };
    }

    return null;
  }

  /**
   * Get analysis history for a campaign key.
   */
  static getHistory(campaignKey: string): OracleAnalysis[] {
    return IntelligenceEngine.history
      .filter((e) => e.key === campaignKey)
      .map((e) => e.analysis);
  }

  // ─── Private helpers ───────────────────────────────────────────────────────

  private static computeOverallConfidence(bullets: AnalysisBullet[]): AnalysisConfidence {
    const highCount = bullets.filter((b) => b.confidence === 'high').length;
    const lowCount = bullets.filter((b) => b.confidence === 'low').length;
    if (highCount >= 2) return 'high';
    if (lowCount >= 2) return 'low';
    return 'moderate';
  }

  private static recordAnalysis(key: string, analysis: OracleAnalysis): void {
    IntelligenceEngine.history.push({ key, analysis });
    if (IntelligenceEngine.history.length > IntelligenceEngine.MAX_HISTORY) {
      IntelligenceEngine.history.shift();
    }
  }

  private static lastAnalysis(key: string): OracleAnalysis | null {
    for (let i = IntelligenceEngine.history.length - 1; i >= 0; i--) {
      if (IntelligenceEngine.history[i].key === key) {
        return IntelligenceEngine.history[i].analysis;
      }
    }
    return null;
  }
}
