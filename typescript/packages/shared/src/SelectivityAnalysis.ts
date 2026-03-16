// SelectivityAnalysis.ts — Cross-platform selectivity analysis types
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

/** Driver of selectivity between targets. */
export type SelectivityDriver = 'enthalpic' | 'entropic' | 'mixed' | 'inconclusive';

/** Delta-delta-G between two targets. */
export interface DeltaDeltaG {
  /** First target identifier */
  targetA: string;
  /** Second target identifier */
  targetB: string;
  /** Delta-delta-G value (kcal/mol) */
  ddg: number;
}

/** Selectivity analysis across targets. */
export interface SelectivityAnalysis {
  /** Preferred target identifier */
  preferredTarget: string;
  /** Free energy of binding to preferred target (kcal/mol) */
  deltaG: number;
  /** Thermodynamic driver of selectivity */
  driver: SelectivityDriver;
  /** Explanation of selectivity */
  explanation: string;
  /** Suggestion for improving selectivity by design */
  designSuggestion: string;
}
