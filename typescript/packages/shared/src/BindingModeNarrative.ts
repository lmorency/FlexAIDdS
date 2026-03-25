// BindingModeNarrative.ts — Cross-platform binding mode narrative types
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

/** Description of a single binding mode with characterization and optimization hint. */
export interface ModeDescription {
  /** Characterization of this binding mode */
  characterization: string;
  /** Optimization hint for improving this mode */
  optimizationHint: string;
}

/** Narrative summary of binding modes with selectivity insight. */
export interface BindingModeNarrative {
  /** Descriptions for each binding mode */
  modeDescriptions: ModeDescription[];
  /** Selectivity insight across modes */
  selectivityInsight: string;
  /** Confidence in the narrative (0.0-1.0) */
  confidence: number;
}
