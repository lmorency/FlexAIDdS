// CleftAssessment.ts — Cross-platform cleft assessment types
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

/** Druggability tier for a binding cleft. */
export type DruggabilityTier = 'high' | 'moderate' | 'low' | 'undruggable';

/** Assessment of a binding cleft's druggability and properties. */
export interface CleftAssessment {
  /** Druggability classification */
  druggability: DruggabilityTier;
  /** Summary of the cleft assessment */
  summary: string;
  /** Suggested ligand properties for this cleft */
  suggestedLigandProperties: string;
  /** Warnings about the cleft */
  warnings: string[];
}
