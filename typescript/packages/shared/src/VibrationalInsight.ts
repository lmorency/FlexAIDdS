// VibrationalInsight.ts — Cross-platform vibrational insight types
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

/** Insight into vibrational modes and their impact on binding. */
export interface VibrationalInsight {
  /** Description of the dominant motion */
  dominantMotionDescription: string;
  /** Impact on binding thermodynamics */
  bindingImpact: string;
  /** Implication for drug design */
  designImplication: string;
  /** Whether binding is entropically driven */
  isEntropicallyDriven: boolean;
}
