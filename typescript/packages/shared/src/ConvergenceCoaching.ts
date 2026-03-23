// ConvergenceCoaching.ts — Cross-platform convergence coaching types
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

/** Advice for GA convergence strategy. */
export type GAAdvice = 'continueRun' | 'stopEarly' | 'increasePopulation' | 'increaseMutationRate' | 'restart';

/** Coaching recommendation for GA convergence. */
export interface ConvergenceCoaching {
  /** Recommended action */
  advice: GAAdvice;
  /** Reasoning behind the advice */
  reasoning: string;
  /** Estimated generations remaining, or null if unknown */
  estimatedGenerationsRemaining: number | null;
  /** Confidence in the coaching advice (0.0-1.0) */
  confidence: number;
}
