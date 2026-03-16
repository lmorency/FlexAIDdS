// GAProgressSnapshot.ts — Cross-platform GA progress snapshot types
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

/** Snapshot of GA generational progress for convergence analysis. */
export interface GAProgressSnapshot {
  /** Current generation number */
  currentGeneration: number;
  /** Maximum generations configured */
  maxGenerations: number;
  /** Best fitness (CF score, negative = better) at current generation */
  bestFitness: number;
  /** Mean fitness of current population */
  meanFitness: number;
  /** Population diversity (Shannon entropy of fitness distribution) */
  populationDiversity: number;
  /** Generations since last improvement in best fitness */
  generationsSinceImprovement: number;
  /** Best fitness trajectory (last 10 generations, oldest first) */
  fitnessTrajectory: number[];
  /** Diversity trajectory (last 10 generations, oldest first) */
  diversityTrajectory: number[];
  /** Whether fitness is still improving (slope of recent trajectory) */
  isImproving: boolean;
  /** Whether diversity has collapsed (below threshold) */
  isDiversityCollapsed: boolean;
  /** Population size (number of chromosomes) */
  populationSize: number;
}
