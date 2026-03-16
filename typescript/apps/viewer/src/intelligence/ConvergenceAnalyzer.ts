// ConvergenceAnalyzer.ts — Rule-based GA convergence coaching
//
// Ports the Swift RuleBasedConvergenceCoach to TypeScript for web parity.
// Monitors GA progress and advises when to stop, adjust parameters,
// or continue running.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import type { GAProgressSnapshot } from '@bonhomme/shared';
import type { ConvergenceCoaching, GAAdvice } from '@bonhomme/shared';

export class ConvergenceAnalyzer {
  /**
   * Assess GA convergence and provide coaching advice.
   */
  static coach(snapshot: GAProgressSnapshot): ConvergenceCoaching {
    const progress = snapshot.currentGeneration / Math.max(snapshot.maxGenerations, 1);
    const stagnationRatio =
      snapshot.generationsSinceImprovement / Math.max(snapshot.maxGenerations, 1);

    // Early run: always continue
    if (progress < 0.2) {
      return {
        advice: 'continueRun',
        reasoning: `Run is early (${(progress * 100).toFixed(0)}% complete). Allow more exploration time.`,
        estimatedGenerationsRemaining: snapshot.maxGenerations - snapshot.currentGeneration,
        confidence: 0.7,
      };
    }

    // Diversity collapsed + stagnated -> restart or increase mutation
    if (snapshot.isDiversityCollapsed && stagnationRatio > 0.15) {
      if (snapshot.populationSize < 200) {
        return {
          advice: 'increasePopulation',
          reasoning: `Diversity collapsed with only ${snapshot.populationSize} chromosomes. Increase population to maintain exploration.`,
          estimatedGenerationsRemaining: null,
          confidence: 0.8,
        };
      }
      return {
        advice: 'increaseMutationRate',
        reasoning: `Diversity collapsed at generation ${snapshot.currentGeneration} despite ${snapshot.populationSize} chromosomes. Increase mutation rate to escape local minimum.`,
        estimatedGenerationsRemaining: null,
        confidence: 0.75,
      };
    }

    // Long stagnation -> stop early
    if (stagnationRatio > 0.25 && !snapshot.isImproving) {
      return {
        advice: 'stopEarly',
        reasoning: `No improvement for ${snapshot.generationsSinceImprovement} generations (${(stagnationRatio * 100).toFixed(0)}% of run). Best fitness ${snapshot.bestFitness.toFixed(2)} kcal/mol is likely the global optimum.`,
        estimatedGenerationsRemaining: 0,
        confidence: 0.85,
      };
    }

    // Still improving -> continue
    if (snapshot.isImproving) {
      const remaining = snapshot.maxGenerations - snapshot.currentGeneration;
      return {
        advice: 'continueRun',
        reasoning: `Fitness still improving. Best: ${snapshot.bestFitness.toFixed(2)} kcal/mol. ${remaining} generations remaining.`,
        estimatedGenerationsRemaining: remaining,
        confidence: 0.8,
      };
    }

    // Default: continue but with low confidence
    return {
      advice: 'continueRun',
      reasoning: `Run at ${(progress * 100).toFixed(0)}% progress. Fitness plateaued but diversity maintained — may still improve.`,
      estimatedGenerationsRemaining: snapshot.maxGenerations - snapshot.currentGeneration,
      confidence: 0.5,
    };
  }
}
