// resultLoader.ts — Parse FlexAID output files into TypeScript types
//
// Port of python/flexaidds/results_io.py for browser/Node.js usage.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import type { DockingResult, BindingModeResult, ThermodynamicResult } from './types.js';

/**
 * Parse a FlexAID .rrd (result) file into a DockingResult.
 *
 * File format (pipe-delimited):
 * ```
 * CLUSTER|1|CF|-10.50|POSES|25|RMSD|1.23
 * POSE|1|CF|-10.50|RANK|1|RMSD|0.00
 * ...
 * ```
 */
export function parseResultFile(content: string): DockingResult {
  const lines = content.split('\n').filter((l) => l.trim().length > 0);
  const modes: BindingModeResult[] = [];
  let currentMode: Partial<BindingModeResult> | null = null;
  let poseCount = 0;

  for (const line of lines) {
    const parts = line.split('|').map((p) => p.trim());

    if (parts[0] === 'CLUSTER') {
      // Save previous mode
      if (currentMode) {
        modes.push(finishMode(currentMode, poseCount));
      }

      currentMode = {};
      poseCount = 0;

      // Parse cluster fields
      for (let i = 1; i < parts.length - 1; i += 2) {
        const key = parts[i];
        const val = parts[i + 1];
        if (key === 'CF') currentMode.freeEnergy = parseFloat(val);
        if (key === 'POSES') poseCount = parseInt(val, 10);
      }
    } else if (parts[0] === 'POSE') {
      poseCount++;
    }
  }

  // Save last mode
  if (currentMode) {
    modes.push(finishMode(currentMode, poseCount));
  }

  // Compute global thermodynamics from all modes
  const globalThermo = computeGlobalThermo(modes);

  return {
    bindingModes: modes,
    globalThermodynamics: globalThermo,
    temperature: 300.0,  // Default; should be read from input file
    populationSize: modes.reduce((s, m) => s + m.size, 0),
    timestamp: new Date().toISOString(),
  };
}

/**
 * Parse a JSON docking result (as produced by Swift/Python serialization).
 */
export function parseDockingResultJSON(json: string): DockingResult {
  return JSON.parse(json) as DockingResult;
}

function finishMode(partial: Partial<BindingModeResult>, poseCount: number): BindingModeResult {
  return {
    size: poseCount || 1,
    freeEnergy: partial.freeEnergy ?? 0,
    entropy: partial.entropy ?? 0,
    enthalpy: partial.enthalpy ?? partial.freeEnergy ?? 0,
    heatCapacity: partial.heatCapacity ?? 0,
    thermodynamics: partial.thermodynamics,
  };
}

function computeGlobalThermo(modes: BindingModeResult[]): ThermodynamicResult {
  if (modes.length === 0) {
    return {
      temperature: 300, logZ: 0, freeEnergy: 0,
      meanEnergy: 0, meanEnergySq: 0,
      heatCapacity: 0, entropy: 0, stdEnergy: 0,
    };
  }

  // Simple aggregation — for accurate results, use StatMechEngine
  const totalSize = modes.reduce((s, m) => s + m.size, 0);
  const weightedEnergy = modes.reduce((s, m) => s + m.freeEnergy * m.size, 0) / totalSize;

  return {
    temperature: 300,
    logZ: 0,
    freeEnergy: Math.min(...modes.map((m) => m.freeEnergy)),
    meanEnergy: weightedEnergy,
    meanEnergySq: 0,
    heatCapacity: modes.reduce((s, m) => s + m.heatCapacity, 0) / modes.length,
    entropy: modes.reduce((s, m) => s + m.entropy, 0) / modes.length,
    stdEnergy: 0,
  };
}
