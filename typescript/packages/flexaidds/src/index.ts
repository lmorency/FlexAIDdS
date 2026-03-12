// index.ts — FlexAIDdS TypeScript SDK entry point
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

export type {
  ThermodynamicResult,
  VibrationalEntropyResult,
  WHAMBinResult,
  TIPoint,
  PoseResult,
  BindingModeResult,
  DockingResult,
  BindingEntropyScore,
  WorkChunk,
  DeviceCapability,
} from './types.js';

export { StatMechEngine } from './StatMechEngine.js';
export { parseResultFile, parseDockingResultJSON } from './resultLoader.js';
export { kB_kcal, kB_SI, hbar_SI, NA } from './constants.js';
