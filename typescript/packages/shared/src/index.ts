// index.ts — Shared types entry point
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

export type {
  BindingPopulation,
  BindingMode,
  Pose,
  Thermodynamics,
  TargetModification,
  HealthCorrelation,
} from './BindingPopulation.js';

export { serializePopulation, deserializePopulation } from './BindingPopulation.js';
