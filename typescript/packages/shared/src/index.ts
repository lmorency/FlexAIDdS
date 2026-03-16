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
  ShannonEntropyDecomposition,
} from './BindingPopulation.js';

export type {
  RefereeSeverity,
  RefereeCategory,
  RefereeFinding,
  RefereeVerdict,
  TemperatureSensitivity,
  ComparativeVerdict,
} from './RefereeVerdict.js';

export type {
  ModeDescription,
  BindingModeNarrative,
} from './BindingModeNarrative.js';

export type {
  DruggabilityTier,
  CleftAssessment,
} from './CleftAssessment.js';

export type {
  GAAdvice,
  ConvergenceCoaching,
} from './ConvergenceCoaching.js';

export type {
  FleetExplanation,
} from './FleetExplanation.js';

export type {
  HealthEntropyInsight,
} from './HealthEntropyInsight.js';

export type {
  VibrationalInsight,
} from './VibrationalInsight.js';

export type {
  SelectivityDriver,
  DeltaDeltaG,
  SelectivityAnalysis,
} from './SelectivityAnalysis.js';

export type {
  CampaignSummary,
} from './CampaignSummary.js';

export type {
  PoseQualityReport,
} from './PoseQualityReport.js';

export { serializePopulation, deserializePopulation } from './BindingPopulation.js';
