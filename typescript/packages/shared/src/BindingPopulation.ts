// BindingPopulation.ts — Canonical BindingPopulation interface for cross-platform exchange
//
// This is the single source of truth for binding population data shared between
// the Swift native layer, TypeScript PWA, and fleet dashboard.
// JSON serialization matches the Swift Codable output format.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

/** Global binding population — the ensemble of all binding modes. */
export interface BindingPopulation {
  /** All binding modes, sorted by free energy */
  modes: BindingMode[];

  /** Global ensemble thermodynamics */
  globalThermodynamics: Thermodynamics;

  /** Temperature used for the simulation (K) */
  temperature: number;

  /** Total number of poses across all modes */
  totalPoses: number;

  /** Shannon configurational entropy of the population */
  shannonS: number;

  /** Whether the population entropy has collapsed */
  isCollapsed: boolean;

  /** Delta-G landscape between modes (upper triangle) */
  deltaGMatrix?: number[][];

  /** Heat capacity curve data points (T vs Cv) */
  heatCapacityCurve?: Array<{ temperature: number; cv: number }>;

  /** PTM/glycan modifications on the target chain */
  targetModifications?: TargetModification[];
}

/** A single binding mode (cluster of poses). */
export interface BindingMode {
  /** Mode index (0-based) */
  index: number;

  /** Number of poses in this mode */
  size: number;

  /** Helmholtz free energy F (kcal/mol) */
  freeEnergy: number;

  /** Conformational entropy S (kcal mol^-1 K^-1) */
  entropy: number;

  /** Boltzmann-weighted mean energy <E> (kcal/mol) */
  enthalpy: number;

  /** Heat capacity C_v */
  heatCapacity: number;

  /** Boltzmann probability of this mode in the population */
  probability: number;

  /** Representative pose (lowest energy in cluster) */
  representative?: Pose;

  /** Full thermodynamic analysis */
  thermodynamics?: Thermodynamics;
}

/** A molecular pose. */
export interface Pose {
  /** Chromosome index in the GA */
  chromIndex: number;

  /** OPTICS clustering order */
  order: number;

  /** Complementarity function score (kcal/mol) */
  cf: number;

  /** Boltzmann weight within the binding mode */
  boltzmannWeight: number;

  /** Cartesian coordinates of the ligand atoms */
  coordinates?: number[][];
}

/** Thermodynamic properties of an ensemble. */
export interface Thermodynamics {
  /** Temperature (K) */
  temperature: number;
  /** ln(Z) */
  logZ: number;
  /** Helmholtz F (kcal/mol) */
  freeEnergy: number;
  /** <E> (kcal/mol) */
  meanEnergy: number;
  /** <E^2> */
  meanEnergySq: number;
  /** C_v */
  heatCapacity: number;
  /** S (kcal mol^-1 K^-1) */
  entropy: number;
  /** sigma_E (kcal/mol) */
  stdEnergy: number;
}

/** PTM or glycan modification on the target chain. */
export interface TargetModification {
  /** Modification type (e.g., "N-glycosylation", "phosphorylation") */
  type: string;

  /** Residue name (e.g., "ASN", "SER") */
  residueName: string;

  /** Residue number */
  residueNumber: number;

  /** Chain ID */
  chainID: string;

  /** Glycan composition (for glycosylation, e.g., "Man5GlcNAc2") */
  composition?: string;

  /** Effect on population (delta-F, delta-S) */
  effect?: {
    deltaFreeEnergy: number;
    deltaEntropy: number;
  };
}

/** Decomposed Shannon entropy from ShannonThermoStack for referee analysis. */
export interface ShannonEntropyDecomposition {
  /** Configurational entropy from GA ensemble histogram (nats) */
  configurational: number;
  /** Torsional vibrational entropy from ENCoM modes (kcal/mol/K) */
  vibrational: number;
  /** Combined -T*S entropy contribution to free energy (kcal/mol) */
  entropyContribution: number;
  /** Whether the Shannon entropy has reached a convergence plateau */
  isConverged: boolean;
  /** Relative change in entropy over the last convergence window */
  convergenceRate: number;
  /** Hardware backend used (e.g., "Metal", "AVX-512", "OpenMP", "scalar") */
  hardwareBackend: string;
  /** Number of non-zero histogram bins */
  occupiedBins: number;
  /** Total histogram bins used */
  totalBins: number;
  /** Per-binding-mode Shannon entropy breakdown (nats, indexed by mode) */
  perModeEntropy: number[];
  /** Top 5 most populated histogram bins */
  dominantBins: Array<{ center: number; probability: number }>;
}

/** Entropy-health correlation for cross-platform exchange. */
export interface HealthCorrelation {
  /** Binding entropy score */
  shannonS: number;
  /** HRV SDNN (ms) */
  hrvSDNN?: number;
  /** Sleep duration (hours) */
  sleepHours?: number;
  /** Resting heart rate (bpm) */
  restingHeartRate?: number;
  /** ISO 8601 timestamp */
  timestamp: string;
  /** Shannon entropy decomposition from ShannonThermoStack */
  shannonDecomposition?: ShannonEntropyDecomposition;
  /** Fitness recommendation */
  fitnessRecommendation?: {
    intensity: 'gentle' | 'moderate' | 'vigorous';
    activities: string[];
  };
  /** Intelligence oracle analysis (3-4 bullets) */
  oracleAnalysis?: string[];
}

/** Serialize a BindingPopulation to JSON (matching Swift Codable format). */
export function serializePopulation(pop: BindingPopulation): string {
  return JSON.stringify(pop, null, 2);
}

/** Deserialize a BindingPopulation from JSON. */
export function deserializePopulation(json: string): BindingPopulation {
  return JSON.parse(json) as BindingPopulation;
}
