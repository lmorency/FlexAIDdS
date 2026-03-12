// types.ts — TypeScript interfaces matching Swift models 1:1
//
// These types are the canonical TypeScript representation of FlexAIDdS
// data structures. They match the Swift Codable output format for
// cross-platform data exchange.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

/** Full thermodynamic analysis of a conformational ensemble. */
export interface ThermodynamicResult {
  /** Temperature in Kelvin */
  temperature: number;
  /** Natural log of the partition function ln(Z) */
  logZ: number;
  /** Helmholtz free energy F = -kT ln Z (kcal/mol) */
  freeEnergy: number;
  /** Boltzmann-weighted mean energy <E> (kcal/mol) */
  meanEnergy: number;
  /** Mean squared energy <E^2> */
  meanEnergySq: number;
  /** Heat capacity C_v */
  heatCapacity: number;
  /** Conformational entropy S (kcal mol^-1 K^-1) */
  entropy: number;
  /** Standard deviation of energy (kcal/mol) */
  stdEnergy: number;
}

/** Vibrational entropy result from ENCoM calculation. */
export interface VibrationalEntropyResult {
  /** Vibrational entropy (kcal mol^-1 K^-1) */
  entropy: number;
  /** Vibrational entropy in SI units (J mol^-1 K^-1) */
  entropySI: number;
  /** Effective frequency omega_eff (rad/s) */
  effectiveFrequency: number;
  /** Number of non-zero normal modes (3N - 6) */
  modeCount: number;
  /** Temperature (K) */
  temperature: number;
}

/** A bin from weighted histogram analysis (WHAM). */
export interface WHAMBinResult {
  /** Center of the coordinate bin */
  coordCenter: number;
  /** Count of samples in this bin */
  count: number;
  /** Free energy at this coordinate (kcal/mol) */
  freeEnergy: number;
}

/** A point for thermodynamic integration (TI). */
export interface TIPoint {
  /** Coupling parameter lambda in [0, 1] */
  lambda: number;
  /** Ensemble average <dV/dlambda> at this lambda */
  dVdLambda: number;
}

/** Lightweight view of a molecular pose. */
export interface PoseResult {
  /** Index in the chromosome array */
  chromIndex: number;
  /** OPTICS clustering order */
  order: number;
  /** Reachability distance */
  reachDist: number;
  /** Complementarity function score (kcal/mol) */
  cf: number;
}

/** Summary of a binding mode (cluster of poses). */
export interface BindingModeResult {
  /** Number of poses in this mode */
  size: number;
  /** Helmholtz free energy F (kcal/mol) */
  freeEnergy: number;
  /** Conformational entropy S (kcal mol^-1 K^-1) */
  entropy: number;
  /** Boltzmann-weighted mean energy (kcal/mol) */
  enthalpy: number;
  /** Heat capacity C_v */
  heatCapacity: number;
  /** Full thermodynamic result */
  thermodynamics?: ThermodynamicResult;
}

/** Complete docking result with binding population. */
export interface DockingResult {
  /** All binding modes, sorted by free energy */
  bindingModes: BindingModeResult[];
  /** Global ensemble thermodynamics */
  globalThermodynamics: ThermodynamicResult;
  /** Temperature (K) */
  temperature: number;
  /** GA population size */
  populationSize: number;
  /** ISO 8601 timestamp */
  timestamp: string;
}

/** Entropy-health correlation data. */
export interface BindingEntropyScore {
  id: string;
  /** Shannon configurational entropy */
  shannonS: number;
  /** Temperature (K) */
  temperature: number;
  /** Number of binding modes */
  bindingModeCount: number;
  /** Best free energy (kcal/mol) */
  bestFreeEnergy: number;
  /** Heat capacity */
  heatCapacity: number;
  /** HRV SDNN (ms), if available */
  hrvSDNN?: number;
  /** Resting heart rate (bpm) */
  restingHeartRate?: number;
  /** Sleep duration (hours) */
  sleepHours?: number;
  /** ISO 8601 timestamp */
  timestamp: string;
  /** Whether entropy has collapsed */
  isCollapsed: boolean;
}

/** Fleet work chunk for distributed docking. */
export interface WorkChunk {
  id: string;
  jobID: string;
  index: number;
  totalChunks: number;
  status: 'pending' | 'claimed' | 'running' | 'completed' | 'failed';
  claimedBy?: string;
  gaParameters: {
    numChromosomes: number;
    maxGenerations: number;
    seed: number;
    temperature: number;
  };
}

/** Fleet device capability. */
export interface DeviceCapability {
  deviceID: string;
  model: string;
  estimatedTFLOPS: number;
  hasGPU: boolean;
  availableMemoryGB: number;
  thermalState: 'nominal' | 'fair' | 'serious' | 'critical';
  computeWeight: number;
}
