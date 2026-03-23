// CleftFeatures.ts — Cross-platform binding cleft feature types
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

/** Geometric features of a detected binding cleft for druggability assessment. */
export interface CleftFeatures {
  /** Volume in cubic Angstroms */
  volume: number;
  /** Effective pocket depth (Angstroms) */
  depth: number;
  /** Number of probe spheres defining the pocket */
  sphereCount: number;
  /** Largest probe sphere radius (Angstroms) */
  maxSphereRadius: number;
  /** Fraction of pocket surface that is hydrophobic (0.0-1.0) */
  hydrophobicFraction: number;
  /** Number of anchor residues lining the pocket */
  anchorResidueCount: number;
  /** Pocket shape: 0 = spherical, 1 = elongated */
  elongation: number;
  /** Fraction of pocket exposed to solvent (0.0-1.0) */
  solventExposure: number;
}
