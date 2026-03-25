// FleetExplanation.ts — Cross-platform fleet explanation types
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

/** Explanation of fleet resource allocation and bottleneck analysis. */
export interface FleetExplanation {
  /** Rationale for current resource allocation */
  allocationRationale: string;
  /** Analysis of current bottlenecks */
  bottleneckAnalysis: string;
  /** Recommended action items */
  actionItems: string[];
  /** Estimated completion time */
  estimatedCompletion: string;
}
