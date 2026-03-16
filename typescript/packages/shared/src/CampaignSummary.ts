// CampaignSummary.ts — Cross-platform campaign summary types
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

/** Summary of a docking campaign. */
export interface CampaignSummary {
  /** Unique campaign key */
  campaignKey: string;
  /** Number of runs in the campaign */
  runCount: number;
  /** Narrative of campaign progress */
  progressNarrative: string;
  /** Best result description */
  bestResult: string;
  /** Trend across runs */
  trend: string;
  /** Recommended next step */
  nextStepRecommendation: string;
  /** Whether the campaign is ready for publication */
  readyForPublication: boolean;
}
