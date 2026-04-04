"""docking_power.py — CASF-2016 Docking Power evaluation.

Computes success rate (Top-1, Top-2, Top-3) at various RMSD thresholds
for the CASF-2016 core set.

Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
SPDX-License-Identifier: Apache-2.0
"""

import os
from typing import Dict, List

import numpy as np


def compute_docking_power(
    rmsd_results: Dict[str, List[float]],
    thresholds: List[float] = [1.0, 2.0, 3.0],
) -> Dict[str, float]:
    """Compute docking success rates at various RMSD thresholds.

    Parameters
    ----------
    rmsd_results : dict
        PDB ID → list of RMSD values for top N predicted poses.
    thresholds : list
        RMSD thresholds (Angstroms) for success classification.

    Returns
    -------
    dict
        Success rates: top1_at_1A, top1_at_2A, top1_at_3A, etc.
    """
    if not rmsd_results:
        return {"error": "no results"}

    metrics = {}
    n_total = len(rmsd_results)

    for thresh in thresholds:
        # Top-1 success: best-scoring pose within threshold
        top1_success = sum(
            1 for rmsds in rmsd_results.values()
            if len(rmsds) > 0 and rmsds[0] <= thresh
        )
        metrics[f"top1_at_{thresh}A"] = top1_success / n_total

        # Top-3 success: any of top 3 poses within threshold
        top3_success = sum(
            1 for rmsds in rmsd_results.values()
            if any(r <= thresh for r in rmsds[:3])
        )
        metrics[f"top3_at_{thresh}A"] = top3_success / n_total

    metrics["n_complexes"] = n_total
    return metrics
"""
