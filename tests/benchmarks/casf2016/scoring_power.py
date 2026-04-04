"""scoring_power.py — CASF-2016 Scoring Power evaluation.

Computes Pearson R between FlexAIDdS predicted scores and experimental
binding affinities (pKd) from the CASF-2016 core set (285 complexes).

Requires CASF2016_DATA environment variable pointing to the PDBbind dataset.

Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
SPDX-License-Identifier: Apache-2.0
"""

import os
import json
from typing import Dict, List, Tuple

import numpy as np


def load_experimental_affinities(data_dir: str) -> Dict[str, float]:
    """Load experimental pKd values from CASF-2016 core set.

    Parameters
    ----------
    data_dir : str
        Path to CASF-2016 dataset directory.

    Returns
    -------
    dict
        Mapping of PDB ID → experimental pKd.
    """
    affinity_file = os.path.join(data_dir, "CoreSet.dat")
    affinities = {}

    if not os.path.exists(affinity_file):
        raise FileNotFoundError(
            f"CASF-2016 core set file not found: {affinity_file}\n"
            f"Set CASF2016_DATA to the PDBbind dataset directory."
        )

    with open(affinity_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                pdb_id = parts[0].lower()
                try:
                    pkd = float(parts[3])
                    affinities[pdb_id] = pkd
                except (ValueError, IndexError):
                    pass

    return affinities


def compute_scoring_power(
    predicted: Dict[str, float],
    experimental: Dict[str, float],
) -> Dict[str, float]:
    """Compute Pearson R and related metrics.

    Parameters
    ----------
    predicted : dict
        PDB ID → predicted score.
    experimental : dict
        PDB ID → experimental pKd.

    Returns
    -------
    dict
        Metrics: pearson_r, spearman_rho, rmse, n_complexes.
    """
    common_ids = sorted(set(predicted.keys()) & set(experimental.keys()))

    if len(common_ids) < 2:
        return {"pearson_r": 0.0, "n_complexes": 0, "error": "insufficient data"}

    pred = np.array([predicted[pid] for pid in common_ids])
    expt = np.array([experimental[pid] for pid in common_ids])

    pearson_r = float(np.corrcoef(pred, expt)[0, 1])
    rmse = float(np.sqrt(np.mean((pred - expt) ** 2)))

    # Spearman rank correlation
    from scipy.stats import spearmanr
    rho, pval = spearmanr(pred, expt)

    return {
        "pearson_r": pearson_r,
        "spearman_rho": float(rho),
        "rmse": rmse,
        "n_complexes": len(common_ids),
    }


def run_scoring_power_benchmark(
    data_dir: str,
    results_dir: str,
) -> Dict:
    """Run the full CASF-2016 scoring power evaluation.

    Parameters
    ----------
    data_dir : str
        CASF-2016 dataset directory.
    results_dir : str
        Directory containing FlexAIDdS docking results (one per PDB ID).

    Returns
    -------
    dict
        Complete benchmark results.
    """
    experimental = load_experimental_affinities(data_dir)

    # Load predicted scores from results directory
    predicted = {}
    for pdb_id in experimental:
        result_file = os.path.join(results_dir, f"{pdb_id}_result.json")
        if os.path.exists(result_file):
            with open(result_file) as f:
                data = json.load(f)
                predicted[pdb_id] = data.get("best_score", 0.0)

    metrics = compute_scoring_power(predicted, experimental)
    return metrics
"""
