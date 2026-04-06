#!/usr/bin/env python3
"""run_crossdock.py — Cross-docking validation benchmark.

Evaluates FlexAIDdS performance on non-native complex structures where
the receptor conformation was crystallized with a different ligand.

Usage:
    python run_crossdock.py --results /path/to/results --pairs pairs.csv

Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np


def load_crossdock_pairs(pairs_file: str) -> List[Dict]:
    """Load cross-docking pairs from a CSV file.

    Expected format: ligand_pdb,receptor_pdb,reference_pdb
    """
    pairs = []
    with open(pairs_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split(",")
            if len(parts) >= 3:
                pairs.append({
                    "ligand": parts[0],
                    "receptor": parts[1],
                    "reference": parts[2],
                })
    return pairs


def compute_crossdock_metrics(
    rmsd_values: List[float],
    chi_angle_deviations: List[float] = None,
) -> Dict:
    """Compute cross-docking performance metrics.

    Parameters
    ----------
    rmsd_values : list
        RMSD (Angstroms) of predicted pose vs reference for each pair.
    chi_angle_deviations : list, optional
        Side-chain chi-angle deviations (degrees).

    Returns
    -------
    dict
        success_rate_2A, mean_rmsd, median_rmsd, chi_rmsd
    """
    if not rmsd_values:
        return {"error": "no results"}

    rmsds = np.array(rmsd_values)
    metrics = {
        "n_pairs": len(rmsds),
        "success_rate_2A": float(np.mean(rmsds <= 2.0)),
        "success_rate_3A": float(np.mean(rmsds <= 3.0)),
        "mean_rmsd": float(np.mean(rmsds)),
        "median_rmsd": float(np.median(rmsds)),
    }

    if chi_angle_deviations:
        chis = np.array(chi_angle_deviations)
        metrics["mean_chi_deviation"] = float(np.mean(chis))
        metrics["chi_within_30deg"] = float(np.mean(chis <= 30.0))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Cross-docking benchmark")
    parser.add_argument("--results", required=True, help="Results directory")
    parser.add_argument("--pairs", required=True, help="Cross-docking pairs CSV")
    parser.add_argument("--output", default="crossdock_report.json")
    args = parser.parse_args()

    pairs = load_crossdock_pairs(args.pairs)
    print(f"[CrossDock] Loaded {len(pairs)} pairs")

    # Placeholder: collect RMSD values from results
    report = {
        "n_pairs": len(pairs),
        "status": "pending_results",
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[CrossDock] Report written to {args.output}")


if __name__ == "__main__":
    main()
