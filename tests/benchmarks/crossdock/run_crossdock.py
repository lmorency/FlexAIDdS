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


def collect_rmsd_from_results(
    results_dir: str,
    pairs: List[Dict],
) -> Tuple[List[float], List[Dict]]:
    """Collect RMSD values from FlexAIDdS docking results for each pair.

    For each cross-docking pair, looks for a subdirectory matching the
    ligand PDB ID under results_dir, loads the top-ranked pose, and
    extracts its RMSD to the crystal reference.

    Parameters
    ----------
    results_dir : str
        Top-level directory containing per-pair result subdirectories.
    pairs : list of dict
        Cross-docking pairs from load_crossdock_pairs().

    Returns
    -------
    rmsd_values : list of float
        RMSD for each pair that had results.
    per_pair_details : list of dict
        Per-pair detail records (ligand, receptor, rmsd, status).
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "python"))
    from flexaidds import load_results

    rmsd_values = []
    per_pair_details = []

    for pair in pairs:
        ligand_id = pair["ligand"]
        detail = {
            "ligand": ligand_id,
            "receptor": pair["receptor"],
            "reference": pair["reference"],
            "status": "no_results",
            "rmsd": None,
        }

        # Look for results in subdirectory named by ligand PDB ID
        pair_dir = os.path.join(results_dir, ligand_id)
        if not os.path.isdir(pair_dir):
            # Also try ligand_receptor naming convention
            pair_dir = os.path.join(
                results_dir, f"{ligand_id}_{pair['receptor']}"
            )

        if not os.path.isdir(pair_dir):
            per_pair_details.append(detail)
            continue

        try:
            result = load_results(pair_dir)
            top_mode = result.top_mode()
            if top_mode is None:
                detail["status"] = "no_poses"
                per_pair_details.append(detail)
                continue

            # Use symmetry-corrected RMSD if available, otherwise raw
            top_pose = top_mode.poses[0] if top_mode.poses else None
            if top_pose is None:
                detail["status"] = "no_poses"
                per_pair_details.append(detail)
                continue

            rmsd = top_pose.rmsd_sym if top_pose.rmsd_sym is not None else top_pose.rmsd_raw
            if rmsd is not None:
                rmsd_values.append(rmsd)
                detail["rmsd"] = rmsd
                detail["status"] = "ok"
            else:
                detail["status"] = "no_rmsd"

        except Exception as e:
            detail["status"] = f"error: {e}"

        per_pair_details.append(detail)

    return rmsd_values, per_pair_details


def main():
    parser = argparse.ArgumentParser(description="Cross-docking benchmark")
    parser.add_argument("--results", required=True, help="Results directory")
    parser.add_argument("--pairs", required=True, help="Cross-docking pairs CSV")
    parser.add_argument("--output", default="crossdock_report.json")
    args = parser.parse_args()

    pairs = load_crossdock_pairs(args.pairs)
    print(f"[CrossDock] Loaded {len(pairs)} pairs")

    rmsd_values, per_pair_details = collect_rmsd_from_results(
        args.results, pairs
    )

    n_with_results = sum(1 for d in per_pair_details if d["status"] == "ok")
    print(f"[CrossDock] {n_with_results}/{len(pairs)} pairs have RMSD results")

    if rmsd_values:
        metrics = compute_crossdock_metrics(rmsd_values)
        print(f"[CrossDock] Success rate (<2Å): {metrics['success_rate_2A']:.1%}")
        print(f"[CrossDock] Mean RMSD: {metrics['mean_rmsd']:.2f} Å")
        print(f"[CrossDock] Median RMSD: {metrics['median_rmsd']:.2f} Å")
    else:
        metrics = {"error": "no RMSD values collected"}
        print("[CrossDock] No RMSD values collected from results")

    report = {
        "n_pairs": len(pairs),
        "n_with_results": n_with_results,
        "metrics": metrics,
        "per_pair": per_pair_details,
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[CrossDock] Report written to {args.output}")


if __name__ == "__main__":
    main()
