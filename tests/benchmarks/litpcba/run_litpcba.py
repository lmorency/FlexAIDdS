#!/usr/bin/env python3
"""run_litpcba.py — LIT-PCBA unbiased virtual screening benchmark.

Usage:
    export LITPCBA_DATA=/path/to/LIT-PCBA
    python run_litpcba.py --results /path/to/results [--output report.json]

Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import json
import os
import sys

import numpy as np

from metrics import evaluate_target


def load_litpcba_target(data_dir: str, target: str):
    """Load actives and inactives for a LIT-PCBA target."""
    actives_file = os.path.join(data_dir, target, "actives.smi")
    inactives_file = os.path.join(data_dir, target, "inactives.smi")

    actives = set()
    if os.path.exists(actives_file):
        with open(actives_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    actives.add(parts[0])

    inactives = set()
    if os.path.exists(inactives_file):
        with open(inactives_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    inactives.add(parts[0])

    return actives, inactives


def load_docking_scores(results_dir: str, target: str, compounds: set) -> dict:
    """Load FlexAIDdS docking scores for a target's compounds.

    Looks for per-compound result subdirectories under
    ``results_dir/target/`` and extracts the top-ranked CF score.

    Parameters
    ----------
    results_dir : str
        Top-level results directory.
    target : str
        Target name (subdirectory under results_dir).
    compounds : set
        Set of compound identifiers (SMILES) to look up.

    Returns
    -------
    dict
        Mapping of compound identifier → docking score (CF, lower is better).
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "python"))
    from flexaidds import load_results

    target_dir = os.path.join(results_dir, target)
    scores = {}

    if not os.path.isdir(target_dir):
        return scores

    # Each compound may have its own subdirectory
    for entry in os.listdir(target_dir):
        compound_dir = os.path.join(target_dir, entry)
        if not os.path.isdir(compound_dir):
            continue

        try:
            result = load_results(compound_dir)
            top_mode = result.top_mode()
            if top_mode is None or not top_mode.poses:
                continue

            top_pose = top_mode.poses[0]
            score = top_pose.cf if top_pose.cf is not None else top_pose.cf_app
            if score is not None:
                scores[entry] = score
        except Exception:
            continue

    return scores


def main():
    parser = argparse.ArgumentParser(description="LIT-PCBA benchmark")
    parser.add_argument("--results", required=True, help="Results directory")
    parser.add_argument("--output", default="litpcba_report.json")
    args = parser.parse_args()

    data_dir = os.environ.get("LITPCBA_DATA")
    if not data_dir:
        print("ERROR: Set LITPCBA_DATA environment variable")
        sys.exit(1)

    # Discover targets
    targets = [d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))]

    print(f"[LIT-PCBA] Found {len(targets)} targets")

    report = {"targets": []}
    for target in sorted(targets):
        print(f"  Evaluating {target}...")
        actives, inactives = load_litpcba_target(data_dir, target)

        if not actives:
            report["targets"].append({
                "target": target,
                "status": "no_actives",
            })
            continue

        all_compounds = actives | inactives
        docking_scores = load_docking_scores(
            args.results, target, all_compounds
        )

        if not docking_scores:
            report["targets"].append({
                "target": target,
                "status": "no_results",
                "n_actives": len(actives),
                "n_inactives": len(inactives),
            })
            continue

        # Build score and label arrays for compounds with docking results
        scored_compounds = []
        for compound, score in docking_scores.items():
            if compound in actives:
                scored_compounds.append((score, 1))
            elif compound in inactives:
                scored_compounds.append((score, 0))

        if not scored_compounds:
            report["targets"].append({
                "target": target,
                "status": "no_matched_compounds",
            })
            continue

        scores_arr = np.array([s for s, _ in scored_compounds])
        labels_arr = np.array([l for _, l in scored_compounds])

        metrics = evaluate_target(scores_arr, labels_arr, target)
        metrics["status"] = "ok"
        metrics["n_scored"] = len(scored_compounds)
        report["targets"].append(metrics)

        print(f"    {target}: EF1%={metrics['EF_1pct']:.1f} "
              f"AUROC={metrics['AUROC']:.3f} "
              f"({metrics['n_scored']} scored)")

    # Aggregate summary
    evaluated = [t for t in report["targets"] if t.get("status") == "ok"]
    if evaluated:
        report["summary"] = {
            "n_targets_evaluated": len(evaluated),
            "mean_EF1": float(np.mean([t["EF_1pct"] for t in evaluated])),
            "mean_AUROC": float(np.mean([t["AUROC"] for t in evaluated])),
            "mean_BEDROC": float(np.mean([t["BEDROC_20"] for t in evaluated])),
        }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[LIT-PCBA] Report written to {args.output}")


if __name__ == "__main__":
    main()
