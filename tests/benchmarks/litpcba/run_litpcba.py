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
        # Placeholder: load results and evaluate
        # In a real pipeline, this would parse FlexAIDdS docking scores
        report["targets"].append({
            "target": target,
            "status": "pending_results",
        })

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[LIT-PCBA] Report written to {args.output}")


if __name__ == "__main__":
    main()
