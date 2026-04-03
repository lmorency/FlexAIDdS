#!/usr/bin/env python3
"""run_casf2016.py — Main driver for CASF-2016 benchmark evaluation.

Usage:
    export CASF2016_DATA=/path/to/PDBbind/CASF-2016
    python run_casf2016.py --results /path/to/flexaids_results [--output report.json]

Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import json
import os
import sys

from scoring_power import run_scoring_power_benchmark
from docking_power import compute_docking_power


def main():
    parser = argparse.ArgumentParser(description="CASF-2016 benchmark evaluation")
    parser.add_argument("--results", required=True, help="FlexAIDdS results directory")
    parser.add_argument("--output", default="casf2016_report.json", help="Output JSON report")
    args = parser.parse_args()

    data_dir = os.environ.get("CASF2016_DATA")
    if not data_dir:
        print("ERROR: Set CASF2016_DATA environment variable to PDBbind CASF-2016 path")
        sys.exit(1)

    print(f"[CASF-2016] Data: {data_dir}")
    print(f"[CASF-2016] Results: {args.results}")

    report = {}

    # Scoring power
    try:
        scoring = run_scoring_power_benchmark(data_dir, args.results)
        report["scoring_power"] = scoring
        print(f"  Scoring Power: Pearson R = {scoring.get('pearson_r', 'N/A'):.3f}")
    except Exception as e:
        report["scoring_power"] = {"error": str(e)}
        print(f"  Scoring Power: ERROR - {e}")

    # Write report
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[CASF-2016] Report written to {args.output}")


if __name__ == "__main__":
    main()
"""
