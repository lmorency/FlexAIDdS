"""Command-line entry point for the ``flexaidds`` package.

Invoked as::

    python -m flexaidds <results_dir> [--json]

Scans *results_dir* for FlexAID∆S docking output PDB files and prints a
human-readable summary of the binding modes to stdout.  Pass ``--json`` to
emit a machine-readable JSON payload instead.

Exit codes:
    0 – success
    1 – unhandled error (propagated as an exception)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .results import load_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m flexaidds",
        description="Inspect FlexAID∆S docking result directories from Python.",
    )
    parser.add_argument("results_dir", type=Path, help="Directory containing docking result PDB files")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a human summary.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    result = load_results(args.results_dir)
    if args.json:
        payload = {
            "source_dir": str(result.source_dir),
            "temperature": result.temperature,
            "n_modes": result.n_modes,
            "metadata": result.metadata,
            "binding_modes": result.to_records(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"Results directory: {result.source_dir}")
    print(f"Binding modes: {result.n_modes}")
    if result.temperature is not None:
        print(f"Temperature: {result.temperature} K")
    top = result.top_mode()
    if top is not None:
        print(
            "Top mode: "
            f"mode_id={top.mode_id}, rank={top.rank}, n_poses={top.n_poses}, "
            f"free_energy={top.free_energy}, best_cf={top.best_cf}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
