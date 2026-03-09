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
from typing import Optional

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
    parser.add_argument(
        "--csv",
        metavar="PATH",
        default=None,
        help="Write binding-mode summary to a CSV file at PATH.",
    )
    parser.add_argument(
        "--top",
        metavar="N",
        type=int,
        default=None,
        help="Show only the top N binding modes in the summary table (default: all).",
    )
    return parser


def _fmt(value: object, width: int = 10, precision: int = 3) -> str:
    if value is None:
        return "-".center(width)
    if isinstance(value, float):
        return f"{value:{width}.{precision}f}"
    return str(value).rjust(width)


def _print_table(result, top_n: Optional[int]) -> None:
    modes = result.binding_modes
    if top_n is not None and top_n > 0:
        modes = modes[:top_n]

    header = (
        f"{'Mode':>5}  {'Rank':>5}  {'N_poses':>7}  "
        f"{'F (kcal/mol)':>14}  {'H (kcal/mol)':>14}  "
        f"{'S (kcal/mol·K)':>16}  {'Best CF':>10}"
    )
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)
    for mode in modes:
        print(
            f"{mode.mode_id:>5}  {mode.rank:>5}  {mode.n_poses:>7}  "
            f"{_fmt(mode.free_energy, 14)}  {_fmt(mode.enthalpy, 14)}  "
            f"{_fmt(mode.entropy, 16, 6)}  {_fmt(mode.best_cf, 10)}"
        )
    print(separator)


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

    if args.csv is not None:
        result.to_csv(args.csv)
        print(f"Wrote {result.n_modes} binding mode(s) to {args.csv}")
        return 0

    print(f"Results directory: {result.source_dir}")
    print(f"Binding modes: {result.n_modes}")
    if result.temperature is not None:
        print(f"Temperature: {result.temperature} K")
    print()

    if not result.binding_modes:
        return 0

    # Table header
    header = (
        f"{'mode':>5}  {'rank':>4}  {'poses':>5}  "
        f"{'best_cf':>10}  {'free_energy':>12}  "
        f"{'enthalpy':>10}  {'entropy':>12}"
    )
    print(header)
    print("-" * len(header))

    # Sort by rank for display
    _print_table(result, args.top)

    top = result.top_mode()
    for mode in sorted(result.binding_modes, key=lambda m: m.rank):
        cf_str = f"{mode.best_cf:10.4f}" if mode.best_cf is not None else f"{'N/A':>10}"
        fe_str = f"{mode.free_energy:12.4f}" if mode.free_energy is not None else f"{'N/A':>12}"
        h_str = f"{mode.enthalpy:10.4f}" if mode.enthalpy is not None else f"{'N/A':>10}"
        s_str = f"{mode.entropy:12.6f}" if mode.entropy is not None else f"{'N/A':>12}"
        marker = " *" if top is not None and mode.mode_id == top.mode_id else ""
        print(
            f"{mode.mode_id:>5}  {mode.rank:>4}  {mode.n_poses:>5}  "
            f"{cf_str}  {fe_str}  {h_str}  {s_str}{marker}"
            f"\nTop mode: mode_id={top.mode_id}, rank={top.rank}, "
            f"n_poses={top.n_poses}, free_energy={top.free_energy}, "
            f"best_cf={top.best_cf}"
        )

    if top is not None:
        print(f"\n* = top-ranked mode (mode_id={top.mode_id})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
