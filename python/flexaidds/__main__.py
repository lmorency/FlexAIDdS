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
from pathlib import Path
from typing import Optional

from .__version__ import __version__
from .results import load_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m flexaidds",
        description="Inspect FlexAID∆S docking result directories from Python.",
    )
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
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
        print(result.to_json(sort_keys=True))
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

    _print_table(result, args.top)

    top = result.top_mode()
    if top is not None:
        print(
            f"\nTop mode: mode_id={top.mode_id}, "
            f"free_energy={top.free_energy}, best_cf={top.best_cf}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
