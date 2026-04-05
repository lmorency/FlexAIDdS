#!/usr/bin/env python3
"""
dataset_runner.py — orchestrate FlexAIDdS benchmark evaluations across
multiple datasets.

This module centralises the invocation of the various dataset‐specific
benchmarks included in the FlexAIDdS repository.  Each benchmark has its
own entry point script (e.g. ``run_casf2016.py`` for CASF‑2016,
``run_crossdock.py`` for cross‑docking and ``run_litpcba.py`` for LIT‑PCBA)
which expects certain environment variables or input files.  Running all
benchmarks manually can be tedious; this runner automates the process
and supports distributed execution via ``concurrent.futures``.

The datasets supported by FlexAIDdS at the time of writing include
PDBbind (general/refined/core), ITC‑187, Binding MOAD, BindingDB,
ChEMBL and DUD‑E/DEKOIS 2.0, as described in the dataset adapter
documentation【58701120988327†L5-L12】.  In addition, the test suite provides
three representative benchmarks: CASF‑2016, CrossDock and LIT‑PCBA,
each of which has its own driver script.  These driver scripts require
the user to set environment variables pointing to the location of the
reference data and to specify where the FlexAIDdS docking results live.

This runner exposes a command‑line interface to execute any combination
of benchmarks either sequentially or in parallel.  It spawns a
sub‑process for each benchmark to avoid interference between tests.

Example usage::

    # Evaluate all benchmarks sequentially
    python dataset_runner.py --results ./results --data-root /data --output reports

    # Evaluate only CASF‑2016 and CrossDock in parallel with up to 2 workers
    python dataset_runner.py --results ./results \
        --data-root /data --benchmarks casf2016 crossdock \
        --output reports --max-workers 2

Prerequisites
-------------
The individual benchmark scripts expect the following environment
variables to be set (or will terminate with an error):

* ``CASF2016_DATA`` – path to the PDBbind CASF‑2016 data set【88467566722458†L1-L6】.
* ``LITPCBA_DATA`` – path to the LIT‑PCBA data set【8809063097560†L1-L6】.

The CrossDock benchmark additionally requires a CSV file enumerating
ligand/receptor pairs via the ``--pairs`` option【991969378609606†L7-L8】.  When using
this runner, supply the path via ``--crossdock-pairs``.

Distributed execution
---------------------
To leverage multiple CPU cores or machines, the runner uses the
``ProcessPoolExecutor`` from ``concurrent.futures``.  Each benchmark is
invoked in its own sub‑process; this isolates their execution and
allows them to run concurrently.  On a cluster, the same script can be
wrapped by a workload manager to dispatch individual benchmarks to
different nodes.

"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict


def run_casf2016(results_dir: str, output_dir: Path,
                 system_workers: int = 0) -> Dict[str, str]:
    """Run the CASF‑2016 benchmark.

    Parameters
    ----------
    results_dir : str
        Directory containing FlexAIDdS docking results for CASF‑2016.
    output_dir : Path
        Directory in which to place the output JSON report.
    system_workers : int
        Per-system parallelism (0 = auto-detect).

    Returns
    -------
    dict
        A dictionary containing the benchmark name and path to the report.
    """
    output_file = output_dir / "casf2016_report.json"
    cmd = [sys.executable,
           str(Path(__file__).parent / "tests/benchmarks/casf2016/run_casf2016.py"),
           "--results", results_dir,
           "--output", str(output_file)]
    if system_workers is not None:
        cmd.extend(["--max-workers", str(system_workers)])
    subprocess.run(cmd, check=True)
    return {"benchmark": "casf2016", "report": str(output_file)}


def run_crossdock(results_dir: str, pairs_file: str, output_dir: Path,
                  system_workers: int = 0) -> Dict[str, str]:
    """Run the CrossDock benchmark.

    Parameters
    ----------
    results_dir : str
        Directory containing FlexAIDdS docking results for cross‑docking.
    pairs_file : str
        Path to the CSV file listing ligand, receptor and reference PDB codes
        used for cross‑docking.
    output_dir : Path
        Directory in which to place the output JSON report.
    system_workers : int
        Per-system parallelism (0 = auto-detect).

    Returns
    -------
    dict
        A dictionary containing the benchmark name and path to the report.
    """
    output_file = output_dir / "crossdock_report.json"
    cmd = [sys.executable,
           str(Path(__file__).parent / "tests/benchmarks/crossdock/run_crossdock.py"),
           "--results", results_dir,
           "--pairs", pairs_file,
           "--output", str(output_file)]
    if system_workers is not None:
        cmd.extend(["--max-workers", str(system_workers)])
    subprocess.run(cmd, check=True)
    return {"benchmark": "crossdock", "report": str(output_file)}


def run_litpcba(results_dir: str, output_dir: Path,
                system_workers: int = 0) -> Dict[str, str]:
    """Run the LIT‑PCBA benchmark.

    Parameters
    ----------
    results_dir : str
        Directory containing FlexAIDdS docking results for LIT‑PCBA.
    output_dir : Path
        Directory in which to place the output JSON report.
    system_workers : int
        Per-system parallelism (0 = auto-detect).

    Returns
    -------
    dict
        A dictionary containing the benchmark name and path to the report.
    """
    output_file = output_dir / "litpcba_report.json"
    cmd = [sys.executable,
           str(Path(__file__).parent / "tests/benchmarks/litpcba/run_litpcba.py"),
           "--results", results_dir,
           "--output", str(output_file)]
    if system_workers is not None:
        cmd.extend(["--max-workers", str(system_workers)])
    subprocess.run(cmd, check=True)
    return {"benchmark": "litpcba", "report": str(output_file)}


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(
        description="Run multiple FlexAIDdS benchmarks either sequentially or in parallel."
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to the directory containing docking results. Each benchmark expects its own sub‑directory within this root."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root directory containing all benchmark data. The runner will set dataset‑specific environment variables based on this path."
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        choices=["casf2016", "crossdock", "litpcba"],
        default=["casf2016", "crossdock", "litpcba"],
        help="Benchmarks to run. Defaults to all."
    )
    parser.add_argument(
        "--crossdock-pairs",
        default=None,
        help="CSV file containing cross‑docking pairs. Required when running the crossdock benchmark."
    )
    parser.add_argument(
        "--output",
        default="benchmark_reports",
        help="Directory where benchmark reports will be written."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of worker processes for suite-level parallel execution."
    )
    parser.add_argument(
        "--system-workers",
        type=int,
        default=0,
        help=(
            "Per-system parallelism within each benchmark suite. "
            "0 = auto-detect optimal count for this machine."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variables for dataset locations
    # CASF‑2016 and LIT‑PCBA require explicit environment variables【88467566722458†L1-L6】【8809063097560†L1-L6】.
    os.environ.setdefault("CASF2016_DATA", os.path.join(args.data_root, "CASF-2016"))
    os.environ.setdefault("LITPCBA_DATA", os.path.join(args.data_root, "LIT-PCBA"))

    sw = args.system_workers

    tasks: List = []
    for benchmark in args.benchmarks:
        if benchmark == "casf2016":
            result_path = os.path.join(args.results, "casf2016")
            tasks.append((run_casf2016, result_path, None, sw))
        elif benchmark == "crossdock":
            if not args.crossdock_pairs:
                raise ValueError("--crossdock-pairs must be specified for the crossdock benchmark")
            result_path = os.path.join(args.results, "crossdock")
            tasks.append((run_crossdock, result_path, args.crossdock_pairs, sw))
        elif benchmark == "litpcba":
            result_path = os.path.join(args.results, "litpcba")
            tasks.append((run_litpcba, result_path, None, sw))

    results = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_bm = {}
        for func, res_path, extra, sys_w in tasks:
            if func is run_crossdock:
                future = executor.submit(func, res_path, extra, output_dir, sys_w)
            else:
                future = executor.submit(func, res_path, output_dir, sys_w)
            future_to_bm[future] = func.__name__
        for future in as_completed(future_to_bm):
            bm_name = future_to_bm[future]
            try:
                result = future.result()
                results.append(result)
                print(f"[Runner] Completed {bm_name}: report at {result['report']}")
            except Exception as exc:
                print(f"[Runner] Benchmark {bm_name} generated an exception: {exc}")

    # Summarise results
    summary_path = output_dir / "summary.json"
    try:
        import json
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Runner] Summary written to {summary_path}")
    except Exception as exc:
        print(f"[Runner] Could not write summary: {exc}")


if __name__ == "__main__":
    main()