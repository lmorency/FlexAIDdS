"""
benchmarks/run.py
==================
CLI entry point for FlexAIDdS benchmarks.

Usage
-----
    # Via module invocation
    python -m benchmarks.run --dataset casf2016 --tier 1

    # Direct execution
    python benchmarks/run.py --dataset casf2016 --tier 1

    # MPI distributed
    mpirun -n 32 python -m benchmarks.run --dataset casf2016

    # List available datasets
    python -m benchmarks.run --list-datasets

    # Dry-run (test CI plumbing without FlexAID binary)
    python -m benchmarks.run --dataset casf2016 --tier 1 --dry-run

Environment variables
---------------------
    FLEXAIDS_DATA       Root directory for dataset files (default: /data/flexaids)
    FLEXAID_BIN         Path to FlexAID binary
    MPI_DISABLED        Set to "1" to force serial execution even if mpi4py present
    BENCHMARK_TIMEOUT   Per-target timeout override (seconds)

Exit codes
----------
    0   All targets succeeded
    1   One or more target failures
    2   Configuration / dataset error
"""

from __future__ import annotations

import os
import pathlib
import sys

# Ensure the repo root is on the path regardless of how we're invoked.
_REPO_ROOT = pathlib.Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.DatasetRunner import DatasetRunner, DatasetConfig, main, _RANK


def _tier_from_env() -> int:
    """Read BENCHMARK_TIER env var; defaults to 2."""
    raw = os.environ.get("BENCHMARK_TIER", "2")
    try:
        val = int(raw)
        if val in (1, 2):
            return val
    except ValueError:
        pass
    return 2


def _timeout_from_env() -> int:
    """Read BENCHMARK_TIMEOUT env var; defaults to 600."""
    raw = os.environ.get("BENCHMARK_TIMEOUT", "600")
    try:
        return max(10, int(raw))
    except ValueError:
        return 600


if __name__ == "__main__":
    # When invoked as `python -m benchmarks.run` or `python benchmarks/run.py`
    # delegate entirely to DatasetRunner.main() which handles arg parsing.
    sys.exit(main())
