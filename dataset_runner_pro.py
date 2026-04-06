#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
dataset_runner_pro.py — robust orchestration of FlexAIDdS benchmarks.

This script is an enhanced version of the simple ``dataset_runner`` that ships with
FlexAIDdS.  It adds input validation, configurable logging, timeouts and more
granular error handling to make the benchmark process as fail‑safe as possible.

It supports the same three benchmark suites used in the test harness—CASF‑2016,
CrossDock and LIT‑PCBA—and wraps each invocation in validation and timeout
controls.  Use this as a template for production or large‑scale benchmarking
jobs where robustness is critical.

Example usage::

    # run all benchmarks with default settings
    python dataset_runner_pro.py --results ./results --data-root /data --output reports

    # run only crossdock with a 10 minute timeout per benchmark
    python dataset_runner_pro.py \
        --benchmarks crossdock \
        --results ./results \
        --data-root /data \
        --crossdock-pairs pairs.csv \
        --timeout 600 \
        --output reports

"""

from __future__ import annotations

import argparse
import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Iterable, Optional, Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed


# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO, log_file: Optional[Path] = None) -> None:
    """Configure root logging.

    Parameters
    ----------
    level : int
        Logging level (e.g. logging.INFO or logging.DEBUG).
    log_file : Optional[Path]
        Optional path to a file to which logs should also be written.  If
        provided, a FileHandler will be added alongside the StreamHandler.
    """
    handlers: List[logging.Handler] = []
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    handlers.append(stream_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        handlers.append(file_handler)
    logging.basicConfig(level=level, handlers=handlers)



def discover_benchmarks() -> List[str]:
    """Discover available benchmark scripts under tests/benchmarks.

    This scans the ``tests/benchmarks`` directory relative to this file
    for any ``run_*.py`` scripts and returns the list of benchmark names
    (without the ``run_`` prefix).  The list is deduplicated and
    alphabetically sorted.

    Returns
    -------
    List[str]
        Available benchmark names.
    """
    benchmarks: List[str] = []
    base = Path(__file__).parent / "tests" / "benchmarks"
    if base.exists():
        for d in base.iterdir():
            if d.is_dir():
                for script in d.glob("run_*.py"):
                    name = script.stem.replace("run_", "")
                    if name not in benchmarks:
                        benchmarks.append(name)
    return sorted(benchmarks)


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------
def ensure_dir(path: Path, description: str) -> Path:
    """Ensure that ``path`` exists and is a directory.

    Raises a ``FileNotFoundError`` if the path is invalid.

    Parameters
    ----------
    path : Path
        Path to check.
    description : str
        Human‑readable description used in error messages.

    Returns
    -------
    Path
        The validated directory path.
    """
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{description} '{path}' does not exist or is not a directory")
    return path



def ensure_file(path: Path, description: str) -> Path:
    """Ensure that ``path`` exists and is a file.

    Raises a ``FileNotFoundError`` if the path is invalid.

    Parameters
    ----------
    path : Path
        Path to check.
    description : str
        Human‑readable description used in error messages.

    Returns
    -------
    Path
        The validated file path.
    """
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{description} '{path}' does not exist or is not a file")
    return path


# -----------------------------------------------------------------------------
# Benchmark runners
# -----------------------------------------------------------------------------
def run_script(cmd: Iterable[str], timeout: Optional[int]) -> None:
    """Run a subprocess command with optional timeout and propagate failures.

    Parameters
    ----------
    cmd : Iterable[str]
        Command and arguments to execute.
    timeout : Optional[int]
        Timeout in seconds for the subprocess.  ``None`` means no timeout.
    """
    try:
        subprocess.run(cmd, check=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout} seconds: {' '.join(cmd)}")
        raise
    except subprocess.CalledProcessError as exc:
        logger.error(f"Command failed with exit code {exc.returncode}: {' '.join(cmd)}")
        raise



def run_casf2016(results_dir: Path, data_root: Path, output_dir: Path, timeout: Optional[int]) -> Dict[str, str]:
    """Run the CASF‑2016 benchmark with validation.

    Parameters
    ----------
    results_dir : Path
        Directory containing docking results for CASF‑2016.
    data_root : Path
        Root directory containing benchmark datasets.  ``CASF2016_DATA`` is
        resolved to ``data_root / 'CASF-2016'``.
    output_dir : Path
        Directory to write the JSON report.
    timeout : Optional[int]
        Timeout (in seconds) for the subprocess.

    Returns
    -------
    dict
        A dictionary with the benchmark name and report path.
    """
    results_dir = ensure_dir(results_dir, "CASF‑2016 results directory")
    casf_data = ensure_dir(data_root / "CASF-2016", "CASF‑2016 dataset")
    os.environ["CASF2016_DATA"] = str(casf_data)
    output_file = output_dir / "casf2016_report.json"
    cmd = [sys.executable,
           str(Path(__file__).parent / "tests/benchmarks/casf2016/run_casf2016.py"),
           "--results", str(results_dir),
           "--output", str(output_file)]
    run_script(cmd, timeout)
    return {"benchmark": "casf2016", "report": str(output_file)}



def run_crossdock(results_dir: Path, pairs_file: Path, data_root: Path, output_dir: Path, timeout: Optional[int]) -> Dict[str, str]:
    """Run the CrossDock benchmark with validation.

    Parameters
    ----------
    results_dir : Path
        Directory containing docking results for cross‑docking.
    pairs_file : Path
        CSV file listing ligand/receptor pairs.
    data_root : Path
        Unused for CrossDock but accepted for API consistency.
    output_dir : Path
        Directory to write the JSON report.
    timeout : Optional[int]
        Timeout (in seconds) for the subprocess.

    Returns
    -------
    dict
        A dictionary with the benchmark name and report path.
    """
    results_dir = ensure_dir(results_dir, "CrossDock results directory")
    pairs_file = ensure_file(pairs_file, "CrossDock pairs CSV")
    output_file = output_dir / "crossdock_report.json"
    cmd = [sys.executable,
           str(Path(__file__).parent / "tests/benchmarks/crossdock/run_crossdock.py"),
           "--results", str(results_dir),
           "--pairs", str(pairs_file),
           "--output", str(output_file)]
    run_script(cmd, timeout)
    return {"benchmark": "crossdock", "report": str(output_file)}



def run_litpcba(results_dir: Path, data_root: Path, output_dir: Path, timeout: Optional[int]) -> Dict[str, str]:
    """Run the LIT‑PCBA benchmark with validation.

    Parameters
    ----------
    results_dir : Path
        Directory containing docking results for LIT‑PCBA.
    data_root : Path
        Root directory containing benchmark datasets.  ``LITPCBA_DATA`` is
        resolved to ``data_root / 'LIT-PCBA'``.
    output_dir : Path
        Directory to write the JSON report.
    timeout : Optional[int]
        Timeout (in seconds) for the subprocess.

    Returns
    -------
    dict
        A dictionary with the benchmark name and report path.
    """
    results_dir = ensure_dir(results_dir, "LIT‑PCBA results directory")
    litpcba_data = ensure_dir(data_root / "LIT-PCBA", "LIT‑PCBA dataset")
    os.environ["LITPCBA_DATA"] = str(litpcba_data)
    output_file = output_dir / "litpcba_report.json"
    cmd = [sys.executable,
           str(Path(__file__).parent / "tests/benchmarks/litpcba/run_litpcba.py"),
           "--results", str(results_dir),
           "--output", str(output_file)]
    run_script(cmd, timeout)
    return {"benchmark": "litpcba", "report": str(output_file)}


# Mapping of benchmark names to runner functions and parameter counts
BENCHMARKS = {
    "casf2016": (run_casf2016, 0),    # (function, number of extra positional args)
    "crossdock": (run_crossdock, 1),  # expects pairs_file
    "litpcba": (run_litpcba, 0),
}



def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command‑line arguments.

    Parameters
    ----------
    argv : Optional[List[str]]
        Argument list for testing purposes.  ``None`` means use ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run FlexAIDdS benchmarks with robust error handling.")
    # ``--results`` and ``--data-root`` are required for running benchmarks
    # but optional when only listing benchmarks via ``--list-benchmarks``.
    parser.add_argument("--results", required=False, help="Root directory containing docking results.")
    parser.add_argument("--data-root", required=False, help="Root directory of benchmark datasets.")
    parser.add_argument("--benchmarks", nargs="*", choices=list(BENCHMARKS.keys()), default=list(BENCHMARKS.keys()), help="Benchmarks to run.")
    parser.add_argument("--crossdock-pairs", help="CSV file containing cross‑docking pairs. Required for the crossdock benchmark.")
    parser.add_argument("--output", default="benchmark_reports", help="Directory where reports will be written.")
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum number of parallel workers.")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds for each benchmark subprocess (0 or omitted = no timeout).")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging verbosity.")
    parser.add_argument("--log-file", type=str, help="Optional file to append log messages to.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing benchmarks.")
    parser.add_argument(
        "--summary-format",
        choices=["json", "yaml", "html", "both", "all"],
        default="json",
        help=(
            "Format of the summary report. 'json' writes JSON, 'yaml' writes YAML, "
            "'html' writes an HTML table, 'both' writes JSON and YAML, and 'all' "
            "writes JSON, YAML and HTML."
        ),
    )
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="List available benchmarks and exit without running anything.",
    )
    return parser.parse_args(argv)



def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    # If the user just wants to list benchmarks, print them and exit
    if args.list_benchmarks:
        # Discover available benchmark scripts dynamically and list them
        print("Available benchmarks:")
        for name in discover_benchmarks():
            print(f"  {name}")
        return

    # Configure logging based on user options
    configure_logging(getattr(logging, args.log_level.upper()), Path(args.log_file) if args.log_file else None)

    # Validate results and data_root
    if not args.results:
        logger.error("--results is required unless --list-benchmarks is specified.")
        return
    if not args.data_root:
        logger.error("--data-root is required unless --list-benchmarks is specified.")
        return
    results_root = Path(args.results)
    data_root = Path(args.data_root)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    # Validate root directories
    ensure_dir(results_root, "Results root")
    ensure_dir(data_root, "Data root")

    logger.info(f"Running benchmarks: {', '.join(args.benchmarks)}")
    logger.info(f"Results root: {results_root}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Output directory: {output_root}")

    tasks: List = []
    for bm in args.benchmarks:
        func, extra_args = BENCHMARKS[bm]
        result_dir = results_root / bm
        if bm == "crossdock":
            if not args.crossdock_pairs:
                logger.warning("CrossDock selected but --crossdock-pairs not provided; skipping this benchmark.")
                continue
            pairs_file = Path(args.crossdock_pairs)
            tasks.append((func, result_dir, pairs_file))
        else:
            tasks.append((func, result_dir, None))

    if args.dry_run:
        logger.info("Dry run mode enabled — benchmarks will not be executed.")
        for func, res_dir, extra in tasks:
            logger.info(f"Would run {func.__name__} on results in '{res_dir}'")
        return

    # Run benchmarks with parallel workers
    results: List[Dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {}
        for func, res_dir, extra in tasks:
            if func is run_crossdock:
                future = executor.submit(func, res_dir, extra, data_root, output_root, args.timeout)
            else:
                future = executor.submit(func, res_dir, data_root, output_root, args.timeout)
            future_map[future] = func.__name__
        for future in as_completed(future_map):
            bm_name = future_map[future]
            try:
                res = future.result()
                results.append(res)
                logger.info(f"{bm_name} completed successfully: {res['report']}")
            except Exception as exc:
                logger.error(f"{bm_name} failed: {exc}")

    # Write summary file(s) depending on requested format
    base = output_root / "summary"
    wrote_any = False
    # JSON output
    if args.summary_format in ("json", "both", "all"):
        try:
            import json
            json_path = base.with_suffix(".json")
            with json_path.open("w", encoding="utf-8") as fp:
                json.dump(results, fp, indent=2)
            logger.info(f"JSON summary written to {json_path}")
            wrote_any = True
        except Exception as exc:
            logger.error(f"Failed to write JSON summary: {exc}")
    # YAML output
    if args.summary_format in ("yaml", "both", "all"):
        try:
            import yaml  # type: ignore
            yaml_path = base.with_suffix(".yaml")
            with yaml_path.open("w", encoding="utf-8") as fp:
                yaml.safe_dump(results, fp, sort_keys=False)
            logger.info(f"YAML summary written to {yaml_path}")
            wrote_any = True
        except Exception as exc:
            logger.error(f"Failed to write YAML summary: {exc}")
    # HTML output
    if args.summary_format in ("html", "all"):
        try:
            html_path = base.with_suffix(".html")
            with html_path.open("w", encoding="utf-8") as fp:
                fp.write("<html><head><title>Benchmark Summary</title></head><body>\n")
                fp.write("<h1>Benchmark Summary</h1>\n")
                fp.write("<table border='1'>\n")
                fp.write("<tr><th>Benchmark</th><th>Status</th><th>Return Code</th><th>Duration (s)</th><th>Report</th><th>Error</th></tr>\n")
                for r in results:
                    fp.write("<tr>")
                    fp.write(f"<td>{r.get('benchmark','')}</td>")
                    fp.write(f"<td>{r.get('status','')}</td>")
                    fp.write(f"<td>{r.get('returncode','')}</td>")
                    fp.write(f"<td>{r.get('duration_s','')}</td>")
                    report_path = r.get('report') or ''
                    error_msg = r.get('error') or ''
                    fp.write(f"<td>{report_path}</td>")
                    fp.write(f"<td>{error_msg}</td>")
                    fp.write("</tr>\n")
                fp.write("</table></body></html>\n")
            logger.info(f"HTML summary written to {html_path}")
            wrote_any = True
        except Exception as exc:
            logger.error(f"Failed to write HTML summary: {exc}")
    if not wrote_any:
        logger.warning("No summary files were written.")


if __name__ == "__main__":
    main()
