"""Unified CLI for legacy benchmark families on top of the next-generation control plane."""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
from pathlib import Path

from benchmarks.DatasetRunner import BenchmarkReport, _git_sha, _runner_info
from benchmarks.nextgen.runner import DatasetRunner
from benchmarks.nextgen.unify_simple import build_unifier


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.run_unified",
        description="Unified legacy benchmark evaluation over the next-generation benchmark control plane",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", "-d", metavar="SLUG")
    group.add_argument("--all", "-a", action="store_true")
    parser.add_argument("--tier", "-t", type=int, choices=[1, 2], default=2)
    parser.add_argument("--metric", "-m", default=None)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--datasets-dir", default="benchmarks/datasets")
    parser.add_argument("--results-dir", default="results/benchmarks")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--binary", default=None)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--max-job-attempts", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--report-prefix", default=None)
    return parser


def _apply_unified_metrics(runner: DatasetRunner, dataset_result, metric_name: str | None):
    config = dataset_result.config
    tier = dataset_result.tier
    runtime_dirs = runner._dataset_runtime_dirs(config, tier)
    state_path = runtime_dirs["state"] / f"{config.slug}_tier{tier}_state.json"
    jsonl_path = runtime_dirs["results"] / f"{config.slug}_tier{tier}_targets.jsonl"
    state_records = runner._load_state_records(state_path)
    poses = runner._read_target_result_jsonl(jsonl_path)
    unifier = build_unifier()
    return unifier.evaluate(
        config=config,
        result=dataset_result,
        poses=poses,
        state_records=state_records,
        requested_metrics=[metric_name] if metric_name else None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.data_dir:
        os.environ["FLEXAIDDS_BENCHMARK_DATA"] = args.data_dir

    runner = DatasetRunner(
        datasets_dir=args.datasets_dir,
        results_dir=args.results_dir,
        binary=args.binary,
        temperature=args.temperature,
        n_workers=args.workers,
        use_mpi=args.distributed,
        cache_dir=args.data_dir,
        bootstrap_ci=args.bootstrap,
        n_bootstrap=args.n_bootstrap,
        dry_run=args.dry_run,
        max_job_attempts=args.max_job_attempts,
    )

    if args.all:
        report = runner.run_all(
            tier=args.tier,
            distributed=args.distributed,
            n_nodes=args.nodes,
            metric_subset=[args.metric] if args.metric else None,
        )
        if runner._mpi_root:
            report.datasets = [_apply_unified_metrics(runner, dr, args.metric) for dr in report.datasets]
    else:
        try:
            dataset_result = (
                runner.collect_single(dataset_slug=args.dataset, tier=args.tier, metric=args.metric)
                if args.collect_only
                else runner.run_single(dataset_slug=args.dataset, tier=args.tier, metric=args.metric)
            )
        except FileNotFoundError as exc:
            logging.error("%s", exc)
            return 2
        if runner._mpi_root:
            dataset_result = _apply_unified_metrics(runner, dataset_result, args.metric)
        report = BenchmarkReport(
            datasets=[dataset_result],
            generated_at=datetime.datetime.utcnow().isoformat() + "Z",
            git_sha=_git_sha(),
            host=os.uname().nodename,
            runner_info=_runner_info(),
        )

    if not runner._mpi_root:
        return 0

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    prefix = args.report_prefix or str(Path(args.results_dir) / f"unified_report_{timestamp}")
    json_path, md_path = report.save(prefix)
    print(f"\nReport saved:\n  JSON: {json_path}\n  Markdown: {md_path}\n")
    print(report.to_markdown())

    any_regression = any(any(dr.regression_flags.values()) for dr in report.datasets)
    return 1 if any_regression else 0


if __name__ == "__main__":
    sys.exit(main())
