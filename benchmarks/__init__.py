"""FlexAIDdS distributed benchmarking suite.

Import the main orchestrator::

    from benchmarks.DatasetRunner import DatasetRunner
    runner = DatasetRunner(datasets_dir="benchmarks/datasets", results_dir="results/")
    report = runner.run_all(tier=1)
    print(report.to_markdown())

Or use the CLI::

    python -m benchmarks.run --dataset casf2016 --tier 1
    python -m benchmarks.run --all --tier 2 --distributed --nodes 4
"""

from .DatasetRunner import (
    DatasetConfig,
    DatasetResult,
    BenchmarkReport,
    DatasetRunner,
)
from .metrics import (
    PoseScore,
    entropy_rescue_rate,
    enrichment_factor,
    log_auc,
    scoring_power,
    docking_power,
    target_specificity_zscore,
    hit_rate_top_n,
    bootstrap_ci,
    compute_all_metrics,
)

__all__ = [
    # Orchestrator
    "DatasetRunner",
    "DatasetConfig",
    "DatasetResult",
    "BenchmarkReport",
    # Metrics
    "PoseScore",
    "entropy_rescue_rate",
    "enrichment_factor",
    "log_auc",
    "scoring_power",
    "docking_power",
    "target_specificity_zscore",
    "hit_rate_top_n",
    "bootstrap_ci",
    "compute_all_metrics",
]
