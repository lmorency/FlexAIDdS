"""Backward-compatibility shim. Real code lives in flexaidds.dataset_runner.metrics."""

from flexaidds.dataset_runner.metrics import *  # noqa: F401,F403
from flexaidds.dataset_runner.metrics import (  # noqa: F401
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
