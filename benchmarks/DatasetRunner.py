"""Backward-compatibility shim. Real code lives in flexaidds.dataset_runner.runner."""

from flexaidds.dataset_runner.runner import *  # noqa: F401,F403
from flexaidds.dataset_runner.runner import (  # noqa: F401
    DatasetConfig,
    DatasetResult,
    TargetResult,
    BenchmarkReport,
    DatasetRunner,
    _git_sha,
    _runner_info,
    _parse_remark_float,
)
