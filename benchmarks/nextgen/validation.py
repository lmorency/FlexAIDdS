from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from benchmarks.DatasetRunner import DatasetConfig


@dataclass
class ValidationReport:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class PreflightValidator:
    def validate(
        self,
        *,
        config: DatasetConfig,
        binary: str,
        dry_run: bool,
        requested_states: Optional[List[str]] = None,
        requested_metrics: Optional[List[str]] = None,
        scheduler_backend: str = "local",
    ) -> ValidationReport:
        report = ValidationReport(ok=True)
        states = requested_states or config.structural_states

        if not config.targets:
            report.errors.append(f"Dataset {config.slug} has no targets configured.")
        if not states:
            report.errors.append(f"Dataset {config.slug} has no structural states configured.")

        if requested_metrics:
            unknown = sorted(set(requested_metrics) - set(config.metrics or requested_metrics))
            if unknown:
                report.errors.append(
                    f"Dataset {config.slug} requested unsupported metrics: {', '.join(unknown)}"
                )

        if not dry_run:
            resolved = shutil.which(binary) if not Path(binary).is_file() else binary
            if not resolved:
                report.errors.append(f"Docking binary not found: {binary}")

            if config.data_dir is None:
                report.errors.append(f"Dataset {config.slug} has no data_dir configured.")
            elif not Path(config.data_dir).exists():
                report.errors.append(f"Dataset directory does not exist: {config.data_dir}")

        if scheduler_backend == "slurm" and not shutil.which("sbatch"):
            report.warnings.append("SLURM executor requested but sbatch is unavailable in PATH.")
        if scheduler_backend == "pbs" and not shutil.which("qsub"):
            report.warnings.append("PBS executor requested but qsub is unavailable in PATH.")

        if config.slug == "casf2016" and not dry_run and not os.environ.get("CASF2016_DATA") and not str(config.data_dir or ""):
            report.warnings.append("CASF-2016 dataset path is not exported through CASF2016_DATA.")
        if config.slug == "litpcba" and not dry_run and not os.environ.get("LITPCBA_DATA") and not str(config.data_dir or ""):
            report.warnings.append("LIT-PCBA dataset path is not exported through LITPCBA_DATA.")

        report.ok = not report.errors
        return report
