from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from benchmarks.DatasetRunner import DatasetConfig

from .executors import BaseExecutor
from .models import ExecutionResult, PlannedJob, SubmissionArtifact, TargetJobRecord
from .policy import JobStatus


class SchedulerExecutorBase(BaseExecutor):
    submit_command = ""

    def _scheduler_available(self) -> bool:
        return bool(self.submit_command) and shutil.which(self.submit_command) is not None

    def _build_script_lines(self, runner, job: PlannedJob, *, config: DatasetConfig, tier: int) -> List[str]:
        return [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {runner.repo_root}",
            (
                f"python -m benchmarks.run_modern --dataset {config.slug} --tier {tier} "
                f"--workers 1 --report-prefix {runner.results_dir / 'scheduler_runs' / job.job_id}"
            ),
            "",
        ]

    def _submission_artifact(self, runner, job: PlannedJob, *, config: DatasetConfig, tier: int) -> SubmissionArtifact:
        script_path = runner.runtime_root / config.slug / f"tier{tier}" / "manifests" / f"{job.job_id}.{self.name}.sh"
        runner._atomic_write_text(script_path, "\n".join(self._build_script_lines(runner, job, config=config, tier=tier)))
        return SubmissionArtifact(
            executor=self.name,
            dataset_slug=config.slug,
            tier=tier,
            manifest_path=str(script_path),
            script_path=str(script_path),
            command=self.submit_command,
            job_count=1,
        )

    def execute_jobs(self, runner, jobs: List[PlannedJob], *, config: DatasetConfig, tier: int) -> List[ExecutionResult]:
        results: List[ExecutionResult] = []
        available = self._scheduler_available()
        for job in jobs:
            artifact = self._submission_artifact(runner, job, config=config, tier=tier)
            error = "" if available else f"scheduler backend unavailable: {self.submit_command}"
            results.append(
                ExecutionResult(
                    record=TargetJobRecord(
                        job_id=job.job_id,
                        dataset_slug=job.dataset_slug,
                        tier=job.tier,
                        target_id=job.target_id,
                        structural_state=job.structural_state,
                        status=JobStatus.QUEUED.value if available else JobStatus.FAILED.value,
                        assigned_executor=self.name,
                        started_at=runner._utc_now(),
                        finished_at=runner._utc_now(),
                        log_path=artifact.script_path,
                        error=error,
                        attempt=1,
                    ),
                    poses=[],
                )
            )
        return results


class SlurmExecutor(SchedulerExecutorBase):
    name = "slurm"
    submit_command = "sbatch"


class PBSExecutor(SchedulerExecutorBase):
    name = "pbs"
    submit_command = "qsub"
