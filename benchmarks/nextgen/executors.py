
from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from benchmarks.DatasetRunner import DatasetConfig

from .models import ExecutionResult, PlannedJob, TargetJobRecord
from .policy import JobStatus


class BaseExecutor(ABC):
    name = "base"

    @abstractmethod
    def execute_jobs(
        self,
        runner: "NextGenRunnerProtocol",
        jobs: List[PlannedJob],
        *,
        config: DatasetConfig,
        tier: int,
    ) -> List[ExecutionResult]:
        raise NotImplementedError


class LocalThreadExecutor(BaseExecutor):
    name = "local-thread"

    def execute_jobs(
        self,
        runner: "NextGenRunnerProtocol",
        jobs: List[PlannedJob],
        *,
        config: DatasetConfig,
        tier: int,
    ) -> List[ExecutionResult]:
        if not jobs:
            return []
        if runner.n_workers <= 1 or len(jobs) <= 1:
            return [runner._execute_planned_job(job, config=config, tier=tier) for job in jobs]

        results: List[ExecutionResult] = []
        with ThreadPoolExecutor(max_workers=runner.n_workers) as pool:
            future_map = {
                pool.submit(runner._execute_planned_job, job, config=config, tier=tier): job
                for job in jobs
            }
            for future in as_completed(future_map):
                job = future_map[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(
                        ExecutionResult(
                            record=TargetJobRecord(
                                job_id=job.job_id,
                                dataset_slug=job.dataset_slug,
                                tier=job.tier,
                                target_id=job.target_id,
                                structural_state=job.structural_state,
                                status=JobStatus.FAILED.value,
                                assigned_executor=self.name,
                                started_at=runner._utc_now(),
                                finished_at=runner._utc_now(),
                                log_path=str(job.log_path),
                                error=str(exc),
                            ),
                            poses=[],
                        )
                    )
        return results


class MPIExecutorAdapter(BaseExecutor):
    name = "mpi-rank"

    def execute_jobs(
        self,
        runner: "NextGenRunnerProtocol",
        jobs: List[PlannedJob],
        *,
        config: DatasetConfig,
        tier: int,
    ) -> List[ExecutionResult]:
        return [runner._execute_planned_job(job, config=config, tier=tier) for job in jobs]


class NextGenRunnerProtocol:
    n_workers: int

    def _utc_now(self) -> str: ...
    def _execute_planned_job(self, job: PlannedJob, *, config: DatasetConfig, tier: int) -> ExecutionResult: ...
