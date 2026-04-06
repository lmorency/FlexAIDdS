
from __future__ import annotations

import json
import logging
import socket
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from benchmarks.DatasetRunner import DatasetConfig, DatasetResult, DatasetRunner as LegacyDatasetRunner, _git_sha, _split_targets
from benchmarks.metrics import PoseScore, compute_all_metrics

from .executors import BaseExecutor, LocalThreadExecutor, MPIExecutorAdapter
from .models import ExecutionResult, PlannedJob, TargetJobRecord
from .policy import JobStatus, RetryPolicy
from .runtime import RuntimeMixin

logger = logging.getLogger(__name__)


class DatasetRunner(RuntimeMixin, LegacyDatasetRunner):
    def __init__(self, *args, max_job_attempts: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.retry_policy = RetryPolicy(max_attempts=max(1, int(max_job_attempts)))
        self._git_sha_value = _git_sha(self.repo_root)
        self.runtime_root = self.results_dir
        self.dataset_results_dir = self.runtime_root / "results"
        self.logs_dir = self.runtime_root / "logs"
        self.manifests_dir = self.runtime_root / "manifests" / "generated"
        self.state_dir = self.runtime_root / "state"
        self.tmp_dir = self.runtime_root / "tmp"
        self.cache_runtime_dir = self.runtime_root / "cache"
        for path in (
            self.dataset_results_dir,
            self.logs_dir,
            self.manifests_dir,
            self.state_dir,
            self.tmp_dir,
            self.cache_runtime_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _select_executor(self) -> BaseExecutor:
        return MPIExecutorAdapter() if self._mpi_comm is not None else LocalThreadExecutor()

    def _plan_jobs(self, *, config: DatasetConfig, tier: int, targets: List[str], states: List[str], state_records: Dict[str, Dict[str, object]], runtime_dirs: Dict[str, Path]) -> Tuple[List[PlannedJob], int]:
        jobs: List[PlannedJob] = []
        skipped = 0
        for target_id in targets:
            for state in states:
                job_id = self._job_id(config.slug, tier, target_id, state)
                if state_records.get(job_id, {}).get("status") == JobStatus.SUCCEEDED.value:
                    skipped += 1
                    continue
                jobs.append(
                    PlannedJob(
                        job_id=job_id,
                        dataset_slug=config.slug,
                        tier=tier,
                        target_id=target_id,
                        structural_state=state,
                        log_path=runtime_dirs["logs"] / f"{job_id}.json",
                    )
                )
        return jobs, skipped

    def _execute_planned_job(self, job: PlannedJob, *, config: DatasetConfig, tier: int) -> ExecutionResult:
        try:
            record, poses = self._run_target_state_job(
                config=config,
                tier=tier,
                target_id=job.target_id,
                structural_state=job.structural_state,
                log_path=job.log_path,
                attempt=1,
            )
            return ExecutionResult(record=record, poses=poses)
        except Exception as exc:
            return ExecutionResult(
                record=TargetJobRecord(
                    job_id=job.job_id,
                    dataset_slug=job.dataset_slug,
                    tier=job.tier,
                    target_id=job.target_id,
                    structural_state=job.structural_state,
                    status=JobStatus.FAILED.value,
                    assigned_executor=self._select_executor().name,
                    started_at=self._utc_now(),
                    finished_at=self._utc_now(),
                    log_path=str(job.log_path),
                    error=str(exc),
                    attempt=1,
                ),
                poses=[],
            )

    def _run_target_state_job(self, *, config: DatasetConfig, tier: int, target_id: str, structural_state: str, log_path: Path, attempt: int = 1) -> Tuple[TargetJobRecord, List[PoseScore]]:
        record = TargetJobRecord(
            job_id=self._job_id(config.slug, tier, target_id, structural_state),
            dataset_slug=config.slug,
            tier=tier,
            target_id=target_id,
            structural_state=structural_state,
            status=JobStatus.RUNNING.value,
            assigned_executor=self._job_executor_name(),
            started_at=self._utc_now(),
            log_path=str(log_path),
            attempt=attempt,
        )
        poses: List[PoseScore] = []
        error = ""
        started = time.monotonic()
        try:
            receptor = None
            if config.data_dir and not self.dry_run:
                receptor = self._find_receptor(target_id, config.data_dir, structural_state)
                if receptor is None:
                    error = f"missing receptor for {target_id}/{structural_state}"
            ligands: List[Path] = []
            if not error and config.data_dir and not self.dry_run:
                ligands = self._find_ligands(target_id, config.data_dir)
                if not ligands:
                    error = f"no ligands found for {target_id}"
            if not error:
                poses = self._dock_target(
                    target_id,
                    receptor or Path("/dev/null"),
                    ligands or [Path(f"{target_id}.mol2")],
                    structural_state=structural_state,
                )
                if not poses:
                    error = f"no poses produced for {target_id}/{structural_state}"
        except Exception as exc:
            error = str(exc)

        record.finished_at = self._utc_now()
        record.duration_seconds = time.monotonic() - started
        record.num_poses = len(poses)
        record.error = error
        record.status = JobStatus.SUCCEEDED.value if poses and not error else JobStatus.FAILED.value
        self._atomic_write_text(
            log_path,
            json.dumps(
                {
                    "job": record.to_dict(),
                    "clean_room_policy_version": self.clean_room_policy().version,
                    "poses": [p.__dict__ for p in poses],
                },
                indent=2,
                sort_keys=True,
            ),
        )
        return record, poses

    def _should_retry_job(self, record: TargetJobRecord) -> bool:
        return (
            record.status == JobStatus.FAILED.value
            and record.attempt < self.retry_policy.max_attempts
            and self.retry_policy.is_retryable(record.error)
        )

    def _merge_job_outcomes(self, execution_results: List[ExecutionResult], *, state_records: Dict[str, Dict[str, object]], dataset_slug: str, tier: int, state_path: Path, jsonl_path: Path, all_poses: List[PoseScore], completed: List[str], failed: List[str]) -> None:
        for result in execution_results:
            record = result.record
            state_records[record.job_id] = record.to_dict()
            if result.poses:
                all_poses.extend(result.poses)
                completed.append(record.target_id)
                self._append_target_result_jsonl(jsonl_path, record, result.poses)
            else:
                failed.append(record.target_id)
            self._save_state_records(state_path, state_records, dataset_slug=dataset_slug, tier=tier)

    def _retry_failed_jobs(self, *, executor: BaseExecutor, config: DatasetConfig, tier: int, execution_results: List[ExecutionResult], state_records: Dict[str, Dict[str, object]], state_path: Path, jsonl_path: Path, all_poses: List[PoseScore], completed: List[str], failed: List[str]) -> None:
        pending = list(execution_results)
        while pending:
            self._merge_job_outcomes(
                pending,
                state_records=state_records,
                dataset_slug=config.slug,
                tier=tier,
                state_path=state_path,
                jsonl_path=jsonl_path,
                all_poses=all_poses,
                completed=completed,
                failed=failed,
            )
            retry_jobs = [
                PlannedJob(
                    job_id=result.record.job_id,
                    dataset_slug=result.record.dataset_slug,
                    tier=result.record.tier,
                    target_id=result.record.target_id,
                    structural_state=result.record.structural_state,
                    log_path=Path(result.record.log_path),
                )
                for result in pending
                if self._should_retry_job(result.record)
            ]
            if not retry_jobs:
                break
            retry_results: List[ExecutionResult] = []
            for job in retry_jobs:
                prior_attempt = int(state_records.get(job.job_id, {}).get("attempt", 1))
                try:
                    record, poses = self._run_target_state_job(
                        config=config,
                        tier=tier,
                        target_id=job.target_id,
                        structural_state=job.structural_state,
                        log_path=job.log_path,
                        attempt=prior_attempt + 1,
                    )
                    retry_results.append(ExecutionResult(record=record, poses=poses))
                except Exception as exc:
                    retry_results.append(
                        ExecutionResult(
                            record=TargetJobRecord(
                                job_id=job.job_id,
                                dataset_slug=job.dataset_slug,
                                tier=job.tier,
                                target_id=job.target_id,
                                structural_state=job.structural_state,
                                status=JobStatus.FAILED.value,
                                assigned_executor=executor.name,
                                started_at=self._utc_now(),
                                finished_at=self._utc_now(),
                                log_path=str(job.log_path),
                                error=str(exc),
                                attempt=prior_attempt + 1,
                            ),
                            poses=[],
                        )
                    )
            pending = retry_results

    def collect_dataset_runtime(self, config: DatasetConfig, tier: int = 2, metric_subset: Optional[List[str]] = None) -> DatasetResult:
        requested_metrics = metric_subset or config.metrics or None
        runtime_dirs = self._dataset_runtime_dirs(config, tier)
        state_records = self._load_state_records(runtime_dirs["state"] / f"{config.slug}_tier{tier}_state.json")
        poses = self._read_target_result_jsonl(runtime_dirs["results"] / f"{config.slug}_tier{tier}_targets.jsonl")
        result = DatasetResult(
            config=config,
            tier=tier,
            targets_attempted=list(config.tier1_targets() if tier == 1 else config.targets),
            timestamp=self._utc_now(),
            git_sha=self._git_sha_value,
            host=socket.gethostname(),
        )
        result.targets_completed = sorted({r.get("target_id", "") for r in state_records.values() if r.get("status") == JobStatus.SUCCEEDED.value} - {""})
        result.targets_failed = sorted(({r.get("target_id", "") for r in state_records.values() if r.get("status") == JobStatus.FAILED.value} - {""}) - set(result.targets_completed))
        if poses:
            result.metrics = compute_all_metrics(poses, requested=requested_metrics)
            if self.do_bootstrap:
                result.ci_95 = self._compute_bootstrap_cis(poses, requested_metrics)
        result.check_regressions()
        return result

    def run_dataset(self, config: DatasetConfig, tier: int = 2, metric_subset: Optional[List[str]] = None, structural_states: Optional[List[str]] = None) -> DatasetResult:
        started = time.monotonic()
        targets = config.tier1_targets() if tier == 1 else config.targets
        states = structural_states or config.structural_states
        requested_metrics = metric_subset or config.metrics or None
        result = DatasetResult(
            config=config,
            tier=tier,
            targets_attempted=list(targets),
            timestamp=self._utc_now(),
            git_sha=self._git_sha_value,
            host=socket.gethostname(),
        )
        if not targets:
            return result

        runtime_dirs = self._dataset_runtime_dirs(config, tier)
        manifest_path = runtime_dirs["manifests"] / f"{config.slug}_tier{tier}_manifest.json"
        state_path = runtime_dirs["state"] / f"{config.slug}_tier{tier}_state.json"
        jsonl_path = runtime_dirs["results"] / f"{config.slug}_tier{tier}_targets.jsonl"
        script_path = runtime_dirs["manifests"] / f"{config.slug}_tier{tier}_{self._select_executor().name}.sh"

        if self._mpi_root:
            self._write_manifest(manifest_path, config=config, tier=tier, targets=list(targets), states=list(states), requested_metrics=requested_metrics)

        existing_records = self._load_state_records(state_path) if self._mpi_root else {}
        my_targets = _split_targets(targets, self._mpi_rank, self._mpi_size)
        state_records = dict(existing_records)
        jobs, skipped = self._plan_jobs(config=config, tier=tier, targets=list(my_targets), states=list(states), state_records=state_records, runtime_dirs=runtime_dirs)

        if self._mpi_root:
            self._write_submission_script(script_path, config=config, tier=tier, manifest_path=manifest_path, executor_name=self._select_executor().name, job_count=len(jobs))
        if skipped:
            logger.info("Dataset %s: skipping %d previously completed job(s)", config.slug, skipped)

        all_poses: List[PoseScore] = []
        completed: List[str] = []
        failed: List[str] = []
        executor = self._select_executor()
        execution_results = executor.execute_jobs(self, jobs, config=config, tier=tier)
        self._retry_failed_jobs(
            executor=executor,
            config=config,
            tier=tier,
            execution_results=execution_results,
            state_records=state_records,
            state_path=state_path,
            jsonl_path=jsonl_path,
            all_poses=all_poses,
            completed=completed,
            failed=failed,
        )

        if self._mpi_comm is not None:
            gathered = self._mpi_comm.gather((all_poses, completed, failed, state_records), root=0)
            if self._mpi_root:
                all_poses, completed, failed = [], [], []
                merged_records: Dict[str, Dict[str, object]] = {}
                for poses_i, completed_i, failed_i, records_i in gathered or []:
                    all_poses.extend(poses_i)
                    completed.extend(completed_i)
                    failed.extend(failed_i)
                    merged_records.update(records_i)
                state_records = merged_records
                self._save_state_records(state_path, state_records, dataset_slug=config.slug, tier=tier)

        if self._mpi_root:
            result.targets_completed = sorted(set(completed))
            result.targets_failed = sorted(set(failed) - set(result.targets_completed))
            if all_poses:
                result.metrics = compute_all_metrics(all_poses, requested=requested_metrics)
                if self.do_bootstrap:
                    result.ci_95 = self._compute_bootstrap_cis(all_poses, requested_metrics)
            result.check_regressions()

        result.duration_seconds = time.monotonic() - started
        return result

    def _save_dataset_result(self, result: DatasetResult) -> None:
        out_path = self.dataset_results_dir / f"{result.config.slug}_tier{result.tier}.json"
        self._atomic_write_text(out_path, json.dumps(result.to_dict(), indent=2, sort_keys=True))

    def collect_single(self, dataset_slug: str, tier: int = 2, metric: Optional[str] = None) -> DatasetResult:
        config = self.load_dataset_config(self.datasets_dir / f"{dataset_slug}.yaml")
        return self.collect_dataset_runtime(config, tier=tier, metric_subset=[metric] if metric else None)
