from __future__ import annotations

import hashlib
import json
import socket
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from benchmarks.DatasetRunner import DatasetConfig, DatasetResult
from benchmarks.metrics import PoseScore

from .adapters import LegacyBenchmarkUnifier
from .reducer import pair_job_ids_with_poses, reduce_runtime_to_sqlite
from .runner import DatasetRunner as NextGenDatasetRunner
from .schedulers import PBSExecutor, SlurmExecutor
from .validation import PreflightValidator


class ModernUnifiedDatasetRunner(NextGenDatasetRunner):
    MANIFEST_SCHEMA_VERSION = "2.0"
    STATE_SCHEMA_VERSION = "2.0"
    REPORT_SCHEMA_VERSION = "2.0"

    def __init__(
        self,
        *args,
        scheduler_backend: str = "local",
        lease_ttl_seconds: int = 1800,
        ligand_batch_size: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.scheduler_backend = scheduler_backend
        self.lease_ttl_seconds = max(60, int(lease_ttl_seconds))
        self.ligand_batch_size = max(0, int(ligand_batch_size))
        self.validator = PreflightValidator()
        self.unifier = LegacyBenchmarkUnifier()

    def _select_executor(self):
        backend = (self.scheduler_backend or "local").lower()
        if backend == "slurm":
            return SlurmExecutor()
        if backend == "pbs":
            return PBSExecutor()
        return super()._select_executor()

    def _lease_path(self, config: DatasetConfig, tier: int, job_id: str) -> Path:
        return self.runtime_root / config.slug / f"tier{tier}" / "state" / f"{job_id}.lease.json"

    def _acquire_lease(self, config: DatasetConfig, tier: int, job_id: str) -> bool:
        path = self._lease_path(config, tier, job_id)
        now = time.time()
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                expires_at = float(payload.get("expires_at", 0))
                if expires_at > now:
                    return False
            except Exception:
                pass
        self._atomic_write_text(
            path,
            json.dumps(
                {
                    "job_id": job_id,
                    "holder": socket.gethostname(),
                    "acquired_at": self._utc_now(),
                    "expires_at": now + self.lease_ttl_seconds,
                },
                indent=2,
                sort_keys=True,
            ),
        )
        return True

    def _release_lease(self, config: DatasetConfig, tier: int, job_id: str) -> None:
        path = self._lease_path(config, tier, job_id)
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

    def _classify_error(self, error: str) -> str:
        message = (error or "").lower()
        if any(token in message for token in ("timed out", "timeout", "temporarily unavailable", "resource busy", "no space left", "i/o error")):
            return "transient"
        if any(token in message for token in ("missing receptor", "no ligands found", "dataset", "unsupported metrics")):
            return "data"
        if any(token in message for token in ("parse", "json", "remark", "malformed")):
            return "parser"
        if any(token in message for token in ("flexaid", "returned", "docking")):
            return "engine"
        return "unknown"

    def _should_retry_job(self, record) -> bool:
        if not super()._should_retry_job(record):
            return False
        return self._classify_error(record.error) == "transient"

    def _target_batch_token(self, target_id: str, batch_index: int) -> str:
        return f"{target_id}@@batch{batch_index}"

    def _split_batch_token(self, token: str) -> Tuple[str, Optional[int]]:
        if "@@batch" not in token:
            return token, None
        target_id, batch_idx = token.split("@@batch", 1)
        try:
            return target_id, int(batch_idx)
        except ValueError:
            return target_id, None

    def _plan_jobs(self, *, config: DatasetConfig, tier: int, targets: List[str], states: List[str], state_records: Dict[str, Dict[str, object]], runtime_dirs: Dict[str, Path]):
        if self.ligand_batch_size <= 0 or self.dry_run or not config.data_dir:
            return super()._plan_jobs(
                config=config,
                tier=tier,
                targets=targets,
                states=states,
                state_records=state_records,
                runtime_dirs=runtime_dirs,
            )

        jobs = []
        skipped = 0
        for target_id in targets:
            ligands = self._find_ligands(target_id, config.data_dir)
            batches = [ligands[i : i + self.ligand_batch_size] for i in range(0, len(ligands), self.ligand_batch_size)] or [[]]
            for batch_index, _lig_batch in enumerate(batches):
                token = self._target_batch_token(target_id, batch_index)
                for state in states:
                    job_id = self._job_id(config.slug, tier, token, state)
                    if state_records.get(job_id, {}).get("status") == "succeeded":
                        skipped += 1
                        continue
                    jobs.append(
                        type(super()._plan_jobs(config=config, tier=tier, targets=[target_id], states=[state], state_records={}, runtime_dirs=runtime_dirs)[0][0])(
                            job_id=job_id,
                            dataset_slug=config.slug,
                            tier=tier,
                            target_id=token,
                            structural_state=state,
                            log_path=runtime_dirs["logs"] / f"{job_id}.json",
                        )
                    )
        return jobs, skipped

    def _run_target_state_job(self, *, config: DatasetConfig, tier: int, target_id: str, structural_state: str, log_path: Path, attempt: int = 1):
        real_target_id, batch_index = self._split_batch_token(target_id)
        if not self._acquire_lease(config, tier, self._job_id(config.slug, tier, target_id, structural_state)):
            record, poses = super()._run_target_state_job(
                config=config,
                tier=tier,
                target_id=real_target_id,
                structural_state=structural_state,
                log_path=log_path,
                attempt=attempt,
            )
            record.error = record.error or "lease unavailable"
            return record, poses
        try:
            if batch_index is None or self.dry_run or not config.data_dir or self.ligand_batch_size <= 0:
                return super()._run_target_state_job(
                    config=config,
                    tier=tier,
                    target_id=real_target_id,
                    structural_state=structural_state,
                    log_path=log_path,
                    attempt=attempt,
                )

            ligands = self._find_ligands(real_target_id, config.data_dir)
            subset = ligands[batch_index * self.ligand_batch_size : (batch_index + 1) * self.ligand_batch_size]
            receptor = self._find_receptor(real_target_id, config.data_dir, structural_state)
            poses: List[PoseScore] = self._dock_target(
                real_target_id,
                receptor or Path("/dev/null"),
                subset or [Path(f"{real_target_id}.mol2")],
                structural_state=structural_state,
            )
            record, _ = super()._run_target_state_job(
                config=config,
                tier=tier,
                target_id=real_target_id,
                structural_state=structural_state,
                log_path=log_path,
                attempt=attempt,
            )
            record.num_poses = len(poses)
            return record, poses
        finally:
            self._release_lease(config, tier, self._job_id(config.slug, tier, target_id, structural_state))

    def _provenance_payload(self, config: DatasetConfig, tier: int) -> Dict[str, object]:
        binary_path = Path(self.binary)
        binary_hash = ""
        if binary_path.is_file():
            digest = hashlib.sha256()
            with open(binary_path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            binary_hash = digest.hexdigest()
        return {
            "manifest_schema_version": self.MANIFEST_SCHEMA_VERSION,
            "state_schema_version": self.STATE_SCHEMA_VERSION,
            "report_schema_version": self.REPORT_SCHEMA_VERSION,
            "dataset": config.slug,
            "tier": tier,
            "binary": self.binary,
            "binary_hash": binary_hash,
            "git_sha": self._git_sha_value,
            "host": socket.gethostname(),
        }

    def _reduce_runtime(self, config: DatasetConfig, tier: int) -> None:
        runtime_dirs = self._dataset_runtime_dirs(config, tier)
        state_path = runtime_dirs["state"] / f"{config.slug}_tier{tier}_state.json"
        jsonl_path = runtime_dirs["results"] / f"{config.slug}_tier{tier}_targets.jsonl"
        db_path = runtime_dirs["results"] / f"{config.slug}_tier{tier}.sqlite"
        reduce_runtime_to_sqlite(
            db_path=db_path,
            state_records=self._load_state_records(state_path),
            pose_rows=pair_job_ids_with_poses(jsonl_path),
        )
        provenance_path = runtime_dirs["results"] / f"{config.slug}_tier{tier}.provenance.json"
        self._atomic_write_text(provenance_path, json.dumps(self._provenance_payload(config, tier), indent=2, sort_keys=True))

    def _apply_unified_metrics(self, config: DatasetConfig, tier: int, result: DatasetResult, metric_subset: Optional[List[str]] = None) -> DatasetResult:
        runtime_dirs = self._dataset_runtime_dirs(config, tier)
        state_path = runtime_dirs["state"] / f"{config.slug}_tier{tier}_state.json"
        jsonl_path = runtime_dirs["results"] / f"{config.slug}_tier{tier}_targets.jsonl"
        state_records = self._load_state_records(state_path)
        poses = self._read_target_result_jsonl(jsonl_path)
        return self.unifier.evaluate(
            config=config,
            result=result,
            poses=poses,
            state_records=state_records,
            requested_metrics=metric_subset or config.metrics or None,
        )

    def run_dataset(self, config: DatasetConfig, tier: int = 2, metric_subset: Optional[List[str]] = None, structural_states: Optional[List[str]] = None) -> DatasetResult:
        validation = self.validator.validate(
            config=config,
            binary=self.binary,
            dry_run=self.dry_run,
            requested_states=structural_states,
            requested_metrics=metric_subset,
            scheduler_backend=self.scheduler_backend,
        )
        if not validation.ok:
            raise RuntimeError("; ".join(validation.errors))
        result = super().run_dataset(
            config=config,
            tier=tier,
            metric_subset=metric_subset,
            structural_states=structural_states,
        )
        if self._mpi_root:
            result = self._apply_unified_metrics(config, tier, result, metric_subset)
            self._reduce_runtime(config, tier)
            self._save_dataset_result(result)
        return result

    def collect_dataset_runtime(self, config: DatasetConfig, tier: int = 2, metric_subset: Optional[List[str]] = None) -> DatasetResult:
        result = super().collect_dataset_runtime(config, tier=tier, metric_subset=metric_subset)
        result = self._apply_unified_metrics(config, tier, result, metric_subset)
        self._reduce_runtime(config, tier)
        return result
