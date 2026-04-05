
from __future__ import annotations

import datetime
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from benchmarks.DatasetRunner import DatasetConfig
from benchmarks.metrics import PoseScore

from .models import SubmissionArtifact, TargetJobRecord
from .policy import CleanRoomPolicy, JobStatus


class RuntimeMixin:
    runtime_root: Path
    repo_root: Path
    temperature: float
    binary: str

    def clean_room_policy(self) -> CleanRoomPolicy:
        return CleanRoomPolicy()

    @staticmethod
    def _utc_now() -> str:
        return datetime.datetime.utcnow().isoformat() + "Z"

    @staticmethod
    def _job_id(dataset_slug: str, tier: int, target_id: str, state: str) -> str:
        raw = f"{dataset_slug}|tier{tier}|{target_id}|{state}"
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        safe_target = target_id.replace("/", "_")
        return f"{dataset_slug}_tier{tier}_{safe_target}_{state}_{digest}"

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            delete=False,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as fh:
            fh.write(content)
            tmp_name = fh.name
        os.replace(tmp_name, path)

    def _append_jsonl_atomic(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\\n")

    def _dataset_runtime_dirs(self, config: DatasetConfig, tier: int) -> Dict[str, Path]:
        root = self.runtime_root / config.slug / f"tier{tier}"
        dirs = {
            "root": root,
            "logs": root / "logs",
            "results": root / "results",
            "manifests": root / "manifests",
            "state": root / "state",
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return dirs

    def _job_executor_name(self) -> str:
        return f"mpi-rank-{self._mpi_rank}" if self._mpi_comm is not None else "local-thread"

    def _load_state_records(self, state_path: Path) -> Dict[str, Dict[str, Any]]:
        if not state_path.is_file():
            return {}
        try:
            return json.loads(state_path.read_text(encoding="utf-8")).get("records", {})
        except Exception:
            return {}

    def _save_state_records(self, state_path: Path, records: Dict[str, Dict[str, Any]], *, dataset_slug: str, tier: int) -> None:
        payload = {
            "dataset": dataset_slug,
            "tier": tier,
            "updated_at": self._utc_now(),
            "policy_version": self.clean_room_policy().version,
            "records": records,
        }
        self._atomic_write_text(state_path, json.dumps(payload, indent=2, sort_keys=True))

    def _write_manifest(self, manifest_path: Path, *, config: DatasetConfig, tier: int, targets: List[str], states: List[str], requested_metrics: List[str] | None) -> None:
        jobs = [
            TargetJobRecord(
                job_id=self._job_id(config.slug, tier, target_id, state),
                dataset_slug=config.slug,
                tier=tier,
                target_id=target_id,
                structural_state=state,
                status=JobStatus.QUEUED.value,
                assigned_executor=self._job_executor_name(),
            ).to_dict()
            for target_id in targets
            for state in states
        ]
        payload = {
            "dataset": config.slug,
            "dataset_name": config.name,
            "tier": tier,
            "generated_at": self._utc_now(),
            "git_sha": getattr(self, "_git_sha_value", None),
            "temperature": self.temperature,
            "binary": self.binary,
            "requested_metrics": requested_metrics,
            "targets": targets,
            "structural_states": states,
            "clean_room_policy": self.clean_room_policy().to_dict(),
            "jobs": jobs,
        }
        self._atomic_write_text(manifest_path, json.dumps(payload, indent=2, sort_keys=True))

    def _write_submission_script(self, script_path: Path, *, config: DatasetConfig, tier: int, manifest_path: Path, executor_name: str, job_count: int) -> SubmissionArtifact:
        command = f"python -m benchmarks.run_nextgen --dataset {config.slug} --tier {tier} --manifest {manifest_path} --executor {executor_name}"
        script = "\\n".join(["#!/usr/bin/env bash", "set -euo pipefail", f"cd {self.repo_root}", command, ""])
        self._atomic_write_text(script_path, script)
        try:
            os.chmod(script_path, 0o755)
        except Exception:
            pass
        return SubmissionArtifact(
            executor=executor_name,
            dataset_slug=config.slug,
            tier=tier,
            manifest_path=str(manifest_path),
            script_path=str(script_path),
            command=command,
            job_count=job_count,
        )

    def _append_target_result_jsonl(self, jsonl_path: Path, record: TargetJobRecord, poses: List[PoseScore]) -> None:
        self._append_jsonl_atomic(
            jsonl_path,
            {
                "job": record.to_dict(),
                "clean_room_policy_version": self.clean_room_policy().version,
                "poses": [p.__dict__ for p in poses],
            },
        )

    def _read_target_result_jsonl(self, jsonl_path: Path) -> List[PoseScore]:
        if not jsonl_path.is_file():
            return []
        poses: List[PoseScore] = []
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                payload = json.loads(line)
                poses.extend(PoseScore(**pose) for pose in payload.get("poses", []))
        return poses
