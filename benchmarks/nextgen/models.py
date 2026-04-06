
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from benchmarks.metrics import PoseScore

from .policy import JobStatus


@dataclass
class TargetJobRecord:
    job_id: str
    dataset_slug: str
    tier: int
    target_id: str
    structural_state: str
    status: str = JobStatus.QUEUED.value
    assigned_executor: str = "local-thread"
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0
    num_poses: int = 0
    log_path: str = ""
    error: str = ""
    attempt: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlannedJob:
    job_id: str
    dataset_slug: str
    tier: int
    target_id: str
    structural_state: str
    log_path: Path


@dataclass
class ExecutionResult:
    record: TargetJobRecord
    poses: List[PoseScore] = field(default_factory=list)


@dataclass
class SubmissionArtifact:
    executor: str
    dataset_slug: str
    tier: int
    manifest_path: str
    script_path: str
    command: str
    job_count: int
