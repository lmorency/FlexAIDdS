from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from benchmarks.metrics import PoseScore


def ensure_runtime_database(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                dataset_slug TEXT NOT NULL,
                tier INTEGER NOT NULL,
                target_id TEXT NOT NULL,
                structural_state TEXT NOT NULL,
                status TEXT NOT NULL,
                assigned_executor TEXT,
                started_at TEXT,
                finished_at TEXT,
                duration_seconds REAL,
                num_poses INTEGER,
                log_path TEXT,
                error TEXT,
                attempt INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS poses (
                job_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                ligand_id TEXT NOT NULL,
                pose_rank INTEGER NOT NULL,
                rmsd REAL,
                enthalpy_score REAL,
                entropy_correction REAL,
                total_score REAL,
                is_active INTEGER,
                exp_affinity REAL,
                structural_state TEXT,
                PRIMARY KEY (job_id, ligand_id, pose_rank)
            )
            """
        )
        conn.commit()


def load_jsonl_poses(jsonl_path: Path) -> List[PoseScore]:
    if not jsonl_path.is_file():
        return []
    poses: List[PoseScore] = []
    with open(jsonl_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            for pose in payload.get("poses", []):
                poses.append(PoseScore(**pose))
    return poses


def pair_job_ids_with_poses(jsonl_path: Path) -> List[Tuple[str, PoseScore]]:
    if not jsonl_path.is_file():
        return []
    rows: List[Tuple[str, PoseScore]] = []
    with open(jsonl_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            job = payload.get("job", {})
            job_id = str(job.get("job_id", ""))
            for pose in payload.get("poses", []):
                rows.append((job_id, PoseScore(**pose)))
    return rows


def reduce_runtime_to_sqlite(
    *,
    db_path: Path,
    state_records: Dict[str, Dict[str, object]],
    pose_rows: Iterable[Tuple[str, PoseScore]],
) -> None:
    ensure_runtime_database(db_path)
    with sqlite3.connect(db_path) as conn:
        for job_id, record in state_records.items():
            conn.execute(
                """
                INSERT OR REPLACE INTO jobs (
                    job_id, dataset_slug, tier, target_id, structural_state, status,
                    assigned_executor, started_at, finished_at, duration_seconds,
                    num_poses, log_path, error, attempt
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    record.get("dataset_slug"),
                    record.get("tier"),
                    record.get("target_id"),
                    record.get("structural_state"),
                    record.get("status"),
                    record.get("assigned_executor"),
                    record.get("started_at"),
                    record.get("finished_at"),
                    record.get("duration_seconds"),
                    record.get("num_poses"),
                    record.get("log_path"),
                    record.get("error"),
                    record.get("attempt"),
                ),
            )
        for job_id, pose in pose_rows:
            conn.execute(
                """
                INSERT OR REPLACE INTO poses (
                    job_id, target_id, ligand_id, pose_rank, rmsd,
                    enthalpy_score, entropy_correction, total_score,
                    is_active, exp_affinity, structural_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    pose.target_id,
                    pose.ligand_id,
                    pose.pose_rank,
                    pose.rmsd,
                    pose.enthalpy_score,
                    pose.entropy_correction,
                    pose.total_score,
                    1 if pose.is_active else 0,
                    pose.exp_affinity,
                    pose.structural_state,
                ),
            )
        conn.commit()
