from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol

import numpy as np

from benchmarks.DatasetRunner import DatasetConfig, DatasetResult
from benchmarks.metrics import PoseScore, compute_all_metrics


class DatasetAdapter(Protocol):
    slug: str

    def apply(
        self,
        *,
        config: DatasetConfig,
        result: DatasetResult,
        poses: List[PoseScore],
        state_records: Dict[str, Dict[str, object]],
        requested_metrics: Optional[List[str]],
    ) -> DatasetResult:
        ...


@dataclass(frozen=True)
class Casf2016Adapter:
    slug: str = "casf2016"

    def apply(
        self,
        *,
        config: DatasetConfig,
        result: DatasetResult,
        poses: List[PoseScore],
        state_records: Dict[str, Dict[str, object]],
        requested_metrics: Optional[List[str]],
    ) -> DatasetResult:
        metrics = dict(compute_all_metrics(poses, requested=requested_metrics)) if poses else {}
        top1 = _best_pose_per_target(poses)
        exp = np.array([p.exp_affinity for p in top1 if p.exp_affinity is not None], dtype=float)
        pred = np.array([p.total_score for p in top1 if p.exp_affinity is not None], dtype=float)
        if exp.size >= 2 and pred.size == exp.size:
            metrics.setdefault("casf_top1_pearson_r", _safe_corrcoef(pred, exp))
            metrics.setdefault("casf_top1_rmse", float(np.sqrt(np.mean((pred - exp) ** 2))))
        metrics.setdefault("casf_n_targets_total", float(len(config.targets)))
        metrics.setdefault("casf_n_targets_scored", float(len(top1)))
        metrics.setdefault("casf_state_records", float(len(state_records)))
        result.metrics = metrics
        result.check_regressions()
        return result


@dataclass(frozen=True)
class CrossDockAdapter:
    slug: str = "crossdock"

    def apply(
        self,
        *,
        config: DatasetConfig,
        result: DatasetResult,
        poses: List[PoseScore],
        state_records: Dict[str, Dict[str, object]],
        requested_metrics: Optional[List[str]],
    ) -> DatasetResult:
        metrics = dict(compute_all_metrics(poses, requested=requested_metrics)) if poses else {}
        top1 = _best_pose_per_target(poses)
        rmsds = np.array([p.rmsd for p in top1 if p.rmsd is not None and p.rmsd >= 0], dtype=float)
        if rmsds.size:
            metrics.setdefault("crossdock_success_rate_2A", float(np.mean(rmsds <= 2.0)))
            metrics.setdefault("crossdock_success_rate_3A", float(np.mean(rmsds <= 3.0)))
            metrics.setdefault("crossdock_mean_rmsd", float(np.mean(rmsds)))
            metrics.setdefault("crossdock_median_rmsd", float(np.median(rmsds)))
        metrics.setdefault("crossdock_n_pairs_attempted", float(len(config.targets)))
        metrics.setdefault("crossdock_n_pairs_scored", float(len(top1)))
        metrics.setdefault("crossdock_state_records", float(len(state_records)))
        result.metrics = metrics
        result.check_regressions()
        return result


@dataclass(frozen=True)
class LitPcbaAdapter:
    slug: str = "litpcba"

    def apply(
        self,
        *,
        config: DatasetConfig,
        result: DatasetResult,
        poses: List[PoseScore],
        state_records: Dict[str, Dict[str, object]],
        requested_metrics: Optional[List[str]],
    ) -> DatasetResult:
        metrics = dict(compute_all_metrics(poses, requested=requested_metrics)) if poses else {}
        target_ids = sorted({p.target_id for p in poses})
        metrics.setdefault("litpcba_targets_scored", float(len(target_ids)))
        metrics.setdefault("litpcba_state_records", float(len(state_records)))
        data_dir = config.data_dir or Path(os.environ.get("LITPCBA_DATA", ""))
        if data_dir and Path(data_dir).is_dir():
            active_counts = []
            inactive_counts = []
            for target in target_ids:
                active_counts.append(_count_lines(Path(data_dir) / target / "actives.smi"))
                inactive_counts.append(_count_lines(Path(data_dir) / target / "inactives.smi"))
            metrics.setdefault("litpcba_actives_total", float(sum(active_counts)))
            metrics.setdefault("litpcba_inactives_total", float(sum(inactive_counts)))
        result.metrics = metrics
        result.check_regressions()
        return result


class LegacyBenchmarkUnifier:
    def __init__(self) -> None:
        self._registry: Dict[str, DatasetAdapter] = {
            "casf2016": Casf2016Adapter(),
            "crossdock": CrossDockAdapter(),
            "litpcba": LitPcbaAdapter(),
        }

    def adapter_for(self, slug: str) -> Optional[DatasetAdapter]:
        return self._registry.get(slug.lower())

    def evaluate(
        self,
        *,
        config: DatasetConfig,
        result: DatasetResult,
        poses: List[PoseScore],
        state_records: Dict[str, Dict[str, object]],
        requested_metrics: Optional[List[str]],
    ) -> DatasetResult:
        adapter = self.adapter_for(config.slug)
        if adapter is None:
            if poses:
                result.metrics = dict(compute_all_metrics(poses, requested=requested_metrics))
                result.check_regressions()
            return result
        return adapter.apply(
            config=config,
            result=result,
            poses=poses,
            state_records=state_records,
            requested_metrics=requested_metrics,
        )


def _best_pose_per_target(poses: Iterable[PoseScore]) -> List[PoseScore]:
    best: Dict[str, PoseScore] = {}
    for pose in poses:
        current = best.get(pose.target_id)
        if current is None or pose.pose_rank < current.pose_rank:
            best[pose.target_id] = pose
    return [best[key] for key in sorted(best)]


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with open(path, encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())
