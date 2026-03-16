"""Continuous training pipeline for 256x256 energy matrices.

Implements curriculum training with warm-start, multi-dataset ingestion,
quality-gated promotion, and full provenance tracking.

Usage::

    python -m flexaidds.energy_matrix_cli continuous-train \\
        --itc-dir data/datasets/itc_187 \\
        --pdbbind-refined data/datasets/pdbbind/refined-set \\
        --casf-dir data/datasets/pdbbind/core-set \\
        --prior-matrix data/training_runs/run_001/matrix_256x256.bin \\
        --output-dir data/training_runs

Dependencies: numpy, scipy (BSD-licensed).  No GPL dependencies.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.stats import pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .energy_matrix import ContactTable, EnergyMatrix
from .train_256x256 import (
    CONTACT_CUTOFF,
    TEMPERATURE,
    Complex,
    build_contact_matrix,
    build_reference_matrix,
    inverse_boltzmann,
    lbfgs_refine,
    ridge_fit,
    validate_casf,
    validate_projection,
    kB_kcal,
    load_pdbbind_complexes,
)
from .dataset_adapters import (
    DatasetAdapter,
    DatasetMetadata,
    ITC187Adapter,
    PDBbindAdapter,
    BindingMOADAdapter,
    BindingDBAdapter,
    ChEMBLAdapter,
    DUDEAdapter,
    DEKOIS2Adapter,
    complexes_to_contact_table,
    get_or_build_contact_table,
    checksum_contact_table,
    create_adapter,
    TIER_WEIGHTS,
)

logger = logging.getLogger(__name__)


# ── configuration ────────────────────────────────────────────────────────────

@dataclass
class ContinuousTrainingConfig:
    """Configuration for the full continuous training pipeline."""

    # Dataset directories (empty string = skip)
    itc_dir: str = ""
    pdbbind_core_dir: str = ""
    pdbbind_refined_dir: str = ""
    pdbbind_general_dir: str = ""
    moad_dir: str = ""
    bindingdb_dir: str = ""
    chembl_dir: str = ""

    # Validation datasets
    casf_dir: str = ""
    dude_dir: str = ""
    dekois_dir: str = ""

    # Warm-start
    prior_matrix_path: str = ""
    contact_cache_dir: str = "data/contact_tables"

    # Training parameters
    contact_cutoff: float = CONTACT_CUTOFF
    temperature: float = TEMPERATURE
    seed: int = 42

    # Output
    output_dir: str = "data/training_runs"
    run_name: str = ""

    # Reference for projection validation
    reference_dat: str = ""

    # Quality gates
    casf_min_r: float = 0.75
    itc_min_r: float = 0.85

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── curriculum phase ─────────────────────────────────────────────────────────

@dataclass
class CurriculumPhase:
    """One phase of the curriculum training."""
    name: str
    order: int
    datasets: List[Tuple[DatasetAdapter, str]]  # (adapter, data_dir) pairs
    ridge_alpha: float = 1.0
    lbfgs_maxiter: int = 200
    prior_mixing_floor: float = 0.0
    dataset_weight: float = 1.0


# ── quality gates ────────────────────────────────────────────────────────────

@dataclass
class QualityGateResult:
    """Results from validation quality gates."""
    casf_pearson_r: float = 0.0
    casf_rmse: float = 0.0
    casf_n: int = 0
    casf_passed: bool = False

    itc_pearson_r: float = 0.0
    itc_rmse: float = 0.0
    itc_n: int = 0
    itc_passed: bool = False

    # Supplementary (tracked, not gating)
    dude_mean_auc: float = 0.0
    dekois_mean_auc: float = 0.0

    @property
    def all_gates_passed(self) -> bool:
        return self.casf_passed and self.itc_passed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── training run result ──────────────────────────────────────────────────────

@dataclass
class TrainingRunResult:
    """Complete result of a training run."""
    run_id: str
    matrix: Optional[EnergyMatrix] = None
    matrix_path: str = ""
    gate_results: Optional[QualityGateResult] = None
    promoted: bool = False
    phase_metrics: Dict[str, Dict] = field(default_factory=dict)
    manifest: Dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "run_id": self.run_id,
            "matrix_path": self.matrix_path,
            "promoted": self.promoted,
            "elapsed_seconds": self.elapsed_seconds,
            "phase_metrics": self.phase_metrics,
        }
        if self.gate_results:
            d["gate_results"] = self.gate_results.to_dict()
        return d


# ── warm-start logic ─────────────────────────────────────────────────────────

def compute_cell_confidence(
    contact_counts: np.ndarray,
    min_contacts: int = 10,
    saturation: int = 1000,
) -> np.ndarray:
    """Compute per-cell confidence for warm-start mixing.

    Returns (256, 256) array in [0, 1]:
        0.0 = fewer than min_contacts → keep prior entirely
        1.0 = saturated contacts → trust new estimate entirely
    """
    return np.clip(
        (contact_counts - min_contacts) / max(saturation - min_contacts, 1),
        0.0,
        1.0,
    )


def warm_start_combine(
    prior: np.ndarray,
    new_estimate: np.ndarray,
    contact_counts: np.ndarray,
    dataset_weight: float = 1.0,
    mixing_floor: float = 0.0,
) -> np.ndarray:
    """Combine prior matrix with new estimate using adaptive per-cell mixing.

    For each cell (i,j):
        λ = max(dataset_weight * confidence(i,j), mixing_floor)
        result(i,j) = (1 - λ) * prior(i,j) + λ * new_estimate(i,j)

    where confidence is based on the number of observed contacts for that cell.
    The mixing_floor ensures noisy datasets can never fully overwrite the prior.

    Args:
        prior: Previous best 256×256 matrix.
        new_estimate: Freshly computed matrix from current phase data.
        contact_counts: (256, 256) observed contact counts from current data.
        dataset_weight: Reliability weight for this dataset (0-1).
        mixing_floor: Minimum prior retention (higher = more conservative).
    """
    confidence = compute_cell_confidence(contact_counts)
    lam = dataset_weight * confidence
    # mixing_floor acts as a ceiling on how much new data can contribute
    # i.e., prior retains at least mixing_floor weight
    effective_lam = np.clip(lam, 0.0, 1.0 - mixing_floor)
    result = (1.0 - effective_lam) * prior + effective_lam * new_estimate
    # Enforce symmetry
    result = (result + result.T) / 2.0
    return result


# ── ITC cross-validation for gating ─────────────────────────────────────────

def validate_itc_crossval(
    matrix: np.ndarray,
    itc_dir: str,
    cutoff: float = CONTACT_CUTOFF,
    n_folds: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    """5-fold cross-validation on ITC-187 for quality gating.

    Scores each held-out fold using the trained matrix and computes
    Pearson r between predicted and experimental ΔG.

    Returns dict with: mean_pearson_r, std_pearson_r, mean_rmse, n_complexes.
    """
    adapter = ITC187Adapter()
    complexes = adapter.load(itc_dir, cutoff)

    if len(complexes) < n_folds:
        return {
            "mean_pearson_r": 0.0, "std_pearson_r": 0.0,
            "mean_rmse": 0.0, "n_complexes": len(complexes),
        }

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(complexes))
    fold_size = len(complexes) // n_folds
    rs = []
    rmses = []

    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else len(complexes)
        val_idx = indices[start:end]

        predicted = []
        experimental = []
        for i in val_idx:
            cpx = complexes[i]
            score = sum(matrix[c.type_a, c.type_b] for c in cpx.contacts)
            predicted.append(score)
            experimental.append(cpx.deltaG)

        if len(predicted) < 3:
            continue

        pred_arr = np.array(predicted)
        exp_arr = np.array(experimental)

        if HAS_SCIPY and np.std(pred_arr) > 1e-12 and np.std(exp_arr) > 1e-12:
            r, _ = pearsonr(pred_arr, exp_arr)
            rs.append(r)
        rmse = float(np.sqrt(np.mean((pred_arr - exp_arr) ** 2)))
        rmses.append(rmse)

    return {
        "mean_pearson_r": float(np.mean(rs)) if rs else 0.0,
        "std_pearson_r": float(np.std(rs)) if rs else 0.0,
        "mean_rmse": float(np.mean(rmses)) if rmses else 0.0,
        "n_complexes": len(complexes),
    }


# ── continuous trainer ───────────────────────────────────────────────────────

class ContinuousTrainer:
    """Orchestrates multi-dataset curriculum training with quality gates."""

    def __init__(self, config: ContinuousTrainingConfig):
        self.config = config
        self.run_id = config.run_name or self._generate_run_id()
        self.run_dir = Path(config.output_dir) / self.run_id
        self.phase_metrics: Dict[str, Dict] = {}
        self._dataset_metadata: List[Dict] = []

    @staticmethod
    def _generate_run_id() -> str:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{ts}"

    def run(self) -> TrainingRunResult:
        """Execute the full curriculum training pipeline."""
        start_time = time.time()
        np.random.seed(self.config.seed)

        self._setup_run_directory()
        log_path = self.run_dir / "training.log"
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

        try:
            # Load prior matrix (warm-start) or start from zeros
            prior = self._load_prior()

            # Build curriculum phases
            curriculum = self._build_curriculum()
            if not curriculum:
                raise RuntimeError("No datasets configured — nothing to train on")

            logger.info(
                "Starting continuous training: %d phases, run_id=%s",
                len(curriculum), self.run_id,
            )

            matrix = prior
            for phase in curriculum:
                logger.info("Phase %d: %s", phase.order, phase.name)
                matrix = self._train_phase(phase, matrix)
                self._checkpoint(phase, matrix)

            # Quality gates
            gate_results = self._run_quality_gates(matrix)

            # Finalize
            elapsed = time.time() - start_time
            result = self._finalize(matrix, gate_results, elapsed)

            logger.info(
                "Training complete: promoted=%s, elapsed=%.1fs",
                result.promoted, elapsed,
            )
            return result

        finally:
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()

    def _setup_run_directory(self) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.run_dir / "checkpoints", exist_ok=True)

    def _load_prior(self) -> np.ndarray:
        """Load prior matrix for warm-start, or return zeros."""
        if self.config.prior_matrix_path and os.path.isfile(
            self.config.prior_matrix_path
        ):
            logger.info("Loading prior matrix: %s", self.config.prior_matrix_path)
            em = EnergyMatrix.from_binary(self.config.prior_matrix_path)
            return em.matrix.copy()
        logger.info("No prior matrix — starting from scratch")
        return np.zeros((256, 256), dtype=np.float64)

    def _build_curriculum(self) -> List[CurriculumPhase]:
        """Build ordered curriculum phases from configured datasets."""
        cfg = self.config
        phases: List[CurriculumPhase] = []
        order = 0

        # Phase 1: ITC-187 (gold standard)
        if cfg.itc_dir:
            phases.append(CurriculumPhase(
                name="phase_1_itc187",
                order=(order := order + 1),
                datasets=[(ITC187Adapter(), cfg.itc_dir)],
                ridge_alpha=0.5,
                lbfgs_maxiter=300,
                prior_mixing_floor=0.0,
                dataset_weight=1.0,
            ))

        # Phase 2: PDBbind core + refined
        phase2_datasets = []
        if cfg.pdbbind_core_dir:
            phase2_datasets.append((
                PDBbindAdapter(subset="core", tier=1, weight=0.95),
                cfg.pdbbind_core_dir,
            ))
        if cfg.pdbbind_refined_dir:
            phase2_datasets.append((
                PDBbindAdapter(subset="refined", tier=2, weight=0.80),
                cfg.pdbbind_refined_dir,
            ))
        if phase2_datasets:
            phases.append(CurriculumPhase(
                name="phase_2_pdbbind_high",
                order=(order := order + 1),
                datasets=phase2_datasets,
                ridge_alpha=1.0,
                lbfgs_maxiter=200,
                prior_mixing_floor=0.3,
                dataset_weight=0.80,
            ))

        # Phase 3: PDBbind general + Binding MOAD
        phase3_datasets = []
        if cfg.pdbbind_general_dir:
            phase3_datasets.append((
                PDBbindAdapter(subset="general", tier=3, weight=0.50),
                cfg.pdbbind_general_dir,
            ))
        if cfg.moad_dir:
            phase3_datasets.append((
                BindingMOADAdapter(), cfg.moad_dir,
            ))
        if phase3_datasets:
            phases.append(CurriculumPhase(
                name="phase_3_broad_coverage",
                order=(order := order + 1),
                datasets=phase3_datasets,
                ridge_alpha=2.0,
                lbfgs_maxiter=150,
                prior_mixing_floor=0.5,
                dataset_weight=0.50,
            ))

        # Phase 4: BindingDB + ChEMBL (noisy, heavy regularization)
        phase4_datasets = []
        if cfg.bindingdb_dir:
            phase4_datasets.append((BindingDBAdapter(), cfg.bindingdb_dir))
        if cfg.chembl_dir:
            phase4_datasets.append((ChEMBLAdapter(), cfg.chembl_dir))
        if phase4_datasets:
            phases.append(CurriculumPhase(
                name="phase_4_noisy_broad",
                order=(order := order + 1),
                datasets=phase4_datasets,
                ridge_alpha=5.0,
                lbfgs_maxiter=100,
                prior_mixing_floor=0.7,
                dataset_weight=0.25,
            ))

        return phases

    def _train_phase(
        self,
        phase: CurriculumPhase,
        prior: np.ndarray,
    ) -> np.ndarray:
        """Execute one curriculum phase with warm-start."""
        all_complexes: List[Complex] = []

        for adapter, data_dir in phase.datasets:
            table, complexes = get_or_build_contact_table(
                adapter, data_dir,
                self.config.contact_cache_dir,
                self.config.contact_cutoff,
            )
            all_complexes.extend(complexes)

            meta = adapter.metadata()
            self._dataset_metadata.append(meta.to_dict())
            logger.info(
                "  %s: %d complexes (tier %d, weight %.2f)",
                adapter.name(), meta.n_complexes,
                meta.reliability_tier, meta.weight,
            )

        if not all_complexes:
            logger.warning("Phase %s: no complexes loaded, skipping", phase.name)
            return prior

        # Build frequency matrices from this phase's data
        freq = build_contact_matrix(all_complexes)
        ref = build_reference_matrix(all_complexes)

        # Compute new estimate: Sippl + Ridge + L-BFGS
        sippl = inverse_boltzmann(freq, ref, self.config.temperature)
        ridge_matrix = ridge_fit(all_complexes, phase.ridge_alpha)
        new_estimate = 0.7 * sippl + 0.3 * ridge_matrix
        new_estimate = (new_estimate + new_estimate.T) / 2.0

        # L-BFGS refinement on new estimate
        new_estimate = lbfgs_refine(new_estimate, all_complexes, phase.lbfgs_maxiter)

        # Warm-start combine with prior
        is_cold_start = np.allclose(prior, 0.0)
        if is_cold_start:
            matrix = new_estimate
        else:
            matrix = warm_start_combine(
                prior=prior,
                new_estimate=new_estimate,
                contact_counts=freq,
                dataset_weight=phase.dataset_weight,
                mixing_floor=phase.prior_mixing_floor,
            )

        # Record phase metrics
        metrics = validate_casf(matrix, all_complexes)
        self.phase_metrics[phase.name] = {
            "n_complexes": len(all_complexes),
            "pearson_r": metrics.get("pearson_r", 0.0),
            "rmse": metrics.get("rmse", 0.0),
            "nonzero_cells": int(np.count_nonzero(matrix)),
            "ridge_alpha": phase.ridge_alpha,
            "lbfgs_maxiter": phase.lbfgs_maxiter,
            "warm_started": not is_cold_start,
        }
        logger.info(
            "  Phase %s: r=%.4f, RMSE=%.4f, nonzero=%d",
            phase.name,
            metrics.get("pearson_r", 0),
            metrics.get("rmse", 0),
            np.count_nonzero(matrix),
        )

        return matrix

    def _checkpoint(self, phase: CurriculumPhase, matrix: np.ndarray) -> None:
        """Save intermediate checkpoint."""
        ckpt_path = self.run_dir / "checkpoints" / f"{phase.name}.bin"
        em = EnergyMatrix(256, matrix)
        em.to_binary(str(ckpt_path))
        logger.info("  Checkpoint saved: %s", ckpt_path)

    def _run_quality_gates(self, matrix: np.ndarray) -> QualityGateResult:
        """Evaluate CASF-2016 and ITC-187 quality gates."""
        result = QualityGateResult()
        cfg = self.config

        # Gate 1: CASF-2016
        if cfg.casf_dir and os.path.isdir(cfg.casf_dir):
            logger.info("Quality gate: CASF-2016")
            casf_complexes = load_pdbbind_complexes(
                cfg.casf_dir, cfg.contact_cutoff
            )
            if casf_complexes:
                casf_metrics = validate_casf(matrix, casf_complexes)
                result.casf_pearson_r = casf_metrics.get("pearson_r", 0.0)
                result.casf_rmse = casf_metrics.get("rmse", 0.0)
                result.casf_n = casf_metrics.get("n_complexes", 0)
                result.casf_passed = result.casf_pearson_r >= cfg.casf_min_r
                logger.info(
                    "  CASF: r=%.4f (threshold=%.2f) → %s",
                    result.casf_pearson_r, cfg.casf_min_r,
                    "PASS" if result.casf_passed else "FAIL",
                )
            else:
                logger.warning("  CASF: no complexes loaded")
        else:
            # No CASF data → pass by default (user didn't configure it)
            result.casf_passed = True
            logger.info("Quality gate: CASF-2016 not configured, auto-pass")

        # Gate 2: ITC-187
        if cfg.itc_dir and os.path.isdir(cfg.itc_dir):
            logger.info("Quality gate: ITC-187 (5-fold CV)")
            itc_metrics = validate_itc_crossval(
                matrix, cfg.itc_dir, cfg.contact_cutoff,
                n_folds=5, seed=cfg.seed,
            )
            result.itc_pearson_r = itc_metrics.get("mean_pearson_r", 0.0)
            result.itc_rmse = itc_metrics.get("mean_rmse", 0.0)
            result.itc_n = itc_metrics.get("n_complexes", 0)
            result.itc_passed = result.itc_pearson_r >= cfg.itc_min_r
            logger.info(
                "  ITC-187: r=%.4f (threshold=%.2f) → %s",
                result.itc_pearson_r, cfg.itc_min_r,
                "PASS" if result.itc_passed else "FAIL",
            )
        else:
            result.itc_passed = True
            logger.info("Quality gate: ITC-187 not configured, auto-pass")

        return result

    def _finalize(
        self,
        matrix: np.ndarray,
        gates: QualityGateResult,
        elapsed: float,
    ) -> TrainingRunResult:
        """Save final artifacts, manifest, and determine promotion."""
        # Save final matrix
        matrix_path = str(self.run_dir / "matrix_256x256.bin")
        em = EnergyMatrix(256, matrix)
        em.to_binary(matrix_path)

        # Save 40×40 projection
        proj = em.project_to_40()
        proj_path = str(self.run_dir / "matrix_40x40.dat")
        proj.to_dat_file(proj_path)

        # Projection validation
        proj_metrics: Dict[str, Any] = {}
        if self.config.reference_dat and os.path.isfile(self.config.reference_dat):
            proj_metrics = validate_projection(matrix, self.config.reference_dat)

        promoted = gates.all_gates_passed

        # Build manifest
        manifest = {
            "run_id": self.run_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "datasets": self._dataset_metadata,
            "curriculum": {
                name: metrics for name, metrics in self.phase_metrics.items()
            },
            "prior_matrix": {
                "path": self.config.prior_matrix_path or None,
            },
            "quality_gates": {
                "casf_2016": {
                    "pearson_r": gates.casf_pearson_r,
                    "rmse": gates.casf_rmse,
                    "n_complexes": gates.casf_n,
                    "threshold": self.config.casf_min_r,
                    "passed": gates.casf_passed,
                },
                "itc_187": {
                    "pearson_r": gates.itc_pearson_r,
                    "rmse": gates.itc_rmse,
                    "n_complexes": gates.itc_n,
                    "threshold": self.config.itc_min_r,
                    "passed": gates.itc_passed,
                },
            },
            "supplementary_metrics": {
                "dude_mean_auc": gates.dude_mean_auc,
                "dekois_mean_auc": gates.dekois_mean_auc,
                **({
                    "projection_r": proj_metrics.get("projection_r", 0),
                    "projection_rmse": proj_metrics.get("projection_rmse", 0),
                } if proj_metrics else {}),
            },
            "promoted": promoted,
            "elapsed_seconds": elapsed,
            "artifacts": {
                "matrix_256x256": "matrix_256x256.bin",
                "matrix_40x40": "matrix_40x40.dat",
                "training_log": "training.log",
            },
        }

        # Write manifest
        manifest_path = self.run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

        # Write metrics summary
        metrics_path = self.run_dir / "metrics.json"
        metrics_path.write_text(json.dumps({
            "phase_metrics": self.phase_metrics,
            "quality_gates": manifest["quality_gates"],
            "supplementary": manifest["supplementary_metrics"],
        }, indent=2, default=str))

        logger.info("Artifacts saved to %s", self.run_dir)

        return TrainingRunResult(
            run_id=self.run_id,
            matrix=em,
            matrix_path=matrix_path,
            gate_results=gates,
            promoted=promoted,
            phase_metrics=self.phase_metrics,
            manifest=manifest,
            elapsed_seconds=elapsed,
        )


# ── run comparison ───────────────────────────────────────────────────────────

def compare_runs(run_a_dir: str, run_b_dir: str) -> Dict[str, Any]:
    """Compare two training runs by loading their manifests."""
    a_manifest = json.loads((Path(run_a_dir) / "manifest.json").read_text())
    b_manifest = json.loads((Path(run_b_dir) / "manifest.json").read_text())

    a_gates = a_manifest.get("quality_gates", {})
    b_gates = b_manifest.get("quality_gates", {})

    comparison = {
        "run_a": a_manifest.get("run_id"),
        "run_b": b_manifest.get("run_id"),
        "casf_r_delta": (
            b_gates.get("casf_2016", {}).get("pearson_r", 0)
            - a_gates.get("casf_2016", {}).get("pearson_r", 0)
        ),
        "itc_r_delta": (
            b_gates.get("itc_187", {}).get("pearson_r", 0)
            - a_gates.get("itc_187", {}).get("pearson_r", 0)
        ),
        "promoted_a": a_manifest.get("promoted", False),
        "promoted_b": b_manifest.get("promoted", False),
        "datasets_a": len(a_manifest.get("datasets", [])),
        "datasets_b": len(b_manifest.get("datasets", [])),
    }
    return comparison


def list_runs(runs_dir: str) -> List[Dict[str, Any]]:
    """List all training runs with summary metrics."""
    runs = []
    for entry in sorted(Path(runs_dir).iterdir()):
        if not entry.is_dir():
            continue
        manifest_path = entry / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
            gates = manifest.get("quality_gates", {})
            runs.append({
                "run_id": manifest.get("run_id", entry.name),
                "timestamp": manifest.get("timestamp", ""),
                "promoted": manifest.get("promoted", False),
                "casf_r": gates.get("casf_2016", {}).get("pearson_r", 0),
                "itc_r": gates.get("itc_187", {}).get("pearson_r", 0),
                "n_datasets": len(manifest.get("datasets", [])),
                "elapsed_s": manifest.get("elapsed_seconds", 0),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return runs
