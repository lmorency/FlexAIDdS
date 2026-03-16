# Continuous Training Methodology for 256×256 Energy Matrices

**Author**: FlexAIDdS Development Team
**Date**: 2026-03-16
**Status**: Implementation Plan
**Scope**: Multi-dataset ingestion, curriculum training, warm-start, quality-gated promotion

---

## 1. Executive Summary

This document defines a methodology for continuously training FlexAIDdS 256×256 soft contact energy matrices from the most exhaustive experimental binding affinity datasets available. The pipeline:

1. **Ingests 7 datasets** — PDBbind (general/refined/core), Binding MOAD, BindingDB, ChEMBL, DEKOIS 2.0, DUD-E, and ITC-187
2. **Applies curriculum training** — high-quality data first (ITC-187 → PDBbind refined → general → MOAD → BindingDB → ChEMBL), with dataset-specific reliability weights
3. **Warm-starts** from previous best matrix on each new training round
4. **Quality-gates** every candidate matrix through CASF-2016 (Pearson *r* ≥ 0.75) and ITC-187 ΔG correlation (*r* ≥ 0.85) before promotion
5. **Versions** every training run with full provenance (datasets, parameters, metrics, artifacts)
6. **CLI-driven** — manual invocation consistent with existing `energy_matrix_cli.py` patterns

---

## 2. Dataset Inventory

### 2.1 Dataset Specifications

| Dataset | Size | Data Type | Affinity Measure | 3D Structures | Reliability Tier | Weight |
|---------|------|-----------|------------------|---------------|-----------------|--------|
| **ITC-187** | 187 complexes | Calorimetric | ΔH, TΔS, ΔG (direct) | Yes (PDB) | Tier 1 (Gold) | 1.00 |
| **PDBbind Core** | ~300 complexes | X-ray + affinity | Kd/Ki/IC50 → pKd | Yes (PDB) | Tier 1 | 0.95 |
| **PDBbind Refined** | ~5,300 complexes | X-ray + affinity | Kd/Ki/IC50 → pKd | Yes (PDB) | Tier 2 | 0.80 |
| **PDBbind General** | ~23,000 complexes | X-ray + affinity | Kd/Ki/IC50 → pKd | Yes (PDB) | Tier 3 | 0.50 |
| **Binding MOAD** | ~38,000 entries | X-ray + affinity | Kd/Ki/IC50/EC50 | Yes (PDB) | Tier 3 | 0.45 |
| **BindingDB** | ~2.9M entries | Assorted assays | Ki/Kd/IC50/EC50 | Partial (PDB cross-ref) | Tier 4 | 0.25 |
| **ChEMBL** | ~2.4M bioactivities | Assorted assays | IC50/Ki/Kd/% inhibition | Requires PDB cross-ref | Tier 4 | 0.20 |
| **DEKOIS 2.0** | 81 targets × ~1,300 decoys | VS benchmark | Active/decoy labels | Yes (SMILES → 3D) | Validation only | — |
| **DUD-E** | 102 targets × ~22,000 decoys | VS benchmark | Active/decoy labels | Yes (SMILES → 3D) | Validation only | — |

### 2.2 Dataset Roles

- **Training**: ITC-187, PDBbind (all tiers), Binding MOAD, BindingDB, ChEMBL
- **Validation gates**: CASF-2016 (subset of PDBbind core, held out), ITC-187 (cross-validated)
- **Supplementary validation**: DUD-E enrichment, DEKOIS 2.0 enrichment (not gating, but tracked)

### 2.3 Dataset Directory Structure

```
data/
├── datasets/
│   ├── pdbbind/
│   │   ├── refined-set/          # PDBbind refined set
│   │   │   ├── INDEX/INDEX_general_PL_data.2020
│   │   │   ├── 1a0q/1a0q_protein.pdb
│   │   │   ├── 1a0q/1a0q_ligand.mol2
│   │   │   └── ...
│   │   ├── general-set/          # PDBbind general set
│   │   └── core-set/             # PDBbind core set (CASF-2016)
│   ├── binding_moad/
│   │   ├── every.csv             # Binding MOAD master index
│   │   └── structures/           # PDB files
│   ├── bindingdb/
│   │   └── BindingDB_All.tsv     # Full BindingDB export
│   ├── chembl/
│   │   └── chembl_activities.csv # ChEMBL bioactivity data
│   ├── itc_187/
│   │   ├── itc_index.csv         # PDB code, ΔH, TΔS, ΔG
│   │   └── structures/           # PDB + ligand files
│   ├── dekois2/
│   │   └── targets/              # Per-target active/decoy sets
│   └── dude/
│       └── targets/              # Per-target active/decoy sets
├── contact_tables/               # Cached ContactTable JSON per dataset
│   ├── pdbbind_refined_v2020.json
│   ├── binding_moad_2024.json
│   └── ...
└── training_runs/                # Versioned training artifacts
    ├── run_001/
    │   ├── manifest.json         # Full provenance record
    │   ├── matrix_256x256.bin    # Trained matrix (SHNN format)
    │   ├── matrix_40x40.dat      # Legacy projection
    │   ├── metrics.json          # All validation metrics
    │   └── training.log          # Full training log
    └── run_002/
        └── ...
```

---

## 3. Multi-Dataset Adapter Architecture

### 3.1 Adapter Interface

Each dataset requires a **DatasetAdapter** that normalizes heterogeneous formats into the common `Complex` dataclass used by the existing `train_256x256.py` pipeline.

```python
# python/flexaidds/dataset_adapters.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Iterator
from pathlib import Path

@dataclass
class DatasetMetadata:
    """Provenance metadata for a dataset."""
    name: str                          # e.g. "pdbbind_refined"
    version: str                       # e.g. "v2020"
    source_url: str                    # Download URL
    n_complexes: int = 0
    reliability_tier: int = 1          # 1 (gold) to 4 (noisy)
    weight: float = 1.0               # Training weight
    date_loaded: str = ""

class DatasetAdapter(ABC):
    """Base class for dataset-specific loaders."""

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(self, data_dir: str, cutoff: float = 4.5) -> List[Complex]: ...

    @abstractmethod
    def metadata(self) -> DatasetMetadata: ...

    def to_contact_table(self, complexes: List[Complex]) -> ContactTable:
        """Convert loaded complexes to a ContactTable for incremental merging."""
        ...
```

### 3.2 Dataset-Specific Adapters

| Adapter Class | Dataset | Input Format | Notes |
|---------------|---------|-------------|-------|
| `PDBbindAdapter` | PDBbind refined/general/core | PDB + MOL2 + INDEX file | Already implemented in `train_256x256.py`; refactor into adapter |
| `BindingMOADAdapter` | Binding MOAD | PDB + `every.csv` index | Parse Kd/Ki/IC50 from MOAD CSV; download PDB structures |
| `BindingDBAdapter` | BindingDB | TSV export + PDB cross-ref | Filter entries with PDB codes; convert Ki/Kd to ΔG |
| `ChEMBLAdapter` | ChEMBL | CSV export + PDB cross-ref | Filter protein-ligand with PDB mapping; IC50→Ki→ΔG conversion |
| `ITC187Adapter` | ITC-187 | CSV index + PDB structures | Direct ΔH/TΔS/ΔG values — highest quality |
| `DEKOIS2Adapter` | DEKOIS 2.0 | SDF + active/decoy labels | Validation only — no affinity values |
| `DUDEAdapter` | DUD-E | SDF/MOL2 + active/decoy labels | Validation only — no affinity values |

### 3.3 Affinity Normalization

All datasets normalize binding affinity to a common scale:

```python
def normalize_affinity(value: float, unit: str, temperature: float = 298.15) -> float:
    """Convert heterogeneous affinity measures to ΔG (kcal/mol).

    Supported units: 'Kd', 'Ki', 'IC50', 'pKd', 'pKi', 'pIC50', 'deltaG'.

    Conversions:
        Kd/Ki  → ΔG = RT·ln(Kd)       [Kd in molar]
        IC50   → Ki ≈ IC50/2  (Cheng-Prusoff approx) → ΔG
        pKd    → Kd = 10^(-pKd)        → ΔG
        deltaG → passthrough
    """
```

### 3.4 Contact Table Caching

Building contact tables from large datasets (23K+ complexes) is expensive. The pipeline caches `ContactTable` JSON files per dataset version:

```python
def get_or_build_contact_table(
    adapter: DatasetAdapter,
    data_dir: str,
    cache_dir: str,
    cutoff: float = 4.5,
    force_rebuild: bool = False,
) -> ContactTable:
    """Load cached ContactTable or build from scratch."""
    cache_path = Path(cache_dir) / f"{adapter.name()}.json"
    if cache_path.exists() and not force_rebuild:
        return ContactTable.load(str(cache_path))
    complexes = adapter.load(data_dir, cutoff)
    # Build ContactTable using existing KnowledgeBasedTrainer
    trainer = KnowledgeBasedTrainer(ntypes=256, distance_cutoff=cutoff)
    for cpx in complexes:
        coords, types, mask = _complex_to_arrays(cpx)
        trainer.add_structure(coords, types, mask)
    table = trainer.get_contact_table()
    table.save(str(cache_path))
    return table
```

---

## 4. Curriculum Training Strategy

### 4.1 Training Phases

The curriculum trains in order of decreasing data quality, warm-starting each phase from the previous result:

```
Phase 1: ITC-187 (Tier 1, weight 1.00)
    → Establishes thermodynamically-grounded baseline
    → Teaches correct ΔG scale and sign conventions
    → Output: matrix_phase1.bin

Phase 2: PDBbind Core + Refined (Tier 1-2, weight 0.95/0.80)
    → Warm-start from Phase 1
    → Broadens chemical diversity while maintaining quality
    → Output: matrix_phase2.bin

Phase 3: PDBbind General + Binding MOAD (Tier 3, weight 0.50/0.45)
    → Warm-start from Phase 2
    → Fills sparse cells in the 256×256 matrix
    → Output: matrix_phase3.bin

Phase 4: BindingDB + ChEMBL (Tier 4, weight 0.25/0.20)
    → Warm-start from Phase 3
    → Maximum coverage of chemical space
    → Heavy regularization to prevent noise from corrupting signal
    → Output: matrix_phase4.bin (final candidate)
```

### 4.2 Warm-Start Mechanism

Each phase inherits the previous matrix as a Bayesian prior:

```python
def warm_start_training(
    prior_matrix: np.ndarray,         # Previous best 256×256
    new_complexes: List[Complex],     # New data for this phase
    dataset_weight: float,            # Reliability weight (0-1)
    ridge_alpha: float = 1.0,
    lbfgs_maxiter: int = 200,
    temperature: float = 298.15,
) -> np.ndarray:
    """Train with warm-start from prior matrix.

    The prior is incorporated as:
        combined = (1 - λ) * prior + λ * new_estimate
    where λ = dataset_weight * (n_new_contacts / (n_new_contacts + n_prior_contacts))

    This ensures:
    - High-quality prior dominates when new data is sparse/noisy
    - New data has more influence when it's abundant and reliable
    - The mixing ratio adapts per-cell based on contact coverage
    """
```

### 4.3 Per-Cell Adaptive Weighting

Not all cells in the 256×256 matrix have equal support. Cells with many observed contacts get updated more aggressively than sparse cells:

```python
def compute_cell_confidence(
    contact_counts: np.ndarray,       # (256, 256) observed counts
    min_contacts: int = 10,           # Below this, keep prior
    saturation: int = 1000,           # Above this, trust new data fully
) -> np.ndarray:
    """Compute per-cell confidence for warm-start mixing.

    Returns (256, 256) array in [0, 1]:
        0.0 = no new contacts → keep prior entirely
        1.0 = saturated contacts → trust new estimate entirely
    """
    return np.clip(
        (contact_counts - min_contacts) / (saturation - min_contacts),
        0.0, 1.0
    )
```

### 4.4 Regularization Schedule

As training progresses through noisier datasets, regularization increases:

| Phase | Ridge α | L-BFGS max iter | Prior mixing floor |
|-------|---------|-----------------|-------------------|
| 1 (ITC-187) | 0.5 | 300 | 0.0 (no prior) |
| 2 (PDBbind refined) | 1.0 | 200 | 0.3 |
| 3 (PDBbind general + MOAD) | 2.0 | 150 | 0.5 |
| 4 (BindingDB + ChEMBL) | 5.0 | 100 | 0.7 |

The "prior mixing floor" ensures that noisy datasets can never completely overwrite the signal from high-quality early phases.

---

## 5. Training Pipeline Implementation

### 5.1 New Module: `continuous_training.py`

```python
# python/flexaidds/continuous_training.py

@dataclass
class ContinuousTrainingConfig:
    """Configuration for the full continuous training pipeline."""

    # Dataset directories (empty string = skip)
    itc_dir: str = ""
    pdbbind_core_dir: str = ""
    pdbbind_refined_dir: str = ""
    pdbbind_general_dir: str = ""
    moad_dir: str = ""
    bindingdb_path: str = ""
    chembl_path: str = ""

    # Validation datasets
    casf_dir: str = ""                    # CASF-2016 for gating
    dude_dir: str = ""                    # DUD-E for enrichment tracking
    dekois_dir: str = ""                  # DEKOIS for enrichment tracking

    # Warm-start
    prior_matrix_path: str = ""           # Path to previous best matrix
    contact_cache_dir: str = "data/contact_tables"

    # Training parameters
    contact_cutoff: float = 4.5
    temperature: float = 298.15
    seed: int = 42

    # Output
    output_dir: str = "data/training_runs"
    run_name: str = ""                    # Auto-generated if empty

    # Reference for projection validation
    reference_dat: str = ""

    # Quality gates
    casf_min_r: float = 0.75             # Minimum CASF-2016 Pearson r
    itc_min_r: float = 0.85              # Minimum ITC-187 ΔG correlation r


class ContinuousTrainer:
    """Orchestrates multi-dataset curriculum training with quality gates."""

    def __init__(self, config: ContinuousTrainingConfig):
        self.config = config
        self.run_id = config.run_name or self._generate_run_id()
        self.run_dir = Path(config.output_dir) / self.run_id
        self.metrics: Dict[str, Any] = {}
        self.adapters: List[DatasetAdapter] = []

    def _generate_run_id(self) -> str:
        """Generate unique run ID: run_YYYYMMDD_HHMMSS."""
        ...

    def register_adapters(self) -> None:
        """Register all configured dataset adapters."""
        ...

    def run(self) -> TrainingRunResult:
        """Execute the full curriculum training pipeline.

        Returns TrainingRunResult with:
        - Final matrix (256×256 binary)
        - All phase-by-phase metrics
        - Pass/fail on quality gates
        - Full provenance manifest
        """
        self._setup_run_directory()
        self._register_adapters()

        matrix = self._load_prior()

        # Curriculum phases
        for phase in self._build_curriculum():
            matrix = self._train_phase(phase, matrix)
            self._checkpoint(phase, matrix)

        # Quality gates
        gate_results = self._run_quality_gates(matrix)

        # Save final artifacts
        result = self._finalize(matrix, gate_results)
        return result

    def _build_curriculum(self) -> List[CurriculumPhase]:
        """Build the ordered curriculum from configured datasets."""
        ...

    def _train_phase(
        self,
        phase: CurriculumPhase,
        prior: Optional[np.ndarray],
    ) -> np.ndarray:
        """Execute one curriculum phase with warm-start."""
        ...

    def _run_quality_gates(self, matrix: np.ndarray) -> QualityGateResult:
        """Evaluate CASF-2016 and ITC-187 gates."""
        ...

    def _checkpoint(self, phase: CurriculumPhase, matrix: np.ndarray) -> None:
        """Save intermediate checkpoint after each phase."""
        ...

    def _finalize(
        self,
        matrix: np.ndarray,
        gates: QualityGateResult,
    ) -> TrainingRunResult:
        """Save final artifacts, manifest, and promotion status."""
        ...
```

### 5.2 Supporting Dataclasses

```python
@dataclass
class CurriculumPhase:
    """One phase of the curriculum training."""
    name: str                          # e.g. "phase_1_itc187"
    order: int                         # Execution order
    adapters: List[DatasetAdapter]     # Datasets in this phase
    ridge_alpha: float = 1.0
    lbfgs_maxiter: int = 200
    prior_mixing_floor: float = 0.0    # Min retention of prior
    dataset_weight: float = 1.0        # Reliability weight

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

@dataclass
class TrainingRunResult:
    """Complete result of a training run."""
    run_id: str
    matrix: EnergyMatrix
    matrix_path: str
    gate_results: QualityGateResult
    promoted: bool                     # True if all gates passed
    phase_metrics: Dict[str, Dict]     # Per-phase validation metrics
    manifest: Dict                     # Full provenance record
    elapsed_seconds: float = 0.0
```

### 5.3 Training Run Manifest

Each training run produces a `manifest.json` for full reproducibility:

```json
{
    "run_id": "run_20260316_143022",
    "timestamp": "2026-03-16T14:30:22Z",
    "flexaidds_version": "0.5.0",
    "git_commit": "abc1234",

    "config": {
        "contact_cutoff": 4.5,
        "temperature": 298.15,
        "seed": 42
    },

    "datasets": [
        {
            "name": "itc_187",
            "version": "v1.0",
            "n_complexes": 187,
            "tier": 1,
            "weight": 1.0,
            "checksum": "sha256:abcdef..."
        },
        {
            "name": "pdbbind_refined",
            "version": "v2020",
            "n_complexes": 5316,
            "tier": 2,
            "weight": 0.80,
            "checksum": "sha256:123456..."
        }
    ],

    "curriculum": [
        {
            "phase": 1,
            "name": "phase_1_itc187",
            "datasets": ["itc_187"],
            "ridge_alpha": 0.5,
            "lbfgs_maxiter": 300,
            "prior_mixing_floor": 0.0,
            "metrics": {
                "pearson_r": 0.92,
                "rmse": 1.3,
                "nonzero_cells": 8421
            }
        }
    ],

    "prior_matrix": {
        "path": "data/training_runs/run_001/matrix_256x256.bin",
        "run_id": "run_001",
        "checksum": "sha256:fedcba..."
    },

    "quality_gates": {
        "casf_2016": {
            "pearson_r": 0.79,
            "rmse": 1.82,
            "n_complexes": 285,
            "threshold": 0.75,
            "passed": true
        },
        "itc_187": {
            "pearson_r": 0.91,
            "rmse": 1.25,
            "n_complexes": 187,
            "threshold": 0.85,
            "passed": true
        }
    },

    "supplementary_metrics": {
        "dude_mean_auc": 0.72,
        "dekois_mean_auc": 0.68,
        "projection_40_r": 0.95
    },

    "promoted": true,
    "elapsed_seconds": 3847.2,

    "artifacts": {
        "matrix_256x256": "matrix_256x256.bin",
        "matrix_40x40": "matrix_40x40.dat",
        "contact_tables": ["itc_187.json", "pdbbind_refined.json"],
        "training_log": "training.log"
    }
}
```

---

## 6. Quality Gate Framework

### 6.1 Gate Definitions

**Gate 1: CASF-2016 Scoring Power** (mandatory)
- Metric: Pearson *r* between predicted score and experimental ΔG
- Threshold: *r* ≥ 0.75
- Dataset: CASF-2016 core set (~285 complexes, held out from training)
- Methodology: Score each complex as `Σ matrix[type_a][type_b]` over all contacts; correlate with experimental ΔG

**Gate 2: ITC-187 ΔG Correlation** (mandatory)
- Metric: Pearson *r* between predicted score and calorimetric ΔG
- Threshold: *r* ≥ 0.85
- Dataset: ITC-187 (5-fold cross-validation; train on 4 folds, validate on held-out fold, report mean)
- Gold standard: ITC provides direct thermodynamic measurements, not derived from IC50/Ki

### 6.2 Supplementary Metrics (tracked, not gating)

- **DUD-E mean AUC**: Virtual screening enrichment across 102 targets
- **DEKOIS 2.0 mean AUC**: VS enrichment across 81 targets
- **256→40 projection correlation**: Agreement with legacy matrix
- **Per-phase Pearson *r***: Training set performance at each curriculum phase
- **Sparse cell count**: Number of 256×256 cells with < 10 contacts (coverage metric)
- **FastOPTICS super-cluster count**: Redundancy in the type system

### 6.3 Gate Enforcement

```python
def evaluate_quality_gates(
    matrix: np.ndarray,
    config: ContinuousTrainingConfig,
) -> QualityGateResult:
    """Run all quality gates and return pass/fail verdicts."""

    result = QualityGateResult()

    # Gate 1: CASF-2016
    if config.casf_dir:
        casf_complexes = load_pdbbind_complexes(config.casf_dir, config.contact_cutoff)
        casf_metrics = validate_casf(matrix, casf_complexes)
        result.casf_pearson_r = casf_metrics["pearson_r"]
        result.casf_rmse = casf_metrics["rmse"]
        result.casf_n = casf_metrics["n_complexes"]
        result.casf_passed = result.casf_pearson_r >= config.casf_min_r

    # Gate 2: ITC-187 (5-fold cross-validation)
    if config.itc_dir:
        itc_metrics = validate_itc_crossval(matrix, config.itc_dir,
                                             config.contact_cutoff, n_folds=5)
        result.itc_pearson_r = itc_metrics["mean_pearson_r"]
        result.itc_rmse = itc_metrics["mean_rmse"]
        result.itc_n = itc_metrics["n_complexes"]
        result.itc_passed = result.itc_pearson_r >= config.itc_min_r

    return result
```

### 6.4 Promotion Policy

A matrix is **promoted** (copied to `data/production/current_matrix.bin`) only if:

1. Both mandatory gates pass (CASF *r* ≥ 0.75, ITC *r* ≥ 0.85)
2. No regression > 0.02 on either metric compared to the currently promoted matrix
3. The run completes without errors

If gates fail, the matrix is retained in the run directory for diagnosis but not promoted. The training log records which gate(s) failed and by how much.

---

## 7. CLI Interface

### 7.1 New Subcommands

Add to `energy_matrix_cli.py`:

```bash
# Full curriculum training pipeline
python -m flexaidds.energy_matrix_cli continuous-train \
    --itc-dir data/datasets/itc_187 \
    --pdbbind-refined data/datasets/pdbbind/refined-set \
    --pdbbind-general data/datasets/pdbbind/general-set \
    --pdbbind-core data/datasets/pdbbind/core-set \
    --moad-dir data/datasets/binding_moad \
    --bindingdb data/datasets/bindingdb/BindingDB_All.tsv \
    --chembl data/datasets/chembl/chembl_activities.csv \
    --casf-dir data/datasets/pdbbind/core-set \
    --prior-matrix data/training_runs/run_001/matrix_256x256.bin \
    --output-dir data/training_runs \
    --seed 42 \
    -v

# Build/update contact table cache for a single dataset
python -m flexaidds.energy_matrix_cli build-contacts \
    --dataset pdbbind_refined \
    --data-dir data/datasets/pdbbind/refined-set \
    --output data/contact_tables/pdbbind_refined.json \
    --cutoff 4.5

# Run quality gates on an existing matrix
python -m flexaidds.energy_matrix_cli validate-gates \
    --matrix data/training_runs/run_003/matrix_256x256.bin \
    --casf-dir data/datasets/pdbbind/core-set \
    --itc-dir data/datasets/itc_187 \
    --casf-threshold 0.75 \
    --itc-threshold 0.85

# Compare two training runs
python -m flexaidds.energy_matrix_cli compare-runs \
    --run-a data/training_runs/run_001 \
    --run-b data/training_runs/run_002

# List all training runs with metrics
python -m flexaidds.energy_matrix_cli list-runs \
    --runs-dir data/training_runs \
    --sort-by casf_r
```

### 7.2 Single-Phase Training (for debugging/experimentation)

```bash
# Train a single phase (e.g., just PDBbind refined, warm-starting from a prior)
python -m flexaidds.energy_matrix_cli train-phase \
    --dataset pdbbind_refined \
    --data-dir data/datasets/pdbbind/refined-set \
    --prior-matrix data/training_runs/run_001/matrix_256x256.bin \
    --ridge-alpha 1.0 \
    --lbfgs-maxiter 200 \
    --prior-mixing-floor 0.3 \
    --dataset-weight 0.80 \
    --output phase2_test.bin
```

---

## 8. Versioning and Provenance

### 8.1 Run Identification

- Format: `run_YYYYMMDD_HHMMSS` (auto-generated) or user-specified name
- Each run gets its own directory under `data/training_runs/`
- The `manifest.json` in each run directory provides complete reproducibility

### 8.2 Dataset Checksums

Contact tables are checksummed (SHA-256) to detect if source data changed between runs:

```python
def checksum_contact_table(table: ContactTable) -> str:
    """SHA-256 of the serialized contact table."""
    import hashlib
    data = json.dumps(table.to_dict(), sort_keys=True).encode()
    return f"sha256:{hashlib.sha256(data).hexdigest()[:16]}"
```

### 8.3 Matrix Lineage

Each manifest records which prior matrix was used:

```
run_001 (no prior)          → matrix_001.bin
    ↓
run_002 (prior: run_001)    → matrix_002.bin  [PROMOTED]
    ↓
run_003 (prior: run_002)    → matrix_003.bin  [GATE FAIL: ITC r=0.83]
    ↓
run_004 (prior: run_002)    → matrix_004.bin  [PROMOTED]  ← retrained with adjusted α
```

### 8.4 Production Matrix Pointer

The current "production" matrix is tracked via a symlink or pointer file:

```
data/production/
├── current_matrix.bin → ../training_runs/run_004/matrix_256x256.bin
├── current_40x40.dat → ../training_runs/run_004/matrix_40x40.dat
└── promotion_log.jsonl   # Append-only log of promotions
```

---

## 9. Implementation Roadmap

### Phase A: Core Infrastructure (Week 1-2)

1. **`dataset_adapters.py`** — Base class + PDBbind adapter (refactored from existing code)
2. **`continuous_training.py`** — `ContinuousTrainer`, `CurriculumPhase`, warm-start logic
3. **Quality gate framework** — `validate_itc_crossval()`, gate enforcement
4. **Manifest generation** — JSON provenance with checksums

### Phase B: Dataset Adapters (Week 2-3)

5. **`ITC187Adapter`** — Calorimetric data loader with ΔH/TΔS/ΔG parsing
6. **`BindingMOADAdapter`** — MOAD CSV parser + PDB structure loader
7. **`BindingDBAdapter`** — TSV parser with PDB cross-referencing + affinity normalization
8. **`ChEMBLAdapter`** — CSV parser with PDB mapping + IC50→Ki→ΔG conversion
9. **`DEKOIS2Adapter`** + **`DUDEAdapter`** — Validation-only adapters for enrichment

### Phase C: CLI + Testing (Week 3-4)

10. **CLI commands** — `continuous-train`, `build-contacts`, `validate-gates`, `compare-runs`, `list-runs`
11. **Unit tests** — Synthetic adapters, warm-start correctness, gate logic, manifest schema
12. **Integration test** — End-to-end curriculum with tiny synthetic datasets

### Phase D: Calibration + Validation (Week 4+)

13. **Run with real ITC-187** — Establish Phase 1 baseline
14. **Run with PDBbind refined** — Phase 2 warm-start
15. **Gate calibration** — Tune thresholds based on real performance
16. **Full curriculum** — All datasets, end-to-end

---

## 10. Dataset-Specific Ingestion Details

### 10.1 ITC-187

**Source**: Collected from literature by Freire and others; curated set of 187 protein-ligand complexes with direct isothermal titration calorimetry measurements.

**Index format** (CSV):
```
pdb_code,deltaH,TdeltaS,deltaG,Kd,temperature,reference
1a0q,-12.3,-4.1,-8.2,9.5e-7,298.15,Freire2009
```

**Parsing**: Direct ΔG values — no conversion needed. The ITC-187 adapter produces complexes with `deltaG` populated from the index, and `pKd = -log10(Kd)`.

**Special handling**: 5-fold cross-validation for gate assessment (train on 80%, validate on 20%).

### 10.2 Binding MOAD

**Source**: Binding MOAD (Mother of All Databases) — curated collection of well-resolved protein-ligand complexes from the PDB with binding data.

**Index format** (`every.csv`):
```
PDB_ID,Ligand_ID,Binding_Data,Kd_Ki,pKd_pKi,Year,...
1a0q,ATP,Kd=9.5e-7M,9.5e-7,6.02,2003,...
```

**Parsing**:
- Parse binding data string to extract value and unit
- Normalize to ΔG via `normalize_affinity()`
- Structure files: download from PDB or use local mirror
- Ligand extraction: use HETATM records matching Ligand_ID

### 10.3 BindingDB

**Source**: BindingDB — public database of measured binding affinities for drug-like molecules.

**Format**: Tab-separated with columns including:
- `Ligand SMILES`, `Target Name`, `Ki (nM)`, `Kd (nM)`, `IC50 (nM)`, `PDB ID(s)`

**Filtering**:
1. Require non-null PDB ID
2. Prefer Kd > Ki > IC50 (in order of thermodynamic rigor)
3. Filter by resolution (< 3.0 Å recommended)
4. Deduplicate by PDB code + ligand

**Parsing**:
- Convert nM to M (×10⁻⁹)
- Apply `normalize_affinity()` for ΔG
- Cross-reference PDB for 3D structures
- Use existing MOL2 ligand parsing or generate from SMILES via RDKit (optional dependency)

### 10.4 ChEMBL

**Source**: ChEMBL — manually curated chemical biology database of bioactive molecules.

**Filtering** (critical — ChEMBL is very large):
1. Require `standard_type` in {Ki, Kd, IC50}
2. Require `standard_units` = 'nM'
3. Require `pchembl_value` is not null (quality filter)
4. Require associated PDB structure via UniProt→PDB mapping
5. Require `assay_type` = 'B' (binding assay, not functional)

**Parsing**:
- `pchembl_value` = -log10(molar value) → direct pKd equivalent
- ΔG = -RT × ln(10) × pchembl_value
- Structure acquisition: PDB cross-reference via target UniProt ID

### 10.5 DUD-E and DEKOIS 2.0 (Validation Only)

These datasets provide active/decoy pairs for virtual screening enrichment testing. They are not used in training but tracked as supplementary metrics.

**DUD-E**: 102 targets, each with ~230 actives and ~10,000 property-matched decoys.
**DEKOIS 2.0**: 81 targets, each with 40 actives and ~1,260 decoys with demanding physicochemical matching.

**Evaluation**: For each target, score all actives and decoys using the trained matrix, compute AUC (Wilcoxon-Mann-Whitney statistic), and report mean AUC across all targets.

---

## 11. Testing Strategy

### 11.1 Unit Tests (no real datasets needed)

```python
# python/tests/test_continuous_training.py

class TestAffinityNormalization:
    """Test Kd/Ki/IC50/pKd → ΔG conversion."""
    def test_kd_to_dg(self): ...
    def test_pki_to_dg(self): ...
    def test_ic50_cheng_prusoff(self): ...

class TestWarmStart:
    """Test warm-start mixing logic."""
    def test_no_prior_returns_new_estimate(self): ...
    def test_high_weight_dataset_dominates(self): ...
    def test_low_weight_preserves_prior(self): ...
    def test_per_cell_confidence_scaling(self): ...
    def test_mixing_floor_respected(self): ...

class TestCurriculumBuilder:
    """Test curriculum phase construction."""
    def test_phases_ordered_by_tier(self): ...
    def test_empty_datasets_skipped(self): ...
    def test_regularization_increases_per_phase(self): ...

class TestQualityGates:
    """Test gate evaluation and promotion logic."""
    def test_both_gates_pass_promotes(self): ...
    def test_casf_fail_blocks_promotion(self): ...
    def test_itc_fail_blocks_promotion(self): ...
    def test_regression_detection(self): ...

class TestManifest:
    """Test manifest generation and schema."""
    def test_manifest_includes_all_datasets(self): ...
    def test_manifest_records_checksums(self): ...
    def test_manifest_roundtrip_json(self): ...

class TestDatasetAdapters:
    """Test individual adapters with synthetic data."""
    def test_pdbbind_adapter_synthetic(self): ...
    def test_itc187_adapter_synthetic(self): ...
    def test_moad_adapter_synthetic(self): ...
    def test_bindingdb_adapter_synthetic(self): ...
    def test_chembl_adapter_synthetic(self): ...
```

### 11.2 Integration Test

```python
class TestContinuousTrainingEndToEnd:
    """End-to-end with tiny synthetic datasets."""
    def test_full_curriculum_synthetic(self):
        """Create 5 synthetic complexes per dataset, run full pipeline."""
        ...

    def test_warm_start_improves_over_cold(self):
        """Warm-started run should not regress vs cold start."""
        ...

    def test_gate_failure_prevents_promotion(self):
        """With deliberately bad matrix, gates should fail."""
        ...
```

---

## 12. Performance Considerations

### 12.1 Contact Table Caching

The most expensive operation is contact enumeration (KD-tree queries over millions of atom pairs). Contact tables are cached per dataset version:

- **PDBbind refined** (~5,300 complexes): ~15 minutes to build, 50 MB JSON
- **PDBbind general** (~23,000 complexes): ~1 hour, 200 MB JSON
- **Binding MOAD** (~38,000 entries): ~2 hours, 300 MB JSON
- **Subsequent runs**: Seconds (load cached JSON)

### 12.2 Matrix Training

- **Inverse Boltzmann**: O(N²) where N=256 → instant
- **Ridge regression**: O(K × N²) where K=number of complexes → seconds
- **L-BFGS**: O(iter × K × N²) → minutes
- **Full 4-phase curriculum**: ~10-30 minutes (with cached contact tables)

### 12.3 Memory

- Contact tables: ~50-300 MB per dataset
- 256×256 matrix: 256 KB
- Complex objects: ~1 KB each → ~40 MB for 38K MOAD complexes
- Peak: ~1 GB (loading all datasets simultaneously; can be reduced by streaming)

---

## 13. Future Extensions

These are **out of scope** for the initial implementation but documented for future consideration:

1. **Automated dataset downloads** — Scripts to fetch PDBbind, MOAD, BindingDB releases
2. **CI-triggered retraining** — GitHub Actions workflow on new dataset releases
3. **Ensemble matrices** — Average multiple training runs for robustness
4. **Active learning** — Identify which new structures would most reduce matrix uncertainty
5. **Per-target fine-tuning** — Specialized matrices for CNS receptors, kinases, etc.
6. **Distance-dependent training** — Extend 256×256 to 256×256×D distance shells
7. **Transfer learning from ML potentials** — Initialize from GNN-predicted contacts
8. **Bayesian matrix estimation** — Full posterior over matrix cells, not just point estimates
