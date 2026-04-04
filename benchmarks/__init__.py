"""
FlexAIDdS benchmark package.

Sub-modules
-----------
DatasetRunner   Distributed orchestrator (MPI + serial)
metrics         Evaluation metrics (entropy_rescue_rate, EF, LogAUC, ...)
ligand_prep     SMILES → Mol2 → .inp preparation pipeline
run             CLI entry point
"""

from benchmarks.metrics import (
    bootstrap_ci,
    docking_power,
    enrichment_factor,
    entropy_rescue_rate,
    hit_rate_top_n,
    log_auc,
    scoring_power,
    target_specificity_zscore,
)

__all__ = [
    "bootstrap_ci",
    "docking_power",
    "enrichment_factor",
    "entropy_rescue_rate",
    "hit_rate_top_n",
    "log_auc",
    "scoring_power",
    "target_specificity_zscore",
]
