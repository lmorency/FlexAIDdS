"""metrics.py — Virtual screening metrics for LIT-PCBA evaluation.

Computes enrichment factors (EF1%, EF5%), BEDROC, and AUROC for
unbiased virtual screening validation.

Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
SPDX-License-Identifier: Apache-2.0
"""

from typing import List, Tuple

import numpy as np


def enrichment_factor(
    scores: np.ndarray,
    labels: np.ndarray,
    fraction: float = 0.01,
) -> float:
    """Compute enrichment factor at a given fraction.

    Parameters
    ----------
    scores : np.ndarray
        Predicted scores (lower = better binding).
    labels : np.ndarray
        Binary activity labels (1 = active, 0 = inactive).
    fraction : float
        Top fraction to evaluate (e.g., 0.01 for EF1%).

    Returns
    -------
    float
        Enrichment factor.
    """
    n_total = len(scores)
    n_actives = np.sum(labels)
    if n_total == 0 or n_actives == 0:
        return 0.0

    # Sort by score (ascending = better)
    order = np.argsort(scores)
    n_top = max(1, int(n_total * fraction))

    top_labels = labels[order[:n_top]]
    hits_in_top = np.sum(top_labels)

    expected = n_actives / n_total
    observed = hits_in_top / n_top

    return float(observed / expected) if expected > 0 else 0.0


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute Area Under the ROC Curve.

    Parameters
    ----------
    scores : np.ndarray
        Predicted scores (lower = better binding).
    labels : np.ndarray
        Binary activity labels.

    Returns
    -------
    float
        AUROC value.
    """
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, -scores))  # negate: higher = better for sklearn
    except ImportError:
        # Manual AUROC via Wilcoxon-Mann-Whitney
        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)
        if n_pos == 0 or n_neg == 0:
            return 0.5

        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(scores) + 1, dtype=float)

        U = np.sum(ranks[labels == 1]) - n_pos * (n_pos + 1) / 2
        return float(U / (n_pos * n_neg))


def bedroc(
    scores: np.ndarray,
    labels: np.ndarray,
    alpha: float = 20.0,
) -> float:
    """Compute Boltzmann-Enhanced Discrimination of ROC (BEDROC).

    Parameters
    ----------
    scores : np.ndarray
        Predicted scores (lower = better).
    labels : np.ndarray
        Binary activity labels.
    alpha : float
        Exponential weight parameter (default 20.0).

    Returns
    -------
    float
        BEDROC value in [0, 1].
    """
    n = len(scores)
    n_actives = int(np.sum(labels))
    if n == 0 or n_actives == 0:
        return 0.0

    order = np.argsort(scores)
    ranked_labels = labels[order]

    # Positions of actives (1-indexed, fractional)
    active_positions = np.where(ranked_labels == 1)[0] + 1
    frac_positions = active_positions / n

    # BEDROC formula
    sum_exp = np.sum(np.exp(-alpha * frac_positions))
    ra = n_actives / n

    rand = (ra * (1 - np.exp(-alpha))) / (np.exp(alpha / n) - 1)
    fac = (ra * np.sinh(alpha / 2)) / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * ra))
    roc_max = (1 - np.exp(-alpha * ra)) / (1 - np.exp(-alpha))

    if roc_max - rand == 0:
        return 0.0

    return float((sum_exp / n_actives - rand) / (roc_max - rand))


def evaluate_target(
    scores: np.ndarray,
    labels: np.ndarray,
    target_name: str = "unknown",
) -> dict:
    """Evaluate a single LIT-PCBA target.

    Returns
    -------
    dict
        Dictionary with EF1%, EF5%, AUROC, BEDROC, and counts.
    """
    return {
        "target": target_name,
        "n_compounds": len(scores),
        "n_actives": int(np.sum(labels)),
        "EF_1pct": enrichment_factor(scores, labels, 0.01),
        "EF_5pct": enrichment_factor(scores, labels, 0.05),
        "AUROC": auroc(scores, labels),
        "BEDROC_20": bedroc(scores, labels, 20.0),
    }
