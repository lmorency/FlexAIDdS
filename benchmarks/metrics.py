"""
benchmarks/metrics.py
=====================
Evaluation metrics for FlexAIDdS benchmarks.

Primary metric: entropy_rescue_rate — fraction of true binders that ΔS
scoring recovers among targets where pure-enthalpy scoring fails.

All metric functions return plain Python floats or namedtuples so they
can be serialised to JSON without post-processing.
"""

from __future__ import annotations

import math
import random
from typing import Callable, NamedTuple, Sequence


# ---------------------------------------------------------------------------
# Named-tuple result containers
# ---------------------------------------------------------------------------

class CI(NamedTuple):
    """Bootstrap confidence interval."""
    lower: float
    point: float
    upper: float
    alpha: float


class EnrichmentResult(NamedTuple):
    ef: float          # enrichment factor value
    fraction: float    # fraction of database screened
    n_actives_found: int
    n_actives_total: int
    n_screened: int
    n_total: int


class DockingPowerResult(NamedTuple):
    success_rate: float  # fraction of top-ranked poses within RMSD threshold
    n_success: int
    n_total: int
    rmsd_threshold: float


class ScoringPowerResult(NamedTuple):
    pearson_r: float
    rmse: float          # kcal/mol
    n_samples: int


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------

def _mean(values: Sequence[float]) -> float:
    """Population mean."""
    if not values:
        raise ValueError("Empty sequence")
    return sum(values) / len(values)


def _variance(values: Sequence[float]) -> float:
    """Population variance."""
    mu = _mean(values)
    return sum((x - mu) ** 2 for x in values) / len(values)


def _std(values: Sequence[float]) -> float:
    """Population standard deviation."""
    return math.sqrt(_variance(values))


def _rank_order(scores: Sequence[float], descending: bool = True) -> list[int]:
    """Return 1-based ranks; higher score → lower rank when descending=True."""
    indexed = sorted(enumerate(scores), key=lambda t: t[1], reverse=descending)
    ranks = [0] * len(scores)
    for rank, (idx, _) in enumerate(indexed, start=1):
        ranks[idx] = rank
    return ranks


# ---------------------------------------------------------------------------
# Primary metric
# ---------------------------------------------------------------------------

def entropy_rescue_rate(
    enthalpy_ranks: Sequence[int],
    entropy_ranks: Sequence[int],
    active_mask: Sequence[bool],
    threshold: int = 10,
) -> float:
    """Compute the ΔS rescue rate.

    The rescue rate is the fraction of true binders (active_mask == True)
    that entropy-augmented scoring (entropy_ranks) places within the top
    *threshold* positions, among those that enthalpy-only scoring
    (enthalpy_ranks) *failed* to place within the threshold.

    Parameters
    ----------
    enthalpy_ranks:
        1-based ranks assigned by enthalpy-only scoring (rank 1 = best).
    entropy_ranks:
        1-based ranks assigned by ΔS-augmented scoring.
    active_mask:
        Boolean mask; True indicates a true binder / active compound.
    threshold:
        Top-N cutoff; a pose is considered "found" if its rank ≤ threshold.

    Returns
    -------
    float in [0, 1].  Returns NaN if there are no rescuable actives.
    """
    if len(enthalpy_ranks) != len(entropy_ranks) != len(active_mask):
        raise ValueError("All input sequences must have the same length")
    if not any(active_mask):
        return float("nan")

    missed_by_enthalpy = [
        i for i, (er, am) in enumerate(zip(enthalpy_ranks, active_mask))
        if am and er > threshold
    ]
    if not missed_by_enthalpy:
        return float("nan")  # nothing to rescue

    rescued = sum(
        1 for i in missed_by_enthalpy if entropy_ranks[i] <= threshold
    )
    return rescued / len(missed_by_enthalpy)


# ---------------------------------------------------------------------------
# Virtual screening metrics
# ---------------------------------------------------------------------------

def enrichment_factor(
    scores: Sequence[float],
    labels: Sequence[int],
    fraction: float = 0.01,
) -> EnrichmentResult:
    """Compute enrichment factor at a given fraction of the database.

    Parameters
    ----------
    scores:
        Docking scores (more negative = better binding).
    labels:
        Binary labels: 1 = active, 0 = decoy/inactive.
    fraction:
        Fraction of the ranked database to examine (e.g. 0.01 for EF1%).

    Returns
    -------
    EnrichmentResult namedtuple.
    """
    n_total = len(scores)
    n_screened = max(1, round(fraction * n_total))
    n_actives_total = sum(labels)

    if n_actives_total == 0:
        raise ValueError("No actives in label vector")

    ranked = sorted(zip(scores, labels), key=lambda t: t[0])  # ascending = better
    top = ranked[:n_screened]
    n_actives_found = sum(lbl for _, lbl in top)

    expected = n_actives_total / n_total
    ef = (n_actives_found / n_screened) / expected if expected > 0 else 0.0

    return EnrichmentResult(
        ef=ef,
        fraction=fraction,
        n_actives_found=n_actives_found,
        n_actives_total=n_actives_total,
        n_screened=n_screened,
        n_total=n_total,
    )


def log_auc(
    scores: Sequence[float],
    labels: Sequence[int],
    min_fraction: float = 0.001,
) -> float:
    """Compute the semi-log ROC AUC (LogAUC) metric.

    Integrates the ROC curve over a logarithmic x-axis (false-positive rate)
    in [min_fraction, 1], normalised so a random classifier scores 0.
    A perfect classifier scores ~1.

    Reference: Mysinger & Shoichet (2010) J. Chem. Inf. Model. 50, 1561.

    Parameters
    ----------
    scores:
        Docking scores (ascending = better, i.e. more negative = better).
    labels:
        Binary labels: 1 = active, 0 = decoy.
    min_fraction:
        Lower FPR cutoff on the log axis (default 0.1%).

    Returns
    -------
    float: LogAUC value.  Returns NaN for degenerate inputs.
    """
    n_total = len(scores)
    n_actives = sum(labels)
    n_decoys = n_total - n_actives

    if n_actives == 0 or n_decoys == 0:
        return float("nan")

    ranked = sorted(zip(scores, labels), key=lambda t: t[0])

    tpr_points: list[tuple[float, float]] = [(0.0, 0.0)]
    n_tp = 0
    n_fp = 0
    for _, lbl in ranked:
        if lbl == 1:
            n_tp += 1
        else:
            n_fp += 1
        fpr = n_fp / n_decoys
        tpr = n_tp / n_actives
        if fpr >= min_fraction:
            tpr_points.append((fpr, tpr))

    if len(tpr_points) < 2:
        return float("nan")

    # Trapezoidal rule on log(fpr) axis
    log_auc_val = 0.0
    for i in range(1, len(tpr_points)):
        fpr0, tpr0 = tpr_points[i - 1]
        fpr1, tpr1 = tpr_points[i]
        fpr0 = max(fpr0, min_fraction)
        fpr1 = max(fpr1, min_fraction)
        if fpr1 <= fpr0:
            continue
        log_auc_val += (tpr0 + tpr1) / 2.0 * (math.log10(fpr1) - math.log10(fpr0))

    # Normalise by log10(1/min_fraction)
    normaliser = math.log10(1.0 / min_fraction)
    log_auc_val /= normaliser

    # Subtract random baseline (log10(1/min_fraction) / 2 / normaliser)
    random_baseline = 0.5  # under log-uniform sampling
    return log_auc_val - random_baseline


# ---------------------------------------------------------------------------
# Docking / scoring power
# ---------------------------------------------------------------------------

def scoring_power(
    predicted_scores: Sequence[float],
    experimental_values: Sequence[float],
) -> ScoringPowerResult:
    """Pearson r and RMSE between predicted scores and experimental ΔG/pKi.

    Parameters
    ----------
    predicted_scores:
        Model-predicted binding scores (ΔG estimate, kcal/mol).
    experimental_values:
        Experimental binding free energies or pKi/pIC50 values.

    Returns
    -------
    ScoringPowerResult namedtuple.
    """
    n = len(predicted_scores)
    if n != len(experimental_values):
        raise ValueError("predicted_scores and experimental_values must have the same length")
    if n < 2:
        raise ValueError("Need at least 2 samples for scoring power")

    pred = list(predicted_scores)
    expt = list(experimental_values)

    mean_p = _mean(pred)
    mean_e = _mean(expt)
    std_p = _std(pred)
    std_e = _std(expt)

    if std_p == 0 or std_e == 0:
        pearson_r = float("nan")
    else:
        cov = sum((pred[i] - mean_p) * (expt[i] - mean_e) for i in range(n)) / n
        pearson_r = cov / (std_p * std_e)
        pearson_r = max(-1.0, min(1.0, pearson_r))

    rmse = math.sqrt(sum((pred[i] - expt[i]) ** 2 for i in range(n)) / n)

    return ScoringPowerResult(pearson_r=pearson_r, rmse=rmse, n_samples=n)


def docking_power(
    rmsd_values: Sequence[float],
    threshold: float = 2.0,
) -> DockingPowerResult:
    """Success rate: fraction of top-ranked poses within RMSD threshold.

    Parameters
    ----------
    rmsd_values:
        RMSD of top-ranked pose vs crystal structure for each target (Å).
    threshold:
        Success threshold in Å (default 2.0 Å, CASF convention).

    Returns
    -------
    DockingPowerResult namedtuple.
    """
    n_total = len(rmsd_values)
    if n_total == 0:
        raise ValueError("Empty RMSD list")

    n_success = sum(1 for r in rmsd_values if r <= threshold)
    return DockingPowerResult(
        success_rate=n_success / n_total,
        n_success=n_success,
        n_total=n_total,
        rmsd_threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Specificity / selectivity
# ---------------------------------------------------------------------------

def target_specificity_zscore(
    scores_target: Sequence[float],
    scores_background: Sequence[float],
) -> float:
    """Z-score of a compound's score against a target vs a background panel.

    Used for selectivity / specificity analysis across a target family.

    Parameters
    ----------
    scores_target:
        Docking scores of the compound against the target of interest.
    scores_background:
        Docking scores of the same compound across background targets.

    Returns
    -------
    Z-score: (mean_target - mean_background) / std_background.
    A high positive Z-score indicates selectivity for the target.
    """
    if not scores_background:
        raise ValueError("Background scores must be non-empty")

    mean_t = _mean(scores_target)
    mean_bg = _mean(scores_background)
    std_bg = _std(scores_background)

    if std_bg == 0:
        return float("nan")
    return (mean_t - mean_bg) / std_bg


def hit_rate_top_n(
    scores: Sequence[float],
    labels: Sequence[int],
    n: int = 10,
) -> float:
    """Fraction of true actives in the top-N scored compounds.

    Parameters
    ----------
    scores:
        Docking scores (ascending = better).
    labels:
        Binary labels: 1 = active, 0 = inactive.
    n:
        Number of top compounds to consider.

    Returns
    -------
    float in [0, 1].
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    n = min(n, len(scores))
    ranked = sorted(zip(scores, labels), key=lambda t: t[0])
    top_n_labels = [lbl for _, lbl in ranked[:n]]
    return sum(top_n_labels) / n


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    metric_fn: Callable[..., float],
    data: list,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> CI:
    """Non-parametric bootstrap confidence interval for a scalar metric.

    Parameters
    ----------
    metric_fn:
        A callable that accepts a list of the same structure as *data* and
        returns a single float metric value.
    data:
        The dataset as a list of items (each item will be resampled with
        replacement).
    n_boot:
        Number of bootstrap replicates (default 1000).
    alpha:
        Significance level; CI covers (1-alpha)*100 % (default 0.05 → 95%).
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    CI namedtuple with lower, point, upper, alpha fields.

    Notes
    -----
    Uses the percentile method.  For small samples or highly skewed
    distributions, consider BCa (bias-corrected accelerated) instead.
    """
    rng = random.Random(seed)
    point = metric_fn(data)

    replicates: list[float] = []
    n = len(data)
    for _ in range(n_boot):
        sample = [data[rng.randint(0, n - 1)] for _ in range(n)]
        try:
            val = metric_fn(sample)
            if not math.isnan(val) and not math.isinf(val):
                replicates.append(val)
        except Exception:
            pass  # skip degenerate resamples

    if not replicates:
        return CI(lower=float("nan"), point=point, upper=float("nan"), alpha=alpha)

    replicates.sort()
    lo_idx = int(math.floor((alpha / 2) * len(replicates)))
    hi_idx = int(math.ceil((1 - alpha / 2) * len(replicates))) - 1
    hi_idx = min(hi_idx, len(replicates) - 1)

    return CI(
        lower=replicates[lo_idx],
        point=point,
        upper=replicates[hi_idx],
        alpha=alpha,
    )
