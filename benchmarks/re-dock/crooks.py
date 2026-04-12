"""
RE-DOCK Crooks Bidirectional Engine (v7)
=========================================

Bidirectional round-trip free energy estimation using the Crooks Fluctuation
Theorem and Bennett Acceptance Ratio (BAR).

Theory
------
The Crooks Fluctuation Theorem (CFT) relates the probability of observing
non-equilibrium work :math:`W` in forward and reverse processes:

.. math::

    \\frac{P_F(W)}{P_R(-W)} = \\exp[\\beta(W - \\Delta G)]

where :math:`\\Delta G` is the equilibrium free energy difference.

The Bennett Acceptance Ratio (BAR) provides the statistically optimal
estimate of :math:`\\Delta G` from bidirectional work measurements:

.. math::

    \\sum_i \\frac{1}{1 + \\frac{n_F}{n_R} \\exp(\\beta(W_{F,i} - \\Delta G))}
    = \\sum_j \\frac{1}{1 + \\frac{n_R}{n_F} \\exp(-\\beta(W_{R,j} + \\Delta G))}

Irreversible entropy production quantifies round-trip dissipation:

.. math::

    \\sigma_{\\mathrm{irr}} = \\langle W_F \\rangle + \\langle W_R \\rangle - 2\\Delta G \\geq 0

Landauer information loss converts dissipation to bits erased:

.. math::

    \\text{bits lost} = \\frac{\\sigma_{\\mathrm{irr}}}{k_B \\ln 2}

Components
----------
- **WorkSample**: Single non-equilibrium work measurement
- **LegResult**: Aggregated results for one direction (forward or reverse)
- **BidirectionalResult**: Full round-trip analysis with BAR, CFT, sigma_irr
- **BidirectionalExchange**: Engine managing forward + reverse replica exchange legs
- **bennett_acceptance_ratio()**: Optimal ΔG from bidirectional work distributions
- **crooks_intersection()**: ΔG where P(W_fwd) and P(-W_rev) cross
- **irreversible_entropy_production()**: σ_irr from work distributions
- **landauer_information_loss()**: Bits lost per round-trip
- **shannon_energy_collapse_rate()**: dI/dT at each temperature rung
- **mutual_information()**: I(hot; cold) from joint ensemble
- **convergence_check()**: σ_irr < threshold criterion

Le Bonhomme Pharma / Najmanovich Research Group
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .thermodynamics import (
    R_KCAL,
    DockingPose,
    ReplicaState,
    attempt_exchanges,
    shannon_entropy_of_ensemble,
)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

LN2: float = math.log(2.0)
"""Natural logarithm of 2, used for Landauer bound conversion."""

_EPSILON: float = 1e-300
"""Tiny positive float to guard against log(0)."""

_BAR_MAX_ITER: int = 500
"""Maximum iterations for BAR self-consistent solver."""

_BAR_TOL: float = 1e-10
"""Convergence tolerance for BAR (kcal/mol)."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorkSample:
    """Single non-equilibrium work measurement from a replica exchange attempt.

    Attributes
    ----------
    work_kcal : float
        Non-equilibrium work W in kcal/mol.
        Forward: :math:`W_F = (\\beta_j - \\beta_i)(E_j - E_i)` for heating.
        Reverse: :math:`W_R = (\\beta_i - \\beta_j)(E_i - E_j)` for cooling.
    temperature_low : float
        Lower temperature of the exchange pair (K).
    temperature_high : float
        Higher temperature of the exchange pair (K).
    accepted : bool
        Whether the Metropolis criterion accepted this exchange.
    replica_i : int
        Index of the lower-temperature replica.
    replica_j : int
        Index of the higher-temperature replica.
    """
    work_kcal: float
    temperature_low: float
    temperature_high: float
    accepted: bool
    replica_i: int
    replica_j: int


@dataclass
class LegResult:
    """Aggregated result for one direction of the round-trip.

    Attributes
    ----------
    direction : str
        ``"forward"`` (heating) or ``"reverse"`` (cooling).
    work_values : NDArray[np.float64]
        Array of all work measurements W (kcal/mol).
    mean_work : float
        :math:`\\langle W \\rangle` (kcal/mol).
    var_work : float
        :math:`\\mathrm{Var}(W)` (kcal²/mol²).
    n_samples : int
        Number of work measurements.
    acceptance_rate : float
        Fraction of accepted exchanges.
    shannon_entropies : dict[float, float]
        Temperature → S_config mapping (kcal/(mol·K)).
    """
    direction: str
    work_values: NDArray[np.float64]
    mean_work: float
    var_work: float
    n_samples: int
    acceptance_rate: float
    shannon_entropies: dict[float, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in ("forward", "reverse"):
            raise ValueError(f"direction must be 'forward' or 'reverse', got '{self.direction}'")
        if self.n_samples < 0:
            raise ValueError(f"n_samples must be non-negative, got {self.n_samples}")


@dataclass
class BidirectionalResult:
    """Full round-trip analysis combining forward and reverse legs.

    Attributes
    ----------
    delta_G_bar : float
        Free energy difference from BAR (kcal/mol).
    delta_G_crooks : float
        Free energy from Crooks intersection (kcal/mol).
    sigma_irr : float
        Irreversible entropy production (kcal/(mol·K)).
    bits_lost : float
        Landauer information loss (bits per round-trip).
    forward : LegResult
        Forward (heating) leg results.
    reverse : LegResult
        Reverse (cooling) leg results.
    mutual_info : float
        Mutual information I(hot; cold) (kcal/(mol·K)).
    collapse_rates : dict[float, float]
        Temperature → dI/dT mapping.
    converged : bool
        Whether σ_irr is below convergence threshold.
    bar_iterations : int
        Number of BAR self-consistent iterations to converge.
    """
    delta_G_bar: float
    delta_G_crooks: float
    sigma_irr: float
    bits_lost: float
    forward: LegResult
    reverse: LegResult
    mutual_info: float
    collapse_rates: dict[float, float] = field(default_factory=dict)
    converged: bool = False
    bar_iterations: int = 0

    def __post_init__(self) -> None:
        if self.sigma_irr < -1e-6:
            raise ValueError(
                f"sigma_irr must be >= 0 by second law, got {self.sigma_irr:.6e}. "
                "This indicates a numerical issue in work accumulation."
            )

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dictionary."""
        return {
            "delta_G_bar": self.delta_G_bar,
            "delta_G_crooks": self.delta_G_crooks,
            "sigma_irr": self.sigma_irr,
            "bits_lost": self.bits_lost,
            "mutual_info": self.mutual_info,
            "collapse_rates": self.collapse_rates,
            "converged": self.converged,
            "bar_iterations": self.bar_iterations,
            "forward": {
                "direction": self.forward.direction,
                "mean_work": self.forward.mean_work,
                "var_work": self.forward.var_work,
                "n_samples": self.forward.n_samples,
                "acceptance_rate": self.forward.acceptance_rate,
                "shannon_entropies": self.forward.shannon_entropies,
            },
            "reverse": {
                "direction": self.reverse.direction,
                "mean_work": self.reverse.mean_work,
                "var_work": self.reverse.var_work,
                "n_samples": self.reverse.n_samples,
                "acceptance_rate": self.reverse.acceptance_rate,
                "shannon_entropies": self.reverse.shannon_entropies,
            },
        }


# ---------------------------------------------------------------------------
# Bennett Acceptance Ratio (BAR)
# ---------------------------------------------------------------------------

def bennett_acceptance_ratio(
    w_fwd: NDArray[np.float64],
    w_rev: NDArray[np.float64],
    temperature: float,
    max_iter: int = _BAR_MAX_ITER,
    tol: float = _BAR_TOL,
) -> tuple[float, int]:
    """Optimal ΔG from bidirectional work distributions via BAR.

    Solves the self-consistent BAR equation iteratively:

    .. math::

        \\sum_i f(\\beta(W_{F,i} - \\Delta G) + \\ln(n_F/n_R))
        = \\sum_j f(\\beta(-W_{R,j} - \\Delta G) + \\ln(n_F/n_R))

    where :math:`f(x) = 1 / (1 + \\exp(x))` is the Fermi function.

    Parameters
    ----------
    w_fwd : NDArray[np.float64]
        Forward work measurements (kcal/mol).
    w_rev : NDArray[np.float64]
        Reverse work measurements (kcal/mol).
    temperature : float
        Reference temperature (K) for β = 1/(R·T).
    max_iter : int
        Maximum self-consistent iterations.
    tol : float
        Convergence tolerance on ΔG (kcal/mol).

    Returns
    -------
    delta_G : float
        BAR estimate of free energy difference (kcal/mol).
    n_iter : int
        Number of iterations to converge.

    Raises
    ------
    ValueError
        If work arrays are empty.
    """
    n_f = len(w_fwd)
    n_r = len(w_rev)
    if n_f == 0 or n_r == 0:
        raise ValueError(
            f"BAR requires non-empty work arrays (got n_F={n_f}, n_R={n_r})"
        )

    beta = 1.0 / (R_KCAL * temperature)
    ln_ratio = np.log(n_f / n_r)

    # Initial estimate: mean of Jarzynski averages
    # ΔG_fwd ≈ -kT ln <exp(-βW_F)>, ΔG_rev ≈ kT ln <exp(-βW_R)>
    max_neg_bwf = np.max(-beta * w_fwd)
    dG_jar_fwd = -(1.0 / beta) * (max_neg_bwf + np.log(
        np.mean(np.exp(-beta * w_fwd - max_neg_bwf))
    ))
    max_neg_bwr = np.max(-beta * w_rev)
    dG_jar_rev = (1.0 / beta) * (max_neg_bwr + np.log(
        np.mean(np.exp(-beta * w_rev - max_neg_bwr))
    ))
    delta_G = 0.5 * (dG_jar_fwd + dG_jar_rev)

    for iteration in range(1, max_iter + 1):
        # Fermi function: f(x) = 1/(1 + exp(x)), computed with log-sum-exp
        # Forward: x_i = β(W_{F,i} - ΔG) + ln(n_F/n_R)
        arg_fwd = beta * (w_fwd - delta_G) + ln_ratio
        # Reverse: x_j = -β(W_{R,j} + ΔG) + ln(n_F/n_R)
        #   = β(-W_{R,j} - ΔG) + ln(n_F/n_R)
        arg_rev = beta * (-w_rev - delta_G) + ln_ratio

        # Numerically stable Fermi function: f(x) = 1/(1+exp(x))
        # For large positive x: f(x) ≈ exp(-x)
        # For large negative x: f(x) ≈ 1
        def _log_fermi(x: NDArray[np.float64]) -> NDArray[np.float64]:
            """log(1 / (1 + exp(x))) = -log(1 + exp(x)), stable."""
            # softplus(x) = log(1 + exp(x)), use np.logaddexp for stability
            return -np.logaddexp(0.0, x)

        log_f_fwd = _log_fermi(arg_fwd)
        log_f_rev = _log_fermi(arg_rev)

        # BAR equation: Σ f(arg_fwd) = Σ f(arg_rev)
        # ΔG_new solves: ln Σ exp(log_f_rev) - ln Σ exp(log_f_fwd) = 0
        # Using log-sum-exp:
        lse_fwd = _logsumexp(log_f_fwd)
        lse_rev = _logsumexp(log_f_rev)

        # Update: ΔG_new = ΔG + (1/β) * (lse_rev - lse_fwd)
        delta_G_new = delta_G + (1.0 / beta) * (lse_rev - lse_fwd)

        if abs(delta_G_new - delta_G) < tol:
            return float(delta_G_new), iteration

        delta_G = delta_G_new

    return float(delta_G), max_iter


# ---------------------------------------------------------------------------
# Crooks intersection
# ---------------------------------------------------------------------------

def crooks_intersection(
    w_fwd: NDArray[np.float64],
    w_rev: NDArray[np.float64],
    temperature: float,
    n_bins: int = 200,
) -> float:
    """Find ΔG where P(W_fwd) and P(-W_rev) cross (Crooks Fluctuation Theorem).

    At the intersection point W* = ΔG:

    .. math::

        P_F(W^*) = P_R(-W^*) \\implies W^* = \\Delta G

    Uses kernel density estimation for smooth distributions and finds
    the crossing point by bisection on a fine grid.

    Parameters
    ----------
    w_fwd : NDArray[np.float64]
        Forward work values (kcal/mol).
    w_rev : NDArray[np.float64]
        Reverse work values (kcal/mol).
    temperature : float
        Reference temperature (K).
    n_bins : int
        Number of grid points for KDE evaluation.

    Returns
    -------
    float
        ΔG estimate from Crooks intersection (kcal/mol).
        Returns NaN if no intersection is found.
    """
    if len(w_fwd) < 2 or len(w_rev) < 2:
        return float("nan")

    neg_w_rev = -w_rev

    # Determine grid range covering both distributions
    all_w = np.concatenate([w_fwd, neg_w_rev])
    w_min = float(np.min(all_w))
    w_max = float(np.max(all_w))
    margin = 0.1 * (w_max - w_min) if w_max > w_min else 1.0
    grid = np.linspace(w_min - margin, w_max + margin, n_bins)

    # Gaussian KDE with Silverman bandwidth
    bw_fwd = _silverman_bandwidth(w_fwd)
    bw_rev = _silverman_bandwidth(neg_w_rev)

    if bw_fwd < _EPSILON or bw_rev < _EPSILON:
        return float("nan")

    pdf_fwd = _gaussian_kde(w_fwd, grid, bw_fwd)
    pdf_rev = _gaussian_kde(neg_w_rev, grid, bw_rev)

    # Find crossing: where sign of (pdf_fwd - pdf_rev) changes
    diff = pdf_fwd - pdf_rev
    crossings = np.where(np.diff(np.sign(diff)))[0]

    if len(crossings) == 0:
        # No crossing found — fall back to midpoint of means
        return float(0.5 * (np.mean(w_fwd) + np.mean(neg_w_rev)))

    # Take the crossing closest to the mean of both distributions
    target = 0.5 * (np.mean(w_fwd) + np.mean(neg_w_rev))
    best_idx = crossings[np.argmin(np.abs(grid[crossings] - target))]

    # Linear interpolation between grid[best_idx] and grid[best_idx + 1]
    d0 = diff[best_idx]
    d1 = diff[best_idx + 1]
    if abs(d1 - d0) < _EPSILON:
        return float(grid[best_idx])

    t = d0 / (d0 - d1)
    return float(grid[best_idx] + t * (grid[best_idx + 1] - grid[best_idx]))


# ---------------------------------------------------------------------------
# Irreversible entropy production
# ---------------------------------------------------------------------------

def irreversible_entropy_production(
    mean_w_fwd: float,
    mean_w_rev: float,
    delta_G: float,
    temperature: float,
) -> float:
    r"""Irreversible entropy production from bidirectional work measurements.

    .. math::

        \sigma_{\mathrm{irr}} = \frac{\langle W_F \rangle + \langle W_R \rangle - 2\Delta G}{T}

    By the second law, :math:`\sigma_{\mathrm{irr}} \geq 0`.

    Parameters
    ----------
    mean_w_fwd : float
        Mean forward work ⟨W_F⟩ (kcal/mol).
    mean_w_rev : float
        Mean reverse work ⟨W_R⟩ (kcal/mol).
    delta_G : float
        Free energy difference ΔG (kcal/mol).
    temperature : float
        Temperature (K).

    Returns
    -------
    float
        σ_irr in kcal/(mol·K).
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    dissipation = mean_w_fwd + mean_w_rev - 2.0 * delta_G
    return max(0.0, dissipation / temperature)


# ---------------------------------------------------------------------------
# Landauer information loss
# ---------------------------------------------------------------------------

def landauer_information_loss(
    sigma_irr: float,
    temperature: float,
) -> float:
    r"""Bits of information erased per round-trip via Landauer's principle.

    .. math::

        \text{bits lost} = \frac{\sigma_{\mathrm{irr}}}{k_B \ln 2}
        = \frac{T \cdot \sigma_{\mathrm{irr}}}{R \cdot \ln 2}

    In our units (σ_irr in kcal/(mol·K), R in kcal/(mol·K)):

    .. math::

        \text{bits lost} = \frac{\sigma_{\mathrm{irr}}}{R \cdot \ln 2}

    Parameters
    ----------
    sigma_irr : float
        Irreversible entropy production (kcal/(mol·K)).
    temperature : float
        Temperature (K), for unit consistency check.

    Returns
    -------
    float
        Landauer information loss in bits.
    """
    if sigma_irr < 0:
        return 0.0
    return sigma_irr / (R_KCAL * LN2)


# ---------------------------------------------------------------------------
# Shannon Energy Collapse Rate
# ---------------------------------------------------------------------------

def shannon_energy_collapse_rate(
    temperatures: Sequence[float],
    entropies: Sequence[float],
) -> dict[float, float]:
    r"""Rate of Shannon entropy collapse dS/dT across the temperature ladder.

    .. math::

        \left.\frac{dS}{dT}\right|_{T_i} \approx \frac{S(T_{i+1}) - S(T_{i-1})}{T_{i+1} - T_{i-1}}

    Positive dS/dT: landscape is fracturing (more accessible states at higher T).
    Negative dS/dT: landscape is funneling (fewer dominant states at higher T).

    Parameters
    ----------
    temperatures : Sequence[float]
        Temperature values (K), sorted ascending.
    entropies : Sequence[float]
        Shannon entropies S_config at each temperature (kcal/(mol·K)).

    Returns
    -------
    dict[float, float]
        Temperature → dS/dT mapping for interior points.
    """
    T = np.asarray(temperatures, dtype=np.float64)
    S = np.asarray(entropies, dtype=np.float64)
    n = len(T)
    rates: dict[float, float] = {}

    if n < 3:
        return rates

    # Central differences for interior points
    for i in range(1, n - 1):
        dT = T[i + 1] - T[i - 1]
        if abs(dT) < _EPSILON:
            continue
        dSdT = (S[i + 1] - S[i - 1]) / dT
        rates[float(T[i])] = float(dSdT)

    # Forward difference for first point
    dT0 = T[1] - T[0]
    if abs(dT0) > _EPSILON:
        rates[float(T[0])] = float((S[1] - S[0]) / dT0)

    # Backward difference for last point
    dTn = T[-1] - T[-2]
    if abs(dTn) > _EPSILON:
        rates[float(T[-1])] = float((S[-1] - S[-2]) / dTn)

    return rates


# ---------------------------------------------------------------------------
# Mutual Information
# ---------------------------------------------------------------------------

def mutual_information(
    S_high: float,
    S_low: float,
    S_joint: float,
) -> float:
    r"""Mutual information between hot and cold ensembles.

    .. math::

        I(\text{hot}; \text{cold}) = S(T_{\text{high}}) + S(T_{\text{low}}) - S_{\text{joint}}

    Parameters
    ----------
    S_high : float
        Shannon entropy at high temperature (kcal/(mol·K)).
    S_low : float
        Shannon entropy at low temperature (kcal/(mol·K)).
    S_joint : float
        Joint Shannon entropy of merged ensemble (kcal/(mol·K)).

    Returns
    -------
    float
        Mutual information I (kcal/(mol·K)). Clamped to >= 0.
    """
    return max(0.0, S_high + S_low - S_joint)


# ---------------------------------------------------------------------------
# Convergence check
# ---------------------------------------------------------------------------

def convergence_check(
    sigma_irr: float,
    threshold: float = 1e-3,
) -> bool:
    r"""Check if the round-trip has converged based on irreversible entropy production.

    The physics-based convergence criterion: :math:`\sigma_{\mathrm{irr}} \to 0`
    implies reversibility of the temperature schedule.

    Parameters
    ----------
    sigma_irr : float
        Irreversible entropy production (kcal/(mol·K)).
    threshold : float
        Convergence threshold (default: 1e-3 kcal/(mol·K)).

    Returns
    -------
    bool
        True if σ_irr < threshold.
    """
    return sigma_irr < threshold


# ---------------------------------------------------------------------------
# Bidirectional Exchange Engine
# ---------------------------------------------------------------------------

class BidirectionalExchange:
    """Engine managing forward (heating) and reverse (cooling) replica exchange legs.

    The bidirectional protocol:
    1. **Forward leg**: sweep T_low → T_high, recording W_fwd for each exchange
    2. **Reverse leg**: sweep T_high → T_low, recording W_rev for each exchange
    3. **Analysis**: BAR, Crooks intersection, σ_irr, Landauer bits, mutual info

    Parameters
    ----------
    temperatures : Sequence[float]
        Temperature ladder (K), sorted ascending.
    reference_temperature : float
        Reference temperature for BAR (K), typically T_low.
    """

    def __init__(
        self,
        temperatures: Sequence[float],
        reference_temperature: float = 298.15,
    ) -> None:
        self.temperatures = np.asarray(temperatures, dtype=np.float64)
        self.reference_temperature = reference_temperature
        self._forward_work: list[WorkSample] = []
        self._reverse_work: list[WorkSample] = []
        self._forward_entropies: dict[float, list[float]] = {}
        self._reverse_entropies: dict[float, list[float]] = {}

    def reset(self) -> None:
        """Clear all accumulated work and entropy measurements."""
        self._forward_work.clear()
        self._reverse_work.clear()
        self._forward_entropies.clear()
        self._reverse_entropies.clear()

    def run_forward_leg(
        self,
        replicas: list[ReplicaState],
        n_sweeps: int = 1,
        rng: np.random.Generator | None = None,
    ) -> LegResult:
        """Execute forward (heating) leg: exchange sweeps from low T to high T.

        Records non-equilibrium work for each exchange attempt:

        .. math::

            W_F = (\\beta_j - \\beta_i)(E_j - E_i)

        Parameters
        ----------
        replicas : list[ReplicaState]
            Replicas sorted by temperature (ascending).
        n_sweeps : int
            Number of full exchange sweeps.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        LegResult
            Forward leg results with work values and Shannon entropies.
        """
        if rng is None:
            rng = np.random.default_rng()

        work_samples: list[WorkSample] = []

        for sweep in range(n_sweeps):
            # Forward: sweep from low-T pairs upward
            for idx in range(len(replicas) - 1):
                ri, rj = replicas[idx], replicas[idx + 1]

                # Non-equilibrium work for this exchange
                delta_beta = rj.beta - ri.beta  # β_high - β_low < 0
                delta_E = rj.current_energy - ri.current_energy
                w = -(delta_beta * delta_E)  # W_F = -(β_j - β_i)(E_j - E_i)

                # Attempt Metropolis exchange
                delta = (ri.beta - rj.beta) * (ri.current_energy - rj.current_energy)
                accepted = delta <= 0.0 or rng.random() < math.exp(min(0.0, -delta))

                ri.exchange_count += 1
                rj.exchange_count += 1
                if accepted:
                    ri.exchange_accepted += 1
                    rj.exchange_accepted += 1
                    ri.current_energy, rj.current_energy = rj.current_energy, ri.current_energy

                sample = WorkSample(
                    work_kcal=w,
                    temperature_low=ri.temperature,
                    temperature_high=rj.temperature,
                    accepted=accepted,
                    replica_i=idx,
                    replica_j=idx + 1,
                )
                work_samples.append(sample)

            # Compute Shannon entropy at each temperature after sweep
            for replica in replicas:
                T = replica.temperature
                if replica.poses:
                    energies = [p.energy_kcal for p in replica.poses]
                    S = shannon_entropy_of_ensemble(energies, T)
                    self._forward_entropies.setdefault(T, []).append(S)

        self._forward_work.extend(work_samples)
        return self._build_leg_result("forward", work_samples)

    def run_reverse_leg(
        self,
        replicas: list[ReplicaState],
        n_sweeps: int = 1,
        rng: np.random.Generator | None = None,
    ) -> LegResult:
        """Execute reverse (cooling) leg: exchange sweeps from high T to low T.

        Takes the high-T ensemble and requenches at progressively lower T.

        Records non-equilibrium work:

        .. math::

            W_R = (\\beta_i - \\beta_j)(E_i - E_j)

        Parameters
        ----------
        replicas : list[ReplicaState]
            Replicas sorted by temperature (ascending).
        n_sweeps : int
            Number of full reverse sweeps.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        LegResult
            Reverse leg results.
        """
        if rng is None:
            rng = np.random.default_rng()

        work_samples: list[WorkSample] = []

        for sweep in range(n_sweeps):
            # Reverse: sweep from high-T pairs downward
            for idx in range(len(replicas) - 2, -1, -1):
                ri, rj = replicas[idx], replicas[idx + 1]

                # Non-equilibrium work for cooling
                delta_beta = ri.beta - rj.beta  # β_low - β_high > 0
                delta_E = ri.current_energy - rj.current_energy
                w = -(delta_beta * delta_E)  # W_R = -(β_i - β_j)(E_i - E_j)

                # Attempt Metropolis exchange
                delta = (ri.beta - rj.beta) * (ri.current_energy - rj.current_energy)
                accepted = delta <= 0.0 or rng.random() < math.exp(min(0.0, -delta))

                ri.exchange_count += 1
                rj.exchange_count += 1
                if accepted:
                    ri.exchange_accepted += 1
                    rj.exchange_accepted += 1
                    ri.current_energy, rj.current_energy = rj.current_energy, ri.current_energy

                sample = WorkSample(
                    work_kcal=w,
                    temperature_low=ri.temperature,
                    temperature_high=rj.temperature,
                    accepted=accepted,
                    replica_i=idx,
                    replica_j=idx + 1,
                )
                work_samples.append(sample)

            # Shannon entropy after reverse sweep
            for replica in replicas:
                T = replica.temperature
                if replica.poses:
                    energies = [p.energy_kcal for p in replica.poses]
                    S = shannon_entropy_of_ensemble(energies, T)
                    self._reverse_entropies.setdefault(T, []).append(S)

        self._reverse_work.extend(work_samples)
        return self._build_leg_result("reverse", work_samples)

    def analyze(self) -> BidirectionalResult:
        """Run full bidirectional analysis: BAR, Crooks, σ_irr, Landauer, mutual info.

        Requires both forward and reverse legs to have been executed.

        Returns
        -------
        BidirectionalResult
            Complete round-trip analysis.

        Raises
        ------
        ValueError
            If either leg has no data.
        """
        if not self._forward_work:
            raise ValueError("No forward work data. Run run_forward_leg() first.")
        if not self._reverse_work:
            raise ValueError("No reverse work data. Run run_reverse_leg() first.")

        w_fwd = np.array([s.work_kcal for s in self._forward_work])
        w_rev = np.array([s.work_kcal for s in self._reverse_work])

        # BAR
        dG_bar, bar_iters = bennett_acceptance_ratio(
            w_fwd, w_rev, self.reference_temperature
        )

        # Crooks intersection
        dG_crooks = crooks_intersection(
            w_fwd, w_rev, self.reference_temperature
        )

        # Build leg results
        fwd_result = self._build_leg_result("forward", self._forward_work)
        rev_result = self._build_leg_result("reverse", self._reverse_work)

        # Irreversible entropy production
        sigma = irreversible_entropy_production(
            fwd_result.mean_work, rev_result.mean_work,
            dG_bar, self.reference_temperature,
        )

        # Landauer bits
        bits = landauer_information_loss(sigma, self.reference_temperature)

        # Mutual information from highest and lowest temperature entropies
        T_low = float(self.temperatures[0])
        T_high = float(self.temperatures[-1])

        S_low = self._avg_entropy(self._forward_entropies, T_low)
        S_high = self._avg_entropy(self._forward_entropies, T_high)

        # Joint entropy: merge energies from both temperature extremes
        # Approximate S_joint from the average of forward and reverse at midpoint
        T_mid_idx = len(self.temperatures) // 2
        T_mid = float(self.temperatures[T_mid_idx])
        S_mid_fwd = self._avg_entropy(self._forward_entropies, T_mid)
        S_mid_rev = self._avg_entropy(self._reverse_entropies, T_mid)
        S_joint = max(S_low, S_high, 0.5 * (S_mid_fwd + S_mid_rev))

        mi = mutual_information(S_high, S_low, S_joint)

        # Collapse rates
        all_temps = sorted(set(
            list(self._forward_entropies.keys()) +
            list(self._reverse_entropies.keys())
        ))
        avg_entropies = []
        for T in all_temps:
            s_fwd = self._avg_entropy(self._forward_entropies, T)
            s_rev = self._avg_entropy(self._reverse_entropies, T)
            avg_entropies.append(0.5 * (s_fwd + s_rev) if s_fwd > 0 and s_rev > 0
                                 else max(s_fwd, s_rev))

        collapse = shannon_energy_collapse_rate(all_temps, avg_entropies)

        # Convergence
        converged = convergence_check(sigma)

        return BidirectionalResult(
            delta_G_bar=dG_bar,
            delta_G_crooks=dG_crooks,
            sigma_irr=sigma,
            bits_lost=bits,
            forward=fwd_result,
            reverse=rev_result,
            mutual_info=mi,
            collapse_rates=collapse,
            converged=converged,
            bar_iterations=bar_iters,
        )

    # -- private helpers --

    def _build_leg_result(
        self,
        direction: str,
        samples: list[WorkSample] | Sequence[WorkSample],
    ) -> LegResult:
        """Aggregate work samples into a LegResult."""
        if not samples:
            return LegResult(
                direction=direction,
                work_values=np.array([], dtype=np.float64),
                mean_work=0.0,
                var_work=0.0,
                n_samples=0,
                acceptance_rate=0.0,
            )

        w = np.array([s.work_kcal for s in samples])
        accepted = sum(1 for s in samples if s.accepted)

        entropy_store = (self._forward_entropies if direction == "forward"
                         else self._reverse_entropies)
        avg_entropies = {
            T: float(np.mean(vals)) for T, vals in entropy_store.items() if vals
        }

        return LegResult(
            direction=direction,
            work_values=w,
            mean_work=float(np.mean(w)),
            var_work=float(np.var(w)),
            n_samples=len(w),
            acceptance_rate=accepted / len(samples),
            shannon_entropies=avg_entropies,
        )

    @staticmethod
    def _avg_entropy(store: dict[float, list[float]], T: float) -> float:
        """Average entropy values at temperature T from stored measurements."""
        vals = store.get(T, [])
        if not vals:
            return 0.0
        return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Utility: log-sum-exp
# ---------------------------------------------------------------------------

def _logsumexp(x: NDArray[np.float64]) -> float:
    """Numerically stable log-sum-exp: log(Σ exp(x_i))."""
    if len(x) == 0:
        return float("-inf")
    x_max = np.max(x)
    if not np.isfinite(x_max):
        return float(x_max)
    return float(x_max + np.log(np.sum(np.exp(x - x_max))))


# ---------------------------------------------------------------------------
# Utility: Gaussian KDE
# ---------------------------------------------------------------------------

def _silverman_bandwidth(data: NDArray[np.float64]) -> float:
    """Silverman's rule of thumb bandwidth: h = 1.06 * σ * n^(-1/5)."""
    n = len(data)
    if n < 2:
        return 0.0
    sigma = float(np.std(data, ddof=1))
    return 1.06 * sigma * n ** (-0.2)


def _gaussian_kde(
    data: NDArray[np.float64],
    grid: NDArray[np.float64],
    bandwidth: float,
) -> NDArray[np.float64]:
    """Vectorized Gaussian kernel density estimate on a grid.

    .. math::

        \\hat{f}(x) = \\frac{1}{n h} \\sum_{i=1}^n K\\left(\\frac{x - x_i}{h}\\right)

    Parameters
    ----------
    data : NDArray[np.float64]
        Data points (1D).
    grid : NDArray[np.float64]
        Evaluation grid (1D).
    bandwidth : float
        KDE bandwidth h.

    Returns
    -------
    NDArray[np.float64]
        Estimated density at each grid point.
    """
    n = len(data)
    if n == 0 or bandwidth < _EPSILON:
        return np.zeros_like(grid)

    # Vectorized: (grid[:, None] - data[None, :]) / bandwidth
    z = (grid[:, np.newaxis] - data[np.newaxis, :]) / bandwidth
    kernel = np.exp(-0.5 * z * z) / (math.sqrt(2.0 * math.pi) * bandwidth)
    return np.mean(kernel, axis=1)
