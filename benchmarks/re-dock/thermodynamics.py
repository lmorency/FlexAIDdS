"""
RE-DOCK Thermodynamics Module
=============================

Van't Hoff analysis, replica exchange thermodynamics, Shannon entropy,
and WHAM free energy surface reconstruction for distributed docking campaigns.

Components
----------
- **DockingPose**: Lightweight pose data (energy, RMSD, coordinates hash)
- **ReplicaState**: Full state of one temperature replica (poses, energies, generation)
- **VantHoffResult**: Fitted ΔH°, ΔS°, ΔCp with statistics

Functions
---------
- geometric_temperature_ladder: Generate T_i = T_min * (T_max/T_min)^(i/(N-1))
- metropolis_exchange_criterion: Δ = (β_i - β_j)(E_i - E_j), accept if Δ<0 or rand<exp(-Δ)
- attempt_exchanges: Sweep adjacent pairs with even/odd alternation
- van_t_hoff_analysis: Linear ln(K) vs 1/T and nonlinear ΔCp fit
- shannon_entropy_of_ensemble: S_config = -k_B Σ p_i ln(p_i), log-sum-exp stable
- free_energy_surface: WHAM over temperature-biased histograms

Temperature ladder: geometric spacing from T_min to T_max across N replicas.
Default: 8 replicas, 298K–600K (β ratio ≈ 0.93 for ~35% exchange acceptance).

Le Bonhomme Pharma / Najmanovich Research Group
"""

from __future__ import annotations

import json
import math
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

R_KCAL = 1.987204e-3   # kcal/(mol·K)
K_BOLTZMANN = 1.380649e-23  # J/K
N_AVOGADRO = 6.02214076e23


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DockingPose:
    """Lightweight representation of a single docked pose."""
    energy_kcal: float
    rmsd_to_xtal: float = float("nan")
    coords_hash: str = ""
    generation: int = 0
    replica_index: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DockingPose":
        return cls(**d)


@dataclass
class ReplicaState:
    """Full thermodynamic state of one temperature replica."""
    replica_index: int
    temperature: float  # Kelvin
    poses: List[DockingPose] = field(default_factory=list)
    best_energy: float = float("inf")
    current_energy: float = float("inf")
    generation: int = 0
    exchange_count: int = 0
    exchange_accepted: int = 0

    @property
    def beta(self) -> float:
        """Inverse temperature β = 1/(k_B T) in (kcal/mol)^{-1}."""
        return 1.0 / (R_KCAL * self.temperature)

    @property
    def acceptance_ratio(self) -> float:
        if self.exchange_count == 0:
            return 0.0
        return self.exchange_accepted / self.exchange_count

    def add_pose(self, pose: DockingPose) -> None:
        self.poses.append(pose)
        if pose.energy_kcal < self.best_energy:
            self.best_energy = pose.energy_kcal
        self.current_energy = pose.energy_kcal

    def to_dict(self) -> dict:
        d = asdict(self)
        d["poses"] = [p.to_dict() for p in self.poses]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ReplicaState":
        poses = [DockingPose.from_dict(p) for p in d.pop("poses", [])]
        state = cls(**{k: v for k, v in d.items() if k != "poses"}, poses=poses)
        return state


@dataclass
class VantHoffResult:
    """Result of Van't Hoff thermodynamic decomposition."""
    delta_H_kcal: float       # ΔH° (kcal/mol)
    delta_S_cal: float         # ΔS° (cal/mol·K)
    delta_Cp_cal: float = 0.0  # ΔCp (cal/mol·K) — nonlinear fit
    T_ref: float = 298.15      # Reference temperature (K)
    r_squared: float = 0.0
    temperatures: List[float] = field(default_factory=list)
    ln_K_values: List[float] = field(default_factory=list)
    residuals: List[float] = field(default_factory=list)

    @property
    def delta_G_kcal(self) -> float:
        """ΔG° at T_ref = ΔH° - T_ref·ΔS°."""
        return self.delta_H_kcal - self.T_ref * self.delta_S_cal / 1000.0

    def delta_G_at(self, T: float) -> float:
        """ΔG(T) = ΔH + ΔCp*(T-Tref) - T*(ΔS + ΔCp*ln(T/Tref)), all in kcal/mol."""
        dCp_kcal = self.delta_Cp_cal / 1000.0
        dS_kcal = self.delta_S_cal / 1000.0
        return (self.delta_H_kcal
                + dCp_kcal * (T - self.T_ref)
                - T * (dS_kcal + dCp_kcal * math.log(T / self.T_ref)))

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Temperature ladder
# ---------------------------------------------------------------------------

def geometric_temperature_ladder(T_min: float, T_max: float, n_replicas: int) -> List[float]:
    """Generate geometrically spaced temperature ladder.

    T_i = T_min * (T_max / T_min)^(i / (N-1))

    This spacing gives roughly uniform exchange acceptance rates when
    energy fluctuations scale as σ_E ~ T (harmonic approximation).

    Parameters
    ----------
    T_min : float
        Lowest temperature (K), typically 298.
    T_max : float
        Highest temperature (K), typically 500–600.
    n_replicas : int
        Number of replicas (≥ 2).

    Returns
    -------
    List[float]
        Sorted temperatures from T_min to T_max.
    """
    if n_replicas < 2:
        return [T_min]
    ratio = T_max / T_min
    return [T_min * ratio ** (i / (n_replicas - 1)) for i in range(n_replicas)]


# ---------------------------------------------------------------------------
# Replica exchange
# ---------------------------------------------------------------------------

def metropolis_exchange_criterion(
    beta_i: float, beta_j: float,
    energy_i: float, energy_j: float,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[bool, float]:
    """Metropolis criterion for replica exchange between replicas i and j.

    Δ = (β_i - β_j)(E_i - E_j)
    Accept if Δ ≤ 0 or rand < exp(-Δ).

    Parameters
    ----------
    beta_i, beta_j : float
        Inverse temperatures 1/(k_B T) in (kcal/mol)^{-1}.
    energy_i, energy_j : float
        Current energies in kcal/mol.
    rng : np.random.Generator, optional
        Random number generator (default: np.random.default_rng()).

    Returns
    -------
    accepted : bool
        Whether the exchange is accepted.
    delta : float
        The Δ value used for the decision.
    """
    if rng is None:
        rng = np.random.default_rng()

    delta = (beta_i - beta_j) * (energy_i - energy_j)

    if delta <= 0.0:
        return True, delta

    prob = math.exp(-delta)
    return rng.random() < prob, delta


def attempt_exchanges(
    replicas: List[ReplicaState],
    even_odd: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[int, int, bool, float]]:
    """Attempt pairwise exchanges between adjacent replicas.

    Uses even/odd alternation to avoid conflicts:
    - even_odd=0: pairs (0,1), (2,3), (4,5), ...
    - even_odd=1: pairs (1,2), (3,4), (5,6), ...

    Accepted exchanges swap the poses/energies between replicas
    (temperatures stay fixed, configurations move).

    Parameters
    ----------
    replicas : List[ReplicaState]
        Sorted by temperature (ascending).
    even_odd : int
        0 for even pairs, 1 for odd pairs.
    rng : np.random.Generator, optional

    Returns
    -------
    List of (i, j, accepted, delta) tuples.
    """
    if rng is None:
        rng = np.random.default_rng()

    results = []
    start = even_odd % 2

    for idx in range(start, len(replicas) - 1, 2):
        i, j = idx, idx + 1
        ri, rj = replicas[i], replicas[j]

        accepted, delta = metropolis_exchange_criterion(
            ri.beta, rj.beta, ri.current_energy, rj.current_energy, rng
        )

        ri.exchange_count += 1
        rj.exchange_count += 1

        if accepted:
            ri.exchange_accepted += 1
            rj.exchange_accepted += 1
            # Swap current energies (configurations move, temperatures stay)
            ri.current_energy, rj.current_energy = rj.current_energy, ri.current_energy

        results.append((i, j, accepted, delta))

    return results


# ---------------------------------------------------------------------------
# Van't Hoff analysis
# ---------------------------------------------------------------------------

def van_t_hoff_analysis(
    temperatures: List[float],
    free_energies_kcal: List[float],
    fit_dCp: bool = False,
    T_ref: float = 298.15,
) -> VantHoffResult:
    """Van't Hoff thermodynamic decomposition from ΔG(T) data.

    Linear fit: ln(K) = -ΔH°/(R·T) + ΔS°/R  where K = exp(-ΔG/(RT))
    Nonlinear (if fit_dCp=True): includes ΔCp heat capacity correction.

    Parameters
    ----------
    temperatures : List[float]
        Temperatures in Kelvin.
    free_energies_kcal : List[float]
        ΔG at each temperature in kcal/mol.
    fit_dCp : bool
        If True, fit 3-parameter model with ΔCp.
    T_ref : float
        Reference temperature for ΔCp model.

    Returns
    -------
    VantHoffResult
    """
    T_arr = np.array(temperatures)
    dG_arr = np.array(free_energies_kcal)

    # ln(K) = -ΔG / (R·T)
    ln_K = -dG_arr / (R_KCAL * T_arr)
    inv_T = 1.0 / T_arr

    if not fit_dCp or len(T_arr) < 4:
        # Linear Van't Hoff: ln(K) = -ΔH/(R) * (1/T) + ΔS/R
        coeffs = np.polyfit(inv_T, ln_K, 1)
        slope, intercept = coeffs[0], coeffs[1]

        delta_H = -slope * R_KCAL * 1000.0   # kcal/mol → cal then back
        delta_S = intercept * R_KCAL * 1000.0  # cal/(mol·K)
        # Convert properly: slope = -ΔH/R (with R in kcal), so ΔH = -slope * R_kcal
        delta_H_kcal = -slope * R_KCAL
        delta_S_cal = intercept * R_KCAL * 1000.0

        # Residuals
        ln_K_fit = np.polyval(coeffs, inv_T)
        residuals = (ln_K - ln_K_fit).tolist()

        # R²
        ss_res = np.sum((ln_K - ln_K_fit) ** 2)
        ss_tot = np.sum((ln_K - np.mean(ln_K)) ** 2)
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return VantHoffResult(
            delta_H_kcal=delta_H_kcal,
            delta_S_cal=delta_S_cal,
            delta_Cp_cal=0.0,
            T_ref=T_ref,
            r_squared=r_sq,
            temperatures=T_arr.tolist(),
            ln_K_values=ln_K.tolist(),
            residuals=residuals,
        )
    else:
        # Nonlinear: ΔG(T) = ΔH + ΔCp*(T-Tref) - T*(ΔS + ΔCp*ln(T/Tref))
        # Fit ΔH, ΔS, ΔCp by least-squares on ΔG
        from scipy.optimize import curve_fit

        def dG_model(T, dH, dS_cal, dCp_cal):
            dS_kcal = dS_cal / 1000.0
            dCp_kcal = dCp_cal / 1000.0
            return dH + dCp_kcal * (T - T_ref) - T * (dS_kcal + dCp_kcal * np.log(T / T_ref))

        p0 = [-10.0, -30.0, -200.0]
        popt, pcov = curve_fit(dG_model, T_arr, dG_arr, p0=p0)
        dH_fit, dS_fit, dCp_fit = popt

        dG_pred = dG_model(T_arr, *popt)
        residuals_dG = (dG_arr - dG_pred).tolist()

        ss_res = np.sum((dG_arr - dG_pred) ** 2)
        ss_tot = np.sum((dG_arr - np.mean(dG_arr)) ** 2)
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return VantHoffResult(
            delta_H_kcal=dH_fit,
            delta_S_cal=dS_fit,
            delta_Cp_cal=dCp_fit,
            T_ref=T_ref,
            r_squared=r_sq,
            temperatures=T_arr.tolist(),
            ln_K_values=ln_K.tolist(),
            residuals=residuals_dG,
        )


# ---------------------------------------------------------------------------
# Shannon entropy
# ---------------------------------------------------------------------------

def shannon_entropy_of_ensemble(
    energies_kcal: List[float],
    temperature: float,
) -> float:
    """Shannon configurational entropy of a Boltzmann-weighted pose ensemble.

    S_config = -k_B Σ p_i ln(p_i)

    where p_i = exp(-E_i / k_BT) / Z, computed with log-sum-exp for stability.

    Parameters
    ----------
    energies_kcal : List[float]
        Pose energies in kcal/mol.
    temperature : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Shannon entropy in kcal/(mol·K).
    """
    if not energies_kcal:
        return 0.0

    E = np.array(energies_kcal)
    beta = 1.0 / (R_KCAL * temperature)

    # log-sum-exp trick
    neg_beta_E = -beta * E
    max_val = np.max(neg_beta_E)
    log_Z = max_val + np.log(np.sum(np.exp(neg_beta_E - max_val)))

    # log(p_i) = -β·E_i - log(Z)
    log_p = neg_beta_E - log_Z
    p = np.exp(log_p)

    # S = -k_B Σ p_i ln(p_i)
    S = -R_KCAL * np.sum(p * log_p)
    return float(S)


# ---------------------------------------------------------------------------
# WHAM free energy surface
# ---------------------------------------------------------------------------

def free_energy_surface(
    replica_energies: List[List[float]],
    temperatures: List[float],
    n_bins: int = 50,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """WHAM (Weighted Histogram Analysis Method) free energy surface.

    Combines histograms from multiple temperature replicas into an
    unbiased free energy profile F(E) at the lowest temperature.

    Parameters
    ----------
    replica_energies : List[List[float]]
        Energy samples from each replica.
    temperatures : List[float]
        Temperature of each replica.
    n_bins : int
        Number of histogram bins.
    max_iter : int
        Maximum WHAM iterations.
    tol : float
        Convergence tolerance on free energies.

    Returns
    -------
    bin_centers : np.ndarray
        Energy bin centers.
    F : np.ndarray
        Free energy at each bin (kcal/mol), relative to minimum.
    """
    n_replicas = len(temperatures)
    betas = [1.0 / (R_KCAL * T) for T in temperatures]

    # Collect all energies to determine bin edges
    all_E = np.concatenate([np.array(e) for e in replica_energies])
    E_min, E_max = np.min(all_E), np.max(all_E)
    edges = np.linspace(E_min, E_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Histograms per replica
    histograms = []
    N_samples = []
    for energies in replica_energies:
        h, _ = np.histogram(energies, bins=edges)
        histograms.append(h.astype(float))
        N_samples.append(len(energies))

    histograms = np.array(histograms)  # (n_replicas, n_bins)
    N_samples = np.array(N_samples, dtype=float)

    # WHAM iteration
    f = np.zeros(n_replicas)  # free energy estimates per replica

    for iteration in range(max_iter):
        # Denominator: Σ_k N_k exp(f_k - β_k E_j)
        # Use log-sum-exp across replicas for each bin
        denom = np.zeros(n_bins)
        for k in range(n_replicas):
            denom += N_samples[k] * np.exp(f[k] - betas[k] * centers)

        # Numerator: total counts per bin
        numer = np.sum(histograms, axis=0)

        # Density of states
        with np.errstate(divide="ignore", invalid="ignore"):
            omega = np.where(denom > 0, numer / denom, 0.0)

        # Update free energies
        f_new = np.zeros(n_replicas)
        for k in range(n_replicas):
            Z_k = np.sum(omega * np.exp(-betas[k] * centers))
            if Z_k > 0:
                f_new[k] = -np.log(Z_k)
            else:
                f_new[k] = f[k]

        # Normalize
        f_new -= f_new[0]

        if np.max(np.abs(f_new - f)) < tol:
            f = f_new
            break
        f = f_new

    # Free energy at reference temperature (lowest)
    beta_ref = betas[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        F = np.where(omega > 0, -np.log(omega) / beta_ref, np.inf)
    F -= np.min(F[np.isfinite(F)])

    return centers, F
