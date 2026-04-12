"""RE-DOCK thermodynamic engines: BAR, Crooks, Van't Hoff, Shannon collapse."""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

K_B = 0.001987204  # kcal/(mol·K)


def _log_sum_exp(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp: log(Σ exp(x_i))."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if not np.isfinite(x_max):
        return float(x_max)
    return x_max + np.log(np.sum(np.exp(x - x_max)))


def _fermi(x: np.ndarray) -> np.ndarray:
    """Fermi function f(x) = 1/(1+exp(x)), numerically stable."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    exp_neg = np.exp(-x[pos])
    out[pos] = exp_neg / (1.0 + exp_neg)
    exp_pos = np.exp(x[neg])
    out[neg] = 1.0 / (1.0 + exp_pos)
    return out


class BennettAcceptanceRatio:
    """Iterative BAR solver for minimum-variance DeltaG estimation.

    BAR equation: sum_i f(beta(W_fwd,i - DeltaG)) = sum_j f(beta(W_rev,j + DeltaG))
    where f(x) = 1/(1+exp(x)) is the Fermi function.
    """

    def __init__(self, max_iterations: int = 200, tolerance: float = 1e-8):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def solve(self, w_fwd: np.ndarray, w_rev: np.ndarray, beta: float) -> float:
        """Return DeltaG estimate via iterative BAR.

        Parameters
        ----------
        w_fwd : array
            Forward work values (state A -> B).
        w_rev : array
            Reverse work values (state B -> A).
        beta : float
            Inverse temperature 1/(k_B T).

        Returns
        -------
        float
            Free energy difference DeltaG.
        """
        w_fwd = np.asarray(w_fwd, dtype=np.float64)
        w_rev = np.asarray(w_rev, dtype=np.float64)
        n_fwd = len(w_fwd)
        n_rev = len(w_rev)

        # Initial guess: exponential average
        dg = (np.mean(w_fwd) - np.mean(w_rev)) / 2.0
        ln_ratio = np.log(n_fwd / n_rev) if n_fwd != n_rev else 0.0

        for _ in range(self.max_iterations):
            # BAR self-consistent equation solved via log-sum-exp
            fwd_args = beta * (w_fwd - dg)
            rev_args = beta * (w_rev + dg)

            # log <f(beta(W_fwd - DG))> via log-sum-exp
            log_fwd = _log_sum_exp(-np.logaddexp(0.0, fwd_args)) - np.log(n_fwd)
            log_rev = _log_sum_exp(-np.logaddexp(0.0, rev_args)) - np.log(n_rev)

            dg_new = (log_rev - log_fwd) / beta + ln_ratio / beta + dg

            if abs(dg_new - dg) < self.tolerance:
                return float(dg_new)
            dg = dg_new

        return float(dg)

    def uncertainty(self, w_fwd: np.ndarray, w_rev: np.ndarray, beta: float,
                    dg: Optional[float] = None) -> float:
        """BAR uncertainty from the variance of Fermi function values."""
        w_fwd = np.asarray(w_fwd, dtype=np.float64)
        w_rev = np.asarray(w_rev, dtype=np.float64)
        if dg is None:
            dg = self.solve(w_fwd, w_rev, beta)

        n_fwd = len(w_fwd)
        n_rev = len(w_rev)

        f_fwd = _fermi(beta * (w_fwd - dg))
        f_rev = _fermi(beta * (w_rev + dg))

        var_fwd = np.var(f_fwd) / n_fwd
        var_rev = np.var(f_rev) / n_rev

        denom_fwd = np.mean(f_fwd) ** 2
        denom_rev = np.mean(f_rev) ** 2

        if denom_fwd < 1e-30 or denom_rev < 1e-30:
            return float('inf')

        var_dg = (var_fwd / denom_fwd + var_rev / denom_rev) / (beta ** 2)
        return float(np.sqrt(max(var_dg, 0.0)))


class CrooksEngine:
    """Crooks fluctuation theorem: P(W_fwd)/P(-W_rev) = exp[beta(W-DeltaG)]"""

    def __init__(self):
        self.w_fwd: List[float] = []
        self.w_rev: List[float] = []

    def add_forward_work(self, w) -> None:
        """Add forward work measurement(s)."""
        if np.ndim(w) == 0:
            self.w_fwd.append(float(w))
        else:
            self.w_fwd.extend(np.asarray(w, dtype=np.float64).tolist())

    def add_reverse_work(self, w) -> None:
        """Add reverse work measurement(s)."""
        if np.ndim(w) == 0:
            self.w_rev.append(float(w))
        else:
            self.w_rev.extend(np.asarray(w, dtype=np.float64).tolist())

    def sigma_irr(self, delta_g: float) -> float:
        """Irreversible entropy production: sigma_irr = <W_fwd> + <W_rev> - 2*DeltaG."""
        mean_fwd = np.mean(self.w_fwd)
        mean_rev = np.mean(self.w_rev)
        return float(mean_fwd + mean_rev - 2.0 * delta_g)

    def landauer_bits(self, delta_g: float, temperature: float = 298.15) -> float:
        """Information-theoretic dissipation: sigma_irr / (k_B T ln 2)."""
        s_irr = self.sigma_irr(delta_g)
        return float(s_irr / (K_B * temperature * np.log(2)))

    def crossing_point(self) -> float:
        """Estimate DeltaG from the crossing point of forward and reverse
        work distributions (histogram intersection)."""
        w_f = np.asarray(self.w_fwd)
        w_r = np.asarray(self.w_rev)

        all_w = np.concatenate([w_f, -w_r])
        lo, hi = np.min(all_w), np.max(all_w)
        bins = np.linspace(lo, hi, 100)

        hist_f, _ = np.histogram(w_f, bins=bins, density=True)
        hist_r, _ = np.histogram(-w_r, bins=bins, density=True)

        centers = 0.5 * (bins[:-1] + bins[1:])
        diff = hist_f - hist_r
        crossings = np.where(np.diff(np.sign(diff)))[0]

        if len(crossings) == 0:
            return float(np.mean(all_w))

        idx = crossings[len(crossings) // 2]
        t = diff[idx] / (diff[idx] - diff[idx + 1]) if diff[idx] != diff[idx + 1] else 0.5
        return float(centers[idx] + t * (centers[idx + 1] - centers[idx]))


class ShannonCollapseRate:
    """Track dI/dT = d/dT [S(T_high) + S(T_low) - S_joint(T)].

    Measures the rate at which mutual information between replicas
    collapses as temperature increases.
    """

    def __init__(self):
        self._data: List[Tuple[float, float]] = []  # (T, S)

    def add_point(self, temperature: float, entropy: float) -> None:
        self._data.append((temperature, entropy))

    def compute_didt(self) -> List[Tuple[float, float]]:
        """Return list of (T, dI/dT) using finite differences on stored data."""
        if len(self._data) < 3:
            return []

        data = sorted(self._data, key=lambda x: x[0])
        temps = np.array([d[0] for d in data])
        entropies = np.array([d[1] for d in data])

        s_min = entropies[0]
        s_max = entropies[-1]
        mutual_info = s_min + s_max - entropies

        didt = []
        for i in range(1, len(temps) - 1):
            dt = temps[i + 1] - temps[i - 1]
            if dt > 0:
                di = (mutual_info[i + 1] - mutual_info[i - 1]) / dt
                didt.append((float(temps[i]), float(di)))

        return didt


def temperature_ladder(t_min: float = 298.0, t_max: float = 600.0,
                       n_replicas: int = 8) -> np.ndarray:
    """Geometric temperature ladder: T_i = T_min * (T_max/T_min)^(i/(n-1))."""
    if n_replicas < 2:
        return np.array([t_min])
    ratio = t_max / t_min
    indices = np.arange(n_replicas, dtype=np.float64)
    return t_min * np.power(ratio, indices / (n_replicas - 1))


def vant_hoff_with_dcp(dh_ref: float, ds_ref: float, dcp: float,
                       t: float, t_ref: float = 298.15) -> float:
    """Extended Van't Hoff with heat capacity correction.

    DeltaG(T) = DeltaH_0 + DeltaCp*(T - T_0) - T*[DeltaS_0 + DeltaCp*ln(T/T_0)]
    """
    dh_t = dh_ref + dcp * (t - t_ref)
    ds_t = ds_ref + dcp * np.log(t / t_ref)
    return float(dh_t - t * ds_t)


def shannon_entropy(energies: np.ndarray, temperature: float) -> float:
    """Shannon entropy S = -k_B sum p_i ln p_i with Boltzmann weights.

    Uses log-sum-exp for numerical stability.
    """
    energies = np.asarray(energies, dtype=np.float64)
    beta = 1.0 / (K_B * temperature)

    neg_beta_e = -beta * energies
    log_z = _log_sum_exp(neg_beta_e)

    log_p = neg_beta_e - log_z
    p = np.exp(log_p)
    mask = p > 0
    s = -K_B * np.sum(p[mask] * log_p[mask])
    return float(s)
