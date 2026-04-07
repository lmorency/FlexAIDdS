"""RE-DOCK orchestrator: bidirectional protocol and benchmark tiers."""
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .thermodynamics import (
    K_B, BennettAcceptanceRatio, CrooksEngine, ShannonCollapseRate,
    temperature_ladder, vant_hoff_with_dcp, shannon_entropy,
)


class Tier(Enum):
    """Benchmark tier definitions."""
    TIER0_1STP = 0       # Single system (1STP) validation
    TIER1_VALIDATION = 1  # Small validation set
    TIER2_ASTEX85 = 2     # Full Astex diverse set (85 systems)


TIER_SYSTEMS = {
    Tier.TIER0_1STP: ["1STP"],
    Tier.TIER1_VALIDATION: ["1STP", "1HVR", "1ABE", "2CPP", "1FKI"],
    Tier.TIER2_ASTEX85: [f"ASTEX_{i:03d}" for i in range(85)],
}


@dataclass
class REDOCKResult:
    """Result from a single RE-DOCK run."""
    pdb_id: str
    tier: Tier
    delta_g_bar: float
    delta_g_bar_err: float
    delta_g_vanthoff: float
    sigma_irr: float
    landauer_bits: float
    temperatures: np.ndarray
    shannon_entropies: np.ndarray
    w_fwd: np.ndarray
    w_rev: np.ndarray
    converged: bool
    didt: List[tuple] = field(default_factory=list)


class BidirectionalProtocol:
    """Bidirectional nonequilibrium work protocol for RE-DOCK.

    Forward: anneal from T_min -> T_max, accumulate work.
    Reverse: anneal from T_max -> T_min, accumulate work.
    """

    def __init__(self, n_replicas: int = 8, t_min: float = 298.0,
                 t_max: float = 600.0, bar_max_iter: int = 200,
                 bar_tol: float = 1e-8):
        self.n_replicas = n_replicas
        self.t_min = t_min
        self.t_max = t_max
        self.bar = BennettAcceptanceRatio(bar_max_iter, bar_tol)
        self.crooks = CrooksEngine()

    def forward_leg(self, temperatures: np.ndarray,
                    energies: np.ndarray) -> np.ndarray:
        """Compute forward work along the temperature ladder.

        W_fwd = sum_i (beta_{i+1} - beta_i) * E_i
        """
        temperatures = np.asarray(temperatures, dtype=np.float64)
        energies = np.asarray(energies, dtype=np.float64)
        n_reps = len(temperatures)
        n_samples = energies.shape[1]

        w_fwd = np.zeros(n_samples)
        for i in range(n_reps - 1):
            beta_curr = 1.0 / (K_B * temperatures[i])
            beta_next = 1.0 / (K_B * temperatures[i + 1])
            w_fwd += (beta_next - beta_curr) * energies[i]

        return w_fwd

    def reverse_leg(self, temperatures: np.ndarray,
                    energies: np.ndarray) -> np.ndarray:
        """Compute reverse work along the temperature ladder (descending)."""
        temperatures = np.asarray(temperatures, dtype=np.float64)
        energies = np.asarray(energies, dtype=np.float64)
        n_reps = len(temperatures)
        n_samples = energies.shape[1]

        w_rev = np.zeros(n_samples)
        for i in range(n_reps - 1, 0, -1):
            beta_curr = 1.0 / (K_B * temperatures[i])
            beta_prev = 1.0 / (K_B * temperatures[i - 1])
            w_rev += (beta_prev - beta_curr) * energies[i]

        return w_rev

    def check_convergence(self, sigma_irr: float,
                          threshold: float = 0.1 * K_B) -> bool:
        """Check if irreversible entropy production is below threshold."""
        return abs(sigma_irr) < threshold

    def run_system(self, pdb_id: str, tier: Tier,
                   energies_fwd: Optional[np.ndarray] = None,
                   energies_rev: Optional[np.ndarray] = None,
                   dh_ref: float = -10.0, ds_ref: float = -0.02,
                   dcp: float = -0.3) -> REDOCKResult:
        """Run full RE-DOCK protocol for a single system."""
        temps = temperature_ladder(self.t_min, self.t_max, self.n_replicas)
        n_samples = 500

        rng = np.random.default_rng(hash(pdb_id) % (2**32))

        if energies_fwd is None:
            energies_fwd = np.array([
                rng.normal(loc=-8.0 + 0.01 * t, scale=2.0, size=n_samples)
                for t in temps
            ])
        if energies_rev is None:
            energies_rev = np.array([
                rng.normal(loc=-8.0 + 0.01 * t, scale=2.0, size=n_samples)
                for t in temps
            ])

        w_fwd = self.forward_leg(temps, energies_fwd)
        w_rev = self.reverse_leg(temps, energies_rev)

        beta_ref = 1.0 / (K_B * self.t_min)
        dg_bar = self.bar.solve(w_fwd, w_rev, beta_ref)
        dg_bar_err = self.bar.uncertainty(w_fwd, w_rev, beta_ref, dg_bar)

        self.crooks = CrooksEngine()
        self.crooks.add_forward_work(w_fwd)
        self.crooks.add_reverse_work(w_rev)

        sig_irr = self.crooks.sigma_irr(dg_bar)
        l_bits = self.crooks.landauer_bits(dg_bar, self.t_min)

        dg_vh = vant_hoff_with_dcp(dh_ref, ds_ref, dcp, self.t_min)

        shannon_s = np.array([
            shannon_entropy(energies_fwd[i], temps[i])
            for i in range(len(temps))
        ])

        collapse = ShannonCollapseRate()
        for i, t in enumerate(temps):
            collapse.add_point(t, shannon_s[i])
        didt = collapse.compute_didt()

        converged = self.check_convergence(sig_irr)

        return REDOCKResult(
            pdb_id=pdb_id,
            tier=tier,
            delta_g_bar=dg_bar,
            delta_g_bar_err=dg_bar_err,
            delta_g_vanthoff=dg_vh,
            sigma_irr=sig_irr,
            landauer_bits=l_bits,
            temperatures=temps,
            shannon_entropies=shannon_s,
            w_fwd=w_fwd,
            w_rev=w_rev,
            converged=converged,
            didt=didt,
        )
