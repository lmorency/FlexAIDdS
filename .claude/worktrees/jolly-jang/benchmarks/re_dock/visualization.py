"""RE-DOCK visualization: Crooks distributions, sigma_irr, dI/dT, entropy collapse."""
import numpy as np
from typing import List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _require_mpl():
    if not HAS_MPL:
        raise ImportError("matplotlib is required for RE-DOCK visualization")


def plot_crooks_distributions(w_fwd: np.ndarray, w_rev: np.ndarray,
                              delta_g: float, ax=None,
                              save_path: Optional[str] = None):
    """Plot forward and reverse work distributions with DeltaG crossing."""
    _require_mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    w_fwd = np.asarray(w_fwd)
    w_rev = np.asarray(w_rev)

    all_w = np.concatenate([w_fwd, -w_rev])
    lo = np.min(all_w) - 1
    hi = np.max(all_w) + 1
    bins = np.linspace(lo, hi, 60)

    ax.hist(w_fwd, bins=bins, density=True, alpha=0.6,
            color="#2196F3", label=r"$P(W_{\mathrm{fwd}})$")
    ax.hist(-w_rev, bins=bins, density=True, alpha=0.6,
            color="#F44336", label=r"$P(-W_{\mathrm{rev}})$")
    ax.axvline(delta_g, color="black", linestyle="--", linewidth=1.5,
               label=rf"$\Delta G = {delta_g:.2f}$ kcal/mol")

    ax.set_xlabel("Work (kcal/mol)")
    ax.set_ylabel("Probability density")
    ax.set_title("Crooks Fluctuation Theorem")
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_sigma_irr_scatter(systems: List[str], sigma_irrs: List[float],
                           ax=None, save_path: Optional[str] = None):
    """Scatter plot of irreversible entropy production per system."""
    _require_mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    x = np.arange(len(systems))
    colors = ["#4CAF50" if s < 0.001 else "#FF9800" if s < 0.01 else "#F44336"
              for s in sigma_irrs]

    ax.bar(x, sigma_irrs, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(r"$\sigma_{\mathrm{irr}}$ (kcal/mol)")
    ax.set_title("Irreversible Entropy Production")
    ax.axhline(0.001, color="gray", linestyle=":", label=r"$0.1 k_B$")
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_didt_vs_kd(didts: List[float], kds: List[float],
                    ax=None, save_path: Optional[str] = None):
    """Plot dI/dT vs experimental Kd (log scale)."""
    _require_mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    didts = np.asarray(didts)
    kds = np.asarray(kds)

    ax.scatter(np.log10(kds), didts, c="#673AB7", s=40, edgecolors="black",
               linewidths=0.5)

    if len(kds) > 2:
        log_kd = np.log10(kds)
        m, b = np.polyfit(log_kd, didts, 1)
        x_fit = np.linspace(log_kd.min(), log_kd.max(), 100)
        ax.plot(x_fit, m * x_fit + b, "k--", alpha=0.5,
                label=f"fit: slope={m:.3f}")
        ax.legend()

    ax.set_xlabel(r"$\log_{10}(K_d / \mathrm{M})$")
    ax.set_ylabel(r"$dI/dT$ (kcal/(mol$\cdot$K$^2$))")
    ax.set_title("Shannon Collapse Rate vs Binding Affinity")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_entropy_collapse(temperatures: np.ndarray, entropies: np.ndarray,
                          ax=None, save_path: Optional[str] = None):
    """Plot Shannon entropy S(T) across the temperature ladder."""
    _require_mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    ax.plot(temperatures, entropies, "o-", color="#009688", linewidth=2,
            markersize=6)
    ax.fill_between(temperatures, entropies, alpha=0.15, color="#009688")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$S(T)$ (kcal/(mol$\cdot$K))")
    ax.set_title("Shannon Entropy Collapse")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_bar_vs_vanthoff(bar_dgs: List[float], vh_dgs: List[float],
                         labels: Optional[List[str]] = None,
                         ax=None, save_path: Optional[str] = None):
    """Parity plot: BAR DeltaG vs Van't Hoff DeltaG."""
    _require_mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    bar_dgs = np.asarray(bar_dgs)
    vh_dgs = np.asarray(vh_dgs)

    ax.scatter(vh_dgs, bar_dgs, c="#E91E63", s=50, edgecolors="black",
               linewidths=0.5, zorder=3)

    lo = min(vh_dgs.min(), bar_dgs.min()) - 1
    hi = max(vh_dgs.max(), bar_dgs.max()) + 1
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y = x")

    if labels:
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (vh_dgs[i], bar_dgs[i]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(r"$\Delta G_{\mathrm{VH}}$ (kcal/mol)")
    ax.set_ylabel(r"$\Delta G_{\mathrm{BAR}}$ (kcal/mol)")
    ax.set_title("BAR vs Van't Hoff Free Energies")
    ax.set_aspect("equal")
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax
