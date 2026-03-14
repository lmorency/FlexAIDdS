"""ITC-style thermogram comparison for FlexAID∆S (Phase 3, deliverable 3.4).

Generates enthalpy-entropy compensation plots and free energy bar charts
comparing FlexAID∆S predictions with experimental ITC data.

Usage:
    PyMOL> flexaids_load_results /path/to/output
    PyMOL> flexaids_itc_plot
    PyMOL> flexaids_itc_compare /path/to/itc_data.csv
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from pymol import cmd
except ImportError as exc:
    raise ImportError("PyMOL not available") from exc


def _parse_itc_csv(csv_path: str) -> Dict[str, float]:
    """Parse a simple ITC experimental data CSV file.

    Expected format (header + data row):
        dG,dH,TdS
        -8.5,-12.3,3.8

    All values in kcal/mol.

    Returns:
        Dictionary with keys 'dG', 'dH', 'TdS'.
    """
    import csv

    result: Dict[str, float] = {}
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for key in ("dG", "dH", "TdS"):
                if key in row and row[key]:
                    result[key] = float(row[key])
            break  # Only read the first data row
    return result


def plot_enthalpy_entropy_compensation(
    output_png: str = "",
) -> None:
    """Plot ΔH vs -TΔS compensation for all loaded binding modes.

    Each binding mode is a scatter point.  If experimental ITC data
    has been loaded via ``flexaids_itc_compare``, it is overlaid as
    a red star.

    Args:
        output_png: Path to save the PNG image.  If empty, uses a temp file
                    and displays it in PyMOL.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib is required for ITC plots. "
              "Install with: pip install matplotlib")
        return

    from . import results_adapter

    result = results_adapter._loaded_result
    if result is None:
        print("ERROR: No results loaded. Use 'flexaids_load_results' first.")
        return

    enthalpies = []
    entropy_terms = []
    labels = []

    temperature = result.temperature or 300.0

    for mode in result.binding_modes:
        if mode.enthalpy is None or mode.entropy is None:
            continue
        h = mode.enthalpy
        t_ds = mode.entropy * (mode.temperature or temperature)
        enthalpies.append(h)
        entropy_terms.append(-t_ds)  # -TΔS convention
        labels.append(f"M{mode.mode_id}")

    if not enthalpies:
        print("ERROR: No modes with enthalpy/entropy data available.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.scatter(enthalpies, entropy_terms, c="steelblue", s=60, zorder=3,
               label="FlexAID∆S modes")

    for i, label in enumerate(labels):
        ax.annotate(label, (enthalpies[i], entropy_terms[i]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=8, color="grey")

    ax.set_xlabel("ΔH (kcal/mol)", fontsize=12)
    ax.set_ylabel("-TΔS (kcal/mol)", fontsize=12)
    ax.set_title("Enthalpy-Entropy Compensation", fontsize=14)

    # Diagonal line: ΔG = ΔH + (-TΔS) = constant
    h_range = max(enthalpies) - min(enthalpies) if len(enthalpies) > 1 else 10.0
    x_line = [min(enthalpies) - 0.2 * h_range, max(enthalpies) + 0.2 * h_range]
    for mode in result.binding_modes:
        if mode.free_energy is not None:
            dg = mode.free_energy
            y_line = [dg - x for x in x_line]
            ax.plot(x_line, y_line, "--", color="lightgrey", linewidth=0.8)
            break

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if not output_png:
        output_png = os.path.join(tempfile.gettempdir(), "flexaids_itc_comp.png")

    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Enthalpy-entropy compensation plot saved: {output_png}")

    # Try to display in PyMOL
    try:
        cmd.load(output_png, "itc_compensation")
        print("Plot loaded into PyMOL as 'itc_compensation'.")
    except Exception:
        pass


def plot_free_energy_comparison(
    itc_csv: str = "",
    output_png: str = "",
) -> None:
    """Plot predicted vs experimental free energy comparison.

    Two-panel figure:
    - Panel A: ΔH vs -TΔS compensation with ITC overlay
    - Panel B: ΔG bar chart comparing prediction vs experiment

    Args:
        itc_csv: Path to ITC experimental data CSV.
        output_png: Path to save the PNG image.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib is required. Install with: pip install matplotlib")
        return

    from . import results_adapter

    result = results_adapter._loaded_result
    if result is None:
        print("ERROR: No results loaded. Use 'flexaids_load_results' first.")
        return

    itc_data: Dict[str, float] = {}
    if itc_csv:
        try:
            itc_data = _parse_itc_csv(itc_csv)
        except Exception as exc:
            print(f"WARNING: Could not parse ITC data: {exc}")

    top = result.top_mode()
    if top is None:
        print("ERROR: No binding modes available.")
        return

    temperature = result.temperature or 300.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Enthalpy-entropy compensation
    ax = axes[0]
    enthalpies = []
    entropy_terms = []
    for mode in result.binding_modes:
        if mode.enthalpy is not None and mode.entropy is not None:
            h = mode.enthalpy
            t_ds = mode.entropy * (mode.temperature or temperature)
            enthalpies.append(h)
            entropy_terms.append(-t_ds)

    if enthalpies:
        ax.scatter(enthalpies, entropy_terms, c="steelblue", s=50,
                   label="FlexAID∆S", zorder=3)

    if "dH" in itc_data and "TdS" in itc_data:
        ax.scatter([itc_data["dH"]], [-itc_data["TdS"]],
                   c="red", marker="*", s=250, zorder=4, label="ITC (exp.)")

    ax.set_xlabel("ΔH (kcal/mol)")
    ax.set_ylabel("-TΔS (kcal/mol)")
    ax.set_title("A) Enthalpy-Entropy Compensation")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Panel B: ΔG bar chart
    ax2 = axes[1]
    bar_labels = ["Predicted"]
    bar_values = [top.free_energy if top.free_energy is not None else 0.0]
    bar_colors = ["steelblue"]

    if "dG" in itc_data:
        bar_labels.append("Experimental")
        bar_values.append(itc_data["dG"])
        bar_colors.append("firebrick")

    bars = ax2.bar(bar_labels, bar_values, color=bar_colors, width=0.5)
    for bar, val in zip(bars, bar_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    ax2.set_ylabel("ΔG (kcal/mol)")
    ax2.set_title("B) Free Energy Comparison")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()

    if not output_png:
        output_png = os.path.join(tempfile.gettempdir(), "flexaids_itc_full.png")

    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"ITC comparison plot saved: {output_png}")

    try:
        cmd.load(output_png, "itc_comparison")
        print("Plot loaded into PyMOL as 'itc_comparison'.")
    except Exception:
        pass
