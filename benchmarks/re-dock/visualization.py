"""
RE-DOCK Visualization Module (v7)
===================================

Chart.js-based standalone HTML dashboard generation for docking campaign analysis.

Functions
---------
- ``vant_hoff_plot_html``: ln(K) vs 1/T with linear regression, ΔH°/ΔS° annotation
- ``convergence_trace_html``: Running ΔG average per replica over generations
- ``shannon_landscape_html``: S_config heatmap over (temperature, target)
- ``exchange_heatmap_html``: Replica exchange acceptance matrix with time evolution
- ``crooks_crossing_plot_html``: v7 — P(W_fwd) and P(-W_rev) with ΔG intersection
- ``information_loss_plot_html``: v7 — σ_irr and Landauer bits across generations
- ``collapse_rate_plot_html``: v7 — dI/dT landscape showing funnel quality

Each function returns a self-contained HTML string with embedded Chart.js (CDN).

Le Bonhomme Pharma / Najmanovich Research Group
"""

from __future__ import annotations

import json
from typing import List, Dict, Optional, Any

CHART_JS_CDN = "https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="{cdn}"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         margin: 2rem; background: #fafafa; color: #333; }}
  .chart-container {{ max-width: 900px; margin: 1rem auto;
                      background: white; padding: 1.5rem; border-radius: 8px;
                      box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
  h1 {{ text-align: center; color: #1a1a2e; }}
  .subtitle {{ text-align: center; color: #666; font-size: 0.9rem; margin-top: -0.5rem; }}
  canvas {{ max-height: 500px; }}
  .stats {{ font-size: 0.85rem; color: #555; margin-top: 0.5rem; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="subtitle">{subtitle}</p>
<div class="chart-container">
<canvas id="chart"></canvas>
<div class="stats">{stats_html}</div>
</div>
<script>
{chart_script}
</script>
</body>
</html>"""


def vant_hoff_plot_html(
    temperatures: List[float],
    ln_K_values: List[float],
    delta_H_kcal: float,
    delta_S_cal: float,
    r_squared: float,
    target_id: str = "",
) -> str:
    """Generate Van't Hoff plot: ln(K) vs 1/T with linear fit overlay.

    Parameters
    ----------
    temperatures : List[float]
        Temperatures in Kelvin.
    ln_K_values : List[float]
        Experimental ln(K_d) values.
    delta_H_kcal : float
        Fitted ΔH° in kcal/mol.
    delta_S_cal : float
        Fitted ΔS° in cal/(mol·K).
    r_squared : float
        Coefficient of determination.
    target_id : str
        PDB ID for labeling.

    Returns
    -------
    str
        Self-contained HTML.
    """
    inv_T = [1.0 / T for T in temperatures]

    # Linear fit line
    R = 1.987204e-3
    fit_ln_K = [(-delta_H_kcal / R) * (1.0 / T) + (delta_S_cal / 1000.0 / R)
                for T in temperatures]

    data_points = json.dumps([{"x": x, "y": y} for x, y in zip(inv_T, ln_K_values)])
    fit_points = json.dumps([{"x": x, "y": y} for x, y in zip(inv_T, fit_ln_K)])

    script = f"""
const ctx = document.getElementById('chart').getContext('2d');
new Chart(ctx, {{
  type: 'scatter',
  data: {{
    datasets: [
      {{
        label: 'Data',
        data: {data_points},
        backgroundColor: 'rgba(220, 53, 69, 0.8)',
        pointRadius: 6,
      }},
      {{
        label: 'Linear fit (R²={r_squared:.4f})',
        data: {fit_points},
        type: 'line',
        borderColor: 'rgba(40, 167, 69, 0.8)',
        borderWidth: 2,
        pointRadius: 0,
        fill: false,
      }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{
      title: {{ display: true, text: "Van\\'t Hoff Plot — {target_id}" }},
    }},
    scales: {{
      x: {{ title: {{ display: true, text: '1/T (K⁻¹)' }} }},
      y: {{ title: {{ display: true, text: 'ln(K)' }} }},
    }}
  }}
}});
"""

    stats = (f"ΔH° = {delta_H_kcal:.2f} kcal/mol | "
             f"ΔS° = {delta_S_cal:.1f} cal/(mol·K) | "
             f"R² = {r_squared:.4f}")

    return HTML_TEMPLATE.format(
        title=f"Van't Hoff Analysis — {target_id}",
        subtitle="RE-DOCK Replica Exchange Distributed Docking",
        cdn=CHART_JS_CDN,
        stats_html=stats,
        chart_script=script,
    )


def convergence_trace_html(
    generations: List[int],
    energies_by_replica: Dict[str, List[float]],
    target_id: str = "",
) -> str:
    """Generate convergence trace: running ΔG per replica over generations.

    Parameters
    ----------
    generations : List[int]
        Generation indices.
    energies_by_replica : Dict[str, List[float]]
        Replica label → energy trace (one value per generation).
    target_id : str
        PDB ID for labeling.

    Returns
    -------
    str
        Self-contained HTML.
    """
    colors = [
        "rgba(220,53,69,0.8)", "rgba(255,127,14,0.8)", "rgba(44,160,44,0.8)",
        "rgba(31,119,180,0.8)", "rgba(148,103,189,0.8)", "rgba(140,86,75,0.8)",
        "rgba(227,119,194,0.8)", "rgba(127,127,127,0.8)",
    ]

    datasets = []
    for i, (label, energies) in enumerate(energies_by_replica.items()):
        color = colors[i % len(colors)]
        data = json.dumps([{"x": g, "y": e} for g, e in zip(generations, energies)])
        datasets.append(f"""{{
          label: '{label}',
          data: {data},
          borderColor: '{color}',
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false,
        }}""")

    datasets_str = ",\n      ".join(datasets)

    script = f"""
const ctx = document.getElementById('chart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{
    datasets: [
      {datasets_str}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{
      title: {{ display: true, text: 'Convergence Trace — {target_id}' }},
    }},
    scales: {{
      x: {{ type: 'linear', title: {{ display: true, text: 'Generation' }} }},
      y: {{ title: {{ display: true, text: 'ΔG (kcal/mol)' }} }},
    }}
  }}
}});
"""

    return HTML_TEMPLATE.format(
        title=f"Convergence Trace — {target_id}",
        subtitle="Running average ΔG per temperature replica",
        cdn=CHART_JS_CDN,
        stats_html=f"{len(energies_by_replica)} replicas, {len(generations)} generations",
        chart_script=script,
    )


def shannon_landscape_html(
    temperatures: List[float],
    target_ids: List[str],
    entropy_matrix: List[List[float]],
) -> str:
    """Generate Shannon entropy heatmap over (temperature, target).

    Parameters
    ----------
    temperatures : List[float]
        Temperature values (columns).
    target_ids : List[str]
        Target PDB IDs (rows).
    entropy_matrix : List[List[float]]
        S_config[target][temperature] in kcal/(mol·K).

    Returns
    -------
    str
        Self-contained HTML with a color-coded table (Chart.js matrix plugin
        not widely available, so we use an HTML table with inline styles).
    """
    # Build HTML table heatmap
    def _color(val: float, vmin: float, vmax: float) -> str:
        if vmax == vmin:
            t = 0.5
        else:
            t = (val - vmin) / (vmax - vmin)
        # Blue (cold/low entropy) to Red (hot/high entropy)
        r = int(255 * t)
        b = int(255 * (1 - t))
        return f"rgb({r}, 80, {b})"

    all_vals = [v for row in entropy_matrix for v in row if v == v]  # skip NaN
    vmin = min(all_vals) if all_vals else 0
    vmax = max(all_vals) if all_vals else 1

    rows_html = ""
    for i, tid in enumerate(target_ids):
        cells = ""
        for j, T in enumerate(temperatures):
            val = entropy_matrix[i][j]
            color = _color(val, vmin, vmax)
            cells += (f'<td style="background:{color}; color:white; '
                      f'text-align:center; padding:8px; font-size:0.8rem;">'
                      f'{val:.4f}</td>')
        rows_html += f"<tr><td style='padding:8px; font-weight:bold;'>{tid}</td>{cells}</tr>"

    header_cells = "".join(f"<th style='padding:8px;'>{T:.0f}K</th>" for T in temperatures)

    table_html = f"""
<table style="border-collapse:collapse; margin:1rem auto; font-family:monospace;">
<tr><th></th>{header_cells}</tr>
{rows_html}
</table>
"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Shannon Entropy Landscape</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 2rem; background: #fafafa; }}
  h1 {{ text-align: center; }}
  .subtitle {{ text-align: center; color: #666; font-size: 0.9rem; }}
</style>
</head><body>
<h1>Shannon Entropy Landscape</h1>
<p class="subtitle">S_config (kcal/mol·K) by target and temperature — RE-DOCK</p>
{table_html}
<p style="text-align:center; font-size:0.8rem; color:#999;">
Blue = low entropy | Red = high entropy
</p>
</body></html>"""

    return html


def exchange_heatmap_html(
    exchange_history: List[dict],
    n_replicas: int = 8,
) -> str:
    """Generate replica exchange acceptance heatmap.

    Parameters
    ----------
    exchange_history : List[dict]
        List of exchange records from BenchmarkCampaign.
    n_replicas : int
        Number of replicas.

    Returns
    -------
    str
        Self-contained HTML with acceptance rate matrix.
    """
    # Build acceptance matrix
    attempts = [[0] * n_replicas for _ in range(n_replicas)]
    accepted = [[0] * n_replicas for _ in range(n_replicas)]

    for record in exchange_history:
        for ex in record.get("exchanges", []):
            i, j = ex["i"], ex["j"]
            attempts[i][j] += 1
            attempts[j][i] += 1
            if ex["accepted"]:
                accepted[i][j] += 1
                accepted[j][i] += 1

    rates = []
    for i in range(n_replicas):
        row = []
        for j in range(n_replicas):
            if attempts[i][j] > 0:
                row.append(accepted[i][j] / attempts[i][j])
            else:
                row.append(0.0)
        rates.append(row)

    # Build HTML table
    def _color(val: float) -> str:
        if val == 0:
            return "rgb(240,240,240)"
        r = int(255 * (1 - val))
        g = int(200 * val)
        return f"rgb({r}, {g}, 80)"

    rows_html = ""
    for i in range(n_replicas):
        cells = ""
        for j in range(n_replicas):
            color = _color(rates[i][j])
            txt = f"{rates[i][j]:.2f}" if attempts[i][j] > 0 else "—"
            cells += (f'<td style="background:{color}; text-align:center; '
                      f'padding:10px; font-size:0.85rem;">{txt}</td>')
        rows_html += f"<tr><td style='padding:8px; font-weight:bold;'>R{i}</td>{cells}</tr>"

    header = "".join(f"<th style='padding:8px;'>R{j}</th>" for j in range(n_replicas))

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Exchange Acceptance Heatmap</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 2rem; background: #fafafa; }}
  h1 {{ text-align: center; }}
  .subtitle {{ text-align: center; color: #666; font-size: 0.9rem; }}
  table {{ border-collapse: collapse; margin: 1rem auto; }}
</style>
</head><body>
<h1>Replica Exchange Acceptance Rates</h1>
<p class="subtitle">Metropolis acceptance between adjacent temperature replicas — RE-DOCK</p>
<table>
<tr><th></th>{header}</tr>
{rows_html}
</table>
<p style="text-align:center; font-size:0.8rem; color:#999;">
Total exchange rounds: {len(exchange_history)}
</p>
</body></html>"""

    return html


# ---------------------------------------------------------------------------
# v7: Crooks crossing plot
# ---------------------------------------------------------------------------

def crooks_crossing_plot_html(
    w_fwd: List[float],
    w_rev: List[float],
    delta_G_bar: float,
    delta_G_crooks: float,
    target_id: str = "",
    n_bins: int = 60,
) -> str:
    """Generate Crooks crossing plot: P(W_fwd) and P(-W_rev) with ΔG intersection.

    Parameters
    ----------
    w_fwd : List[float]
        Forward work values (kcal/mol).
    w_rev : List[float]
        Reverse work values (kcal/mol).
    delta_G_bar : float
        BAR estimate of ΔG (kcal/mol).
    delta_G_crooks : float
        Crooks intersection ΔG (kcal/mol).
    target_id : str
        PDB ID for labeling.
    n_bins : int
        Histogram bins.

    Returns
    -------
    str
        Self-contained HTML.
    """
    import numpy as _np

    w_f = _np.array(w_fwd)
    neg_w_r = -_np.array(w_rev)

    all_w = _np.concatenate([w_f, neg_w_r])
    w_min, w_max = float(_np.min(all_w)), float(_np.max(all_w))
    margin = 0.1 * (w_max - w_min) if w_max > w_min else 1.0
    edges = _np.linspace(w_min - margin, w_max + margin, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    hist_fwd, _ = _np.histogram(w_f, bins=edges, density=True)
    hist_rev, _ = _np.histogram(neg_w_r, bins=edges, density=True)

    labels = json.dumps([f"{c:.2f}" for c in centers])
    fwd_data = json.dumps([float(v) for v in hist_fwd])
    rev_data = json.dumps([float(v) for v in hist_rev])

    script = f"""
const ctx = document.getElementById('chart').getContext('2d');
new Chart(ctx, {{
  type: 'bar',
  data: {{
    labels: {labels},
    datasets: [
      {{
        label: 'P(W_fwd)',
        data: {fwd_data},
        backgroundColor: 'rgba(220, 53, 69, 0.5)',
        borderColor: 'rgba(220, 53, 69, 0.8)',
        borderWidth: 1,
      }},
      {{
        label: 'P(-W_rev)',
        data: {rev_data},
        backgroundColor: 'rgba(40, 167, 69, 0.5)',
        borderColor: 'rgba(40, 167, 69, 0.8)',
        borderWidth: 1,
      }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{
      title: {{ display: true, text: 'Crooks Crossing — {target_id}' }},
    }},
    scales: {{
      x: {{ title: {{ display: true, text: 'Work W (kcal/mol)' }} }},
      y: {{ title: {{ display: true, text: 'Probability density' }} }},
    }}
  }}
}});
"""

    stats = (f"\\u0394G_BAR = {delta_G_bar:.3f} kcal/mol | "
             f"\\u0394G_Crooks = {delta_G_crooks:.3f} kcal/mol | "
             f"n_fwd = {len(w_fwd)} | n_rev = {len(w_rev)}")

    return HTML_TEMPLATE.format(
        title=f"Crooks Crossing Plot — {target_id}",
        subtitle="Forward/reverse work distributions with \\u0394G intersection — RE-DOCK v7",
        cdn=CHART_JS_CDN,
        stats_html=stats,
        chart_script=script,
    )


# ---------------------------------------------------------------------------
# v7: Information loss plot
# ---------------------------------------------------------------------------

def information_loss_plot_html(
    generations: List[int],
    sigma_irr_values: List[float],
    bits_lost_values: List[float],
    target_id: str = "",
) -> str:
    """Generate information loss plot: sigma_irr and bits lost over generations.

    Parameters
    ----------
    generations : List[int]
        Generation indices.
    sigma_irr_values : List[float]
        Irreversible entropy production at each generation.
    bits_lost_values : List[float]
        Landauer bits lost at each generation.
    target_id : str
        PDB ID for labeling.

    Returns
    -------
    str
        Self-contained HTML.
    """
    sigma_data = json.dumps([{"x": g, "y": s} for g, s in zip(generations, sigma_irr_values)])
    bits_data = json.dumps([{"x": g, "y": b} for g, b in zip(generations, bits_lost_values)])

    script = f"""
const ctx = document.getElementById('chart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{
    datasets: [
      {{
        label: '\\u03c3_irr (kcal/mol\\u00b7K)',
        data: {sigma_data},
        borderColor: 'rgba(220, 53, 69, 0.8)',
        backgroundColor: 'rgba(220, 53, 69, 0.1)',
        borderWidth: 2,
        pointRadius: 3,
        fill: true,
        yAxisID: 'y',
      }},
      {{
        label: 'Bits lost (Landauer)',
        data: {bits_data},
        borderColor: 'rgba(31, 119, 180, 0.8)',
        backgroundColor: 'rgba(31, 119, 180, 0.1)',
        borderWidth: 2,
        pointRadius: 3,
        fill: true,
        yAxisID: 'y1',
      }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{
      title: {{ display: true, text: 'Information Loss — {target_id}' }},
    }},
    scales: {{
      x: {{ type: 'linear', title: {{ display: true, text: 'Generation' }} }},
      y: {{ type: 'linear', position: 'left',
            title: {{ display: true, text: '\\u03c3_irr (kcal/mol\\u00b7K)' }} }},
      y1: {{ type: 'linear', position: 'right',
             title: {{ display: true, text: 'Bits lost' }},
             grid: {{ drawOnChartArea: false }} }},
    }}
  }}
}});
"""

    stats = (f"Final \\u03c3_irr = {sigma_irr_values[-1]:.4f} kcal/(mol\\u00b7K) | "
             f"Final bits lost = {bits_lost_values[-1]:.2f}" if sigma_irr_values else "No data")

    return HTML_TEMPLATE.format(
        title=f"Information Loss — {target_id}",
        subtitle="Irreversible entropy production and Landauer bits over generations — RE-DOCK v7",
        cdn=CHART_JS_CDN,
        stats_html=stats,
        chart_script=script,
    )


# ---------------------------------------------------------------------------
# v7: Collapse rate plot
# ---------------------------------------------------------------------------

def collapse_rate_plot_html(
    temperatures: List[float],
    collapse_rates: List[float],
    target_id: str = "",
) -> str:
    """Generate Shannon Energy Collapse rate plot: dS/dT vs temperature.

    Positive dS/dT: landscape fracturing (more states at higher T).
    Negative dS/dT: landscape funneling (fewer dominant states).

    Parameters
    ----------
    temperatures : List[float]
        Temperature values (K).
    collapse_rates : List[float]
        dS/dT at each temperature (kcal/(mol*K^2)).
    target_id : str
        PDB ID for labeling.

    Returns
    -------
    str
        Self-contained HTML.
    """
    data_points = json.dumps([{"x": T, "y": r} for T, r in zip(temperatures, collapse_rates)])

    script = f"""
const ctx = document.getElementById('chart').getContext('2d');
new Chart(ctx, {{
  type: 'scatter',
  data: {{
    datasets: [
      {{
        label: 'dS/dT (collapse rate)',
        data: {data_points},
        backgroundColor: function(context) {{
          const val = context.raw ? context.raw.y : 0;
          return val >= 0 ? 'rgba(220, 53, 69, 0.8)' : 'rgba(40, 167, 69, 0.8)';
        }},
        pointRadius: 8,
        showLine: true,
        borderColor: 'rgba(100, 100, 100, 0.5)',
        borderWidth: 1.5,
        fill: false,
      }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{
      title: {{ display: true, text: 'Shannon Energy Collapse Rate — {target_id}' }},
    }},
    scales: {{
      x: {{ title: {{ display: true, text: 'Temperature (K)' }} }},
      y: {{ title: {{ display: true, text: 'dS/dT (kcal/(mol*K^2))' }} }},
    }}
  }}
}});
"""

    n_funnel = sum(1 for r in collapse_rates if r < 0)
    n_fracture = sum(1 for r in collapse_rates if r >= 0)
    stats = (f"Funneling (dS/dT < 0): {n_funnel} rungs | "
             f"Fracturing (dS/dT >= 0): {n_fracture} rungs")

    return HTML_TEMPLATE.format(
        title=f"Shannon Energy Collapse Rate — {target_id}",
        subtitle="dS/dT landscape: green=funnel, red=fracture — RE-DOCK v7",
        cdn=CHART_JS_CDN,
        stats_html=stats,
        chart_script=script,
    )
