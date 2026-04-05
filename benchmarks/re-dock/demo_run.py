"""
RE-DOCK Demo Run
=================

Simulated distributed docking campaign with synthetic thermodynamics.

Generates realistic mock data for demonstration and testing using the
extended Van't Hoff model:

    ΔG(T) = ΔH + ΔCp*(T - Tref) - T*(ΔS + ΔCp*ln(T/Tref))

Runs 8-replica temperature ladder with Metropolis exchanges and
produces Van't Hoff analysis + Chart.js HTML dashboards.

Usage::

    python -m benchmarks.re_dock.demo_run [--output-dir ./demo_campaign]

Le Bonhomme Pharma / Najmanovich Research Group
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from thermodynamics import (
    geometric_temperature_ladder,
    DockingPose,
    ReplicaState,
    van_t_hoff_analysis,
    shannon_entropy_of_ensemble,
    attempt_exchanges,
    R_KCAL,
)
from visualization import (
    vant_hoff_plot_html,
    convergence_trace_html,
    shannon_landscape_html,
    exchange_heatmap_html,
)


# ---------------------------------------------------------------------------
# Synthetic target parameters
# ---------------------------------------------------------------------------

SYNTHETIC_TARGETS = [
    {"pdb_id": "1a30", "name": "HIV-1 protease",
     "dH": -12.5, "dS": -8.0, "dCp": -350.0},
    {"pdb_id": "1err", "name": "Estrogen receptor",
     "dH": -15.2, "dS": -18.5, "dCp": -280.0},
    {"pdb_id": "2bm2", "name": "Factor Xa",
     "dH": -9.8, "dS": -2.0, "dCp": -200.0},
    {"pdb_id": "3htb", "name": "Thrombin",
     "dH": -7.5, "dS": 3.5, "dCp": -150.0},
    {"pdb_id": "4djh", "name": "Acetylcholinesterase",
     "dH": -18.0, "dS": -22.0, "dCp": -420.0},
]


def synthetic_dG(T: float, dH: float, dS_cal: float, dCp_cal: float,
                 T_ref: float = 298.15) -> float:
    """Extended Van't Hoff ΔG(T) with heat capacity correction.

    ΔG(T) = ΔH + ΔCp*(T-Tref) - T*(ΔS + ΔCp*ln(T/Tref))

    All enthalpy in kcal/mol, entropy in cal/(mol·K).
    """
    dS_kcal = dS_cal / 1000.0
    dCp_kcal = dCp_cal / 1000.0
    return dH + dCp_kcal * (T - T_ref) - T * (dS_kcal + dCp_kcal * math.log(T / T_ref))


def run_demo(output_dir: str = "./demo_campaign", n_generations: int = 20) -> None:
    """Execute a full simulated campaign."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    n_replicas = 8
    T_min, T_max = 298.0, 600.0
    temperatures = geometric_temperature_ladder(T_min, T_max, n_replicas)
    pop_size = 100

    print("RE-DOCK Demo Campaign")
    print(f"  Temperatures: {', '.join(f'{T:.1f}K' for T in temperatures)}")
    print(f"  Targets: {len(SYNTHETIC_TARGETS)}")
    print(f"  Generations: {n_generations}")
    print()

    all_exchange_history = []
    all_results = {}

    for target in SYNTHETIC_TARGETS:
        pid = target["pdb_id"]
        dH = target["dH"]
        dS = target["dS"]
        dCp = target["dCp"]

        print(f"Target {pid} ({target['name']}):")

        # Initialize replicas
        replicas = [
            ReplicaState(replica_index=i, temperature=temperatures[i])
            for i in range(n_replicas)
        ]

        # Track convergence
        convergence = {f"T={T:.0f}K": [] for T in temperatures}

        for gen in range(n_generations):
            # Generate synthetic poses at each temperature
            for i, replica in enumerate(replicas):
                T = replica.temperature
                dG_true = synthetic_dG(T, dH, dS, dCp)

                # Boltzmann-distributed energies around true ΔG
                sigma = math.sqrt(R_KCAL * T) * 2.0  # thermal fluctuation
                energies = rng.normal(dG_true, sigma, size=pop_size)

                for e in energies:
                    replica.add_pose(DockingPose(
                        energy_kcal=float(e),
                        generation=gen,
                        replica_index=i,
                    ))

                convergence[f"T={T:.0f}K"].append(float(np.mean(energies)))

            # Replica exchange
            even_odd = gen % 2
            results = attempt_exchanges(replicas, even_odd=even_odd, rng=rng)
            all_exchange_history.append({
                "target_id": pid,
                "generation": gen,
                "exchanges": [
                    {"i": i, "j": j, "accepted": acc, "delta": float(d)}
                    for i, j, acc, d in results
                ],
            })

        # Van't Hoff analysis
        temps_for_vhoff = []
        dGs_for_vhoff = []
        for replica in replicas:
            energies = [p.energy_kcal for p in replica.poses]
            beta = replica.beta
            E_arr = np.array(energies)
            max_neg = np.max(-beta * E_arr)
            log_Z = max_neg + np.log(np.sum(np.exp(-beta * E_arr - max_neg)))
            dG = -log_Z / beta
            temps_for_vhoff.append(replica.temperature)
            dGs_for_vhoff.append(float(dG))

        vr = van_t_hoff_analysis(temps_for_vhoff, dGs_for_vhoff, fit_dCp=True)
        dG_true_298 = synthetic_dG(298.15, dH, dS, dCp)
        print(f"  ΔH° = {vr.delta_H_kcal:.2f} (true: {dH:.2f}) kcal/mol")
        print(f"  ΔS° = {vr.delta_S_cal:.1f} (true: {dS:.1f}) cal/(mol·K)")
        print(f"  ΔG° = {vr.delta_G_kcal:.2f} (true: {dG_true_298:.2f}) kcal/mol")
        print(f"  ΔCp = {vr.delta_Cp_cal:.0f} (true: {dCp:.0f}) cal/(mol·K)")
        print(f"  R²  = {vr.r_squared:.4f}")

        # Shannon entropy
        S_by_T = []
        for replica in replicas:
            energies = [p.energy_kcal for p in replica.poses]
            S = shannon_entropy_of_ensemble(energies, replica.temperature)
            S_by_T.append(S)
        print(f"  S_config range: [{min(S_by_T):.4f}, {max(S_by_T):.4f}] kcal/(mol·K)")

        # Exchange acceptance
        target_exchanges = [r for r in all_exchange_history if r["target_id"] == pid]
        total_ex = sum(len(r["exchanges"]) for r in target_exchanges)
        total_acc = sum(
            sum(1 for e in r["exchanges"] if e["accepted"])
            for r in target_exchanges
        )
        print(f"  Exchanges: {total_acc}/{total_ex} accepted "
              f"({100*total_acc/total_ex:.0f}%)" if total_ex > 0 else "")
        print()

        all_results[pid] = {
            "vant_hoff": vr,
            "convergence": convergence,
            "replicas": replicas,
            "shannon": S_by_T,
        }

    # -----------------------------------------------------------------------
    # Generate HTML dashboards
    # -----------------------------------------------------------------------
    viz_dir = out / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    for pid, res in all_results.items():
        vr = res["vant_hoff"]

        # Van't Hoff plot
        html = vant_hoff_plot_html(
            vr.temperatures, vr.ln_K_values,
            vr.delta_H_kcal, vr.delta_S_cal, vr.r_squared,
            target_id=pid,
        )
        (viz_dir / f"vanthoff_{pid}.html").write_text(html)

        # Convergence trace
        generations = list(range(n_generations))
        html = convergence_trace_html(
            generations, res["convergence"], target_id=pid,
        )
        (viz_dir / f"convergence_{pid}.html").write_text(html)

    # Shannon landscape (all targets × temperatures)
    target_ids = [t["pdb_id"] for t in SYNTHETIC_TARGETS]
    entropy_matrix = [all_results[pid]["shannon"] for pid in target_ids]
    html = shannon_landscape_html(temperatures, target_ids, entropy_matrix)
    (viz_dir / "shannon_landscape.html").write_text(html)

    # Exchange heatmap
    html = exchange_heatmap_html(all_exchange_history, n_replicas)
    (viz_dir / "exchange_heatmap.html").write_text(html)

    # Summary JSON
    summary = {
        "campaign": "demo",
        "n_targets": len(SYNTHETIC_TARGETS),
        "n_replicas": n_replicas,
        "n_generations": n_generations,
        "temperatures": temperatures,
        "targets": {
            pid: {
                "dH_fit": res["vant_hoff"].delta_H_kcal,
                "dS_fit": res["vant_hoff"].delta_S_cal,
                "dG_fit": res["vant_hoff"].delta_G_kcal,
                "dCp_fit": res["vant_hoff"].delta_Cp_cal,
                "r_squared": res["vant_hoff"].r_squared,
            }
            for pid, res in all_results.items()
        },
    }
    (out / "demo_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Visualizations written to: {viz_dir}/")
    print(f"Summary written to: {out / 'demo_summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RE-DOCK Demo Run")
    parser.add_argument("--output-dir", default="./demo_campaign",
                        help="Output directory (default: ./demo_campaign)")
    parser.add_argument("--generations", type=int, default=20,
                        help="Number of GA generations (default: 20)")
    args = parser.parse_args()
    run_demo(args.output_dir, args.generations)


if __name__ == "__main__":
    main()
