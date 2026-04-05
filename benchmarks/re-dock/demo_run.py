"""
RE-DOCK Demo Run (v7)
======================

Simulated distributed docking campaign with bidirectional round-trip
thermodynamics (Crooks Fluctuation Theorem + Bennett Acceptance Ratio).

Generates realistic mock data for demonstration and testing using the
extended Van't Hoff model:

    DG(T) = DH + DCp*(T - Tref) - T*(DS + DCp*ln(T/Tref))

v7 additions:
- Bidirectional forward/reverse exchange legs with work recording
- BAR and Crooks intersection free energy estimates
- Irreversible entropy production sigma_irr -> 0 convergence
- Landauer information loss (bits erased per round-trip)
- Shannon Energy Collapse rate dI/dT
- New Chart.js dashboards: Crooks crossing, information loss, collapse rate

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

from .thermodynamics import (
    geometric_temperature_ladder,
    DockingPose,
    ReplicaState,
    van_t_hoff_analysis,
    shannon_entropy_of_ensemble,
    attempt_exchanges,
    R_KCAL,
)
from .crooks import (
    BidirectionalExchange,
    bennett_acceptance_ratio,
    crooks_intersection,
    irreversible_entropy_production,
    landauer_information_loss,
    shannon_energy_collapse_rate,
    mutual_information,
    convergence_check,
)
from .visualization import (
    vant_hoff_plot_html,
    convergence_trace_html,
    shannon_landscape_html,
    exchange_heatmap_html,
    crooks_crossing_plot_html,
    information_loss_plot_html,
    collapse_rate_plot_html,
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
    """Extended Van't Hoff DG(T) with heat capacity correction.

    DG(T) = DH + DCp*(T-Tref) - T*(DS + DCp*ln(T/Tref))

    All enthalpy in kcal/mol, entropy in cal/(mol*K).
    """
    dS_kcal = dS_cal / 1000.0
    dCp_kcal = dCp_cal / 1000.0
    return dH + dCp_kcal * (T - T_ref) - T * (dS_kcal + dCp_kcal * math.log(T / T_ref))


def run_demo(output_dir: str = "./demo_campaign", n_generations: int = 20) -> None:
    """Execute a full simulated campaign with bidirectional round-trip analysis."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    n_replicas = 8
    T_min, T_max = 298.0, 600.0
    temperatures = geometric_temperature_ladder(T_min, T_max, n_replicas)
    pop_size = 100

    print("RE-DOCK v7 Demo Campaign (Bidirectional Round-Trip)")
    print(f"  Temperatures: {', '.join(f'{T:.1f}K' for T in temperatures)}")
    print(f"  Targets: {len(SYNTHETIC_TARGETS)}")
    print(f"  Generations: {n_generations}")
    print()

    all_exchange_history: list[dict] = []
    all_results: dict = {}

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

        # Initialize bidirectional engine
        bidir = BidirectionalExchange(
            temperatures=temperatures,
            reference_temperature=temperatures[0],
        )

        # Track convergence and bidirectional metrics per generation
        convergence: dict[str, list[float]] = {f"T={T:.0f}K": [] for T in temperatures}
        sigma_irr_trace: list[float] = []
        bits_lost_trace: list[float] = []

        for gen in range(n_generations):
            # Generate synthetic poses at each temperature
            for i, replica in enumerate(replicas):
                T = replica.temperature
                dG_true = synthetic_dG(T, dH, dS, dCp)

                # Boltzmann-distributed energies around true DG
                sigma = math.sqrt(R_KCAL * T) * 2.0  # thermal fluctuation
                energies = rng.normal(dG_true, sigma, size=pop_size)

                for e in energies:
                    replica.add_pose(DockingPose(
                        energy_kcal=float(e),
                        generation=gen,
                        replica_index=i,
                    ))

                convergence[f"T={T:.0f}K"].append(float(np.mean(energies)))

            # v7: Bidirectional round-trip (forward + reverse legs)
            bidir.run_forward_leg(replicas, n_sweeps=1, rng=rng)
            bidir.run_reverse_leg(replicas, n_sweeps=1, rng=rng)

            # Also run standard exchange for compatibility with v6 viz
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

            # Track sigma_irr per generation (accumulated so far)
            w_fwd = np.array([s.work_kcal for s in bidir._forward_work])
            w_rev = np.array([s.work_kcal for s in bidir._reverse_work])
            if len(w_fwd) > 0 and len(w_rev) > 0:
                try:
                    dG_est, _ = bennett_acceptance_ratio(
                        w_fwd, w_rev, temperatures[0]
                    )
                    sig = irreversible_entropy_production(
                        float(np.mean(w_fwd)), float(np.mean(w_rev)),
                        dG_est, temperatures[0],
                    )
                    bits = landauer_information_loss(sig, temperatures[0])
                    sigma_irr_trace.append(sig)
                    bits_lost_trace.append(bits)
                except (ValueError, FloatingPointError):
                    sigma_irr_trace.append(float("nan"))
                    bits_lost_trace.append(float("nan"))

        # Full bidirectional analysis
        bidir_result = bidir.analyze()

        # Van't Hoff analysis
        temps_for_vhoff: list[float] = []
        dGs_for_vhoff: list[float] = []
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

        # Shannon entropy per temperature
        S_by_T: list[float] = []
        for replica in replicas:
            energies = [p.energy_kcal for p in replica.poses]
            S = shannon_entropy_of_ensemble(energies, replica.temperature)
            S_by_T.append(S)

        # Report
        print(f"  Van't Hoff:")
        print(f"    DH  = {vr.delta_H_kcal:.2f} (true: {dH:.2f}) kcal/mol")
        print(f"    DS  = {vr.delta_S_cal:.1f} (true: {dS:.1f}) cal/(mol*K)")
        print(f"    DG  = {vr.delta_G_kcal:.2f} (true: {dG_true_298:.2f}) kcal/mol")
        print(f"    DCp = {vr.delta_Cp_cal:.0f} (true: {dCp:.0f}) cal/(mol*K)")
        print(f"    R2  = {vr.r_squared:.4f}")
        print(f"  Bidirectional (v7):")
        print(f"    DG_BAR    = {bidir_result.delta_G_bar:.3f} kcal/mol")
        print(f"    DG_Crooks = {bidir_result.delta_G_crooks:.3f} kcal/mol")
        print(f"    sigma_irr = {bidir_result.sigma_irr:.6f} kcal/(mol*K)")
        print(f"    bits_lost = {bidir_result.bits_lost:.2f}")
        print(f"    mutual_info = {bidir_result.mutual_info:.4f} kcal/(mol*K)")
        print(f"    converged = {bidir_result.converged}")
        print(f"    BAR iters = {bidir_result.bar_iterations}")
        print(f"  S_config range: [{min(S_by_T):.4f}, {max(S_by_T):.4f}] kcal/(mol*K)")

        # Exchange acceptance
        target_exchanges = [r for r in all_exchange_history if r["target_id"] == pid]
        total_ex = sum(len(r["exchanges"]) for r in target_exchanges)
        total_acc = sum(
            sum(1 for e in r["exchanges"] if e["accepted"])
            for r in target_exchanges
        )
        if total_ex > 0:
            print(f"  Exchanges: {total_acc}/{total_ex} accepted "
                  f"({100*total_acc/total_ex:.0f}%)")
        print()

        all_results[pid] = {
            "vant_hoff": vr,
            "convergence": convergence,
            "replicas": replicas,
            "shannon": S_by_T,
            "bidir": bidir_result,
            "bidir_engine": bidir,
            "sigma_irr_trace": sigma_irr_trace,
            "bits_lost_trace": bits_lost_trace,
        }

    # -----------------------------------------------------------------------
    # Generate HTML dashboards
    # -----------------------------------------------------------------------
    viz_dir = out / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    for pid, res in all_results.items():
        vr = res["vant_hoff"]
        bidir_res = res["bidir"]
        bidir_eng = res["bidir_engine"]

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

        # v7: Crooks crossing plot
        w_fwd_all = [s.work_kcal for s in bidir_eng._forward_work]
        w_rev_all = [s.work_kcal for s in bidir_eng._reverse_work]
        html = crooks_crossing_plot_html(
            w_fwd_all, w_rev_all,
            bidir_res.delta_G_bar, bidir_res.delta_G_crooks,
            target_id=pid,
        )
        (viz_dir / f"crooks_{pid}.html").write_text(html)

        # v7: Information loss plot
        gen_indices = list(range(len(res["sigma_irr_trace"])))
        html = information_loss_plot_html(
            gen_indices,
            res["sigma_irr_trace"],
            res["bits_lost_trace"],
            target_id=pid,
        )
        (viz_dir / f"info_loss_{pid}.html").write_text(html)

        # v7: Collapse rate plot
        if bidir_res.collapse_rates:
            cr_temps = sorted(bidir_res.collapse_rates.keys())
            cr_vals = [bidir_res.collapse_rates[T] for T in cr_temps]
            html = collapse_rate_plot_html(cr_temps, cr_vals, target_id=pid)
            (viz_dir / f"collapse_{pid}.html").write_text(html)

    # Shannon landscape (all targets x temperatures)
    target_ids = [t["pdb_id"] for t in SYNTHETIC_TARGETS]
    entropy_matrix = [all_results[pid]["shannon"] for pid in target_ids]
    html = shannon_landscape_html(temperatures, target_ids, entropy_matrix)
    (viz_dir / "shannon_landscape.html").write_text(html)

    # Exchange heatmap
    html = exchange_heatmap_html(all_exchange_history, n_replicas)
    (viz_dir / "exchange_heatmap.html").write_text(html)

    # Summary JSON
    summary = {
        "campaign": "demo_v7",
        "version": "7.0",
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
                "dG_BAR": res["bidir"].delta_G_bar,
                "dG_Crooks": res["bidir"].delta_G_crooks,
                "sigma_irr": res["bidir"].sigma_irr,
                "bits_lost": res["bidir"].bits_lost,
                "mutual_info": res["bidir"].mutual_info,
                "converged": res["bidir"].converged,
                "bar_iterations": res["bidir"].bar_iterations,
            }
            for pid, res in all_results.items()
        },
    }
    (out / "demo_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Visualizations written to: {viz_dir}/")
    print(f"Summary written to: {out / 'demo_summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RE-DOCK v7 Demo Run")
    parser.add_argument("--output-dir", default="./demo_campaign",
                        help="Output directory (default: ./demo_campaign)")
    parser.add_argument("--generations", type=int, default=20,
                        help="Number of GA generations (default: 20)")
    args = parser.parse_args()
    run_demo(args.output_dir, args.generations)


if __name__ == "__main__":
    main()
