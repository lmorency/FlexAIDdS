"""RE-DOCK CLI: python -m benchmarks.re_dock --tier 0 --systems 1STP"""
import argparse
import json
import os
import sys
from typing import List

from .orchestrator import BidirectionalProtocol, Tier, TIER_SYSTEMS, REDOCKResult
from .thermodynamics import K_B


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="re-dock",
        description="RE-DOCK: Replica Exchange Distributed Orchestrated Docking Kit",
    )
    parser.add_argument(
        "--tier", type=int, choices=[0, 1, 2], default=0,
        help="Benchmark tier: 0=1STP, 1=validation, 2=Astex85 (default: 0)",
    )
    parser.add_argument(
        "--systems", nargs="*", default=None,
        help="Override system list (PDB IDs). If omitted, uses tier default.",
    )
    parser.add_argument(
        "--n-replicas", type=int, default=8,
        help="Number of temperature replicas (default: 8)",
    )
    parser.add_argument(
        "--t-min", type=float, default=298.0,
        help="Minimum temperature in K (default: 298.0)",
    )
    parser.add_argument(
        "--t-max", type=float, default=600.0,
        help="Maximum temperature in K (default: 600.0)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for results JSON (default: stdout)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON",
    )
    return parser.parse_args(argv)


def result_to_dict(r: REDOCKResult) -> dict:
    return {
        "pdb_id": r.pdb_id,
        "tier": r.tier.name,
        "delta_g_bar": round(r.delta_g_bar, 4),
        "delta_g_bar_err": round(r.delta_g_bar_err, 4),
        "delta_g_vanthoff": round(r.delta_g_vanthoff, 4),
        "sigma_irr": round(r.sigma_irr, 6),
        "landauer_bits": round(r.landauer_bits, 4),
        "converged": r.converged,
        "n_replicas": len(r.temperatures),
        "t_min": round(float(r.temperatures[0]), 2),
        "t_max": round(float(r.temperatures[-1]), 2),
    }


def main(argv: List[str] = None) -> int:
    args = parse_args(argv)

    tier = Tier(args.tier)
    systems = args.systems or TIER_SYSTEMS[tier]

    protocol = BidirectionalProtocol(
        n_replicas=args.n_replicas,
        t_min=args.t_min,
        t_max=args.t_max,
    )

    results = []
    for pdb_id in systems:
        result = protocol.run_system(pdb_id, tier)
        results.append(result)

        if not args.json:
            status = "CONVERGED" if result.converged else "NOT CONVERGED"
            print(f"{pdb_id:>10s}  "
                  f"DG_BAR={result.delta_g_bar:+8.3f} +/- {result.delta_g_bar_err:.3f}  "
                  f"DG_VH={result.delta_g_vanthoff:+8.3f}  "
                  f"sigma_irr={result.sigma_irr:.6f}  "
                  f"bits={result.landauer_bits:.2f}  "
                  f"[{status}]")

    if args.json:
        output = json.dumps([result_to_dict(r) for r in results], indent=2)
        print(output)

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        out_path = os.path.join(args.output, f"redock_tier{args.tier}.json")
        with open(out_path, "w") as f:
            json.dump([result_to_dict(r) for r in results], f, indent=2)
        print(f"\nResults written to {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
