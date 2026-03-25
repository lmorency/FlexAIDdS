"""CLI for energy matrix training and optimization.

Usage::

    python -m flexaidds.energy_matrix_cli train --contacts contacts.json --output trained.dat
    python -m flexaidds.energy_matrix_cli optimize --matrix base.dat --benchmark bench/ -o optimized.dat
    python -m flexaidds.energy_matrix_cli evaluate --matrix test.dat --benchmark bench/
    python -m flexaidds.energy_matrix_cli convert --input matrix.dat --output matrix.json
    python -m flexaidds.energy_matrix_cli continuous-train --itc-dir data/itc_187 --pdbbind-refined data/pdbbind/refined
    python -m flexaidds.energy_matrix_cli build-contacts --dataset pdbbind_refined --data-dir data/pdbbind/refined
    python -m flexaidds.energy_matrix_cli validate-gates --matrix matrix.bin --casf-dir data/casf
    python -m flexaidds.energy_matrix_cli compare-runs --run-a runs/run_001 --run-b runs/run_002
    python -m flexaidds.energy_matrix_cli list-runs --runs-dir data/training_runs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m flexaidds.energy_matrix_cli",
        description="Energy matrix training and optimization for FlexAID∆S.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── train ────────────────────────────────────────────────────────────
    train_p = subparsers.add_parser(
        "train", help="Derive knowledge-based potential from contact frequencies.",
    )
    train_p.add_argument("--contacts", required=True,
                         help="Contact table JSON file.")
    train_p.add_argument("--output", "-o", required=True,
                         help="Output .dat matrix file.")
    train_p.add_argument("--temperature", type=float, default=300.0)
    train_p.add_argument("--pseudocount", type=int, default=1)
    train_p.add_argument("--no-labels", action="store_true",
                         help="Omit TYPE-TYPE labels in output .dat file.")

    # ── optimize ─────────────────────────────────────────────────────────
    opt_p = subparsers.add_parser(
        "optimize", help="Optimize matrix via gradient-free search.",
    )
    opt_p.add_argument("--matrix", required=True,
                       help="Starting .dat matrix file.")
    opt_p.add_argument("--benchmark", required=True,
                       help="Benchmark directory.")
    opt_p.add_argument("--output", "-o", required=True,
                       help="Output optimized .dat file.")
    opt_p.add_argument("--objective", choices=["auc", "rmsd"], default="auc")
    opt_p.add_argument("--optimizer", choices=["de", "cma"], default="de")
    opt_p.add_argument("--max-evals", type=int, default=5000)
    opt_p.add_argument("--workers", type=int, default=1)
    opt_p.add_argument("--seed", type=int, default=None)
    opt_p.add_argument("--binary", default=None,
                       help="Path to FlexAID executable.")

    # ── evaluate ─────────────────────────────────────────────────────────
    eval_p = subparsers.add_parser(
        "evaluate", help="Evaluate a matrix on a benchmark.",
    )
    eval_p.add_argument("--matrix", required=True)
    eval_p.add_argument("--benchmark", required=True)
    eval_p.add_argument("--objective", choices=["auc", "rmsd"], default="auc")
    eval_p.add_argument("--workers", type=int, default=1)
    eval_p.add_argument("--binary", default=None)

    # ── convert ──────────────────────────────────────────────────────────
    conv_p = subparsers.add_parser(
        "convert", help="Convert between .dat and JSON formats.",
    )
    conv_p.add_argument("--input", required=True)
    conv_p.add_argument("--output", "-o", required=True)
    conv_p.add_argument("--format", choices=["dat", "json", "binary"],
                        default="dat",
                        help="Output format (default: dat).")

    # ── continuous-train ───────────────────────────────────────────────
    ct_p = subparsers.add_parser(
        "continuous-train",
        help="Run multi-dataset curriculum training with quality gates.",
    )
    ct_p.add_argument("--itc-dir", default="",
                      help="ITC-187 dataset directory.")
    ct_p.add_argument("--pdbbind-core", default="",
                      help="PDBbind core set directory.")
    ct_p.add_argument("--pdbbind-refined", default="",
                      help="PDBbind refined set directory.")
    ct_p.add_argument("--pdbbind-general", default="",
                      help="PDBbind general set directory.")
    ct_p.add_argument("--moad-dir", default="",
                      help="Binding MOAD directory.")
    ct_p.add_argument("--bindingdb-dir", default="",
                      help="BindingDB directory.")
    ct_p.add_argument("--chembl-dir", default="",
                      help="ChEMBL directory.")
    ct_p.add_argument("--casf-dir", default="",
                      help="CASF-2016 directory for quality gating.")
    ct_p.add_argument("--prior-matrix", default="",
                      help="Path to previous best matrix for warm-start.")
    ct_p.add_argument("--output-dir", default="data/training_runs",
                      help="Directory for training run artifacts.")
    ct_p.add_argument("--run-name", default="",
                      help="Custom run name (auto-generated if empty).")
    ct_p.add_argument("--cache-dir", default="data/contact_tables",
                      help="Contact table cache directory.")
    ct_p.add_argument("--reference-dat", default="",
                      help="Reference 40-type .dat for projection validation.")
    ct_p.add_argument("--casf-threshold", type=float, default=0.75,
                      help="Minimum CASF-2016 Pearson r for gate (default: 0.75).")
    ct_p.add_argument("--itc-threshold", type=float, default=0.85,
                      help="Minimum ITC-187 Pearson r for gate (default: 0.85).")
    ct_p.add_argument("--seed", type=int, default=42)
    ct_p.add_argument("-v", "--verbose", action="store_true")

    # ── build-contacts ─────────────────────────────────────────────────
    bc_p = subparsers.add_parser(
        "build-contacts",
        help="Build/update contact table cache for a dataset.",
    )
    bc_p.add_argument("--dataset", required=True,
                      help="Dataset name (pdbbind_refined, itc_187, etc.).")
    bc_p.add_argument("--data-dir", required=True,
                      help="Path to dataset directory.")
    bc_p.add_argument("--output", "-o", default="",
                      help="Output JSON path (default: cache-dir/dataset.json).")
    bc_p.add_argument("--cache-dir", default="data/contact_tables")
    bc_p.add_argument("--cutoff", type=float, default=4.5)
    bc_p.add_argument("--force", action="store_true",
                      help="Force rebuild even if cache exists.")

    # ── validate-gates ─────────────────────────────────────────────────
    vg_p = subparsers.add_parser(
        "validate-gates",
        help="Run quality gates on an existing matrix.",
    )
    vg_p.add_argument("--matrix", required=True,
                      help="Path to 256x256 binary matrix.")
    vg_p.add_argument("--casf-dir", default="",
                      help="CASF-2016 directory.")
    vg_p.add_argument("--itc-dir", default="",
                      help="ITC-187 directory.")
    vg_p.add_argument("--casf-threshold", type=float, default=0.75)
    vg_p.add_argument("--itc-threshold", type=float, default=0.85)
    vg_p.add_argument("--cutoff", type=float, default=4.5)

    # ── compare-runs ───────────────────────────────────────────────────
    cr_p = subparsers.add_parser(
        "compare-runs",
        help="Compare metrics between two training runs.",
    )
    cr_p.add_argument("--run-a", required=True,
                      help="Path to first run directory.")
    cr_p.add_argument("--run-b", required=True,
                      help="Path to second run directory.")

    # ── list-runs ──────────────────────────────────────────────────────
    lr_p = subparsers.add_parser(
        "list-runs",
        help="List all training runs with metrics.",
    )
    lr_p.add_argument("--runs-dir", default="data/training_runs",
                      help="Directory containing training runs.")
    lr_p.add_argument("--sort-by", choices=["casf_r", "itc_r", "timestamp"],
                      default="timestamp")

    return parser


def _cmd_train(args: argparse.Namespace) -> int:
    from .energy_matrix import ContactTable, KnowledgeBasedTrainer

    if not Path(args.contacts).is_file():
        raise FileNotFoundError(f"Contact table not found: {args.contacts}")
    table = ContactTable.load(args.contacts)
    trainer = KnowledgeBasedTrainer(
        ntypes=table.ntypes,
        temperature=args.temperature,
        pseudocount=args.pseudocount,
    )
    trainer.add_contact_table(table)
    matrix = trainer.derive_potential()
    matrix.to_dat_file(args.output, labels=None if not args.no_labels else
                       [str(i) for i in range(matrix.ntypes)])
    print(f"Wrote {matrix.ntypes}-type potential to {args.output}")
    return 0


def _cmd_optimize(args: argparse.Namespace) -> int:
    from .energy_matrix import EnergyMatrix, EnergyMatrixOptimizer

    if not Path(args.matrix).is_file():
        raise FileNotFoundError(f"Matrix file not found: {args.matrix}")
    if not Path(args.benchmark).is_dir():
        raise FileNotFoundError(f"Benchmark directory not found: {args.benchmark}")
    matrix = EnergyMatrix.from_dat_file(args.matrix)
    benchmark = EnergyMatrixOptimizer.load_benchmark(args.benchmark)
    print(f"Loaded {len(benchmark)} benchmark cases "
          f"({sum(1 for c in benchmark if c.is_active)} actives, "
          f"{sum(1 for c in benchmark if not c.is_active)} decoys)")

    optimizer = EnergyMatrixOptimizer(
        reference_matrix=matrix,
        benchmark=benchmark,
        objective=args.objective,
        optimizer=args.optimizer,
        n_workers=args.workers,
        flexaid_binary=args.binary,
    )

    def progress(iteration: int, score: float, vector: object) -> None:
        print(f"  Iteration {iteration}: score = {score:.4f}")

    result = optimizer.optimize(
        max_evaluations=args.max_evals,
        seed=args.seed,
        callback=progress,
    )

    result.best_matrix.to_dat_file(args.output)
    print(f"\nOptimization complete: score = {result.best_score:.4f}")
    print(f"  Evaluations: {result.n_evaluations}")
    print(f"  Reason: {result.convergence_reason}")
    print(f"  Output: {args.output}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .energy_matrix import EnergyMatrix, EnergyMatrixOptimizer

    if not Path(args.matrix).is_file():
        raise FileNotFoundError(f"Matrix file not found: {args.matrix}")
    if not Path(args.benchmark).is_dir():
        raise FileNotFoundError(f"Benchmark directory not found: {args.benchmark}")
    matrix = EnergyMatrix.from_dat_file(args.matrix)
    benchmark = EnergyMatrixOptimizer.load_benchmark(args.benchmark)

    optimizer = EnergyMatrixOptimizer(
        reference_matrix=matrix,
        benchmark=benchmark,
        objective=args.objective,
        n_workers=args.workers,
        flexaid_binary=args.binary,
    )

    vector = optimizer._matrix_to_vector(matrix)
    score = -optimizer._evaluate_objective(vector)

    print(f"Evaluation: {args.objective} = {score:.4f}")
    print(f"  Matrix: {args.matrix}")
    print(f"  Benchmark: {args.benchmark}")
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    from .energy_matrix import EnergyMatrix, SHNN_MAGIC

    if not Path(args.input).is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    input_path = args.input
    output_path = args.output
    fmt = args.format

    # Detect input format via magic bytes, fall back to .dat text format
    try:
        with open(input_path, "rb") as fh:
            magic = fh.read(4)
        if magic == SHNN_MAGIC:
            matrix = EnergyMatrix.from_binary(input_path)
        else:
            matrix = EnergyMatrix.from_dat_file(input_path)
    except Exception:
        matrix = EnergyMatrix.from_dat_file(input_path)

    if fmt == "dat":
        matrix.to_dat_file(output_path)
    elif fmt == "json":
        data = {
            "ntypes": matrix.ntypes,
            "matrix": matrix.matrix.tolist(),
        }
        try:
            Path(output_path).write_text(json.dumps(data, indent=2, allow_nan=False))
        except ValueError:
            raise ValueError(
                "Matrix contains NaN or Inf values that cannot be serialized to JSON. "
                "Clean the matrix first or use binary format."
            )
    elif fmt == "binary":
        matrix.to_binary(output_path)

    print(f"Converted {input_path} → {output_path} (format: {fmt})")
    return 0


def _cmd_continuous_train(args: argparse.Namespace) -> int:
    from .continuous_training import ContinuousTrainer, ContinuousTrainingConfig

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    config = ContinuousTrainingConfig(
        itc_dir=args.itc_dir,
        pdbbind_core_dir=args.pdbbind_core,
        pdbbind_refined_dir=args.pdbbind_refined,
        pdbbind_general_dir=args.pdbbind_general,
        moad_dir=args.moad_dir,
        bindingdb_dir=args.bindingdb_dir,
        chembl_dir=args.chembl_dir,
        casf_dir=args.casf_dir,
        prior_matrix_path=args.prior_matrix,
        contact_cache_dir=args.cache_dir,
        seed=args.seed,
        output_dir=args.output_dir,
        run_name=args.run_name,
        reference_dat=args.reference_dat,
        casf_min_r=args.casf_threshold,
        itc_min_r=args.itc_threshold,
    )

    trainer = ContinuousTrainer(config)
    result = trainer.run()

    print(f"\nTraining run: {result.run_id}")
    print(f"  Promoted: {result.promoted}")
    if result.gate_results:
        g = result.gate_results
        print(f"  CASF r={g.casf_pearson_r:.4f} (threshold={config.casf_min_r}) "
              f"{'PASS' if g.casf_passed else 'FAIL'}")
        print(f"  ITC  r={g.itc_pearson_r:.4f} (threshold={config.itc_min_r}) "
              f"{'PASS' if g.itc_passed else 'FAIL'}")
    print(f"  Elapsed: {result.elapsed_seconds:.1f}s")
    print(f"  Output: {result.matrix_path}")
    return 0


def _cmd_build_contacts(args: argparse.Namespace) -> int:
    from .dataset_adapters import create_adapter, get_or_build_contact_table

    adapter = create_adapter(args.dataset)
    table, complexes = get_or_build_contact_table(
        adapter, args.data_dir, args.cache_dir,
        cutoff=args.cutoff, force_rebuild=args.force,
    )

    output = args.output or str(Path(args.cache_dir) / f"{adapter.name()}.json")
    table.save(output)
    print(f"Contact table: {table.n_structures} structures, "
          f"{int(table.counts.sum())} contacts → {output}")
    return 0


def _cmd_validate_gates(args: argparse.Namespace) -> int:
    from .energy_matrix import EnergyMatrix, SHNN_MAGIC
    from .continuous_training import (
        ContinuousTrainingConfig,
        validate_itc_crossval,
    )
    from .train_256x256 import load_pdbbind_complexes, validate_casf

    # Load matrix
    try:
        with open(args.matrix, "rb") as fh:
            magic = fh.read(4)
        if magic == SHNN_MAGIC:
            em = EnergyMatrix.from_binary(args.matrix)
        else:
            em = EnergyMatrix.from_dat_file(args.matrix)
    except Exception:
        em = EnergyMatrix.from_dat_file(args.matrix)

    matrix = em.matrix
    all_passed = True

    if args.casf_dir:
        complexes = load_pdbbind_complexes(args.casf_dir, args.cutoff)
        metrics = validate_casf(matrix, complexes)
        r = metrics.get("pearson_r", 0.0)
        passed = r >= args.casf_threshold
        all_passed = all_passed and passed
        print(f"CASF-2016: r={r:.4f} (threshold={args.casf_threshold}) "
              f"{'PASS' if passed else 'FAIL'}")

    if args.itc_dir:
        metrics = validate_itc_crossval(matrix, args.itc_dir, args.cutoff)
        r = metrics.get("mean_pearson_r", 0.0)
        passed = r >= args.itc_threshold
        all_passed = all_passed and passed
        print(f"ITC-187:   r={r:.4f} (threshold={args.itc_threshold}) "
              f"{'PASS' if passed else 'FAIL'}")

    if not args.casf_dir and not args.itc_dir:
        print("No validation datasets specified. Use --casf-dir and/or --itc-dir.")
        return 1

    return 0 if all_passed else 1


def _cmd_compare_runs(args: argparse.Namespace) -> int:
    from .continuous_training import compare_runs

    comparison = compare_runs(args.run_a, args.run_b)
    print(f"Run A: {comparison['run_a']} (promoted={comparison['promoted_a']})")
    print(f"Run B: {comparison['run_b']} (promoted={comparison['promoted_b']})")
    print(f"  CASF r delta: {comparison['casf_r_delta']:+.4f}")
    print(f"  ITC  r delta: {comparison['itc_r_delta']:+.4f}")
    print(f"  Datasets: {comparison['datasets_a']} → {comparison['datasets_b']}")
    return 0


def _cmd_list_runs(args: argparse.Namespace) -> int:
    from .continuous_training import list_runs

    runs = list_runs(args.runs_dir)
    if not runs:
        print(f"No training runs found in {args.runs_dir}")
        return 0

    sort_key = {
        "casf_r": lambda r: r.get("casf_r", 0),
        "itc_r": lambda r: r.get("itc_r", 0),
        "timestamp": lambda r: r.get("timestamp", ""),
    }[args.sort_by]
    runs.sort(key=sort_key, reverse=True)

    print(f"{'Run ID':<30} {'Promoted':<10} {'CASF r':<10} {'ITC r':<10} "
          f"{'Datasets':<10} {'Elapsed':<10}")
    print("-" * 80)
    for r in runs:
        print(f"{r['run_id']:<30} {'YES' if r['promoted'] else 'no':<10} "
              f"{r['casf_r']:<10.4f} {r['itc_r']:<10.4f} "
              f"{r['n_datasets']:<10} {r['elapsed_s']:<10.1f}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    handlers = {
        "train": _cmd_train,
        "optimize": _cmd_optimize,
        "evaluate": _cmd_evaluate,
        "convert": _cmd_convert,
        "continuous-train": _cmd_continuous_train,
        "build-contacts": _cmd_build_contacts,
        "validate-gates": _cmd_validate_gates,
        "compare-runs": _cmd_compare_runs,
        "list-runs": _cmd_list_runs,
    }
    try:
        return handlers[args.command](args)
    except FileNotFoundError as exc:
        print(f"Error: file not found: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"Error: missing dependency: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
