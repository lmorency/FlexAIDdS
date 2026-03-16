"""CLI for energy matrix training and optimization.

Usage::

    python -m flexaidds.energy_matrix_cli train --contacts contacts.json --output trained.dat
    python -m flexaidds.energy_matrix_cli optimize --matrix base.dat --benchmark bench/ -o optimized.dat
    python -m flexaidds.energy_matrix_cli evaluate --matrix test.dat --benchmark bench/
    python -m flexaidds.energy_matrix_cli convert --input matrix.dat --output matrix.json
"""

from __future__ import annotations

import argparse
import json
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


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    handlers = {
        "train": _cmd_train,
        "optimize": _cmd_optimize,
        "evaluate": _cmd_evaluate,
        "convert": _cmd_convert,
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
