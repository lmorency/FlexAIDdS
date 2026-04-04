"""
benchmarks/DatasetRunner.py
============================
FlexAIDdS automated distributed benchmarking orchestrator.

Loads dataset configs from benchmarks/datasets/*.yaml, distributes targets
across MPI ranks (or runs serially), executes FlexAIDdS on each target, and
computes evaluation metrics with bootstrap confidence intervals.

Primary metric: entropy_rescue_rate — the fraction of true binders that
ΔS scoring rescues among targets that pure-enthalpy scoring misses.

Usage
-----
    # Serial (single node)
    python benchmarks/DatasetRunner.py --dataset casf2016 --tier 1

    # MPI distributed (e.g. 32 ranks)
    mpirun -n 32 python benchmarks/DatasetRunner.py --dataset casf2016

    # Via CLI wrapper
    python -m benchmarks.run --dataset casf2016 --tier 1 --output results/

Architecture
------------
- Rank 0 ("coordinator"):
    - Loads dataset config
    - Builds target work list (optionally filtered to tier subset)
    - Scatters work items to worker ranks
    - Gathers results from all ranks
    - Computes aggregate metrics
    - Writes JSON + Markdown report

- Ranks 1..N ("workers"):
    - Receive work items from rank 0
    - For each target: run FlexAIDdS (subprocess) or Python API
    - Collect per-target scores, poses, RMSD
    - Send results back to rank 0

MPI is optional.  If mpi4py is not available (or MPI_DISABLED=1), the runner
falls back to sequential single-process execution.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

# Optional MPI
_MPI_AVAILABLE = False
_COMM = None
_RANK = 0
_SIZE = 1

if os.environ.get("MPI_DISABLED", "0") != "1":
    try:
        from mpi4py import MPI  # type: ignore[import]
        _COMM = MPI.COMM_WORLD
        _RANK = _COMM.Get_rank()
        _SIZE = _COMM.Get_size()
        _MPI_AVAILABLE = True
    except ImportError:
        pass

# Optional YAML loader
try:
    import yaml  # type: ignore[import]
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# Internal benchmark metrics
_BENCHMARKS_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(_BENCHMARKS_DIR.parent))

from benchmarks.metrics import (  # noqa: E402
    CI,
    DockingPowerResult,
    EnrichmentResult,
    ScoringPowerResult,
    bootstrap_ci,
    docking_power,
    enrichment_factor,
    entropy_rescue_rate,
    hit_rate_top_n,
    log_auc,
    scoring_power,
    target_specificity_zscore,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TargetConfig:
    """Lightweight representation of a single docking target."""
    name: str                          # identifier (e.g. PDB ID or internal name)
    receptor_path: pathlib.Path
    ligand_path: Optional[pathlib.Path] = None
    actives_path: Optional[pathlib.Path] = None
    decoys_path: Optional[pathlib.Path] = None
    experimental_dG: Optional[float] = None     # kcal/mol
    experimental_Ki_nM: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetResult:
    """Per-target docking result."""
    name: str
    success: bool
    wall_time_s: float
    enthalpy_scores: list[float] = field(default_factory=list)
    entropy_scores: list[float] = field(default_factory=list)
    combined_scores: list[float] = field(default_factory=list)
    pose_rmsd: list[float] = field(default_factory=list)          # vs crystal
    experimental_dG: Optional[float] = None
    is_active: Optional[bool] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Aggregate report for a full dataset run."""
    dataset_name: str
    dataset_version: str
    tier: int
    n_targets_total: int
    n_targets_success: int
    n_targets_failed: int
    wall_time_s: float
    timestamp: str
    git_commit: str
    hostname: str
    platform: str
    mpi_size: int

    # Aggregate metrics
    entropy_rescue_rate: Optional[float] = None
    entropy_rescue_rate_ci: Optional[CI] = None
    scoring_power: Optional[ScoringPowerResult] = None
    docking_power: Optional[DockingPowerResult] = None
    enrichment_ef1: Optional[EnrichmentResult] = None
    log_auc: Optional[float] = None
    log_auc_ci: Optional[CI] = None

    # Per-target results
    target_results: list[TargetResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dataset config loading
# ---------------------------------------------------------------------------

def _load_yaml(path: pathlib.Path) -> dict[str, Any]:
    """Load a YAML file, falling back to a minimal parser if PyYAML absent."""
    if _YAML_AVAILABLE:
        with path.open() as fh:
            return yaml.safe_load(fh)  # type: ignore[no-any-return]
    # Minimal YAML parser: handles only simple key: value and lists
    raise RuntimeError(
        "PyYAML is required.  Install with: pip install pyyaml"
    )


def _resolve_path(template: str, env_key: str, fallback: str, **kwargs: str) -> pathlib.Path:
    """Resolve a path template using env vars + format kwargs."""
    root = os.environ.get(env_key, fallback)
    resolved = template.replace(f"${{{env_key}}}", root)
    for k, v in kwargs.items():
        resolved = resolved.replace(f"{{{k}}}", v)
    return pathlib.Path(resolved)


class DatasetConfig:
    """Parsed dataset configuration from a YAML file."""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self._raw: dict[str, Any] = _load_yaml(path)
        self.name: str = self._raw["name"]
        self.version: str = str(self._raw.get("version", "unknown"))
        self.description: str = self._raw.get("description", "")
        self.n_targets: int = int(self._raw["n_targets"])
        self.protocol: str = self._raw["protocol"]
        self.tier: int = int(self._raw.get("tier", 2))
        self.tier1_subset: list[str] = self._raw.get("tier1_subset", [])
        self.data: dict[str, Any] = self._raw.get("data", {})
        self.metrics_cfg: dict[str, Any] = self._raw.get("metrics", {})
        self.flexaids_params: dict[str, Any] = self._raw.get("flexaids_params", {})
        self.targets_raw: list[dict[str, Any]] = self._raw.get("targets", [])
        self.reproducibility: dict[str, Any] = self._raw.get("reproducibility", {})

    @classmethod
    def from_name(cls, name: str) -> "DatasetConfig":
        """Load a dataset config by short name from benchmarks/datasets/."""
        datasets_dir = pathlib.Path(__file__).parent / "datasets"
        path = datasets_dir / f"{name}.yaml"
        if not path.exists():
            available = sorted(p.stem for p in datasets_dir.glob("*.yaml"))
            raise FileNotFoundError(
                f"Dataset config not found: {path}\n"
                f"Available datasets: {available}"
            )
        return cls(path)

    def build_targets(
        self,
        tier: int = 2,
        data_root_override: Optional[pathlib.Path] = None,
    ) -> list[TargetConfig]:
        """Build a list of TargetConfig from the raw YAML targets."""
        data_root_env = self.data.get("local_root", "${FLEXAIDS_DATA}/" + self.name)
        data_root = (
            str(data_root_override)
            if data_root_override
            else os.environ.get("FLEXAIDS_DATA", "/data/flexaids")
        )
        data_root_resolved = data_root_env.replace("${FLEXAIDS_DATA}", data_root)

        receptor_tmpl = self.data.get("receptor_subdir", "{name}/receptor.pdb")
        ligand_tmpl = self.data.get("ligand_subdir", "{name}/ligand.mol2")

        # Filter to tier-1 subset if requested
        subset = set(self.tier1_subset) if tier == 1 and self.tier1_subset else set()

        targets: list[TargetConfig] = []
        for raw in self.targets_raw:
            tname = str(raw.get("pdbid") or raw.get("name") or raw.get("bdid") or "unknown")
            if subset and tname not in subset:
                continue

            receptor = pathlib.Path(
                receptor_tmpl.replace("{pdbid}", tname).replace("{name}", tname).replace("{target}", tname)
            )
            if not receptor.is_absolute():
                receptor = pathlib.Path(data_root_resolved) / receptor

            ligand = pathlib.Path(
                ligand_tmpl.replace("{pdbid}", tname).replace("{name}", tname).replace("{target}", tname)
            )
            if not ligand.is_absolute():
                ligand = pathlib.Path(data_root_resolved) / ligand

            dG = raw.get("logKa") or raw.get("dG") or raw.get("dG_kcal_mol")
            if dG and raw.get("logKa"):
                # Convert log10(Ka) → ΔG = -RT ln(Ka)
                import math
                dG = -0.592 * raw["logKa"]  # at 298.15 K, kcal/mol

            tc = TargetConfig(
                name=tname,
                receptor_path=receptor,
                ligand_path=ligand if ligand_tmpl else None,
                experimental_dG=float(dG) if dG is not None else None,
                metadata=dict(raw),
            )
            targets.append(tc)

        # If YAML has no inline targets but has affinity_csv, return minimal stubs
        if not targets and subset:
            for name in (subset if subset else []):
                targets.append(
                    TargetConfig(
                        name=name,
                        receptor_path=pathlib.Path(data_root_resolved) / name / "receptor.pdb",
                    )
                )

        return targets

    @property
    def seed(self) -> int:
        return int(self.reproducibility.get("seed", 42))


# ---------------------------------------------------------------------------
# FlexAIDdS runner
# ---------------------------------------------------------------------------

def _find_flexaid_binary() -> Optional[pathlib.Path]:
    """Locate the FlexAID binary in standard search paths."""
    candidates = [
        pathlib.Path(os.environ.get("FLEXAID_BIN", "")).expanduser(),
        pathlib.Path(__file__).parent.parent / "BIN" / "FlexAID",
        pathlib.Path(__file__).parent.parent / "build" / "FlexAID",
        pathlib.Path("/usr/local/bin/FlexAID"),
    ]
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return c
    # Try PATH
    import shutil
    found = shutil.which("FlexAID")
    return pathlib.Path(found) if found else None


def run_flexaids_target(
    target: TargetConfig,
    params: dict[str, Any],
    work_dir: pathlib.Path,
    binary: Optional[pathlib.Path] = None,
    timeout: int = 3600,
) -> TargetResult:
    """Run FlexAIDdS on a single target and parse the results.

    Parameters
    ----------
    target:
        Target configuration (paths + metadata).
    params:
        FlexAIDdS parameters from the dataset YAML flexaids_params block.
    work_dir:
        Working directory for this target's run.
    binary:
        Path to FlexAID binary.  Auto-discovered if None.
    timeout:
        Maximum wall-clock time in seconds per target.

    Returns
    -------
    TargetResult with scores and poses (or error message if run failed).
    """
    t_start = time.perf_counter()
    work_dir.mkdir(parents=True, exist_ok=True)

    if binary is None:
        binary = _find_flexaid_binary()

    if binary is None or not binary.exists():
        # No binary available — return a synthetic stub for unit-testing
        # the orchestrator logic without a compiled binary.
        logger.warning(
            "FlexAID binary not found for target %s; returning stub result.",
            target.name,
        )
        return TargetResult(
            name=target.name,
            success=False,
            wall_time_s=0.0,
            error="FlexAID binary not found",
            experimental_dG=target.experimental_dG,
        )

    if not target.receptor_path.exists():
        return TargetResult(
            name=target.name,
            success=False,
            wall_time_s=0.0,
            error=f"Receptor not found: {target.receptor_path}",
            experimental_dG=target.experimental_dG,
        )

    # Build config .inp
    config_path = _write_flexaid_config(target, params, work_dir)

    # Build GA .inp
    ga_path = _write_ga_config(params, work_dir)

    cmd = [str(binary), str(config_path), str(ga_path)]
    logger.info("Running FlexAID: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return TargetResult(
            name=target.name,
            success=False,
            wall_time_s=time.perf_counter() - t_start,
            error=f"Timeout after {timeout}s",
            experimental_dG=target.experimental_dG,
        )

    if proc.returncode != 0:
        return TargetResult(
            name=target.name,
            success=False,
            wall_time_s=time.perf_counter() - t_start,
            error=f"FlexAID exited {proc.returncode}: {proc.stderr[-500:]}",
            experimental_dG=target.experimental_dG,
        )

    # Parse output
    result = _parse_flexaid_output(target, work_dir)
    result.wall_time_s = time.perf_counter() - t_start
    return result


def _write_flexaid_config(
    target: TargetConfig,
    params: dict[str, Any],
    work_dir: pathlib.Path,
) -> pathlib.Path:
    """Write a FlexAID main config file (.inp) to work_dir."""
    T = params.get("temperature", 298.15) or 298.15
    n_poses = params.get("n_poses", 10)
    use_entropy = str(params.get("use_entropy", True)).upper()
    entropy_weight = params.get("entropy_weight", 1.0)

    lines = [
        f"RECEPTOR        {target.receptor_path}",
        f"TEMPERATURE     {T}",
        f"NUM_POSES       {n_poses}",
        f"USE_ENTROPY     {use_entropy}",
        f"ENTROPY_WEIGHT  {entropy_weight}",
        f"OUTPUT_DIR      {work_dir}",
        f"TARGET_NAME     {target.name}",
    ]
    if target.ligand_path and target.ligand_path.exists():
        lines.insert(1, f"LIGAND          {target.ligand_path}")

    config_path = work_dir / "config.inp"
    config_path.write_text("\n".join(lines) + "\n")
    return config_path


def _write_ga_config(params: dict[str, Any], work_dir: pathlib.Path) -> pathlib.Path:
    """Write FlexAID GA config file to work_dir."""
    lines = [
        f"ITERATIONS      {params.get('ga_iterations', 200)}",
        f"POPULATION      {params.get('population_size', 100)}",
    ]
    ga_path = work_dir / "ga.inp"
    ga_path.write_text("\n".join(lines) + "\n")
    return ga_path


def _parse_flexaid_output(
    target: TargetConfig,
    work_dir: pathlib.Path,
) -> TargetResult:
    """Parse FlexAID output files and build a TargetResult.

    Looks for:
      - {work_dir}/{name}_results.json  (preferred structured output)
      - {work_dir}/*.rrd                (legacy rank result data)
    """
    # Try JSON output first
    json_out = work_dir / f"{target.name}_results.json"
    if json_out.exists():
        return _parse_json_output(target, json_out)

    # Try RRD files
    rrd_files = sorted(work_dir.glob("*.rrd"))
    if rrd_files:
        return _parse_rrd_output(target, rrd_files)

    return TargetResult(
        name=target.name,
        success=False,
        wall_time_s=0.0,
        error="No output files found",
        experimental_dG=target.experimental_dG,
    )


def _parse_json_output(target: TargetConfig, json_path: pathlib.Path) -> TargetResult:
    """Parse structured JSON output from FlexAIDdS."""
    with json_path.open() as fh:
        data = json.load(fh)

    poses = data.get("poses", [])
    return TargetResult(
        name=target.name,
        success=True,
        wall_time_s=0.0,
        enthalpy_scores=[p.get("enthalpy_score", 0.0) for p in poses],
        entropy_scores=[p.get("entropy_score", 0.0) for p in poses],
        combined_scores=[p.get("combined_score", 0.0) for p in poses],
        pose_rmsd=[p.get("rmsd", float("inf")) for p in poses],
        experimental_dG=target.experimental_dG,
        metadata=data.get("metadata", {}),
    )


def _parse_rrd_output(target: TargetConfig, rrd_files: list[pathlib.Path]) -> TargetResult:
    """Parse legacy .rrd rank-result-data files."""
    enthalpy: list[float] = []
    combined: list[float] = []
    rmsd: list[float] = []

    for rrd in rrd_files:
        with rrd.open() as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        enthalpy.append(float(parts[1]))
                        combined.append(float(parts[2]))
                        rmsd.append(float(parts[-1]) if len(parts) >= 4 else float("inf"))
                    except ValueError:
                        continue

    if not enthalpy:
        return TargetResult(
            name=target.name,
            success=False,
            wall_time_s=0.0,
            error="Empty RRD output",
            experimental_dG=target.experimental_dG,
        )

    entropy_s = [c - e for c, e in zip(combined, enthalpy)]
    return TargetResult(
        name=target.name,
        success=True,
        wall_time_s=0.0,
        enthalpy_scores=enthalpy,
        entropy_scores=entropy_s,
        combined_scores=combined,
        pose_rmsd=rmsd,
        experimental_dG=target.experimental_dG,
    )


# ---------------------------------------------------------------------------
# MPI work distribution
# ---------------------------------------------------------------------------

def _scatter_work(
    targets: list[TargetConfig],
    n_ranks: int,
) -> list[list[TargetConfig]]:
    """Distribute targets across ranks using round-robin assignment."""
    buckets: list[list[TargetConfig]] = [[] for _ in range(n_ranks)]
    for i, t in enumerate(targets):
        buckets[i % n_ranks].append(t)
    return buckets


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def _aggregate_metrics(
    results: list[TargetResult],
    dataset_cfg: DatasetConfig,
    seed: int,
) -> dict[str, Any]:
    """Compute all configured metrics over the collected target results."""
    successful = [r for r in results if r.success]
    if not successful:
        return {"error": "No successful target results to aggregate"}

    agg: dict[str, Any] = {}

    # ΔS rescue rate (primary)
    targets_with_poses = [
        r for r in successful
        if r.enthalpy_scores and r.entropy_scores and r.experimental_dG is not None
    ]
    if targets_with_poses:
        # For rescue rate: treat each target's top pose as the rank-1 result.
        # A target is "missed by enthalpy" if rank1_enthalpy_dG > threshold.
        # Use per-target top-1 enthalpy vs entropy + experimental dG.
        threshold_kcal = -6.0   # approximate Ki=10μM threshold
        active_mask = [
            bool(r.experimental_dG is not None and r.experimental_dG < threshold_kcal)
            for r in successful
        ]
        enthalpy_scores_top1 = [
            min(r.enthalpy_scores) if r.enthalpy_scores else 0.0
            for r in successful
        ]
        entropy_scores_top1 = [
            min(r.combined_scores) if r.combined_scores else 0.0
            for r in successful
        ]
        # Build ranks within this result set
        def _rank(scores: list[float]) -> list[int]:
            idx_sorted = sorted(range(len(scores)), key=lambda i: scores[i])
            ranks = [0] * len(scores)
            for rank, idx in enumerate(idx_sorted, start=1):
                ranks[idx] = rank
            return ranks

        e_ranks = _rank(enthalpy_scores_top1)
        s_ranks = _rank(entropy_scores_top1)
        rescue = entropy_rescue_rate(e_ranks, s_ranks, active_mask, threshold=5)
        agg["entropy_rescue_rate"] = rescue

        # Bootstrap CI on rescue rate
        combined_data = list(zip(e_ranks, s_ranks, active_mask))

        def _rr_from_tuples(data: list) -> float:
            er = [d[0] for d in data]
            sr = [d[1] for d in data]
            am = [d[2] for d in data]
            v = entropy_rescue_rate(er, sr, am, threshold=5)
            import math
            return v if not math.isnan(v) else 0.0

        ci = bootstrap_ci(_rr_from_tuples, combined_data, n_boot=500, seed=seed)
        agg["entropy_rescue_rate_ci"] = {"lower": ci.lower, "point": ci.point, "upper": ci.upper, "alpha": ci.alpha}

    # Scoring power (Pearson r vs ΔG_exp)
    targets_with_exp = [(r.combined_scores[0] if r.combined_scores else None, r.experimental_dG)
                        for r in successful if r.experimental_dG is not None and r.combined_scores]
    if len(targets_with_exp) >= 5:
        pred = [p for p, _ in targets_with_exp]
        expt = [e for _, e in targets_with_exp]
        sp = scoring_power(pred, expt)
        agg["scoring_power"] = {"pearson_r": sp.pearson_r, "rmse": sp.rmse, "n_samples": sp.n_samples}

    # Docking power
    rmsd_top1 = [r.pose_rmsd[0] for r in successful if r.pose_rmsd and r.pose_rmsd[0] != float("inf")]
    if rmsd_top1:
        dp = docking_power(rmsd_top1, threshold=2.0)
        agg["docking_power"] = {
            "success_rate": dp.success_rate,
            "n_success": dp.n_success,
            "n_total": dp.n_total,
            "rmsd_threshold": dp.rmsd_threshold,
        }

    # VS enrichment (where actives/decoys data is present via is_active)
    vs_results = [(min(r.combined_scores) if r.combined_scores else 0.0, int(r.is_active or False))
                  for r in successful if r.is_active is not None and r.combined_scores]
    if len(vs_results) >= 5:
        scores_vs = [s for s, _ in vs_results]
        labels_vs = [l for _, l in vs_results]
        ef_frac = float(dataset_cfg.metrics_cfg.get("ef_fraction", 0.01))
        try:
            ef = enrichment_factor(scores_vs, labels_vs, fraction=ef_frac)
            agg["enrichment_ef"] = {"ef": ef.ef, "fraction": ef.fraction,
                                    "n_actives_found": ef.n_actives_found}
            la = log_auc(scores_vs, labels_vs)
            agg["log_auc"] = la
        except ValueError:
            pass

    return agg


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    """Return current git commit hash (short), or 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=pathlib.Path(__file__).parent,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _write_json_report(report: BenchmarkReport, out_path: pathlib.Path) -> None:
    """Write the full BenchmarkReport to a JSON file."""
    def _default(obj: Any) -> Any:
        if isinstance(obj, pathlib.Path):
            return str(obj)
        if isinstance(obj, CI):
            return {"lower": obj.lower, "point": obj.point, "upper": obj.upper, "alpha": obj.alpha}
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)  # type: ignore[arg-type]
        raise TypeError(f"Not serialisable: {type(obj)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(asdict(report), fh, indent=2, default=_default)
    logger.info("JSON report written to %s", out_path)


def _write_markdown_report(
    report: BenchmarkReport,
    metrics: dict[str, Any],
    out_path: pathlib.Path,
) -> None:
    """Write a human-readable Markdown summary of the benchmark run."""
    lines = [
        f"# FlexAIDdS Benchmark — {report.dataset_name} (tier {report.tier})",
        "",
        f"**Date**: {report.timestamp}  ",
        f"**Commit**: `{report.git_commit}`  ",
        f"**Host**: {report.hostname} ({report.platform})  ",
        f"**MPI ranks**: {report.mpi_size}  ",
        f"**Targets**: {report.n_targets_success}/{report.n_targets_total} succeeded  ",
        f"**Wall time**: {report.wall_time_s:.1f}s",
        "",
        "---",
        "",
        "## Metrics",
        "",
    ]

    # ΔS rescue rate (primary)
    rr = metrics.get("entropy_rescue_rate")
    rr_ci = metrics.get("entropy_rescue_rate_ci")
    if rr is not None:
        import math
        if not math.isnan(float(rr)):
            ci_str = ""
            if rr_ci:
                ci_str = f" [95% CI: {rr_ci['lower']:.3f}–{rr_ci['upper']:.3f}]"
            lines += [
                f"### ΔS Rescue Rate (primary) ★",
                "",
                f"**{rr:.3f}**{ci_str}",
                "",
                "> Fraction of true binders rescued by entropy-augmented scoring",
                "> among those missed by pure-enthalpy scoring (top-5 cutoff).",
                "",
            ]

    sp = metrics.get("scoring_power")
    if sp:
        lines += [
            "### Scoring Power",
            "",
            f"Pearson r = **{sp['pearson_r']:.3f}** | RMSE = {sp['rmse']:.2f} kcal/mol | n = {sp['n_samples']}",
            "",
        ]

    dp = metrics.get("docking_power")
    if dp:
        lines += [
            "### Docking Power",
            "",
            f"Success rate (RMSD ≤ {dp['rmsd_threshold']:.1f} Å) = **{dp['success_rate']:.1%}**"
            f" ({dp['n_success']}/{dp['n_total']})",
            "",
        ]

    ef = metrics.get("enrichment_ef")
    if ef:
        lines += [
            "### Enrichment Factor",
            "",
            f"EF{ef['fraction']*100:.1f}% = **{ef['ef']:.2f}** (n_actives found: {ef['n_actives_found']})",
            "",
        ]

    la = metrics.get("log_auc")
    if la is not None:
        import math
        if not math.isnan(float(la)):
            lines += [
                "### LogAUC",
                "",
                f"**{la:.4f}**",
                "",
            ]

    # Per-target table
    lines += ["---", "", "## Per-Target Results", ""]
    if report.target_results:
        lines += ["| Target | Status | ΔG_exp | top1_score | top1_RMSD | wall(s) |",
                  "|--------|--------|--------|------------|-----------|---------|"]
        for tr in sorted(report.target_results, key=lambda r: (not r.success, r.name)):
            status = "✓" if tr.success else "✗"
            dg_str = f"{tr.experimental_dG:.2f}" if tr.experimental_dG is not None else "—"
            score_str = f"{tr.combined_scores[0]:.3f}" if tr.combined_scores else "—"
            rmsd_str = f"{tr.pose_rmsd[0]:.2f}" if tr.pose_rmsd and tr.pose_rmsd[0] != float("inf") else "—"
            lines.append(
                f"| {tr.name} | {status} | {dg_str} | {score_str} | {rmsd_str} | {tr.wall_time_s:.1f} |"
            )

    if report.errors:
        lines += ["", "## Errors", ""]
        for err in report.errors:
            lines.append(f"- {err}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    logger.info("Markdown report written to %s", out_path)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

class DatasetRunner:
    """Orchestrates a full benchmark run for a single dataset.

    Parameters
    ----------
    dataset_name:
        Short name matching a YAML file in benchmarks/datasets/.
    tier:
        1 = PR sanity (small subset), 2 = full nightly run.
    output_dir:
        Directory for reports and work files.
    data_root:
        Override for dataset data root (env var FLEXAIDS_DATA otherwise).
    flexaid_binary:
        Explicit path to FlexAID binary.
    target_timeout:
        Per-target timeout in seconds.
    dry_run:
        If True, skip actual FlexAIDdS execution (useful for CI plumbing tests).
    """

    def __init__(
        self,
        dataset_name: str,
        tier: int = 2,
        output_dir: Optional[pathlib.Path] = None,
        data_root: Optional[pathlib.Path] = None,
        flexaid_binary: Optional[pathlib.Path] = None,
        target_timeout: int = 600,
        dry_run: bool = False,
    ) -> None:
        self.dataset_name = dataset_name
        self.tier = tier
        self.output_dir = output_dir or pathlib.Path("results") / dataset_name
        self.data_root = data_root
        self.flexaid_binary = flexaid_binary or _find_flexaid_binary()
        self.target_timeout = target_timeout
        self.dry_run = dry_run

        if _RANK == 0:
            self.cfg = DatasetConfig.from_name(dataset_name)
            logging.basicConfig(
                level=logging.INFO,
                format=f"[Rank 0] %(levelname)s %(name)s: %(message)s",
            )
        else:
            self.cfg = None  # workers receive config via MPI bcast

    def run(self) -> BenchmarkReport:
        """Execute the full benchmark and return a BenchmarkReport."""
        t_global_start = time.perf_counter()

        # ---- Rank 0: broadcast config and scatter targets ----
        if _MPI_AVAILABLE:
            self.cfg = _COMM.bcast(self.cfg, root=0)

        if _RANK == 0:
            targets = self.cfg.build_targets(tier=self.tier, data_root_override=self.data_root)
            logger.info(
                "Dataset: %s | tier: %d | n_targets: %d | MPI ranks: %d",
                self.cfg.name, self.tier, len(targets), _SIZE,
            )
        else:
            targets = []

        if _MPI_AVAILABLE:
            targets = _COMM.bcast(targets, root=0)

        # ---- Distribute work ----
        buckets = _scatter_work(targets, _SIZE)
        my_targets = buckets[_RANK]

        # ---- Each rank runs its targets ----
        my_results: list[TargetResult] = []
        for target in my_targets:
            work_dir = self.output_dir / "runs" / target.name
            if self.dry_run:
                result = TargetResult(
                    name=target.name,
                    success=True,
                    wall_time_s=0.0,
                    enthalpy_scores=[-8.0, -7.5, -7.0],
                    entropy_scores=[-1.0, -0.8, -0.5],
                    combined_scores=[-9.0, -8.3, -7.5],
                    pose_rmsd=[1.2, 2.1, 3.5],
                    experimental_dG=target.experimental_dG,
                )
            else:
                result = run_flexaids_target(
                    target=target,
                    params=self.cfg.flexaids_params,
                    work_dir=work_dir,
                    binary=self.flexaid_binary,
                    timeout=self.target_timeout,
                )
            my_results.append(result)

        # ---- Gather results to rank 0 ----
        if _MPI_AVAILABLE:
            all_results_nested = _COMM.gather(my_results, root=0)
            if _RANK == 0:
                all_results: list[TargetResult] = [r for sub in all_results_nested for r in sub]
            else:
                return None  # type: ignore[return-value]
        else:
            all_results = my_results

        # ---- Rank 0: aggregate and report ----
        t_total = time.perf_counter() - t_global_start
        errors = [f"{r.name}: {r.error}" for r in all_results if r.error]
        metrics = _aggregate_metrics(all_results, self.cfg, seed=self.cfg.seed)

        report = BenchmarkReport(
            dataset_name=self.cfg.name,
            dataset_version=self.cfg.version,
            tier=self.tier,
            n_targets_total=len(all_results),
            n_targets_success=sum(1 for r in all_results if r.success),
            n_targets_failed=sum(1 for r in all_results if not r.success),
            wall_time_s=t_total,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            git_commit=_git_commit(),
            hostname=platform.node(),
            platform=platform.system(),
            mpi_size=_SIZE,
            target_results=all_results,
            errors=errors,
        )

        # Attach aggregate metrics
        if "entropy_rescue_rate" in metrics:
            import math
            val = metrics["entropy_rescue_rate"]
            if not math.isnan(float(val)):
                report.entropy_rescue_rate = float(val)
        if "scoring_power" in metrics:
            sp = metrics["scoring_power"]
            report.scoring_power = ScoringPowerResult(**sp)
        if "docking_power" in metrics:
            dp = metrics["docking_power"]
            report.docking_power = DockingPowerResult(**dp)

        # Write outputs
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        json_out = self.output_dir / f"{self.cfg.name}_tier{self.tier}_{ts}.json"
        md_out = self.output_dir / f"{self.cfg.name}_tier{self.tier}_{ts}.md"
        _write_json_report(report, json_out)
        _write_markdown_report(report, metrics, md_out)

        # Print summary to stdout
        self._print_summary(report, metrics)
        return report

    def _print_summary(self, report: BenchmarkReport, metrics: dict[str, Any]) -> None:
        """Print a short summary table to stdout."""
        print("\n" + "=" * 60)
        print(f"  FlexAIDdS Benchmark: {report.dataset_name} (tier {report.tier})")
        print("=" * 60)
        print(f"  Targets:    {report.n_targets_success}/{report.n_targets_total} succeeded")
        print(f"  Wall time:  {report.wall_time_s:.1f}s")
        print(f"  MPI ranks:  {report.mpi_size}")
        print()

        import math
        rr = metrics.get("entropy_rescue_rate")
        if rr is not None and not math.isnan(float(rr)):
            ci = metrics.get("entropy_rescue_rate_ci")
            ci_str = f"  [95% CI: {ci['lower']:.3f}–{ci['upper']:.3f}]" if ci else ""
            print(f"  ★ ΔS Rescue Rate:  {float(rr):.3f}{ci_str}")

        sp = metrics.get("scoring_power")
        if sp:
            print(f"  Scoring r:         {sp['pearson_r']:.3f}  (RMSE {sp['rmse']:.2f} kcal/mol)")

        dp = metrics.get("docking_power")
        if dp:
            print(f"  Docking success:   {dp['success_rate']:.1%}  ({dp['n_success']}/{dp['n_total']})")

        ef = metrics.get("enrichment_ef")
        if ef:
            print(f"  EF{ef['fraction']*100:.1f}%:          {ef['ef']:.2f}")

        la = metrics.get("log_auc")
        if la is not None and not math.isnan(float(la)):
            print(f"  LogAUC:            {float(la):.4f}")

        if report.errors:
            print(f"\n  Errors ({len(report.errors)}):")
            for err in report.errors[:5]:
                print(f"    - {err}")
            if len(report.errors) > 5:
                print(f"    ... and {len(report.errors)-5} more")
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FlexAIDdS DatasetRunner — distributed benchmark orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", "-d",
        required=True,
        help="Dataset name (e.g. casf2016, itc187, dude37)",
    )
    parser.add_argument(
        "--tier", "-t",
        type=int,
        default=2,
        choices=[1, 2],
        help="1 = PR sanity (small subset), 2 = full benchmark (default: 2)",
    )
    parser.add_argument(
        "--output", "-o",
        type=pathlib.Path,
        default=None,
        help="Output directory for reports (default: results/<dataset>)",
    )
    parser.add_argument(
        "--data-root",
        type=pathlib.Path,
        default=None,
        help="Override FLEXAIDS_DATA env var for dataset root",
    )
    parser.add_argument(
        "--binary",
        type=pathlib.Path,
        default=None,
        help="Explicit path to FlexAID binary",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-target timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip actual FlexAIDdS execution; test orchestrator plumbing only",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.list_datasets:
        datasets_dir = pathlib.Path(__file__).parent / "datasets"
        yamls = sorted(datasets_dir.glob("*.yaml"))
        if _RANK == 0:
            print("Available datasets:")
            for y in yamls:
                cfg = DatasetConfig(y)
                print(f"  {cfg.name:<25} {cfg.n_targets:>4} targets  [{cfg.protocol}]")
        return 0

    runner = DatasetRunner(
        dataset_name=args.dataset,
        tier=args.tier,
        output_dir=args.output,
        data_root=args.data_root,
        flexaid_binary=args.binary,
        target_timeout=args.timeout,
        dry_run=args.dry_run,
    )
    report = runner.run()
    if _RANK == 0 and report is not None:
        n_fail = report.n_targets_failed
        return 1 if n_fail > 0 else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
