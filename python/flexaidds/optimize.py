"""optimize.py — Bayesian hyperparameter tuning for FlexAIDdS GA parameters.

Wraps the FlexAID C++ executable to perform automated parameter optimization
using scipy's differential_evolution (CMA-ES compatible). Users supply a
receptor/ligand pair and optionally a known active for validation.

Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
SPDX-License-Identifier: Apache-2.0
"""

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

try:
    from scipy.optimize import differential_evolution
    HAS_SCIPY_OPTIMIZE = True
except ImportError:
    HAS_SCIPY_OPTIMIZE = False


@dataclass
class OptimizationResult:
    """Result of GA hyperparameter optimization."""
    best_params: Dict[str, float]
    best_score: float
    n_iterations: int
    history: List[Dict] = field(default_factory=list)
    converged: bool = False

    def to_json(self) -> str:
        return json.dumps({
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_iterations": self.n_iterations,
            "converged": self.converged,
        }, indent=2)


# Parameter space definition with (min, max) bounds
PARAM_BOUNDS = {
    "num_chromosomes":  (100, 5000),
    "num_generations":  (50, 2000),
    "crossover_rate":   (0.5, 0.99),
    "mutation_rate":    (0.005, 0.15),
    "sharing_alpha":    (0.5, 2.0),
    "entropy_weight":   (0.0, 1.0),
}

PARAM_NAMES = list(PARAM_BOUNDS.keys())


class GAOptimizer:
    """Automated GA parameter tuning via Bayesian optimization.

    Uses scipy.optimize.differential_evolution to search the space of
    GA parameters, running the FlexAID executable for each evaluation.

    Parameters
    ----------
    receptor : str
        Path to receptor PDB file.
    ligand : str
        Path to ligand MOL2/SDF file.
    cleft : str
        Path to cleft/grid file.
    base_config : dict, optional
        Base configuration dictionary (merged with GA param overrides).
    flexaid_exe : str, optional
        Path to FlexAIDdS executable. Defaults to 'FlexAIDdS' on PATH.
    timeout : int
        Maximum seconds per docking run (default: 300).
    """

    def __init__(
        self,
        receptor: str,
        ligand: str,
        cleft: str,
        base_config: Optional[Dict] = None,
        flexaid_exe: Optional[str] = None,
        timeout: int = 300,
    ):
        self.receptor = os.path.abspath(receptor)
        self.ligand = os.path.abspath(ligand)
        self.cleft = os.path.abspath(cleft)
        self.base_config = base_config or {}
        self.flexaid_exe = flexaid_exe or "FlexAIDdS"
        self.timeout = timeout
        self.history: List[Dict] = []

    def objective(self, params: np.ndarray) -> float:
        """Run FlexAID with given GA params, return negative best CF score.

        Lower (more negative) CF is better, so we return positive values
        for minimization. Returns a large penalty on failure.
        """
        param_dict = dict(zip(PARAM_NAMES, params))

        config = dict(self.base_config)
        config.setdefault("ga", {})
        config["ga"]["num_chromosomes"] = int(param_dict["num_chromosomes"])
        config["ga"]["num_generations"] = int(param_dict["num_generations"])
        config["ga"]["crossover_rate"] = param_dict["crossover_rate"]
        config["ga"]["mutation_rate"] = param_dict["mutation_rate"]
        config["ga"]["sharing_alpha"] = param_dict["sharing_alpha"]
        config["ga"]["entropy_weight"] = param_dict["entropy_weight"]

        try:
            score = self._run_docking(config)
            self.history.append({"params": param_dict, "score": score})
            return score  # lower is better
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
            return 1e6  # penalty for failed runs

    def _run_docking(self, config: Dict) -> float:
        """Execute a docking run with given config and return best CF score."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f)

            cmd = [
                self.flexaid_exe,
                self.receptor,
                self.ligand,
                "-c", config_path,
                "-g", self.cleft,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tmpdir,
            )

            # Parse best score from stdout (FlexAID outputs "CF=<value>")
            best_cf = 1e6
            for line in result.stdout.splitlines():
                if "CF=" in line:
                    try:
                        val = float(line.split("CF=")[1].split()[0])
                        if val < best_cf:
                            best_cf = val
                    except (ValueError, IndexError):
                        pass

            return best_cf

    def optimize(
        self,
        n_iterations: int = 50,
        method: str = "differential_evolution",
        seed: int = 42,
    ) -> OptimizationResult:
        """Run the optimization loop.

        Parameters
        ----------
        n_iterations : int
            Maximum function evaluations.
        method : str
            Optimization method (currently 'differential_evolution').
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        OptimizationResult
            Best parameters found and optimization history.
        """
        if not HAS_SCIPY_OPTIMIZE:
            raise ImportError(
                "scipy is required for optimization. "
                "Install with: pip install scipy"
            )

        bounds = [PARAM_BOUNDS[name] for name in PARAM_NAMES]

        result = differential_evolution(
            self.objective,
            bounds=bounds,
            maxiter=n_iterations,
            seed=seed,
            tol=0.01,
            disp=True,
        )

        best_params = dict(zip(PARAM_NAMES, result.x))
        # Convert integer params
        best_params["num_chromosomes"] = int(best_params["num_chromosomes"])
        best_params["num_generations"] = int(best_params["num_generations"])

        return OptimizationResult(
            best_params=best_params,
            best_score=result.fun,
            n_iterations=result.nfev,
            history=self.history,
            converged=result.success,
        )
