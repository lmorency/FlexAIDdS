"""test_optimize.py — Tests for the GA hyperparameter optimizer."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from flexaidds.optimize import (
    GAOptimizer,
    OptimizationResult,
    PARAM_BOUNDS,
    PARAM_NAMES,
)


class TestOptimizationResult:
    def test_creation(self):
        result = OptimizationResult(
            best_params={"num_chromosomes": 500, "mutation_rate": 0.05},
            best_score=-42.5,
            n_iterations=10,
        )
        assert result.best_score == -42.5
        assert result.best_params["num_chromosomes"] == 500
        assert result.converged is False

    def test_to_json(self):
        result = OptimizationResult(
            best_params={"mutation_rate": 0.05},
            best_score=-10.0,
            n_iterations=5,
            converged=True,
        )
        j = json.loads(result.to_json())
        assert j["best_score"] == -10.0
        assert j["converged"] is True


class TestGAOptimizer:
    def test_parameter_bounds(self):
        assert len(PARAM_BOUNDS) == 6
        assert "num_chromosomes" in PARAM_BOUNDS
        assert "mutation_rate" in PARAM_BOUNDS
        for name, (lo, hi) in PARAM_BOUNDS.items():
            assert lo < hi, f"{name}: {lo} >= {hi}"

    def test_param_names_match_bounds(self):
        assert set(PARAM_NAMES) == set(PARAM_BOUNDS.keys())

    @patch("subprocess.run")
    def test_objective_parses_output(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="Generation 10\nCF=-42.500\nbest by energy\n",
            returncode=0,
        )

        with tempfile.NamedTemporaryFile(suffix=".pdb") as pdb, \
             tempfile.NamedTemporaryFile(suffix=".mol2") as mol2, \
             tempfile.NamedTemporaryFile(suffix=".grid") as grid:
            opt = GAOptimizer(pdb.name, mol2.name, grid.name)
            params = np.array([500, 100, 0.8, 0.03, 1.0, 0.5])
            score = opt.objective(params)
            assert score == -42.5

    @patch("subprocess.run")
    def test_objective_handles_failure(self, mock_run):
        mock_run.side_effect = OSError("command not found")

        with tempfile.NamedTemporaryFile(suffix=".pdb") as pdb, \
             tempfile.NamedTemporaryFile(suffix=".mol2") as mol2, \
             tempfile.NamedTemporaryFile(suffix=".grid") as grid:
            opt = GAOptimizer(pdb.name, mol2.name, grid.name)
            params = np.array([500, 100, 0.8, 0.03, 1.0, 0.5])
            score = opt.objective(params)
            assert score == 1e6  # penalty

    def test_history_tracking(self):
        with tempfile.NamedTemporaryFile(suffix=".pdb") as pdb, \
             tempfile.NamedTemporaryFile(suffix=".mol2") as mol2, \
             tempfile.NamedTemporaryFile(suffix=".grid") as grid:
            opt = GAOptimizer(pdb.name, mol2.name, grid.name)
            assert len(opt.history) == 0
