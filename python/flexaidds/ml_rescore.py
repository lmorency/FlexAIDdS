"""ml_rescore.py — ML rescoring bridge for FlexAIDdS docking results.

Extracts Voronoi interaction graphs, Shannon entropy profiles, and
thermodynamic features from docking results for use as input to
machine learning rescoring models.

No model training is included — this module provides featurization only.
Users bring their own models (scikit-learn, PyTorch, etc.).

Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
SPDX-License-Identifier: Apache-2.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np


@dataclass
class VoronoiContactGraph:
    """Representation of pairwise atomic contacts from Voronoi CF scoring."""
    type_pair_counts: np.ndarray  # (N_TYPES, N_TYPES) contact frequency matrix
    type_pair_areas: np.ndarray   # (N_TYPES, N_TYPES) total contact area matrix
    n_contacts: int = 0
    n_types: int = 40  # FlexAID default type count


@dataclass
class ThermoFeatures:
    """Thermodynamic properties extracted from docking results."""
    free_energy: float = 0.0       # F (kcal/mol)
    mean_energy: float = 0.0       # <E> (kcal/mol)
    entropy: float = 0.0           # S (kcal/mol/K)
    heat_capacity: float = 0.0     # Cv (kcal/mol/K²)
    boltzmann_weight: float = 0.0  # normalized Boltzmann probability
    shannon_entropy: float = 0.0   # configurational entropy (nats)


class VoronoiGraphExtractor:
    """Extract Voronoi contact graphs from docking result files.

    Parses .rrd (result) files containing REMARK lines with contact
    type-pair contribution data.
    """

    def __init__(self, n_types: int = 40):
        self.n_types = n_types

    def extract(self, result_file: str) -> VoronoiContactGraph:
        """Extract contact graph from a single result PDB/RRD file.

        Parameters
        ----------
        result_file : str
            Path to docking result file with REMARK CF data.

        Returns
        -------
        VoronoiContactGraph
            Contact type-pair frequency and area matrices.
        """
        graph = VoronoiContactGraph(
            type_pair_counts=np.zeros((self.n_types, self.n_types)),
            type_pair_areas=np.zeros((self.n_types, self.n_types)),
            n_types=self.n_types,
        )

        try:
            with open(result_file, "r") as f:
                for line in f:
                    if not line.startswith("REMARK"):
                        continue
                    parts = line.strip().split()
                    # Parse contribution lines: REMARK CONTRIBUTION type1 type2 area
                    if len(parts) >= 5 and parts[1] == "CONTRIBUTION":
                        try:
                            t1 = int(parts[2]) - 1  # 0-indexed
                            t2 = int(parts[3]) - 1
                            area = float(parts[4])
                            if 0 <= t1 < self.n_types and 0 <= t2 < self.n_types:
                                graph.type_pair_counts[t1, t2] += 1
                                graph.type_pair_areas[t1, t2] += area
                                graph.n_contacts += 1
                        except (ValueError, IndexError):
                            pass
        except (OSError, IOError):
            pass

        return graph

    def extract_batch(self, result_files: List[str]) -> List[VoronoiContactGraph]:
        """Extract contact graphs from multiple result files."""
        return [self.extract(f) for f in result_files]


class ShannonProfileExtractor:
    """Extract Shannon entropy trajectory from GA generation logs."""

    def extract(self, log_file: str) -> np.ndarray:
        """Extract entropy values from generation log output.

        Parses lines matching: [SMFREE] gen=N ... S=<value> ...

        Returns
        -------
        np.ndarray
            Array of entropy values per generation.
        """
        entropies = []
        try:
            with open(log_file, "r") as f:
                for line in f:
                    if "[SMFREE]" in line and "S=" in line:
                        parts = line.strip().split()
                        for part in parts:
                            if part.startswith("S="):
                                try:
                                    entropies.append(float(part[2:]))
                                except ValueError:
                                    pass
        except (OSError, IOError):
            pass

        return np.array(entropies, dtype=np.float64)


class FeatureBuilder:
    """Combine Voronoi graph + Shannon profile + thermodynamic properties
    into a unified feature vector for ML rescoring."""

    def __init__(self, n_types: int = 40, max_generations: int = 500):
        self.n_types = n_types
        self.max_generations = max_generations

    def build(
        self,
        graph: VoronoiContactGraph,
        shannon_profile: Optional[np.ndarray] = None,
        thermo: Optional[ThermoFeatures] = None,
    ) -> np.ndarray:
        """Build a feature vector from docking result components.

        Feature layout:
        - Upper triangle of contact count matrix (flattened): n_types*(n_types+1)/2
        - Upper triangle of contact area matrix (flattened): n_types*(n_types+1)/2
        - Shannon entropy profile (zero-padded to max_generations)
        - Thermodynamic properties (6 values)

        Returns
        -------
        np.ndarray
            1D feature vector.
        """
        # Upper triangle of count and area matrices
        tri_idx = np.triu_indices(self.n_types)
        count_features = graph.type_pair_counts[tri_idx]
        area_features = graph.type_pair_areas[tri_idx]

        # Shannon profile (zero-padded)
        if shannon_profile is not None and len(shannon_profile) > 0:
            profile = np.zeros(self.max_generations)
            n = min(len(shannon_profile), self.max_generations)
            profile[:n] = shannon_profile[:n]
        else:
            profile = np.zeros(self.max_generations)

        # Thermodynamic features
        if thermo is not None:
            thermo_vec = np.array([
                thermo.free_energy,
                thermo.mean_energy,
                thermo.entropy,
                thermo.heat_capacity,
                thermo.boltzmann_weight,
                thermo.shannon_entropy,
            ])
        else:
            thermo_vec = np.zeros(6)

        return np.concatenate([count_features, area_features, profile, thermo_vec])

    def feature_names(self) -> List[str]:
        """Return human-readable feature names."""
        names = []
        for i in range(self.n_types):
            for j in range(i, self.n_types):
                names.append(f"count_{i}_{j}")
        for i in range(self.n_types):
            for j in range(i, self.n_types):
                names.append(f"area_{i}_{j}")
        for g in range(self.max_generations):
            names.append(f"shannon_gen_{g}")
        names.extend([
            "free_energy", "mean_energy", "entropy",
            "heat_capacity", "boltzmann_weight", "shannon_entropy",
        ])
        return names

    def feature_dim(self) -> int:
        """Total number of features."""
        tri = self.n_types * (self.n_types + 1) // 2
        return tri * 2 + self.max_generations + 6


class MLRescorer:
    """Apply a user-provided ML model to rescore docking poses.

    The model must implement a `predict(X)` method compatible with
    scikit-learn or similar interfaces.
    """

    def __init__(self, model: Any = None):
        self.model = model

    def rescore(self, features: np.ndarray) -> float:
        """Rescore a single feature vector.

        Parameters
        ----------
        features : np.ndarray
            Feature vector from FeatureBuilder.build().

        Returns
        -------
        float
            Predicted binding affinity or score.
        """
        if self.model is None:
            raise ValueError("No model loaded. Set self.model first.")
        prediction = self.model.predict(features.reshape(1, -1))
        return float(prediction[0])

    def rescore_batch(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Rescore multiple feature vectors.

        Parameters
        ----------
        feature_matrix : np.ndarray
            (N, D) matrix of feature vectors.

        Returns
        -------
        np.ndarray
            (N,) array of predicted scores.
        """
        if self.model is None:
            raise ValueError("No model loaded. Set self.model first.")
        return np.array(self.model.predict(feature_matrix), dtype=np.float64)
