"""Super-cluster extraction for fast Shannon entropy collapse.

Provides a ``SuperCluster`` class that wraps the C++ FastOPTICS
``SUPER_CLUSTER_ONLY`` mode, falling back to a pure-Python density
filter when the C++ extension is not available.
"""

from __future__ import annotations

import math
from typing import List, Optional

try:
    from . import _core
    _HAS_CORE = _core is not None
except ImportError:
    _core = None
    _HAS_CORE = False


class SuperCluster:
    """Fast super-cluster extraction for Shannon entropy collapse.

    Wraps the lightweight FastOPTICS ``SUPER_CLUSTER_ONLY`` mode to
    identify the dominant energy basin from a conformational ensemble.

    Args:
        energies: List of pose energies (kcal/mol).
        min_pts:  Minimum neighbourhood size for OPTICS (default 4).

    Example:
        >>> sc = SuperCluster(energies, min_pts=4)
        >>> filtered = sc.filter_energies()
        >>> print(f"Kept {len(filtered)}/{len(energies)} poses")
    """

    def __init__(self, energies: List[float], min_pts: int = 4) -> None:
        self._energies = list(energies)
        self._min_pts = min_pts
        self._indices: Optional[List[int]] = None

    def extract(self) -> List[int]:
        """Extract super-cluster point indices.

        Returns:
            List of indices into the original energies list that belong
            to the dominant super-cluster.
        """
        if self._indices is not None:
            return self._indices

        if not self._energies:
            self._indices = []
            return self._indices

        if _HAS_CORE:
            try:
                return self._extract_cpp()
            except (AttributeError, RuntimeError):
                pass

        self._indices = self._extract_python()
        return self._indices

    def filter_energies(self) -> List[float]:
        """Return only energies belonging to the super-cluster."""
        indices = self.extract()
        return [self._energies[i] for i in indices]

    @property
    def n_total(self) -> int:
        """Total number of input energies."""
        return len(self._energies)

    @property
    def n_selected(self) -> int:
        """Number of poses in the super-cluster."""
        return len(self.extract())

    def _extract_cpp(self) -> List[int]:
        """Delegate to C++ FastOPTICSLight via _core bindings."""
        points = [_core.Point([e]) for e in self._energies]
        foptics = _core.FastOPTICSLight(points, self._min_pts)
        sc = foptics.extract_super_cluster(_core.ClusterMode.SUPER_CLUSTER_ONLY)
        self._indices = [int(i) for i in sc]
        return self._indices

    def _extract_python(self) -> List[int]:
        """Pure-Python fallback: keep poses within 0.8 std of the median energy."""
        if len(self._energies) < 2:
            return list(range(len(self._energies)))

        n = len(self._energies)
        sorted_e = sorted(self._energies)
        median = sorted_e[n // 2]

        mean = sum(self._energies) / n
        var = sum((e - mean) ** 2 for e in self._energies) / n
        std = math.sqrt(var) if var > 0 else 1.0

        cutoff = 0.8 * std
        indices = [i for i, e in enumerate(self._energies)
                   if abs(e - median) <= cutoff]

        # If the filter is too aggressive, return the full ensemble
        # (matches C++ behavior where empty extractSuperCluster falls back
        # to the unfiltered set)
        if len(indices) < self._min_pts:
            indices = list(range(n))

        return indices

    def __repr__(self) -> str:
        return (f"<SuperCluster n_selected={self.n_selected}/"
                f"{self.n_total} min_pts={self._min_pts}>")
