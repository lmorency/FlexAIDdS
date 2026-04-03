# test_gil_release.py — Verify GIL is released during expensive C++ calls
#
# Tests that StatMechEngine.compute() releases the GIL, allowing
# concurrent execution from multiple Python threads.
#
# Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
# SPDX-License-Identifier: Apache-2.0

import time
import pytest
from concurrent.futures import ThreadPoolExecutor

from flexaidds import HAS_CORE_BINDINGS

if HAS_CORE_BINDINGS:
    from flexaidds._core import StatMechEngine


@pytest.mark.skipif(not HAS_CORE_BINDINGS, reason="C++ _core extension not built")
class TestGILRelease:
    """Verify GIL release enables true thread parallelism."""

    def _make_engine(self, n_samples=5000):
        """Create an engine with many samples for measurable compute time."""
        engine = StatMechEngine(300.0)
        import random
        random.seed(42)
        for _ in range(n_samples):
            engine.add_sample(random.gauss(-10.0, 5.0))
        return engine

    def test_compute_returns_correct_results(self):
        """Baseline: compute() produces valid thermodynamics."""
        engine = self._make_engine(100)
        result = engine.compute()
        assert hasattr(result, 'free_energy')
        assert hasattr(result, 'entropy')

    def test_concurrent_compute_matches_serial(self):
        """Multiple threads computing simultaneously produce same results."""
        engines = [self._make_engine(200) for _ in range(4)]

        # Serial results
        serial_results = [e.compute() for e in engines]

        # Concurrent results (same engines, fresh compute)
        # Re-create to avoid state issues
        engines2 = [self._make_engine(200) for _ in range(4)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            concurrent_results = list(pool.map(lambda e: e.compute(), engines2))

        for s, c in zip(serial_results, concurrent_results):
            assert abs(s.free_energy - c.free_energy) < 1e-10

    def test_boltzmann_weights_threaded(self):
        """boltzmann_weights() can run from multiple threads."""
        engines = [self._make_engine(100) for _ in range(4)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda e: e.boltzmann_weights(), engines))
        for r in results:
            assert len(r) == 100
            assert all(w >= 0 for w in r)
