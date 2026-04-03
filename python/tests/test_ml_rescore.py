"""test_ml_rescore.py — Tests for ML rescoring bridge."""

import os
import tempfile

import numpy as np
import pytest

from flexaidds.ml_rescore import (
    VoronoiContactGraph,
    VoronoiGraphExtractor,
    ShannonProfileExtractor,
    FeatureBuilder,
    ThermoFeatures,
    MLRescorer,
)


class TestVoronoiGraphExtractor:
    def test_extract_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
            f.write("ATOM      1  N   ALA A   1       0.000   0.000   0.000\n")
            f.write("END\n")
            fname = f.name

        try:
            extractor = VoronoiGraphExtractor(n_types=40)
            graph = extractor.extract(fname)
            assert graph.n_contacts == 0
            assert graph.type_pair_counts.shape == (40, 40)
        finally:
            os.unlink(fname)

    def test_extract_with_contributions(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
            f.write("REMARK CONTRIBUTION 1 2 3.14\n")
            f.write("REMARK CONTRIBUTION 1 2 2.71\n")
            f.write("REMARK CONTRIBUTION 5 10 1.00\n")
            fname = f.name

        try:
            extractor = VoronoiGraphExtractor(n_types=40)
            graph = extractor.extract(fname)
            assert graph.n_contacts == 3
            assert graph.type_pair_counts[0, 1] == 2  # type 1→0, type 2→1
            assert abs(graph.type_pair_areas[0, 1] - 5.85) < 0.01
            assert graph.type_pair_counts[4, 9] == 1
        finally:
            os.unlink(fname)

    def test_nonexistent_file(self):
        extractor = VoronoiGraphExtractor()
        graph = extractor.extract("/nonexistent/path.pdb")
        assert graph.n_contacts == 0

    def test_batch_extract(self):
        extractor = VoronoiGraphExtractor()
        results = extractor.extract_batch([])
        assert results == []


class TestShannonProfileExtractor:
    def test_extract_from_log(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("[SMFREE] gen=1  F=-10.0  <E>=-5.0  S=0.003000  Cv=0.1\n")
            f.write("[SMFREE] gen=50  F=-12.0  <E>=-6.0  S=0.004500  Cv=0.2\n")
            f.write("Some other output\n")
            fname = f.name

        try:
            extractor = ShannonProfileExtractor()
            profile = extractor.extract(fname)
            assert len(profile) == 2
            assert abs(profile[0] - 0.003) < 1e-6
            assert abs(profile[1] - 0.0045) < 1e-6
        finally:
            os.unlink(fname)

    def test_empty_log(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("No entropy data here\n")
            fname = f.name

        try:
            extractor = ShannonProfileExtractor()
            profile = extractor.extract(fname)
            assert len(profile) == 0
        finally:
            os.unlink(fname)


class TestFeatureBuilder:
    def test_feature_dimensions(self):
        builder = FeatureBuilder(n_types=10, max_generations=100)
        dim = builder.feature_dim()
        tri = 10 * 11 // 2  # 55
        assert dim == tri * 2 + 100 + 6  # 55*2 + 100 + 6 = 216

    def test_build_with_all_components(self):
        builder = FeatureBuilder(n_types=5, max_generations=10)
        graph = VoronoiContactGraph(
            type_pair_counts=np.ones((5, 5)),
            type_pair_areas=np.ones((5, 5)) * 2.0,
            n_types=5,
        )
        profile = np.array([0.1, 0.2, 0.3])
        thermo = ThermoFeatures(
            free_energy=-10.0, entropy=0.003, heat_capacity=0.1
        )

        features = builder.build(graph, profile, thermo)
        assert features.ndim == 1
        assert len(features) == builder.feature_dim()

    def test_build_without_optional(self):
        builder = FeatureBuilder(n_types=5, max_generations=10)
        graph = VoronoiContactGraph(
            type_pair_counts=np.zeros((5, 5)),
            type_pair_areas=np.zeros((5, 5)),
            n_types=5,
        )
        features = builder.build(graph)
        assert len(features) == builder.feature_dim()
        # All should be zero
        assert np.allclose(features, 0.0)

    def test_feature_names(self):
        builder = FeatureBuilder(n_types=5, max_generations=10)
        names = builder.feature_names()
        assert len(names) == builder.feature_dim()
        assert names[-1] == "shannon_entropy"
        assert names[-6] == "free_energy"


class TestMLRescorer:
    def test_no_model_raises(self):
        rescorer = MLRescorer()
        with pytest.raises(ValueError, match="No model loaded"):
            rescorer.rescore(np.zeros(10))

    def test_mock_model(self):
        class MockModel:
            def predict(self, X):
                return np.array([-42.0] * len(X))

        rescorer = MLRescorer(model=MockModel())
        score = rescorer.rescore(np.zeros(10))
        assert score == -42.0

    def test_batch_rescore(self):
        class MockModel:
            def predict(self, X):
                return np.sum(X, axis=1)

        rescorer = MLRescorer(model=MockModel())
        features = np.ones((5, 10))
        scores = rescorer.rescore_batch(features)
        assert len(scores) == 5
        assert np.allclose(scores, 10.0)
