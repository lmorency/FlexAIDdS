"""Tests for tencom_results.py — tENCoM output parser.

Covers PDB REMARK parsing, JSON parsing, dataclass defaults,
FlexPopulationResult properties, and edge cases.
No C++ extension needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flexaidds.tencom_results import (
    EigenvalueDiff,
    FlexModeResult,
    FlexPopulationResult,
    parse_tencom_json,
    parse_tencom_pdb,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_pdb(tmp_path: Path) -> Path:
    """Write a minimal tENCoM output PDB with REMARK metadata."""
    pdb = tmp_path / "tencom_mode_0.pdb"
    pdb.write_text(
        "REMARK TENCOM_VERSION=1.0.0\n"
        "REMARK TOOL=tENCoM\n"
        "REMARK MODE_ID=0\n"
        "REMARK MODE_TYPE=reference\n"
        "REMARK SOURCE=/data/receptor.pdb\n"
        "REMARK S_VIB=0.00325\n"
        "REMARK DELTA_S_VIB=0.0\n"
        "REMARK DELTA_F_VIB=0.0\n"
        "REMARK N_MODES=50\n"
        "REMARK N_RESIDUES=100\n"
        "REMARK TEMPERATURE=300.0\n"
        "REMARK FULL_FLEXIBILITY=ON\n"
        "REMARK BFACTORS 1.2 2.3 3.4\n"
        "REMARK DELTA_BFACTORS 0.1 0.2 0.3\n"
        "REMARK PER_RESIDUE_SVIB 0.001 0.002 0.003\n"
        "REMARK PER_RESIDUE_DELTA_SVIB 0.0001 0.0002 0.0003\n"
        "REMARK EIGENVALUE_DIFF MODE=1 DELTA_EIG=0.05 OVERLAP=0.92\n"
        "REMARK EIGENVALUE_DIFF MODE=2 DELTA_EIG=-0.03 OVERLAP=0.87\n"
        "REMARK COMPOSITION ALPHA=35 BETA=20 COIL=45\n"
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C\n"
        "END\n"
    )
    return pdb


@pytest.fixture
def sample_json(tmp_path: Path) -> Path:
    """Write a minimal tENCoM JSON results file."""
    data = {
        "tool": "tENCoM",
        "version": "1.0.0",
        "temperature": 310.0,
        "full_flexibility": True,
        "modes": [
            {
                "mode_id": 0,
                "type": "reference",
                "source": "/data/ref.pdb",
                "S_vib": 0.004,
                "delta_S_vib": 0.0,
                "delta_F_vib": 0.0,
                "n_modes": 30,
                "n_residues": 80,
                "bfactors": [1.0, 2.0],
                "delta_bfactors": [],
                "per_residue_svib": [0.001, 0.002],
                "per_residue_delta_svib": [],
                "composition": {"alpha": 40, "beta": 25, "coil": 35},
                "eigenvalue_diffs": [],
            },
            {
                "mode_id": 1,
                "type": "target",
                "source": "/data/target.pdb",
                "S_vib": 0.0038,
                "delta_S_vib": -0.0002,
                "delta_F_vib": 0.062,
                "n_modes": 30,
                "n_residues": 80,
                "bfactors": [1.1, 2.1],
                "delta_bfactors": [0.1, 0.1],
                "per_residue_svib": [0.0009, 0.0019],
                "per_residue_delta_svib": [-0.0001, -0.0001],
                "composition": {},
                "eigenvalue_diffs": [
                    {"mode": 1, "delta": 0.05, "overlap": 0.91},
                    {"mode": 2, "delta": -0.02, "overlap": None},
                ],
            },
        ],
    }
    json_path = tmp_path / "tencom_results.json"
    json_path.write_text(json.dumps(data))
    return json_path


# ── EigenvalueDiff dataclass ─────────────────────────────────────────────────


class TestEigenvalueDiff:
    def test_defaults(self):
        ed = EigenvalueDiff(mode=1, delta_eigenvalue=0.05)
        assert ed.mode == 1
        assert ed.delta_eigenvalue == pytest.approx(0.05)
        assert ed.overlap is None

    def test_with_overlap(self):
        ed = EigenvalueDiff(mode=3, delta_eigenvalue=-0.1, overlap=0.95)
        assert ed.overlap == pytest.approx(0.95)


# ── FlexModeResult dataclass ────────────────────────────────────────────────


class TestFlexModeResult:
    def test_defaults(self):
        m = FlexModeResult()
        assert m.mode_id == 0
        assert m.mode_type == ""
        assert m.S_vib == 0.0
        assert m.temperature == pytest.approx(300.0)
        assert m.full_flexibility is True
        assert m.bfactors == []
        assert m.eigenvalue_diffs == []
        assert m.composition == {}

    def test_fields_set(self):
        m = FlexModeResult(
            mode_id=2,
            mode_type="target",
            S_vib=0.005,
            delta_S_vib=-0.001,
            n_residues=150,
        )
        assert m.mode_id == 2
        assert m.mode_type == "target"
        assert m.S_vib == pytest.approx(0.005)
        assert m.delta_S_vib == pytest.approx(-0.001)
        assert m.n_residues == 150


# ── FlexPopulationResult dataclass ───────────────────────────────────────────


class TestFlexPopulationResult:
    def test_reference_property(self):
        ref = FlexModeResult(mode_id=0, mode_type="reference")
        tgt = FlexModeResult(mode_id=1, mode_type="target")
        pop = FlexPopulationResult(modes=[ref, tgt])
        assert pop.reference is ref

    def test_reference_returns_none_when_missing(self):
        tgt = FlexModeResult(mode_id=1)
        pop = FlexPopulationResult(modes=[tgt])
        assert pop.reference is None

    def test_targets_property(self):
        ref = FlexModeResult(mode_id=0)
        t1 = FlexModeResult(mode_id=1, delta_F_vib=0.5)
        t2 = FlexModeResult(mode_id=2, delta_F_vib=-0.3)
        pop = FlexPopulationResult(modes=[ref, t1, t2])
        targets = pop.targets
        assert len(targets) == 2
        assert all(t.mode_id > 0 for t in targets)

    def test_sorted_by_free_energy(self):
        ref = FlexModeResult(mode_id=0)
        t1 = FlexModeResult(mode_id=1, delta_F_vib=0.5)
        t2 = FlexModeResult(mode_id=2, delta_F_vib=-0.3)
        t3 = FlexModeResult(mode_id=3, delta_F_vib=0.1)
        pop = FlexPopulationResult(modes=[ref, t1, t2, t3])
        sorted_modes = pop.sorted_by_free_energy()
        assert [m.delta_F_vib for m in sorted_modes] == pytest.approx([-0.3, 0.1, 0.5])

    def test_empty_population(self):
        pop = FlexPopulationResult()
        assert pop.reference is None
        assert pop.targets == []
        assert pop.sorted_by_free_energy() == []


# ── parse_tencom_pdb ─────────────────────────────────────────────────────────


class TestParseTencomPdb:
    def test_parses_key_value_remarks(self, sample_pdb):
        result = parse_tencom_pdb(str(sample_pdb))
        assert result.version == "1.0.0"
        assert result.mode_id == 0
        assert result.mode_type == "reference"
        assert result.source == "/data/receptor.pdb"
        assert result.S_vib == pytest.approx(0.00325)
        assert result.delta_S_vib == pytest.approx(0.0)
        assert result.delta_F_vib == pytest.approx(0.0)
        assert result.n_modes == 50
        assert result.n_residues == 100
        assert result.temperature == pytest.approx(300.0)
        assert result.full_flexibility is True

    def test_parses_bfactors(self, sample_pdb):
        result = parse_tencom_pdb(str(sample_pdb))
        assert result.bfactors == pytest.approx([1.2, 2.3, 3.4])

    def test_parses_delta_bfactors(self, sample_pdb):
        result = parse_tencom_pdb(str(sample_pdb))
        assert result.delta_bfactors == pytest.approx([0.1, 0.2, 0.3])

    def test_parses_per_residue_svib(self, sample_pdb):
        result = parse_tencom_pdb(str(sample_pdb))
        assert result.per_residue_svib == pytest.approx([0.001, 0.002, 0.003])

    def test_parses_per_residue_delta_svib(self, sample_pdb):
        result = parse_tencom_pdb(str(sample_pdb))
        assert result.per_residue_delta_svib == pytest.approx([0.0001, 0.0002, 0.0003])

    def test_parses_eigenvalue_diffs(self, sample_pdb):
        result = parse_tencom_pdb(str(sample_pdb))
        assert len(result.eigenvalue_diffs) == 2
        assert result.eigenvalue_diffs[0].mode == 1
        assert result.eigenvalue_diffs[0].delta_eigenvalue == pytest.approx(0.05)
        assert result.eigenvalue_diffs[0].overlap == pytest.approx(0.92)
        assert result.eigenvalue_diffs[1].mode == 2
        assert result.eigenvalue_diffs[1].delta_eigenvalue == pytest.approx(-0.03)
        assert result.eigenvalue_diffs[1].overlap == pytest.approx(0.87)

    def test_parses_composition(self, sample_pdb):
        result = parse_tencom_pdb(str(sample_pdb))
        assert result.composition == {"alpha": 35, "beta": 20, "coil": 45}

    def test_full_flexibility_off(self, tmp_path):
        pdb = tmp_path / "flex_off.pdb"
        pdb.write_text("REMARK FULL_FLEXIBILITY=OFF\nEND\n")
        result = parse_tencom_pdb(str(pdb))
        assert result.full_flexibility is False

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_tencom_pdb(str(tmp_path / "nonexistent.pdb"))

    def test_empty_pdb_returns_defaults(self, tmp_path):
        pdb = tmp_path / "empty.pdb"
        pdb.write_text("ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C\nEND\n")
        result = parse_tencom_pdb(str(pdb))
        assert result.mode_id == 0
        assert result.S_vib == 0.0
        assert result.bfactors == []
        assert result.eigenvalue_diffs == []

    def test_non_remark_lines_ignored(self, tmp_path):
        pdb = tmp_path / "mixed.pdb"
        pdb.write_text(
            "HEADER test\n"
            "REMARK MODE_ID=5\n"
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C\n"
            "END\n"
        )
        result = parse_tencom_pdb(str(pdb))
        assert result.mode_id == 5


# ── parse_tencom_json ────────────────────────────────────────────────────────


class TestParseTencomJson:
    def test_parses_population_metadata(self, sample_json):
        pop = parse_tencom_json(str(sample_json))
        assert pop.tool == "tENCoM"
        assert pop.version == "1.0.0"
        assert pop.temperature == pytest.approx(310.0)
        assert pop.full_flexibility is True

    def test_parses_mode_count(self, sample_json):
        pop = parse_tencom_json(str(sample_json))
        assert len(pop.modes) == 2

    def test_parses_reference_mode(self, sample_json):
        pop = parse_tencom_json(str(sample_json))
        ref = pop.reference
        assert ref is not None
        assert ref.mode_id == 0
        assert ref.mode_type == "reference"
        assert ref.source == "/data/ref.pdb"
        assert ref.S_vib == pytest.approx(0.004)
        assert ref.n_modes == 30
        assert ref.n_residues == 80
        assert ref.bfactors == pytest.approx([1.0, 2.0])
        assert ref.per_residue_svib == pytest.approx([0.001, 0.002])
        assert ref.composition == {"alpha": 40, "beta": 25, "coil": 35}

    def test_parses_target_mode(self, sample_json):
        pop = parse_tencom_json(str(sample_json))
        targets = pop.targets
        assert len(targets) == 1
        tgt = targets[0]
        assert tgt.mode_id == 1
        assert tgt.delta_S_vib == pytest.approx(-0.0002)
        assert tgt.delta_F_vib == pytest.approx(0.062)
        assert tgt.delta_bfactors == pytest.approx([0.1, 0.1])

    def test_parses_eigenvalue_diffs_from_json(self, sample_json):
        pop = parse_tencom_json(str(sample_json))
        tgt = pop.targets[0]
        assert len(tgt.eigenvalue_diffs) == 2
        assert tgt.eigenvalue_diffs[0].mode == 1
        assert tgt.eigenvalue_diffs[0].delta_eigenvalue == pytest.approx(0.05)
        assert tgt.eigenvalue_diffs[0].overlap == pytest.approx(0.91)
        assert tgt.eigenvalue_diffs[1].overlap is None

    def test_inherits_population_temperature(self, sample_json):
        pop = parse_tencom_json(str(sample_json))
        for mode in pop.modes:
            assert mode.temperature == pytest.approx(310.0)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_tencom_json(str(tmp_path / "nonexistent.json"))

    def test_empty_modes_list(self, tmp_path):
        json_path = tmp_path / "empty.json"
        json_path.write_text(json.dumps({"tool": "tENCoM", "modes": []}))
        pop = parse_tencom_json(str(json_path))
        assert pop.tool == "tENCoM"
        assert len(pop.modes) == 0
        assert pop.reference is None
        assert pop.targets == []

    def test_minimal_json(self, tmp_path):
        json_path = tmp_path / "minimal.json"
        json_path.write_text(json.dumps({}))
        pop = parse_tencom_json(str(json_path))
        assert pop.tool == ""
        assert pop.temperature == pytest.approx(300.0)
        assert len(pop.modes) == 0
