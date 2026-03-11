"""Tests for from_dict / from_json round-trip deserialization on model classes.

Priority 4 coverage.  No C++ extension needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flexaidds.models import BindingModeResult, DockingResult, PoseResult


# ===========================================================================
# PoseResult.from_dict
# ===========================================================================

class TestPoseResultFromDict:
    def test_round_trip_all_fields(self):
        original = PoseResult(
            path=Path("/data/pose_1.pdb"),
            mode_id=2,
            pose_rank=3,
            cf=-42.5,
            cf_app=-41.0,
            rmsd_raw=1.2,
            rmsd_sym=0.9,
            free_energy=-40.0,
            enthalpy=-38.0,
            entropy=0.0033,
            heat_capacity=0.05,
            std_energy=0.3,
            temperature=300.0,
            remarks={"run_id": 7},
        )
        data = {
            "path": str(original.path),
            "mode_id": original.mode_id,
            "pose_rank": original.pose_rank,
            "cf": original.cf,
            "cf_app": original.cf_app,
            "rmsd_raw": original.rmsd_raw,
            "rmsd_sym": original.rmsd_sym,
            "free_energy": original.free_energy,
            "enthalpy": original.enthalpy,
            "entropy": original.entropy,
            "heat_capacity": original.heat_capacity,
            "std_energy": original.std_energy,
            "temperature": original.temperature,
            "remarks": original.remarks,
        }
        restored = PoseResult.from_dict(data)
        assert restored.path == original.path
        assert restored.mode_id == original.mode_id
        assert restored.pose_rank == original.pose_rank
        assert restored.cf == pytest.approx(original.cf)
        assert restored.cf_app == pytest.approx(original.cf_app)
        assert restored.free_energy == pytest.approx(original.free_energy)
        assert restored.temperature == pytest.approx(original.temperature)
        assert restored.remarks == original.remarks

    def test_defaults_for_missing_keys(self):
        restored = PoseResult.from_dict({})
        assert restored.path == Path("")
        assert restored.mode_id == 0
        assert restored.pose_rank == 0
        assert restored.cf is None
        assert restored.free_energy is None
        assert restored.remarks == {}

    def test_accepts_best_pose_path_alias(self):
        restored = PoseResult.from_dict({"best_pose_path": "/data/pose.pdb"})
        assert restored.path == Path("/data/pose.pdb")

    def test_path_field_takes_precedence_over_alias(self):
        restored = PoseResult.from_dict({
            "path": "/real.pdb",
            "best_pose_path": "/alias.pdb",
        })
        assert restored.path == Path("/real.pdb")

    def test_accepts_path_object(self):
        restored = PoseResult.from_dict({"path": Path("/data/pose.pdb")})
        assert restored.path == Path("/data/pose.pdb")


# ===========================================================================
# BindingModeResult.from_dict
# ===========================================================================

class TestBindingModeResultFromDict:
    def test_round_trip_with_poses(self):
        pose_data = {
            "path": "/data/pose_1.pdb",
            "mode_id": 1,
            "pose_rank": 1,
            "cf": -10.0,
        }
        data = {
            "mode_id": 1,
            "rank": 1,
            "poses": [pose_data],
            "free_energy": -9.5,
            "enthalpy": -8.0,
            "entropy": 0.005,
            "best_cf": -10.0,
            "frequency": 15,
            "temperature": 300.0,
            "metadata": {"run_id": 42},
        }
        restored = BindingModeResult.from_dict(data)
        assert restored.mode_id == 1
        assert restored.rank == 1
        assert restored.n_poses == 1
        assert restored.poses[0].cf == pytest.approx(-10.0)
        assert restored.free_energy == pytest.approx(-9.5)
        assert restored.frequency == 15
        assert restored.metadata == {"run_id": 42}

    def test_defaults_for_missing_keys(self):
        restored = BindingModeResult.from_dict({})
        assert restored.mode_id == 0
        assert restored.rank == 0
        assert restored.poses == []
        assert restored.free_energy is None
        assert restored.metadata == {}

    def test_multiple_poses_preserved(self):
        data = {
            "mode_id": 1,
            "rank": 1,
            "poses": [
                {"path": "a.pdb", "mode_id": 1, "pose_rank": 1, "cf": -12.0},
                {"path": "b.pdb", "mode_id": 1, "pose_rank": 2, "cf": -10.0},
            ],
        }
        restored = BindingModeResult.from_dict(data)
        assert restored.n_poses == 2
        assert restored.poses[0].cf == pytest.approx(-12.0)
        assert restored.poses[1].cf == pytest.approx(-10.0)


# ===========================================================================
# DockingResult.from_dict
# ===========================================================================

class TestDockingResultFromDict:
    def test_round_trip_with_nested_modes(self):
        data = {
            "source_dir": "/data/output",
            "temperature": 300.0,
            "metadata": {"n_pose_files": 4},
            "binding_modes": [
                {
                    "mode_id": 1,
                    "rank": 1,
                    "poses": [
                        {"path": "p1.pdb", "mode_id": 1, "pose_rank": 1, "cf": -10.0},
                    ],
                    "free_energy": -9.5,
                },
                {
                    "mode_id": 2,
                    "rank": 2,
                    "poses": [],
                    "free_energy": -7.0,
                },
            ],
        }
        restored = DockingResult.from_dict(data)
        assert restored.source_dir == Path("/data/output")
        assert restored.temperature == pytest.approx(300.0)
        assert restored.n_modes == 2
        assert restored.binding_modes[0].free_energy == pytest.approx(-9.5)
        assert restored.binding_modes[0].n_poses == 1
        assert restored.metadata == {"n_pose_files": 4}

    def test_from_flat_records(self):
        """Accepts flat records from to_records() (no poses key)."""
        data = {
            "source_dir": "/out",
            "binding_modes": [
                {"mode_id": 1, "rank": 1, "free_energy": -10.0, "best_cf": -12.0},
                {"mode_id": 2, "rank": 2, "free_energy": -8.0},
            ],
        }
        restored = DockingResult.from_dict(data)
        assert restored.n_modes == 2
        assert restored.binding_modes[0].free_energy == pytest.approx(-10.0)
        assert restored.binding_modes[0].poses == []
        assert restored.binding_modes[1].free_energy == pytest.approx(-8.0)

    def test_defaults_for_missing_keys(self):
        restored = DockingResult.from_dict({})
        assert restored.source_dir == Path(".")
        assert restored.binding_modes == []
        assert restored.temperature is None
        assert restored.metadata == {}


# ===========================================================================
# DockingResult.from_json – file-based
# ===========================================================================

class TestDockingResultFromJson:
    def _make_result(self) -> DockingResult:
        pose = PoseResult(
            path=Path("/data/pose_1.pdb"),
            mode_id=1,
            pose_rank=1,
            cf=-42.5,
            free_energy=-41.0,
            temperature=300.0,
        )
        mode = BindingModeResult(
            mode_id=1,
            rank=1,
            poses=[pose],
            free_energy=-41.0,
            best_cf=-42.5,
            temperature=300.0,
        )
        return DockingResult(
            source_dir=Path("/data/output"),
            binding_modes=[mode],
            temperature=300.0,
            metadata={"n_pose_files": 1},
        )

    def test_round_trip_via_file(self, tmp_path):
        original = self._make_result()
        json_path = tmp_path / "results.json"
        original.to_json(json_path)

        restored = DockingResult.from_json(json_path)
        assert restored.n_modes == original.n_modes
        assert restored.temperature == pytest.approx(original.temperature)
        assert restored.binding_modes[0].free_energy == pytest.approx(-41.0)
        assert restored.metadata == {"n_pose_files": 1}

    def test_round_trip_via_string(self):
        original = self._make_result()
        json_text = original.to_json()

        restored = DockingResult.from_json(json_text)
        assert restored.n_modes == original.n_modes
        assert restored.temperature == pytest.approx(original.temperature)

    def test_from_json_invalid_string_raises(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            DockingResult.from_json("not valid json {{{")

    def test_round_trip_preserves_mode_count(self, tmp_path):
        pose1 = PoseResult(path=Path("a.pdb"), mode_id=1, pose_rank=1, cf=-10.0)
        pose2 = PoseResult(path=Path("b.pdb"), mode_id=2, pose_rank=1, cf=-8.0)
        mode1 = BindingModeResult(mode_id=1, rank=1, poses=[pose1], free_energy=-9.5)
        mode2 = BindingModeResult(mode_id=2, rank=2, poses=[pose2], free_energy=-7.5)
        original = DockingResult(
            source_dir=tmp_path,
            binding_modes=[mode1, mode2],
            temperature=310.0,
        )
        json_path = tmp_path / "out.json"
        original.to_json(json_path)

        restored = DockingResult.from_json(json_path)
        assert restored.n_modes == 2
        assert restored.binding_modes[0].mode_id == 1
        assert restored.binding_modes[1].mode_id == 2
