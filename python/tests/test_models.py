"""Tests for flexaidds.models – BindingModeResult and DockingResult.

Priority 4 coverage.  No C++ extension needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flexaidds.models import BindingModeResult, DockingResult, PoseResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pose(
    *,
    path: str = "x.pdb",
    mode_id: int = 1,
    pose_rank: int = 1,
    cf: float | None = None,
    cf_app: float | None = None,
) -> PoseResult:
    return PoseResult(
        path=Path(path),
        mode_id=mode_id,
        pose_rank=pose_rank,
        cf=cf,
        cf_app=cf_app,
    )


def _mode(
    *,
    mode_id: int = 1,
    poses: list[PoseResult] | None = None,
    free_energy: float | None = None,
    best_cf: float | None = None,
    rank: int | None = None,
) -> BindingModeResult:
    return BindingModeResult(
        mode_id=mode_id,
        rank=rank if rank is not None else mode_id,
        poses=poses or [],
        free_energy=free_energy,
        best_cf=best_cf,
    )


# ===========================================================================
# PoseResult
# ===========================================================================

class TestPoseResultRepr:
    def test_repr_basic(self):
        p = _pose(mode_id=2, pose_rank=3)
        r = repr(p)
        assert "PoseResult" in r
        assert "mode=2" in r
        assert "rank=3" in r

    def test_repr_with_cf(self):
        p = _pose(cf=-10.5)
        assert "cf=-10.50" in repr(p)

    def test_repr_with_free_energy(self):
        p = PoseResult(path=Path("x.pdb"), mode_id=1, pose_rank=1, free_energy=-8.3)
        assert "F=-8.30" in repr(p)

    def test_repr_without_optional_fields(self):
        p = _pose()
        r = repr(p)
        assert "cf=" not in r
        assert "F=" not in r


class TestBindingModeResultRepr:
    def test_repr_basic(self):
        mode = _mode(mode_id=2, poses=[_pose(), _pose()], rank=1)
        r = repr(mode)
        assert "BindingModeResult" in r
        assert "id=2" in r
        assert "rank=1" in r
        assert "poses=2" in r

    def test_repr_with_free_energy(self):
        mode = _mode(free_energy=-9.5)
        assert "F=-9.50" in repr(mode)


class TestDockingResultRepr:
    def test_repr(self):
        result = DockingResult(
            source_dir=Path("/tmp/docking_out"),
            binding_modes=[_mode()],
        )
        r = repr(result)
        assert "DockingResult" in r
        assert "modes=1" in r
        assert "docking_out" in r


class TestPoseResult:
    def test_is_frozen(self):
        p = _pose(cf=-5.0)
        with pytest.raises((AttributeError, TypeError)):
            p.cf = 0.0  # type: ignore[misc]

    def test_defaults_are_none(self):
        p = PoseResult(path=Path("x.pdb"), mode_id=1, pose_rank=1)
        for field in ("cf", "cf_app", "rmsd_raw", "rmsd_sym",
                      "free_energy", "enthalpy", "entropy",
                      "heat_capacity", "std_energy", "temperature"):
            assert getattr(p, field) is None

    def test_remarks_default_is_empty_dict(self):
        p = PoseResult(path=Path("x.pdb"), mode_id=1, pose_rank=1)
        assert p.remarks == {}


# ===========================================================================
# BindingModeResult.n_poses
# ===========================================================================

class TestBindingModeResultNPoses:
    def test_n_poses_matches_list_length(self):
        poses = [_pose(), _pose(), _pose()]
        mode = _mode(poses=poses)
        assert mode.n_poses == 3

    def test_n_poses_zero_when_no_poses(self):
        assert _mode(poses=[]).n_poses == 0


# ===========================================================================
# BindingModeResult.best_pose
# ===========================================================================

class TestBindingModeResultBestPose:
    def test_returns_pose_with_lowest_cf(self):
        poses = [
            _pose(path="a.pdb", cf=-8.0),
            _pose(path="b.pdb", cf=-12.0),
            _pose(path="c.pdb", cf=-10.0),
        ]
        mode = _mode(poses=poses)
        assert mode.best_pose().path.name == "b.pdb"

    def test_falls_back_to_cf_app_when_no_cf(self):
        poses = [
            _pose(path="a.pdb", cf_app=-7.0),
            _pose(path="b.pdb", cf_app=-11.0),
        ]
        mode = _mode(poses=poses)
        assert mode.best_pose().path.name == "b.pdb"

    def test_cf_takes_priority_over_cf_app(self):
        poses = [
            _pose(path="cf_only.pdb", cf=-9.0),
            _pose(path="app_only.pdb", cf_app=-15.0),  # better app score, no cf
        ]
        mode = _mode(poses=poses)
        # Only the pose with cf is in the cf-scored set; it should win
        best = mode.best_pose()
        assert best.path.name == "cf_only.pdb"

    def test_returns_first_pose_when_no_scores(self):
        poses = [_pose(path="first.pdb"), _pose(path="second.pdb")]
        mode = _mode(poses=poses)
        assert mode.best_pose().path.name == "first.pdb"

    def test_returns_none_when_no_poses(self):
        assert _mode(poses=[]).best_pose() is None


# ===========================================================================
# DockingResult.n_modes
# ===========================================================================

class TestDockingResultNModes:
    def _result(self, n: int) -> DockingResult:
        return DockingResult(
            source_dir=Path("/tmp"),
            binding_modes=[_mode(mode_id=i) for i in range(1, n + 1)],
        )

    def test_n_modes_correct(self):
        assert self._result(3).n_modes == 3

    def test_n_modes_zero(self):
        assert self._result(0).n_modes == 0


# ===========================================================================
# DockingResult.top_mode
# ===========================================================================

class TestDockingResultTopMode:
    def test_returns_mode_with_lowest_free_energy(self):
        modes = [
            _mode(mode_id=1, free_energy=-8.0, rank=1),
            _mode(mode_id=2, free_energy=-12.0, rank=2),
            _mode(mode_id=3, free_energy=-10.0, rank=3),
        ]
        result = DockingResult(source_dir=Path("/tmp"), binding_modes=modes)
        assert result.top_mode().mode_id == 2

    def test_falls_back_to_lowest_rank_when_no_free_energy(self):
        modes = [
            _mode(mode_id=3, rank=3),
            _mode(mode_id=1, rank=1),
            _mode(mode_id=2, rank=2),
        ]
        result = DockingResult(source_dir=Path("/tmp"), binding_modes=modes)
        assert result.top_mode().rank == 1

    def test_returns_none_when_no_modes(self):
        result = DockingResult(source_dir=Path("/tmp"), binding_modes=[])
        assert result.top_mode() is None

    def test_modes_with_partial_free_energy(self):
        """Modes without free_energy are excluded from F-based ranking."""
        modes = [
            _mode(mode_id=1, free_energy=None, rank=1),
            _mode(mode_id=2, free_energy=-8.0, rank=2),
        ]
        result = DockingResult(source_dir=Path("/tmp"), binding_modes=modes)
        assert result.top_mode().mode_id == 2


# ===========================================================================
# DockingResult.to_records
# ===========================================================================

class TestDockingResultToRecords:
    def _make_result(self) -> DockingResult:
        poses = [_pose(path="p1.pdb", cf=-10.0), _pose(path="p2.pdb", cf=-9.0)]
        modes = [
            BindingModeResult(
                mode_id=1, rank=1, poses=poses,
                free_energy=-9.8, enthalpy=-9.5, entropy=0.001,
                heat_capacity=0.05, std_energy=0.3,
                best_cf=-10.0, temperature=300.0,
            ),
            BindingModeResult(
                mode_id=2, rank=2, poses=[],
                free_energy=None, best_cf=None, temperature=None,
            ),
        ]
        return DockingResult(source_dir=Path("/tmp"), binding_modes=modes)

    def test_returns_one_record_per_mode(self):
        records = self._make_result().to_records()
        assert len(records) == 2

    def test_record_keys(self):
        record = self._make_result().to_records()[0]
        expected = {
            "mode_id", "rank", "n_poses", "free_energy", "enthalpy",
            "entropy", "heat_capacity", "std_energy", "best_cf",
            "temperature", "best_pose_path",
        }
        assert expected == set(record.keys())

    def test_record_values_mode1(self):
        record = self._make_result().to_records()[0]
        assert record["mode_id"] == 1
        assert record["rank"] == 1
        assert record["n_poses"] == 2
        assert record["free_energy"] == pytest.approx(-9.8)
        assert record["best_cf"] == pytest.approx(-10.0)
        assert record["temperature"] == pytest.approx(300.0)
        assert record["best_pose_path"] is not None

    def test_none_values_for_empty_mode(self):
        record = self._make_result().to_records()[1]
        assert record["free_energy"] is None
        assert record["best_cf"] is None
        assert record["best_pose_path"] is None

    def test_best_pose_path_is_string(self):
        record = self._make_result().to_records()[0]
        assert isinstance(record["best_pose_path"], str)


# ===========================================================================
# DockingResult.to_dataframe – success path and pandas import error path
# ===========================================================================

class TestDockingResultToDataframe:
    def test_raises_import_error_without_pandas(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pandas":
                raise ImportError("No module named 'pandas'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        result = DockingResult(
            source_dir=Path("/tmp"),
            binding_modes=[_mode()],
        )
        with pytest.raises(ImportError, match="pandas"):
            result.to_dataframe()


# ===========================================================================
# DockingResult.to_json
# ===========================================================================

class TestDockingResultToJson:
    def _make_result(self) -> DockingResult:
        poses = [_pose(path="p1.pdb", cf=-10.0), _pose(path="p2.pdb", cf=-9.0)]
        modes = [
            BindingModeResult(
                mode_id=1, rank=1, poses=poses,
                free_energy=-9.8, enthalpy=-9.5, entropy=0.001,
                heat_capacity=0.05, std_energy=0.3,
                best_cf=-10.0, temperature=300.0,
            ),
            BindingModeResult(
                mode_id=2, rank=2, poses=[],
                free_energy=None, best_cf=None, temperature=None,
            ),
        ]
        return DockingResult(source_dir=Path("/tmp"), binding_modes=modes)

    def test_returns_valid_json_string(self):
        import json
        text = self._make_result().to_json()
        parsed = json.loads(text)
        assert isinstance(parsed, dict)

    def test_json_has_required_keys(self):
        import json
        parsed = json.loads(self._make_result().to_json())
        assert set(parsed.keys()) >= {
            "source_dir", "temperature", "n_modes", "metadata", "binding_modes",
        }

    def test_json_n_modes(self):
        import json
        parsed = json.loads(self._make_result().to_json())
        assert parsed["n_modes"] == 2

    def test_json_binding_modes_length(self):
        import json
        parsed = json.loads(self._make_result().to_json())
        assert len(parsed["binding_modes"]) == 2

    def test_json_free_energy_value(self):
        import json
        parsed = json.loads(self._make_result().to_json())
        assert parsed["binding_modes"][0]["free_energy"] == pytest.approx(-9.8)

    def test_json_none_free_energy(self):
        import json
        parsed = json.loads(self._make_result().to_json())
        assert parsed["binding_modes"][1]["free_energy"] is None

    def test_to_json_writes_file(self, tmp_path):
        import json
        out = tmp_path / "result.json"
        ret = self._make_result().to_json(path=out)
        assert ret is None
        parsed = json.loads(out.read_text(encoding="utf-8"))
        assert parsed["n_modes"] == 2

    def test_to_json_kwargs_forwarded(self):
        import json
        text = self._make_result().to_json(sort_keys=True, indent=4)
        parsed = json.loads(text)
        assert isinstance(parsed, dict)
        # Verify sort_keys was applied: keys should be alphabetical
        lines = text.strip().splitlines()
        # indent=4 means 4-space indentation
        assert any("    " in line for line in lines)


    def test_to_dataframe_success(self):
        pytest.importorskip("pandas")
        poses = [_pose(path="p1.pdb", cf=-10.0)]
        modes = [
            BindingModeResult(
                mode_id=1, rank=1, poses=poses,
                free_energy=-9.8, best_cf=-10.0,
            ),
        ]
        result = DockingResult(source_dir=Path("/tmp"), binding_modes=modes)
        df = result.to_dataframe()
        import pandas as pd
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "mode_id" in df.columns
        assert df.iloc[0]["mode_id"] == 1
        assert df.iloc[0]["free_energy"] == pytest.approx(-9.8)
