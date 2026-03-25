import csv
import json
from pathlib import Path

from flexaidds.models import DockingResult
from flexaidds.results import load_results


def _write_pdb(path: Path, remarks: list[str]) -> None:
    lines = [f"REMARK {line}\n" for line in remarks]
    lines.extend(
        [
            "ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00           C\n",
            "END\n",
        ]
    )
    path.write_text("".join(lines), encoding="utf-8")


def test_load_results_groups_poses_by_mode(tmp_path: Path) -> None:
    _write_pdb(
        tmp_path / "binding_mode_1_pose_1.pdb",
        [
            "binding_mode = 1",
            "pose_rank = 1",
            "CF = -42.5",
            "free_energy = -41.0",
            "enthalpy = -40.0",
            "entropy = 0.0033",
            "temperature = 300.0",
        ],
    )
    _write_pdb(
        tmp_path / "binding_mode_1_pose_2.pdb",
        [
            "binding_mode = 1",
            "pose_rank = 2",
            "CF = -39.0",
            "temperature = 300.0",
        ],
    )
    _write_pdb(
        tmp_path / "binding_mode_2_pose_1.pdb",
        [
            "binding_mode = 2",
            "pose_rank = 1",
            "CF = -35.0",
            "free_energy = -34.2",
            "temperature = 300.0",
        ],
    )

    result = load_results(tmp_path)

    assert result.n_modes == 2
    assert result.temperature == 300.0
    assert result.binding_modes[0].mode_id == 1
    assert result.binding_modes[0].n_poses == 2
    assert result.binding_modes[0].best_cf == -42.5
    assert result.binding_modes[0].free_energy == -41.0
    assert result.binding_modes[1].mode_id == 2
    assert result.binding_modes[1].best_cf == -35.0


def test_load_results_uses_filename_heuristics_when_remarks_missing(tmp_path: Path) -> None:
    _write_pdb(
        tmp_path / "mode_7_pose_3.pdb",
        [
            "CF = -11.5",
            "temperature = 298.15",
        ],
    )
    _write_pdb(
        tmp_path / "cluster-12_conformer-4.pdb",
        [
            "CF = -9.25",
            "temperature = 298.15",
        ],
    )

    result = load_results(tmp_path)
    first_mode = result.binding_modes[0]
    first_pose = first_mode.poses[0]
    second_mode = result.binding_modes[1]
    second_pose = second_mode.poses[0]

    assert first_mode.mode_id == 7
    assert first_pose.pose_rank == 3
    assert first_mode.best_cf == -11.5
    assert second_mode.mode_id == 12
    assert second_pose.pose_rank == 4
    assert second_mode.best_cf == -9.25


def test_from_json_round_trip_string(tmp_path: Path) -> None:
    """to_json() → from_json() round-trips mode-level scalars via string."""
    _write_pdb(
        tmp_path / "binding_mode_1_pose_1.pdb",
        [
            "binding_mode = 1",
            "pose_rank = 1",
            "CF = -42.5",
            "free_energy = -41.0",
            "enthalpy = -40.0",
            "entropy = 0.0033",
            "heat_capacity = 0.012",
            "std_energy = 1.5",
            "temperature = 300.0",
        ],
    )
    _write_pdb(
        tmp_path / "binding_mode_2_pose_1.pdb",
        [
            "binding_mode = 2",
            "pose_rank = 1",
            "CF = -35.0",
            "free_energy = -34.2",
            "temperature = 300.0",
        ],
    )

    original = load_results(tmp_path)
    json_text = original.to_json()
    restored = DockingResult.from_json(json_text)

    assert restored.n_modes == original.n_modes
    assert restored.temperature == original.temperature
    assert str(restored.source_dir) == str(original.source_dir)

    for orig_mode, rest_mode in zip(original.binding_modes, restored.binding_modes):
        assert rest_mode.mode_id == orig_mode.mode_id
        assert rest_mode.rank == orig_mode.rank
        assert rest_mode.free_energy == orig_mode.free_energy
        assert rest_mode.enthalpy == orig_mode.enthalpy
        assert rest_mode.entropy == orig_mode.entropy
        assert rest_mode.heat_capacity == orig_mode.heat_capacity
        assert rest_mode.std_energy == orig_mode.std_energy
        assert rest_mode.best_cf == orig_mode.best_cf
        assert rest_mode.temperature == orig_mode.temperature


def test_from_json_round_trip_file(tmp_path: Path) -> None:
    """to_json(path) → from_json(path) round-trips via a file on disk."""
    _write_pdb(
        tmp_path / "mode_1_pose_1.pdb",
        [
            "binding_mode = 1",
            "pose_rank = 1",
            "CF = -10.0",
            "free_energy = -9.5",
            "temperature = 310.0",
        ],
    )

    original = load_results(tmp_path)
    json_path = tmp_path / "results.json"
    original.to_json(json_path)

    restored = DockingResult.from_json(json_path)

    assert restored.n_modes == 1
    assert restored.temperature == 310.0
    assert restored.binding_modes[0].free_energy == -9.5
    assert restored.binding_modes[0].best_cf == -10.0


def test_from_json_empty_modes() -> None:
    """from_json handles results with no binding modes."""
    payload = json.dumps({
        "source_dir": "/tmp/empty",
        "temperature": 300.0,
        "n_modes": 0,
        "metadata": {},
        "binding_modes": [],
    })
    restored = DockingResult.from_json(payload)
    assert restored.n_modes == 0
    assert restored.temperature == 300.0
