from pathlib import Path

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

    result = load_results(tmp_path)
    mode = result.binding_modes[0]
    pose = mode.poses[0]

    assert mode.mode_id == 7
    assert pose.pose_rank == 3
    assert mode.best_cf == -11.5
