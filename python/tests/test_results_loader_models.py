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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def test_load_results_records_top_mode_and_recursive_count(tmp_path: Path) -> None:
    _write_pdb(
        tmp_path / "nested" / "binding_mode_1_pose_1.pdb",
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
        tmp_path / "nested" / "binding_mode_1_pose_2.pdb",
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
    records = result.to_records()

    assert result.metadata["n_pose_files"] == 3
    assert result.top_mode().mode_id == 1
    assert len(records) == 2
    assert records[0]["mode_id"] == 1
    assert records[0]["n_poses"] == 2
    assert records[0]["best_cf"] == -42.5
    assert records[0]["best_pose_path"].endswith("binding_mode_1_pose_1.pdb")


def test_load_results_promotes_frequency_metadata_to_mode_level(tmp_path: Path) -> None:
    _write_pdb(
        tmp_path / "binding_mode_3_pose_1.pdb",
        [
            "binding_mode = 3",
            "pose_rank = 1",
            "CF = -20.0",
            "frequency = 7",
            "temperature = 298.15",
        ],
    )
    _write_pdb(
        tmp_path / "binding_mode_3_pose_2.pdb",
        [
            "binding_mode = 3",
            "pose_rank = 2",
            "CF = -18.0",
            "frequency = 7",
            "temperature = 298.15",
        ],
    )

    result = load_results(tmp_path)
    mode = result.binding_modes[0]

    assert mode.mode_id == 3
    assert mode.n_poses == 2
    assert mode.frequency == 7
    assert mode.metadata["frequency"] == 7
