"""Tests for the experimental flexaidds.flexmolview module."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from flexaidds.flexmolview import FlexMolView


def _write_demo_pdb(path: Path) -> None:
    lines = [
        "ATOM      1  N   ALA A  10       0.000   0.000   0.000  1.00 10.00           N\n",
        "ATOM      2  CA  ALA A  10       1.000   0.000   0.000  1.00 10.00           C\n",
        "ATOM      3  C   GLY A  11       2.000   0.000   0.000  1.00 10.00           C\n",
        "ATOM      4  O   GLY A  11       3.000   0.000   0.000  1.00 10.00           O\n",
        "HETATM    5  C1  LIG B 101       0.000   2.000   0.000  1.00 20.00           C\n",
        "HETATM    6  O1  HOH B 201       0.000   3.000   0.000  1.00 30.00           O\n",
        "HETATM    7  ZN  ZN  B 301       0.000   4.000   0.000  1.00 40.00          ZN\n",
        "END\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


@pytest.fixture()
def loaded_view(tmp_path: Path) -> tuple[FlexMolView, str]:
    pdb = tmp_path / "demo.pdb"
    _write_demo_pdb(pdb)
    view = FlexMolView()
    name = view.load_pdb(str(pdb), object_name="demo")
    return view, name


def test_load_and_summary(loaded_view: tuple[FlexMolView, str]) -> None:
    view, name = loaded_view
    summary = view.summary(name)
    assert summary["atom_count"] == 7
    assert summary["residue_count"] == 5
    assert summary["chains"] == ["A", "B"]
    assert summary["visible_representations"] == ["lines"]


def test_chain_selection(loaded_view: tuple[FlexMolView, str]) -> None:
    view, name = loaded_view
    sel = view.select("chain A", name)
    assert sel.count == 4


def test_resi_range_selection(loaded_view: tuple[FlexMolView, str]) -> None:
    view, name = loaded_view
    assert view.count_atoms(name, "resi 10-11") == 4
    assert view.count_residues(name, "resi 10-11") == 2


def test_boolean_selection(loaded_view: tuple[FlexMolView, str]) -> None:
    view, name = loaded_view
    sel = view.select("polymer and not name O", name)
    assert sel.count == 3


def test_ligand_solvent_and_ion_partition(loaded_view: tuple[FlexMolView, str]) -> None:
    view, name = loaded_view
    assert view.count_atoms(name, "ligand") == 1
    assert view.count_atoms(name, "solvent") == 1
    # Zn ion must not be counted as ligand or solvent
    assert view.count_atoms(name, "not (polymer or ligand or solvent)") == 1


def test_named_selection_round_trip(loaded_view: tuple[FlexMolView, str]) -> None:
    view, name = loaded_view
    result = view.select("resn LIG", name, name="lig")
    assert result.count == 1
    restored = view.get_selection("lig")
    assert restored.atom_indices == result.atom_indices
    assert restored.object_name == name


def test_show_hide_and_as_representation(loaded_view: tuple[FlexMolView, str]) -> None:
    view, name = loaded_view
    view.show("sticks", name)
    assert view.get_visible_representations(name) == {"lines", "sticks"}
    view.hide("lines", name)
    assert view.get_visible_representations(name) == {"sticks"}
    view.as_representation("cartoon", name)
    assert view.get_visible_representations(name) == {"cartoon"}


def test_center_and_bounds(loaded_view: tuple[FlexMolView, str]) -> None:
    view, name = loaded_view
    center = view.center_of_geometry(name, "polymer")
    assert center == pytest.approx((1.5, 0.0, 0.0))
    lower, upper = view.bounds(name, "ligand or solvent")
    assert lower == pytest.approx((0.0, 2.0, 0.0))
    assert upper == pytest.approx((0.0, 3.0, 0.0))


def test_distance_by_serial(loaded_view: tuple[FlexMolView, str]) -> None:
    view, name = loaded_view
    distance = view.distance_by_serial(name, 1, 5)
    assert distance == pytest.approx(math.sqrt(4.0))


def test_invalid_representation_rejected(loaded_view: tuple[FlexMolView, str]) -> None:
    view, name = loaded_view
    with pytest.raises(ValueError):
        view.show("ribbonz", name)


def test_empty_selection_raises_for_geometry(loaded_view: tuple[FlexMolView, str]) -> None:
    view, name = loaded_view
    with pytest.raises(ValueError):
        view.center_of_geometry(name, "chain Z")
