"""Tests for io.py PDB read/write, FlexAID config I/O, and sphere PDB parsing.

These are pure-Python tests that exercise the general-purpose I/O functions
in flexaidds.io which were previously untested.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from flexaidds.io import (
    Atom,
    PDBStructure,
    SphereRecord,
    read_flexaid_config,
    read_pdb,
    read_sphere_pdb,
    write_flexaid_config,
    write_pdb,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

SAMPLE_PDB = """\
TITLE     Test Structure
REMARK Free_energy = -12.5
REMARK Temperature: 300
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 10.00           C
ATOM      3  C   ALA A   1       3.000   4.000   5.000  0.50  5.00           C
HETATM    4  O   HOH A   2       5.000   6.000   7.000  1.00  0.00           O
END
"""

SAMPLE_SPHERE_PDB = """\
REMARK  Cleft spheres
HETATM    1  SPH SURF    1       1.000   2.000   3.000  1.00  2.50           S
HETATM    2  SPH SURF    1       4.000   5.000   6.000  1.00  1.80           S
HETATM    3  SPH SURF    2       7.000   8.000   9.000  1.00  3.20           S
END
"""

SAMPLE_CONFIG = """\
# FlexAID configuration
PDBNAM receptor.pdb
INPLIG ligand.mol2
TEMPRT 300.0
MAXGEN 1000
NCHROM 100
OPTIMZ ALA_A_42
OPTIMZ GLY_B_15
FLEXSC VAL_A_100
FLEXSC LEU_A_101
DTEFRE
"""


@pytest.fixture
def pdb_path(tmp_path):
    p = tmp_path / "test.pdb"
    p.write_text(SAMPLE_PDB)
    return str(p)


@pytest.fixture
def sphere_path(tmp_path):
    p = tmp_path / "spheres.pdb"
    p.write_text(SAMPLE_SPHERE_PDB)
    return str(p)


@pytest.fixture
def config_path(tmp_path):
    p = tmp_path / "config.inp"
    p.write_text(SAMPLE_CONFIG)
    return str(p)


# ─── read_pdb ────────────────────────────────────────────────────────────────


class TestReadPDB:
    def test_reads_title(self, pdb_path):
        struct = read_pdb(pdb_path)
        assert struct.title == "Test Structure"

    def test_reads_remarks(self, pdb_path):
        struct = read_pdb(pdb_path)
        assert len(struct.remarks) == 2
        assert "Free_energy" in struct.remarks[0]

    def test_reads_atom_count(self, pdb_path):
        struct = read_pdb(pdb_path)
        assert len(struct.atoms) == 4

    def test_reads_atom_records(self, pdb_path):
        struct = read_pdb(pdb_path)
        atom1 = struct.atoms[0]
        assert atom1.record == "ATOM"
        assert atom1.serial == 1
        assert atom1.name == "N"
        assert atom1.resname == "ALA"
        assert atom1.chainid == "A"
        assert atom1.resseq == 1

    def test_reads_hetatm(self, pdb_path):
        struct = read_pdb(pdb_path)
        hetatm = struct.atoms[3]
        assert hetatm.record == "HETATM"
        assert hetatm.name == "O"
        assert hetatm.resname == "HOH"

    def test_reads_coordinates(self, pdb_path):
        struct = read_pdb(pdb_path)
        atom1 = struct.atoms[0]
        assert pytest.approx(atom1.x, abs=0.001) == 1.0
        assert pytest.approx(atom1.y, abs=0.001) == 2.0
        assert pytest.approx(atom1.z, abs=0.001) == 3.0

    def test_reads_bfactor(self, pdb_path):
        struct = read_pdb(pdb_path)
        assert pytest.approx(struct.atoms[1].bfactor, abs=0.01) == 10.0

    def test_reads_occupancy(self, pdb_path):
        struct = read_pdb(pdb_path)
        assert pytest.approx(struct.atoms[2].occupancy, abs=0.01) == 0.50

    def test_reads_element(self, pdb_path):
        struct = read_pdb(pdb_path)
        assert struct.atoms[0].element == "N"
        assert struct.atoms[1].element == "C"

    def test_coords_property(self, pdb_path):
        struct = read_pdb(pdb_path)
        coords = struct.coords
        assert coords.shape == (4, 3)
        np.testing.assert_allclose(coords[0], [1.0, 2.0, 3.0], atol=0.001)

    def test_atom_coords_property(self, pdb_path):
        struct = read_pdb(pdb_path)
        c = struct.atoms[0].coords
        assert isinstance(c, np.ndarray)
        assert c.shape == (3,)


# ─── PDBStructure methods ───────────────────────────────────────────────────


class TestPDBStructure:
    def test_select_chain(self, pdb_path):
        struct = read_pdb(pdb_path)
        chain_a = struct.select_chain("A")
        assert len(chain_a.atoms) == 4  # all atoms are chain A

    def test_select_chain_empty(self, pdb_path):
        struct = read_pdb(pdb_path)
        chain_b = struct.select_chain("B")
        assert len(chain_b.atoms) == 0

    def test_select_residue(self, pdb_path):
        struct = read_pdb(pdb_path)
        res1 = struct.select_residue(1)
        assert len(res1) == 3  # N, CA, C of ALA-1

    def test_select_residue_with_chain(self, pdb_path):
        struct = read_pdb(pdb_path)
        res1 = struct.select_residue(1, chain_id="A")
        assert len(res1) == 3

    def test_get_chain_ids(self, pdb_path):
        struct = read_pdb(pdb_path)
        chains = struct.get_chain_ids()
        assert chains == ["A"]


# ─── write_pdb ───────────────────────────────────────────────────────────────


class TestWritePDB:
    def test_roundtrip(self, pdb_path, tmp_path):
        struct = read_pdb(pdb_path)
        out_path = str(tmp_path / "output.pdb")
        write_pdb(struct, out_path)

        reread = read_pdb(out_path)
        assert len(reread.atoms) == len(struct.atoms)
        assert reread.title == struct.title

    def test_coordinates_preserved(self, pdb_path, tmp_path):
        struct = read_pdb(pdb_path)
        out_path = str(tmp_path / "output.pdb")
        write_pdb(struct, out_path)

        reread = read_pdb(out_path)
        for orig, new in zip(struct.atoms, reread.atoms):
            assert pytest.approx(orig.x, abs=0.01) == new.x
            assert pytest.approx(orig.y, abs=0.01) == new.y
            assert pytest.approx(orig.z, abs=0.01) == new.z

    def test_end_record_present(self, tmp_path):
        struct = PDBStructure(atoms=[], title="Empty")
        out_path = str(tmp_path / "empty.pdb")
        write_pdb(struct, out_path)

        content = open(out_path).read()
        assert "END" in content

    def test_remarks_written(self, pdb_path, tmp_path):
        struct = read_pdb(pdb_path)
        out_path = str(tmp_path / "output.pdb")
        write_pdb(struct, out_path)

        content = open(out_path).read()
        assert "REMARK" in content


# ─── read_sphere_pdb ─────────────────────────────────────────────────────────


class TestReadSpherePDB:
    def test_reads_sphere_count(self, sphere_path):
        spheres = read_sphere_pdb(sphere_path)
        assert len(spheres) == 3

    def test_reads_coordinates(self, sphere_path):
        spheres = read_sphere_pdb(sphere_path)
        assert pytest.approx(spheres[0].x, abs=0.001) == 1.0
        assert pytest.approx(spheres[0].y, abs=0.001) == 2.0
        assert pytest.approx(spheres[0].z, abs=0.001) == 3.0

    def test_reads_radius(self, sphere_path):
        spheres = read_sphere_pdb(sphere_path)
        assert pytest.approx(spheres[0].radius, abs=0.01) == 2.50
        assert pytest.approx(spheres[2].radius, abs=0.01) == 3.20

    def test_reads_cleft_id(self, sphere_path):
        spheres = read_sphere_pdb(sphere_path)
        assert spheres[0].cleft_id == 1
        assert spheres[2].cleft_id == 2

    def test_sphere_coords_property(self, sphere_path):
        spheres = read_sphere_pdb(sphere_path)
        c = spheres[0].coords
        assert isinstance(c, np.ndarray)
        np.testing.assert_allclose(c, [1.0, 2.0, 3.0], atol=0.001)

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.pdb"
        p.write_text("REMARK empty\nEND\n")
        spheres = read_sphere_pdb(str(p))
        assert len(spheres) == 0


# ─── read_flexaid_config ─────────────────────────────────────────────────────


class TestReadFlexaidConfig:
    def test_reads_string_value(self, config_path):
        config = read_flexaid_config(config_path)
        assert config["PDBNAM"] == "receptor.pdb"

    def test_reads_float_value(self, config_path):
        config = read_flexaid_config(config_path)
        assert pytest.approx(config["TEMPRT"], abs=0.1) == 300.0

    def test_reads_int_value(self, config_path):
        config = read_flexaid_config(config_path)
        assert config["MAXGEN"] == 1000
        assert config["NCHROM"] == 100

    def test_reads_list_values(self, config_path):
        config = read_flexaid_config(config_path)
        assert isinstance(config["OPTIMZ"], list)
        assert len(config["OPTIMZ"]) == 2
        assert "ALA_A_42" in config["OPTIMZ"]

    def test_reads_flexsc_list(self, config_path):
        config = read_flexaid_config(config_path)
        assert isinstance(config["FLEXSC"], list)
        assert len(config["FLEXSC"]) == 2

    def test_reads_flag_as_true(self, config_path):
        config = read_flexaid_config(config_path)
        assert config["DTEFRE"] is True

    def test_skips_comments(self, config_path):
        config = read_flexaid_config(config_path)
        # No key should start with '#'
        for key in config:
            assert not key.startswith("#")


# ─── write_flexaid_config ────────────────────────────────────────────────────


class TestWriteFlexaidConfig:
    def test_roundtrip(self, config_path, tmp_path):
        config = read_flexaid_config(config_path)
        out_path = str(tmp_path / "output.inp")
        write_flexaid_config(config, out_path)

        reread = read_flexaid_config(out_path)
        assert reread["PDBNAM"] == config["PDBNAM"]
        assert reread["MAXGEN"] == config["MAXGEN"]

    def test_list_values_roundtrip(self, config_path, tmp_path):
        config = read_flexaid_config(config_path)
        out_path = str(tmp_path / "output.inp")
        write_flexaid_config(config, out_path)

        reread = read_flexaid_config(out_path)
        assert len(reread["OPTIMZ"]) == len(config["OPTIMZ"])
        assert set(reread["OPTIMZ"]) == set(config["OPTIMZ"])

    def test_flag_values_roundtrip(self, config_path, tmp_path):
        config = read_flexaid_config(config_path)
        out_path = str(tmp_path / "output.inp")
        write_flexaid_config(config, out_path)

        reread = read_flexaid_config(out_path)
        assert reread["DTEFRE"] is True
