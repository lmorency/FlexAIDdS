"""pytest configuration and shared fixtures for FlexAID∆S Python tests."""

from __future__ import annotations

import math
import tempfile
import textwrap
from pathlib import Path
from typing import List

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Check whether the compiled C++ extension is available
# ─────────────────────────────────────────────────────────────────────────────

def _has_core() -> bool:
    try:
        from flexaidds import _core
        return _core is not None
    except ImportError:
        return False


requires_core = pytest.mark.skipif(
    not _has_core(),
    reason="_core C++ extension not built; run 'pip install -e python/' first",
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Temporary directory cleaned up after each test."""
    return tmp_path


@pytest.fixture
def sample_energies() -> List[float]:
    """Small set of CF energies (kcal/mol) for ensemble tests."""
    return [-12.5, -11.8, -12.1, -10.9, -13.0, -11.5]


@pytest.fixture
def simple_pdb_file(tmp_path: Path) -> Path:
    """Minimal PDB file with three atoms."""
    content = textwrap.dedent("""\
        ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
        ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00  0.00           C
        HETATM    3  C1  LIG B   1       5.000   6.000   7.000  1.00  0.00           C
        END
    """)
    pdb = tmp_path / "test.pdb"
    pdb.write_text(content)
    return pdb


@pytest.fixture
def simple_mol2_file(tmp_path: Path) -> Path:
    """Minimal MOL2 file with two atoms."""
    content = textwrap.dedent("""\
        @<TRIPOS>MOLECULE
        test_ligand
        2 1 0 0 0
        SMALL
        GASTEIGER

        @<TRIPOS>ATOM
              1 C1          0.0000    0.0000    0.0000 C.3     1  LIG1        0.0000
              2 O1          1.4000    0.0000    0.0000 O.3     1  LIG1       -0.3982
        @<TRIPOS>BOND
             1     1     2    1
    """)
    mol2 = tmp_path / "test.mol2"
    mol2.write_text(content)
    return mol2


@pytest.fixture
def simple_rrd_file(tmp_path: Path) -> Path:
    """Minimal RRD file with three poses (no RMSD column)."""
    content = textwrap.dedent("""\
        # pose_id  cf_score  mode_id  chromosome
        1  -12.500  1  0.5 0.3 0.1
        2  -11.800  1  0.6 0.2 0.1
        3  -10.200  2  0.1 0.9 0.5
    """)
    rrd = tmp_path / "results.rrd"
    rrd.write_text(content)
    return rrd


@pytest.fixture
def rrd_with_rmsd_file(tmp_path: Path) -> Path:
    """RRD file that includes an RMSD column."""
    content = textwrap.dedent("""\
        1  -12.500  0.45  1  0.5 0.3
        2  -11.800  1.20  1  0.6 0.2
        3  -10.200  2.80  2  0.1 0.9
    """)
    rrd = tmp_path / "results_rmsd.rrd"
    rrd.write_text(content)
    return rrd


@pytest.fixture
def flexaid_config_file(tmp_path: Path) -> Path:
    """Minimal FlexAID .inp configuration file."""
    content = textwrap.dedent("""\
        # FlexAID test configuration
        PDBNAM /data/receptor.pdb
        INPLIG /data/ligand.mol2
        METOPT GA
        TEMPER 300
        OPTIMZ -12.0
        OPTIMZ -11.5
        EXCHET
        NRGOUT 10
        ACSWEI 0.75
    """)
    cfg = tmp_path / "test.inp"
    cfg.write_text(content)
    return cfg


@pytest.fixture
def encom_files(tmp_path: Path):
    """Pair of ENCoM eigenvalue/eigenvector files (6 modes, 2 atoms → 6 components each)."""
    eigenvalues = "0.0\n0.0\n0.0\n0.0\n0.0\n0.0\n1.23\n4.56\n9.10\n"
    eigenvectors = "\n".join(
        " ".join(f"{(i + j * 0.1):.4f}" for j in range(6))
        for i in range(9)
    ) + "\n"
    ev_file  = tmp_path / "eigenvalues.txt"
    evc_file = tmp_path / "eigenvectors.txt"
    ev_file.write_text(eigenvalues)
    evc_file.write_text(eigenvectors)
    return ev_file, evc_file
