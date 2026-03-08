"""File I/O utilities for FlexAID∆S.

Provides readers and writers for PDB, MOL2, and FlexAID config files.
"""

import re
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class Atom:
    """Parsed ATOM/HETATM record from a PDB file."""
    serial: int
    name: str
    altloc: str
    resname: str
    chainid: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float
    occupancy: float
    bfactor: float
    element: str
    record: str  # 'ATOM' or 'HETATM'

    @property
    def coords(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class PDBStructure:
    """Parsed PDB structure."""
    atoms: List[Atom] = field(default_factory=list)
    remarks: List[str] = field(default_factory=list)
    title: str = ""

    @property
    def coords(self) -> np.ndarray:
        """All atomic coordinates as (N, 3) array."""
        return np.array([[a.x, a.y, a.z] for a in self.atoms])

    def select_chain(self, chain_id: str) -> "PDBStructure":
        """Return new structure with only atoms from specified chain."""
        filtered = [a for a in self.atoms if a.chainid == chain_id]
        return PDBStructure(atoms=filtered, title=self.title)

    def select_residue(self, resseq: int, chain_id: str = "") -> List[Atom]:
        """Return atoms from a specific residue."""
        return [
            a for a in self.atoms
            if a.resseq == resseq and (not chain_id or a.chainid == chain_id)
        ]

    def get_chain_ids(self) -> List[str]:
        """Return list of unique chain IDs."""
        seen = []
        for a in self.atoms:
            if a.chainid not in seen:
                seen.append(a.chainid)
        return seen


def read_pdb(path: str) -> PDBStructure:
    """Parse a PDB file into a PDBStructure.

    Args:
        path: Path to PDB file.

    Returns:
        Parsed PDBStructure.
    """
    p = Path(path)
    structure = PDBStructure()

    with open(p) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec == "TITLE":
                structure.title = line[10:].strip()
            elif rec == "REMARK":
                structure.remarks.append(line[6:].strip())
            elif rec in ("ATOM", "HETATM"):
                try:
                    atom = Atom(
                        record=rec,
                        serial=int(line[6:11]),
                        name=line[12:16].strip(),
                        altloc=line[16].strip(),
                        resname=line[17:20].strip(),
                        chainid=line[21].strip(),
                        resseq=int(line[22:26]),
                        icode=line[26].strip(),
                        x=float(line[30:38]),
                        y=float(line[38:46]),
                        z=float(line[46:54]),
                        occupancy=float(line[54:60]) if len(line) > 54 else 1.0,
                        bfactor=float(line[60:66]) if len(line) > 60 else 0.0,
                        element=line[76:78].strip() if len(line) > 76 else "",
                    )
                    structure.atoms.append(atom)
                except (ValueError, IndexError):
                    continue

    return structure


def write_pdb(structure: PDBStructure, path: str) -> None:
    """Write a PDBStructure to a PDB file.

    Args:
        structure: Structure to write.
        path: Output file path.
    """
    with open(path, "w") as fh:
        if structure.title:
            fh.write(f"TITLE     {structure.title}\n")
        for remark in structure.remarks:
            fh.write(f"REMARK{remark}\n")
        for atom in structure.atoms:
            fh.write(
                f"{atom.record:<6}{atom.serial:5d} {atom.name:<4}{atom.altloc:1}"
                f"{atom.resname:<3} {atom.chainid:1}{atom.resseq:4d}{atom.icode:1}   "
                f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
                f"{atom.occupancy:6.2f}{atom.bfactor:6.2f}"
                f"          {atom.element:>2}\n"
            )
        fh.write("END\n")


def read_flexaid_config(path: str) -> Dict[str, object]:
    """Parse a FlexAID .inp configuration file into a dictionary.

    The FlexAID config format uses 6-character keywords followed by values.
    Keywords that may appear multiple times (OPTIMZ, FLEXSC) are collected
    into lists.

    Args:
        path: Path to .inp config file.

    Returns:
        Dictionary mapping keyword -> value.
    """
    _list_keys = {"OPTIMZ", "FLEXSC"}
    config: Dict[str, object] = {k: [] for k in _list_keys}

    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if len(line) < 6:
                continue
            keyword = line[:6].strip()
            value = line[7:].strip() if len(line) > 7 else ""

            if keyword in _list_keys:
                config[keyword].append(value)  # type: ignore[attr-defined]
            else:
                # Attempt numeric conversion
                try:
                    config[keyword] = int(value)
                except ValueError:
                    try:
                        config[keyword] = float(value)
                    except ValueError:
                        config[keyword] = value if value else True

    return config


def write_flexaid_config(config: Dict[str, object], path: str) -> None:
    """Write a FlexAID configuration dictionary to a .inp file.

    Args:
        config: Dictionary of keyword -> value mappings.
        path: Output file path.
    """
    _list_keys = {"OPTIMZ", "FLEXSC"}
    with open(path, "w") as fh:
        for keyword, value in config.items():
            if keyword in _list_keys and isinstance(value, list):
                for item in value:
                    fh.write(f"{keyword:<6} {item}\n")
            elif value is True:
                fh.write(f"{keyword}\n")
            elif value is not False and value is not None:
                fh.write(f"{keyword:<6} {value}\n")


@dataclass
class SphereRecord:
    """Sphere record from a FlexAID sphere/cleft PDB file."""
    x: float
    y: float
    z: float
    radius: float
    cleft_id: int = 1

    @property
    def coords(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


def read_sphere_pdb(path: str) -> List[SphereRecord]:
    """Parse a FlexAID sphere PDB (B-factor = radius, resSeq = cleft_id).

    Args:
        path: Path to sphere PDB file.

    Returns:
        List of SphereRecord objects.
    """
    spheres: List[SphereRecord] = []
    with open(path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                radius = float(line[60:66]) if len(line) > 60 else 1.0
                cleft_id = int(line[22:26]) if len(line) > 22 else 1
                spheres.append(SphereRecord(x=x, y=y, z=z,
                                            radius=radius, cleft_id=cleft_id))
            except (ValueError, IndexError):
                continue
    return spheres
